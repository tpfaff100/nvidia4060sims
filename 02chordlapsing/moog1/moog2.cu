/*
 * moog.cu — GPU Minimoog-style synthesizer, fully parameterized
 * Target: RTX 4060 Ti (sm_89)
 *
 * BUILD (after vcvars64.bat):
 *   nvcc -arch=sm_89 -O3 -use_fast_math -o moog moog.cu
 * RUN:
 *   moog                          -> renders built-in demo patch to moog.wav
 *   moog out.wav mypatch.txt      -> renders your patch file
 *   moog out.wav mypatch.txt 96000
 *
 * ARCHITECTURE
 *   Voice-per-thread: every note in the sequence is rendered by ONE CUDA
 *   thread running the complete synth voice sequentially (oscillators ->
 *   FM/ring routing -> nonlinear ladder filter -> VCA), all notes in
 *   parallel. Chorus + reverb are a global stereo bus applied after.
 *
 * FEATURES (all patchable)
 *   • up to 10 oscillators: saw / square (PW) / triangle / sine / noise,
 *     polyBLEP anti-aliased, per-osc semitone + cents detune + level
 *   • FM synthesis: any osc can be phase-modulated by any earlier osc
 *   • RING MOD: any osc multiplied by any earlier osc
 *   • LFO: sine/tri/square, routed to pitch (vibrato) and/or cutoff
 *   • SAMPLE & HOLD: clocked random, routed to pitch and/or cutoff
 *   • MOOG LADDER FILTER: 4-pole 24 dB/oct nonlinear transistor-ladder
 *     model (Huovilainen-style tanh stages), 2x oversampled, resonance
 *     to self-oscillation, filter ADSR + key tracking, input drive
 *     (NOTE: you asked for "Kleiner-Perkins" filtering — that's the VC
 *      firm :) — this is the Moog LADDER, the Minimoog's actual filter.
 *      Sallen-Key, the other classic topology, is the MS-20 sound.)
 *   • amp ADSR, glide/portamento (mono-style, from previous note)
 *   • TREMOLO (dedicated amp LFO), CHORUS (dual modulated delay),
 *     REVERB (Schroeder)
 *
 * PATCH FILE FORMAT (text, one command per line, # comments):
 *   osc    <i> <wave: saw|sqr|tri|sin|noise> <semi> <cents> <level>
 *   pw     <i> <0.05..0.95>          pulse width for sqr
 *   fm     <carrier_i> <mod_i> <index 0..10>    (mod_i < carrier_i)
 *   ring   <i> <src_i>                           (src_i < i)
 *   lfo    <rateHz> <wave: sin|tri|sqr> <toPitch semitones> <toCutoff 0..1>
 *   sh     <rateHz> <toPitch semitones> <toCutoff 0..1>
 *   filter <cutoffHz> <res 0..1.1> <envAmt 0..1> <keytrack 0..1> <drive 0.5..4>
 *   aenv   <A> <D> <S> <R>            seconds / sustain 0..1
 *   fenv   <A> <D> <S> <R>
 *   glide  <seconds>
 *   trem   <rateHz> <depth 0..1>
 *   chorus <rateHz> <depth s> <mix 0..1>
 *   reverb <size 0..1> <wet 0..1>
 *   note   <start_s> <dur_s> <midi> <vel 0..1>
 */

#define _USE_MATH_DEFINES
#include <cuda_runtime.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define SR_DEFAULT 48000
#define TWO_PI     6.28318530717958647692f
#define MAX_OSC    10
#define MAX_NOTES  1024
#define MAX_PATCH  8

#define CUDA_CHECK(x) do{ cudaError_t _=x; if(_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_)); \
    exit(1);} }while(0)

enum Wave { W_SAW=0, W_SQR, W_TRI, W_SIN, W_NOISE };
enum LWave{ L_SIN=0, L_TRI, L_SQR };

struct OscP {
    int   wave;
    float semi, cents, level, pw;
    int   fmSrc;   float fmAmt;    /* -1 = none                        */
    int   ringSrc;                  /* -1 = none                        */
};
struct Patch {
    int   nOsc;
    OscP  osc[MAX_OSC];
    float lfoRate; int lfoWave; float lfoToPitch, lfoToCut;
    float shRate, shToPitch, shToCut;
    float cutoff, res, fenvAmt, keytrack, drive;
    float aA,aD,aS,aR;
    float fA,fD,fS,fR;
    float glide;
    float tremRate, tremDepth;
    float chRate, chDepth, chMix;
    float rvSize, rvWet;
};
struct Note { float t, dur, vel; int midi, prevMidi, patch; };

/* ── device helpers ──────────────────────────────────────────────────*/
__device__ __forceinline__ float hnoise(unsigned i,unsigned s){
    unsigned h=i*0x9E3779B9u^s*0x85EBCA6Bu;
    h^=h>>13; h*=0xC2B2AE35u; h^=h>>16;
    return (float)(int)h*4.6566129e-10f;
}
__device__ __forceinline__ float ftanh(float x){   /* Padé tanh        */
    float x2=x*x;
    return x*(27.f+x2)/(27.f+9.f*x2);
}
__device__ __forceinline__ float polyblep(float t,float dt){
    if(t<dt){ t/=dt; return t+t-t*t-1.f; }
    if(t>1.f-dt){ t=(t-1.f)/dt; return t*t+t+t+1.f; }
    return 0.f;
}
__device__ float adsr(float t,float dur,float A,float D,float S,float R){
    if(t<0.f)return 0.f;
    float e;
    if(t<A)e=t/fmaxf(A,1e-4f);
    else if(t<A+D)e=1.f-(t-A)/fmaxf(D,1e-4f)*(1.f-S);
    else if(t<dur)e=S;
    else{float tr=t-dur; if(tr>R)return 0.f; e=S*(1.f-tr/fmaxf(R,1e-4f));}
    return fmaxf(0.f,fminf(1.f,e));
}

/* ═══════════════ VOICE KERNEL: one thread = one note ═══════════════ */
__global__ void voiceKernel(
    const Patch*__restrict__ pp,
    const Note*__restrict__ notes,int nNotes,
    float*bus,int N,float sr,unsigned seed)
{
    int vi=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if(vi>=nNotes)return;
    const Note nt=notes[vi];
    const Patch P=pp[nt.patch];

    float invSr=1.f/sr;
    int   startSamp=(int)(nt.t*sr);
    float totalDur=nt.dur+P.aR+0.05f;
    int   nSamp=(int)(totalDur*sr);

    /* per-voice state                                                  */
    float ph[MAX_OSC]; float oscOut[MAX_OSC];
    for(int i=0;i<MAX_OSC;i++){ph[i]=hnoise((unsigned)(vi*17+i),seed)*0.5f+0.5f;
                                oscOut[i]=0.f;}
    /* ladder state                                                     */
    float s1=0,s2=0,s3=0,s4=0;

    float baseHz=440.f*exp2f(((float)nt.midi-69.f)/12.f);
    float prevHz=440.f*exp2f(((float)nt.prevMidi-69.f)/12.f);

    for(int i=0;i<nSamp;i++){
        float t=(float)i*invSr;
        float tAbs=nt.t+t;

        /* ── glide ─────────────────────────────────────────────────── */
        float gl=(P.glide>1e-4f)? fminf(1.f,t/P.glide):1.f;
        float noteHz=prevHz*powf(baseHz/prevHz,gl);

        /* ── LFO (free-running on absolute time) ───────────────────── */
        float lph=P.lfoRate*tAbs; lph-=floorf(lph);
        float lfo;
        if(P.lfoWave==L_SIN)      lfo=__sinf(TWO_PI*lph);
        else if(P.lfoWave==L_TRI) lfo=4.f*fabsf(lph-0.5f)-1.f;
        else                      lfo=(lph<0.5f)?1.f:-1.f;

        /* ── SAMPLE & HOLD (clocked random) ────────────────────────── */
        float sh=0.f;
        if(P.shRate>0.01f)
            sh=hnoise((unsigned)floorf(tAbs*P.shRate),seed^0x5AAD);

        /* pitch modulation (semitones -> ratio)                        */
        float pitchMod=exp2f((P.lfoToPitch*lfo+P.shToPitch*sh)/12.f);
        float f0=noteHz*pitchMod;

        /* ── oscillators (in index order so FM/ring see this sample) ─ */
        float mix=0.f;
        for(int o=0;o<P.nOsc;o++){
            const OscP&op=P.osc[o];
            float f=f0*exp2f((op.semi+op.cents*0.01f)/12.f);
            float dt=f*invSr;

            /* FM: phase modulation by an earlier oscillator            */
            float pmod=0.f;
            if(op.fmSrc>=0&&op.fmSrc<o)
                pmod=op.fmAmt*oscOut[op.fmSrc]*0.5f;

            ph[o]+=dt; ph[o]-=floorf(ph[o]);
            float p=ph[o]+pmod; p-=floorf(p); if(p<0.f)p+=1.f;

            float v;
            if(op.wave==W_SAW){
                v=2.f*p-1.f-polyblep(p,dt);
            }else if(op.wave==W_SQR){
                float pw=op.pw;
                v=(p<pw?1.f:-1.f);
                v+=polyblep(p,dt);
                float p2=p-pw; if(p2<0.f)p2+=1.f;
                v-=polyblep(p2,dt);
            }else if(op.wave==W_TRI){
                v=4.f*fabsf(p-0.5f)-1.f;
            }else if(op.wave==W_SIN){
                v=__sinf(TWO_PI*p);
            }else{ /* noise */
                v=hnoise((unsigned)(i*MAX_OSC+o),seed^(unsigned)(vi*31));
            }

            /* ring mod                                                 */
            if(op.ringSrc>=0&&op.ringSrc<o) v*=oscOut[op.ringSrc];

            oscOut[o]=v;
            mix+=v*op.level;
        }

        /* ── filter cutoff modulation ──────────────────────────────── */
        float fenv=adsr(t,nt.dur,P.fA,P.fD,P.fS,P.fR);
        float cut=P.cutoff
            *exp2f( P.fenvAmt*fenv*5.f            /* env: up to +5 oct */
                   +P.lfoToCut*lfo*2.f
                   +P.shToCut*sh*2.f
                   +P.keytrack*log2f(noteHz/261.63f));
        cut=fmaxf(20.f,fminf(0.45f*sr,cut));

        /* ── MOOG LADDER (nonlinear, 2x oversampled) ───────────────── */
        float g=1.f-__expf(-TWO_PI*cut*invSr*0.5f);  /* half-rate step */
        float k=4.f*P.res;
        float x=mix*P.drive;
        float out=s4;
        #pragma unroll
        for(int os=0;os<2;os++){
            float in=ftanh(x-k*s4);
            s1+=g*(in-s1);
            s2+=g*(ftanh(s1)-ftanh(s2));
            s3+=g*(ftanh(s2)-ftanh(s3));
            s4+=g*(ftanh(s3)-ftanh(s4));
        }
        out=s4/fmaxf(0.3f,P.drive*0.7f);             /* gain comp      */

        /* ── VCA: amp ADSR + velocity + tremolo ─────────────────────── */
        float aenv=adsr(t,nt.dur,P.aA,P.aD,P.aS,P.aR);
        if(aenv<=0.f&&t>nt.dur)break;
        float trem=1.f;
        if(P.tremDepth>0.001f){
            float tp=P.tremRate*tAbs; tp-=floorf(tp);
            trem=1.f-P.tremDepth*(0.5f+0.5f*__sinf(TWO_PI*tp));
        }
        float s=out*aenv*nt.vel*trem*0.5f;

        int gI=startSamp+i;
        if(gI>=0&&gI<N) atomicAdd(&bus[gI],s);
    }
}

/* ─────────────────────────────────────────────────────────────────────
   CPU: patch parsing + default demo
   ───────────────────────────────────────────────────────────────────── */
static int waveFromStr(const char*s){
    if(!strncmp(s,"saw",3))return W_SAW;
    if(!strncmp(s,"sqr",3)||!strncmp(s,"squ",3))return W_SQR;
    if(!strncmp(s,"tri",3))return W_TRI;
    if(!strncmp(s,"sin",3))return W_SIN;
    return W_NOISE;
}
static int lwaveFromStr(const char*s){
    if(!strncmp(s,"tri",3))return L_TRI;
    if(!strncmp(s,"sqr",3))return L_SQR;
    return L_SIN;
}

static void defaultPatch(Patch&P,Note*notes,int*nNotes)
{
    memset(&P,0,sizeof(P));
    /* fat 3-saw Minimoog lead + sub sine + FM sparkle osc + ring pair  */
    P.nOsc=6;
    P.osc[0]={W_SAW, 0.f,  -6.f, 0.30f,0.5f,-1,0.f,-1};
    P.osc[1]={W_SAW, 0.f,   0.f, 0.30f,0.5f,-1,0.f,-1};
    P.osc[2]={W_SAW, 0.f,  +7.f, 0.30f,0.5f,-1,0.f,-1};
    P.osc[3]={W_SIN,-12.f,  0.f, 0.35f,0.5f,-1,0.f,-1};   /* sub       */
    P.osc[4]={W_SIN,+12.f,  0.f, 0.10f,0.5f, 1,3.5f,-1};  /* FM'd by 1 */
    P.osc[5]={W_SIN,+19.f,  0.f, 0.06f,0.5f,-1,0.f, 3};   /* ring w/ sub */
    P.lfoRate=5.6f; P.lfoWave=L_SIN; P.lfoToPitch=0.06f; P.lfoToCut=0.05f;
    P.shRate=0.f; P.shToPitch=0.f; P.shToCut=0.f;
    P.cutoff=320.f; P.res=0.62f; P.fenvAmt=0.72f; P.keytrack=0.5f; P.drive=1.4f;
    P.aA=0.004f;P.aD=0.25f;P.aS=0.65f;P.aR=0.25f;
    P.fA=0.002f;P.fD=0.28f;P.fS=0.18f;P.fR=0.3f;
    P.glide=0.045f;
    P.tremRate=4.8f;P.tremDepth=0.0f;
    P.chRate=0.45f;P.chDepth=0.0045f;P.chMix=0.35f;
    P.rvSize=0.7f;P.rvWet=0.16f;

    /* demo line: a funky mono bass/lead sequence, 12.8 s               */
    static const int seq[]={36,36,43,36, 48,46,43,41, 36,36,43,48,
                             51,50,46,43};
    int n=0; float t=0.4f, step=0.4f;
    int prev=36;
    for(int r=0;r<2;r++)
      for(int i=0;i<16;i++){
        float dur=(i%4==3)?step*0.9f:step*0.55f;
        notes[n++]={t,dur,(i%8==0)?1.0f:0.8f,seq[i],prev,0};
        prev=seq[i]; t+=step;
      }
    /* closing held note with full filter bloom                          */
    notes[n++]={t,2.6f,1.0f,36,prev,0};
    *nNotes=n;
}

/* initialise a patch with sane defaults                                 */
static void initPatch(Patch&P){
    memset(&P,0,sizeof(P));
    for(int i=0;i<MAX_OSC;i++){P.osc[i].fmSrc=-1;P.osc[i].ringSrc=-1;
                               P.osc[i].pw=0.5f;}
    P.drive=1.f;P.cutoff=1000.f;P.res=0.3f;
    P.aA=0.005f;P.aD=0.2f;P.aS=0.7f;P.aR=0.2f;
    P.fA=0.005f;P.fD=0.2f;P.fS=0.3f;P.fR=0.2f;
}

/*
 * Multi-patch sequencer format additions:
 *   patch <id>            — following osc/filter/... lines edit patch <id>
 *   tempo <bpm>           — note times/durations below are in BEATS
 *   note  <start> <dur> <midi> <vel> [patch]
 *                         — optional 5th arg selects patch (default:
 *                           the patch currently being edited)
 * FX bus (chorus/reverb) is taken from patch 0.
 */
static int loadPatch(const char*path,Patch*patches,int*nPatchOut,
                     Note*notes,int*nNotes)
{
    FILE*f=fopen(path,"r");
    if(!f)return 0;
    for(int i=0;i<MAX_PATCH;i++)initPatch(patches[i]);
    int nPatch=1, cur=0;
    int prevMidi[MAX_PATCH];
    for(int i=0;i<MAX_PATCH;i++)prevMidi[i]=60;
    float tempo=0.f;                 /* 0 = times in seconds            */
    int n=0;
    char line[256];
    while(fgets(line,sizeof(line),f)){
        char*h=strchr(line,'#'); if(h)*h=0;
        char cmd[32]={0}; if(sscanf(line,"%31s",cmd)!=1)continue;
        Patch&P=patches[cur];
        if(!strcmp(cmd,"patch")){
            int id;
            if(sscanf(line,"patch %d",&id)==1&&id>=0&&id<MAX_PATCH){
                cur=id; if(id+1>nPatch)nPatch=id+1;
            }
        }else if(!strcmp(cmd,"tempo")){
            sscanf(line,"tempo %f",&tempo);
        }else if(!strcmp(cmd,"osc")){
            int i;char w[16];float s,c,l;
            if(sscanf(line,"osc %d %15s %f %f %f",&i,w,&s,&c,&l)==5
               &&i>=0&&i<MAX_OSC){
                P.osc[i].wave=waveFromStr(w);P.osc[i].semi=s;
                P.osc[i].cents=c;P.osc[i].level=l;
                if(i+1>P.nOsc)P.nOsc=i+1;
            }
        }else if(!strcmp(cmd,"pw")){
            int i;float v;
            if(sscanf(line,"pw %d %f",&i,&v)==2&&i>=0&&i<MAX_OSC)
                P.osc[i].pw=v;
        }else if(!strcmp(cmd,"fm")){
            int c2,m;float a;
            if(sscanf(line,"fm %d %d %f",&c2,&m,&a)==3
               &&c2>=0&&c2<MAX_OSC){P.osc[c2].fmSrc=m;P.osc[c2].fmAmt=a;}
        }else if(!strcmp(cmd,"ring")){
            int i,s2;
            if(sscanf(line,"ring %d %d",&i,&s2)==2&&i>=0&&i<MAX_OSC)
                P.osc[i].ringSrc=s2;
        }else if(!strcmp(cmd,"lfo")){
            char w[16];float r,tp,tc;
            if(sscanf(line,"lfo %f %15s %f %f",&r,w,&tp,&tc)==4){
                P.lfoRate=r;P.lfoWave=lwaveFromStr(w);
                P.lfoToPitch=tp;P.lfoToCut=tc;}
        }else if(!strcmp(cmd,"sh")){
            sscanf(line,"sh %f %f %f",&P.shRate,&P.shToPitch,&P.shToCut);
        }else if(!strcmp(cmd,"filter")){
            sscanf(line,"filter %f %f %f %f %f",
                   &P.cutoff,&P.res,&P.fenvAmt,&P.keytrack,&P.drive);
        }else if(!strcmp(cmd,"aenv")){
            sscanf(line,"aenv %f %f %f %f",&P.aA,&P.aD,&P.aS,&P.aR);
        }else if(!strcmp(cmd,"fenv")){
            sscanf(line,"fenv %f %f %f %f",&P.fA,&P.fD,&P.fS,&P.fR);
        }else if(!strcmp(cmd,"glide")){
            sscanf(line,"glide %f",&P.glide);
        }else if(!strcmp(cmd,"trem")){
            sscanf(line,"trem %f %f",&P.tremRate,&P.tremDepth);
        }else if(!strcmp(cmd,"chorus")){
            sscanf(line,"chorus %f %f %f",&P.chRate,&P.chDepth,&P.chMix);
        }else if(!strcmp(cmd,"reverb")){
            sscanf(line,"reverb %f %f",&P.rvSize,&P.rvWet);
        }else if(!strcmp(cmd,"note")){
            float t,d,v;int m,pt=cur;
            int got=sscanf(line,"note %f %f %d %f %d",&t,&d,&m,&v,&pt);
            if(got>=4&&n<MAX_NOTES){
                if(pt<0||pt>=MAX_PATCH)pt=cur;
                if(tempo>0.f){float b=60.f/tempo;t*=b;d*=b;}
                notes[n++]={t,d,v,m,prevMidi[pt],pt};
                prevMidi[pt]=m;
                if(pt+1>nPatch)nPatch=pt+1;
            }
        }
    }
    fclose(f);
    *nNotes=n; *nPatchOut=nPatch;
    return (n>0);
}

/* ── stereo FX bus (CPU): chorus -> reverb ───────────────────────────*/
static void applyChorus(const float*mono,float*L,float*R,int N,float sr,
                        float rate,float depth,float mix)
{
    int maxD=(int)(0.06f*sr);
    float*dl=(float*)calloc(maxD,sizeof(float));
    int wp=0;
    float base=0.018f*sr, dep=depth*sr;
    for(int i=0;i<N;i++){
        dl[wp]=mono[i];
        float lphL=rate*i/sr;
        float dL=base+dep*(0.5f+0.5f*sinf(TWO_PI*lphL));
        float dR=base+dep*(0.5f+0.5f*sinf(TWO_PI*lphL+2.1f));
        auto tap=[&](float d)->float{
            float rpos=(float)wp-d;
            while(rpos<0)rpos+=maxD;
            int i0=(int)rpos; float fr=rpos-i0;
            int i1=(i0+1)%maxD;
            return dl[i0]*(1.f-fr)+dl[i1]*fr;
        };
        L[i]=mono[i]*(1.f-mix)+tap(dL)*mix;
        R[i]=mono[i]*(1.f-mix)+tap(dR)*mix;
        wp=(wp+1)%maxD;
    }
    free(dl);
}

static void applyReverb(float*L,float*R,int N,float sr,float room,float wet)
{
    if(wet<0.005f)return;
    static const float cd[]={0.02972f,0.03168f,0.03396f,0.03668f,
                              0.03856f,0.04024f,0.04228f,0.04384f};
    static const float ad[]={0.0053f,0.0071f,0.0102f,0.0123f};
    int NC=8,NA=4;float fb=0.78f*room+0.1f,damp=0.32f;
    float*cbL[8],*cbR[8],*abL[4],*abR[4];
    int csz[8],asz[4],cpL[8]={},cpR[8]={},apL[4]={},apR[4]={};
    float lpL[8]={},lpR[8]={};
    for(int i=0;i<NC;i++){csz[i]=(int)(cd[i]*sr*(0.8f+0.4f*room));
        cbL[i]=(float*)calloc(csz[i],sizeof(float));
        cbR[i]=(float*)calloc(csz[i]+7,sizeof(float));}
    for(int i=0;i<NA;i++){asz[i]=(int)(ad[i]*sr);
        abL[i]=(float*)calloc(asz[i],sizeof(float));
        abR[i]=(float*)calloc(asz[i],sizeof(float));}
    for(int s=0;s<N;s++){
        float iL=L[s],iR=R[s],oL=0,oR=0;
        for(int i=0;i<NC;i++){
            float dL=cbL[i][cpL[i]];
            lpL[i]=dL*(1.f-damp)+lpL[i]*damp;
            cbL[i][cpL[i]]=iL+lpL[i]*fb;
            cpL[i]=(cpL[i]+1==csz[i])?0:cpL[i]+1;oL+=dL;
            float dR=cbR[i][cpR[i]];
            lpR[i]=dR*(1.f-damp)+lpR[i]*damp;
            cbR[i][cpR[i]]=iR+lpR[i]*fb;
            cpR[i]=(cpR[i]+1==csz[i])?0:cpR[i]+1;oR+=dR;
        }
        for(int i=0;i<NA;i++){
            float bL=abL[i][apL[i]];float vL=oL+bL*0.5f;
            abL[i][apL[i]]=vL;apL[i]=(apL[i]+1==asz[i])?0:apL[i]+1;oL=bL-vL;
            float bR=abR[i][apR[i]];float vR=oR+bR*0.5f;
            abR[i][apR[i]]=vR;apR[i]=(apR[i]+1==asz[i])?0:apR[i]+1;oR=bR-vR;
        }
        L[s]=iL*(1.f-wet)+oL*wet/(float)NC;
        R[s]=iR*(1.f-wet)+oR*wet/(float)NC;
    }
    for(int i=0;i<NC;i++){free(cbL[i]);free(cbR[i]);}
    for(int i=0;i<NA;i++){free(abL[i]);free(abR[i]);}
}

static void writeWav(const char*p,const float*L,const float*R,int N,int sr){
    FILE*f=fopen(p,"wb");if(!f){fprintf(stderr,"open %s fail\n",p);return;}
    int16_t*b=(int16_t*)malloc((size_t)N*4);
    float pk=1e-9f;
    for(int i=0;i<N;i++){pk=fmaxf(pk,fabsf(L[i]));pk=fmaxf(pk,fabsf(R[i]));}
    float g=0.93f/pk;
    for(int i=0;i<N;i++){
        float d=(((float)(rand()&0xFFFF)+(float)(rand()&0xFFFF))/65536.f-1.f)/32768.f;
        b[i*2]=(int16_t)(fmaxf(-1.f,fminf(1.f,L[i]*g+d))*32767.f);
        b[i*2+1]=(int16_t)(fmaxf(-1.f,fminf(1.f,R[i]*g+d))*32767.f);
    }
    uint32_t dS=(uint32_t)N*4,cS=36+dS,bR=(uint32_t)sr*4;
    uint16_t bA=4,bp=16,fm=1,ch=2;uint32_t sc=16;
    fwrite("RIFF",1,4,f);fwrite(&cS,4,1,f);fwrite("WAVEfmt ",1,8,f);
    fwrite(&sc,4,1,f);fwrite(&fm,2,1,f);fwrite(&ch,2,1,f);
    fwrite(&sr,4,1,f);fwrite(&bR,4,1,f);fwrite(&bA,2,1,f);fwrite(&bp,2,1,f);
    fwrite("data",1,4,f);fwrite(&dS,4,1,f);
    fwrite(b,2,(size_t)N*2,f);free(b);fclose(f);
    printf("Wrote %s (%.1f s, %d Hz stereo)\n",p,(double)N/sr,sr);
}

int main(int argc,char**argv){
    const char*out=(argc>1)?argv[1]:"moog.wav";
    const char*patchPath=(argc>2)?argv[2]:NULL;
    int sr=(argc>3)?atoi(argv[3]):SR_DEFAULT;

    Patch*Ps=(Patch*)calloc(MAX_PATCH,sizeof(Patch));
    Note*notes=(Note*)calloc(MAX_NOTES,sizeof(Note));
    int nNotes=0, nPatch=1;
    if(patchPath&&loadPatch(patchPath,Ps,&nPatch,notes,&nNotes))
        printf("Loaded: %s (%d patches, %d notes)\n",patchPath,nPatch,nNotes);
    else{
        if(patchPath)printf("Could not load %s — using demo patch\n",patchPath);
        for(int i=0;i<MAX_PATCH;i++)initPatch(Ps[i]);
        defaultPatch(Ps[0],notes,&nNotes);
        nPatch=1;
        printf("Demo patch: %d osc, %d notes\n",Ps[0].nOsc,nNotes);
    }
    Patch&P=Ps[0];   /* FX bus + banner reference patch                 */

    printf("GPU Minimoog | patches=%d osc0=%d cutoff0=%.0fHz res0=%.2f\n\n",
           nPatch,P.nOsc,P.cutoff,P.res);
    cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("Device: %s (SM %d.%d)\n\n",prop.name,prop.major,prop.minor);

    /* buffer length: use the longest release across patches             */
    float maxR=0.f;
    for(int i=0;i<nPatch;i++)maxR=fmaxf(maxR,Ps[i].aR);
    float endT=0.f;
    for(int i=0;i<nNotes;i++)
        endT=fmaxf(endT,notes[i].t+notes[i].dur+maxR);
    endT+=2.5f;
    int N=(int)(endT*sr);

    Patch*dP;Note*dN;float*dBus;
    CUDA_CHECK(cudaMalloc(&dP,MAX_PATCH*sizeof(Patch)));
    CUDA_CHECK(cudaMalloc(&dN,MAX_NOTES*sizeof(Note)));
    CUDA_CHECK(cudaMalloc(&dBus,(size_t)N*4));
    CUDA_CHECK(cudaMemcpy(dP,Ps,MAX_PATCH*sizeof(Patch),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dN,notes,MAX_NOTES*sizeof(Note),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dBus,0,(size_t)N*4));

    cudaEvent_t e0,e1;CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    int nb=(nNotes+63)/64;
    voiceKernel<<<nb,64>>>(dP,dN,nNotes,dBus,N,(float)sr,12345u);
    CUDA_CHECK(cudaEventRecord(e1));CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    printf("Voice kernel: %.1f ms (%.0fx real-time)\n",ms,endT*1000.f/ms);

    float*mono=(float*)malloc((size_t)N*4);
    CUDA_CHECK(cudaMemcpy(mono,dBus,(size_t)N*4,cudaMemcpyDeviceToHost));

    float*L=(float*)calloc(N,sizeof(float));
    float*R=(float*)calloc(N,sizeof(float));
    printf("FX bus: chorus + reverb...\n");
    applyChorus(mono,L,R,N,(float)sr,P.chRate,P.chDepth,P.chMix);
    applyReverb(L,R,N,(float)sr,P.rvSize,P.rvWet);
    writeWav(out,L,R,N,sr);

    free(Ps);free(notes);free(mono);free(L);free(R);
    cudaFree(dP);cudaFree(dN);cudaFree(dBus);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 0;
}
