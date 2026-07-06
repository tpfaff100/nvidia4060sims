/*
 * musetraffic01.cu — GPU Generative Disco + CPU Freeway Soundscape
 * =================================================================
 * Merges discogen.cu (GPU Minimoog composer/synth) with the macOS
 * freeway layer (CPU, runs after GPU render).
 *
 * BUILD (after vcvars64.bat):
 *   nvcc -arch=sm_89 -O3 -use_fast_math -o musetraffic01 musetraffic01.cu
 *
 * RUN:
 *   musetraffic01 [out.wav] [sr] [seed] [length] [kick] [traffic] [honk] [mix]
 *
 *   musetraffic01
 *       defaults: length 5, kick 1.0, no traffic, mix 0.2
 *
 *   musetraffic01 out.wav 48000 0 5 1.0 0.0 0.0 0.0
 *       music only, no traffic at all
 *
 *   musetraffic01 out.wav 48000 0 5 1.0 0.5 0.2 0.2
 *       light freeway in the background, rare horn, music dominant
 *
 *   musetraffic01 out.wav 48000 0 5 1.0 1.0 0.8 0.4
 *       moderate traffic + honking, balanced with music
 *
 *   musetraffic01 out.wav 48000 0 7 1.5 2.0 2.0 0.6
 *       longer song, heavy kick, gridlock rage, traffic prominent
 *
 *   musetraffic01 out.wav 48000 42 5 1.0 1.5 1.0 0.3
 *       reproduce an exact run — same seed always gives same output
 *
 * PARAMETERS  (positional, all optional — defaults shown)
 *   out.wav    output filename                          [musetraffic01.wav]
 *   sr         sample rate in Hz                        [48000]
 *   seed       0 = different song every run             [0]
 *              any other value = reproducible (printed at startup)
 *   length     song duration scale  1=~1:15  5=~3:00  10=~6:15   [5]
 *   kick       bass drum character                       [1.0]
 *              1.0 = normal  2.0 = punchier/deeper  3.0 = very heavy
 *   traffic    freeway density                           [0.0]
 *              0.0 = none  0.5 = light  1.0 = moderate  2.0 = heavy
 *   honk       horn honking probability                  [0.0]
 *              0.0 = silent  0.5 = occasional  1.0 = normal  2.0 = rage
 *   mix        music vs traffic gain ratio               [0.2]
 *              0.0 = music only   0.5 = equal blend   1.0 = traffic only
 *              (writeWav normalises the final mix so overall loudness
 *               stays consistent regardless of this value)
 *
 * ARCHITECTURE
 *   GPU  voiceKernel: one CUDA thread per note, parallel synthesis of
 *        the 10-osc Minimoog engine (nonlinear ladder filter, FM, ring
 *        mod, polyBLEP, glide, LFO, S&H, tremolo).  All notes render
 *        simultaneously; result atomicAdd-ed into a mono float buffer.
 *
 *   CPU  Composer: Markov-chain harmony, motif development, voice
 *        leading, form grammar, patch generation.  Parameters length
 *        and kick reshape the song structure and drum patch at runtime.
 *
 *   CPU  Freeway layer: physically-modelled traffic rendered after the
 *        GPU pass into a separate stereo buffer, then blended into the
 *        music at the ratio set by mix.  Each car gets exact closed-form
 *        Doppler, era/nation-correct engine harmonics (US V8 half-order
 *        burble, UK/JP inline-4), forward-directive horns (US 350+440 Hz
 *        dual-tone, UK 320+400 Hz, JP single ~470 Hz), ground-reflection
 *        comb filter, and a distant traffic wash underneath.
 *
 *   CPU  FX bus: Schroeder reverb + chorus stereo widener applied to
 *        the music before the traffic layer is blended in.
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
#include <ctime>
#include <algorithm>

#define SR_DEFAULT  48000
#define TWO_PI      6.28318530717958647692f
#define MAX_OSC     10
#define MAX_NOTES   6144
#define MAX_PATCH   8

#define CUDA_CHECK(x) do{ cudaError_t _=x; if(_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_)); \
    exit(1);} }while(0)

/* ══════════════════════════════════════════════════════════════════════
   SYNTH DATA STRUCTURES
   ══════════════════════════════════════════════════════════════════════ */
enum Wave  { W_SAW=0, W_SQR, W_TRI, W_SIN, W_NOISE };
enum LWave { L_SIN=0, L_TRI, L_SQR };

struct OscP {
    int   wave;
    float semi, cents, level, pw;
    int   fmSrc; float fmAmt;
    int   ringSrc;
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

/* ══════════════════════════════════════════════════════════════════════
   GPU DEVICE HELPERS
   ══════════════════════════════════════════════════════════════════════ */
__device__ __forceinline__ float hnoise(unsigned i, unsigned s){
    unsigned h=i*0x9E3779B9u^s*0x85EBCA6Bu;
    h^=h>>13; h*=0xC2B2AE35u; h^=h>>16;
    return (float)(int)h*4.6566129e-10f;
}
__device__ __forceinline__ float ftanh(float x){
    float x2=x*x; return x*(27.f+x2)/(27.f+9.f*x2);
}
__device__ __forceinline__ float polyblep(float t, float dt){
    if(t<dt){t/=dt;return t+t-t*t-1.f;}
    if(t>1.f-dt){t=(t-1.f)/dt;return t*t+t+t+1.f;}
    return 0.f;
}
__device__ float adsr_d(float t,float dur,float A,float D,float S,float R){
    if(t<0.f)return 0.f;
    float e;
    if(t<A)e=t/fmaxf(A,1e-4f);
    else if(t<A+D)e=1.f-(t-A)/fmaxf(D,1e-4f)*(1.f-S);
    else if(t<dur)e=S;
    else{float tr=t-dur;if(tr>R)return 0.f;e=S*(1.f-tr/fmaxf(R,1e-4f));}
    return fmaxf(0.f,fminf(1.f,e));
}

/* ══════════════════════════════════════════════════════════════════════
   GPU VOICE KERNEL  (one thread = one note)
   ══════════════════════════════════════════════════════════════════════ */
__global__ void voiceKernel(
    const Patch*__restrict__ pp,
    const Note*__restrict__ notes, int nNotes,
    float*bus, int N, float sr, unsigned seed)
{
    int vi=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if(vi>=nNotes)return;
    const Note nt=notes[vi];
    const Patch P=pp[nt.patch];

    float invSr=1.f/sr;
    int   startSamp=(int)(nt.t*sr);
    float totalDur=nt.dur+P.aR+0.05f;
    int   nSamp=(int)(totalDur*sr);

    float ph[MAX_OSC],oscOut[MAX_OSC];
    for(int i=0;i<MAX_OSC;i++){
        ph[i]=hnoise((unsigned)(vi*17+i),seed)*0.5f+0.5f;
        oscOut[i]=0.f;
    }
    float s1=0,s2=0,s3=0,s4=0;
    float baseHz=440.f*exp2f(((float)nt.midi-69.f)/12.f);
    float prevHz=440.f*exp2f(((float)nt.prevMidi-69.f)/12.f);

    for(int i=0;i<nSamp;i++){
        float t=(float)i*invSr;
        float tAbs=nt.t+t;

        float gl=(P.glide>1e-4f)?fminf(1.f,t/P.glide):1.f;
        float noteHz=prevHz*powf(baseHz/prevHz,gl);

        float lph=P.lfoRate*tAbs; lph-=floorf(lph);
        float lfo;
        if(P.lfoWave==L_SIN)      lfo=__sinf(TWO_PI*lph);
        else if(P.lfoWave==L_TRI) lfo=4.f*fabsf(lph-0.5f)-1.f;
        else                      lfo=(lph<0.5f)?1.f:-1.f;

        float sh=0.f;
        if(P.shRate>0.01f)
            sh=hnoise((unsigned)floorf(tAbs*P.shRate),seed^0x5AAD);

        float pitchMod=exp2f((P.lfoToPitch*lfo+P.shToPitch*sh)/12.f);
        float f0=noteHz*pitchMod;

        float mix=0.f;
        for(int o=0;o<P.nOsc;o++){
            const OscP&op=P.osc[o];
            float f=f0*exp2f((op.semi+op.cents*0.01f)/12.f);
            float dt=f*invSr;
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
            }else{
                v=hnoise((unsigned)(i*MAX_OSC+o),seed^(unsigned)(vi*31));
            }
            if(op.ringSrc>=0&&op.ringSrc<o) v*=oscOut[op.ringSrc];
            oscOut[o]=v;
            mix+=v*op.level;
        }

        float fenv=adsr_d(t,nt.dur,P.fA,P.fD,P.fS,P.fR);
        float cut=P.cutoff
            *exp2f(P.fenvAmt*fenv*5.f
                  +P.lfoToCut*lfo*2.f
                  +P.shToCut*sh*2.f
                  +P.keytrack*log2f(noteHz/261.63f));
        cut=fmaxf(20.f,fminf(0.45f*sr,cut));

        float g=1.f-__expf(-TWO_PI*cut*invSr*0.5f);
        float k=4.f*P.res;
        float x=mix*P.drive;
        #pragma unroll
        for(int os=0;os<2;os++){
            float in=ftanh(x-k*s4);
            s1+=g*(in-s1);
            s2+=g*(ftanh(s1)-ftanh(s2));
            s3+=g*(ftanh(s2)-ftanh(s3));
            s4+=g*(ftanh(s3)-ftanh(s4));
        }
        float out=s4/fmaxf(0.3f,P.drive*0.7f);

        float aenv=adsr_d(t,nt.dur,P.aA,P.aD,P.aS,P.aR);
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

/* ══════════════════════════════════════════════════════════════════════
   COMPOSER  (CPU)
   ══════════════════════════════════════════════════════════════════════ */
static unsigned g_rng=1;
static void   sr_seed(unsigned s){ g_rng=(s!=0)?s:(unsigned)time(NULL); }
static float  rnd(){ g_rng^=g_rng<<13;g_rng^=g_rng>>17;g_rng^=g_rng<<5;
                     return (float)(g_rng&0xFFFFFF)/16777216.f; }
static float  rr(float a,float b){ return a+(b-a)*rnd(); }
static int    ri(int a,int b){ return a+(int)(rnd()*(float)(b-a+1)*0.9999f); }
static int    pick(const float*w,int n){
    float s=0;for(int i=0;i<n;i++)s+=w[i];
    float x=rnd()*s;
    for(int i=0;i<n;i++){x-=w[i];if(x<=0)return i;}
    return n-1;
}

static const int MINOR[7]={0,2,3,5,7,8,10};
struct Chord{int deg,type;};
static const int CH_ROOT[6]={0,3,4,5,6,2};
static const int CH_TYPE[6]={0,0,2,1,1,1};
static const float MARKOV[6][6]={
    {0.10f,0.22f,0.16f,0.26f,0.20f,0.06f},
    {0.30f,0.06f,0.30f,0.12f,0.16f,0.06f},
    {0.55f,0.08f,0.05f,0.20f,0.08f,0.04f},
    {0.16f,0.14f,0.16f,0.06f,0.42f,0.06f},
    {0.48f,0.08f,0.12f,0.16f,0.06f,0.10f},
    {0.16f,0.20f,0.14f,0.30f,0.14f,0.06f},
};
struct Composer{
    int key,tempo;
    Note*notes; int n;
    int lastMidi[5];
    int vMotifDeg[8]; float vMotifT[8],vMotifD[8]; int vMotifN;
    int cMotifDeg[8]; float cMotifT[8],cMotifD[8]; int cMotifN;
    int strVoice[4],padVoice[4];
    float beat;
};
static void emit(Composer&C,float beat,float durB,int midi,float vel,int patch){
    if(C.n>=MAX_NOTES)return;
    if(midi<12||midi>108)return;
    C.notes[C.n]={beat*C.beat,durB*C.beat,vel,midi,C.lastMidi[patch],patch};
    C.lastMidi[patch]=midi; C.n++;
}
static void chordTones(const Composer&C,Chord c,int oct,int*out){
    int root=C.key+MINOR[c.deg]+12*oct;
    out[0]=root; out[1]=root+((c.type==0)?3:4);
    out[2]=root+7; out[3]=(c.type==2)?root+10:root+12;
}
static int snapChord(const Composer&C,Chord c,int midi){
    int t[4];chordTones(C,c,0,t);
    int best=midi,bd=99;
    for(int o=-24;o<=48;o+=12) for(int i=0;i<4;i++){
        int d=abs(midi-(t[i]+o));if(d<bd){bd=d;best=t[i]+o;}
    }return best;
}
static int snapScale(const Composer&C,int midi){
    int best=midi,bd=99;
    for(int o=-24;o<=60;o+=12) for(int i=0;i<7;i++){
        int c=C.key+MINOR[i]+o,d=abs(midi-c);if(d<bd){bd=d;best=c;}
    }return best;
}
static void makeProg(Chord*prog,int start){
    int cur=start;
    for(int i=0;i<4;i++){
        prog[i]={CH_ROOT[cur],CH_TYPE[cur]};
        if(i==2){float w[6];memcpy(w,MARKOV[cur],sizeof(w));
                 w[2]*=3.f;w[4]*=2.f;cur=pick(w,6);}
        else cur=pick(MARKOV[cur],6);
    }
}
struct RCell{int n;float t[6],d[6];};
static const RCell RHY[]={
    {4,{0,1,2,3},              {0.9f,0.9f,0.9f,0.9f}},
    {5,{0.5f,1,1.5f,2.5f,3},  {0.4f,0.4f,0.9f,0.4f,0.9f}},
    {5,{0,0.5f,1.5f,2,3},     {0.4f,0.9f,0.4f,0.9f,0.9f}},
    {6,{0,0.5f,1,1.5f,2.5f,3},{0.4f,0.4f,0.4f,0.9f,0.4f,0.9f}},
    {4,{0.5f,1.5f,2,3},       {0.9f,0.4f,0.9f,0.9f}},
    {3,{0,1.5f,2.5f},         {1.4f,0.9f,1.4f}},
    {5,{0,1,1.5f,2,2.5f},     {0.9f,0.4f,0.4f,0.4f,1.4f}},
};
#define NRHY (int)(sizeof(RHY)/sizeof(RHY[0]))
static void makeMotif(int*deg,float*tt,float*dd,int*nn,int span){
    int n=0,curDeg=ri(0,6);
    for(int bar=0;bar<2;bar++){
        const RCell&rc=RHY[ri(0,NRHY-1)];
        for(int i=0;i<rc.n&&n<8;i++){
            tt[n]=bar*4+rc.t[i];dd[n]=rc.d[i];deg[n]=curDeg;
            float r=rnd();
            int step=(r<0.42f)?1:(r<0.78f)?-1:(r<0.9f)?ri(2,3):-ri(2,3);
            curDeg+=step;
            if(curDeg>span)curDeg=span-ri(1,2);
            if(curDeg<-1)curDeg=ri(0,1);
            n++;
        }
    }*nn=n;
}
enum Dev{DEV_NONE=0,DEV_SEQ_UP,DEV_SEQ_DN,DEV_INV,DEV_SHIFT};
static void playMotif(Composer&C,const int*deg,const float*tt,const float*dd,
                      int n,float barBeat,Chord ch,int baseMidi,
                      int dev,float vel,int patch)
{
    float shift=(dev==DEV_SHIFT)?0.5f:0.f;
    for(int i=0;i<n;i++){
        int d=deg[i];
        if(dev==DEV_SEQ_UP)d+=1;
        if(dev==DEV_SEQ_DN)d-=1;
        if(dev==DEV_INV)   d=-d+2;
        int midi=baseMidi+((d>=0)?MINOR[d%7]+12*(d/7)
                                 :MINOR[(d%7+7)%7]-12*((-d+6)/7));
        float pos=tt[i];
        bool strong=(fmodf(pos,1.f)<0.01f);
        midi=strong?snapChord(C,ch,midi):snapScale(C,midi);
        float v=vel*(strong?1.f:0.9f)*rr(0.93f,1.f);
        emit(C,barBeat+pos+shift,dd[i],midi,v,patch);
    }
}
static void voiceLead(Composer&C,Chord ch,int*voice,int lowAnchor){
    int t[4];chordTones(C,ch,0,t);
    for(int v2=0;v2<4;v2++){
        int want=t[v2%4];
        int prev=voice[v2]?voice[v2]:lowAnchor+v2*4;
        int best=want,bd=99;
        for(int o=-24;o<=36;o+=12){
            int cand=want+o;
            if(cand<lowAnchor-6||cand>lowAnchor+34)continue;
            int d2=abs(cand-prev);if(d2<bd){bd=d2;best=cand;}
        }voice[v2]=best;
    }
}
static void bassBar(Composer&C,float barBeat,Chord ch,Chord next,float energy){
    int t[4];chordTones(C,ch,0,t);
    int root=t[0];
    while(root>C.key+MINOR[ch.deg]-12+24)root-=12;
    root=snapScale(C,root);if(root<28)root+=12;
    int nt2[4];chordTones(C,next,0,nt2);
    int nroot=nt2[0];
    while(nroot>root+8)nroot-=12;while(nroot<root-8)nroot+=12;
    float r=rnd();
    if(energy<0.4f||r<0.25f){
        emit(C,barBeat,1.8f,root,0.9f,0);
        emit(C,barBeat+2,1.8f,(r<0.5f)?root:root+7,0.85f,0);
    }else if(r<0.6f){
        for(int b=0;b<4;b++){
            emit(C,barBeat+b,0.55f,root,0.95f,0);
            emit(C,barBeat+b+0.5f,0.4f,root+12,0.8f,0);
        }
    }else{
        emit(C,barBeat,    0.9f,root,0.98f,0);
        emit(C,barBeat+0.5f,0.4f,root,0.8f,0);
        emit(C,barBeat+1,  0.9f,root+((rnd()<0.5f)?3:7),0.9f,0);
        emit(C,barBeat+1.5f,0.4f,root,0.8f,0);
        emit(C,barBeat+2,  0.9f,root+7,0.92f,0);
        emit(C,barBeat+2.5f,0.4f,root+((rnd()<0.5f)?10:5),0.8f,0);
        int app=nroot+((nroot>root)?-1:1);
        emit(C,barBeat+3,  0.5f,app,0.9f,0);
        emit(C,barBeat+3.5f,0.5f,nroot,0.85f,0);
    }
}
static void percBar(Composer&C,float barBeat,float energy,int fill,
                    float kickBoost=1.0f){
    for(int b=0;b<4;b++){
        float kvel=fminf(1.0f,(0.95f*energy+0.2f)*kickBoost);
        emit(C,barBeat+b,0.35f,24,kvel,4);
        if(b%2==1)emit(C,barBeat+b,0.22f,38,0.8f*energy+0.15f,4);
    }
    if(fill)for(int i=0;i<8;i++)
        emit(C,barBeat+2+i*0.25f,0.15f,38,0.5f+0.06f*i,4);
}

/* ── patch generators ───────────────────────────────────────────────── */
static void initPatch(Patch&P){
    memset(&P,0,sizeof(P));
    for(int i=0;i<MAX_OSC;i++){P.osc[i].fmSrc=-1;P.osc[i].ringSrc=-1;
                               P.osc[i].pw=0.5f;}
    P.drive=1.f;P.cutoff=1000.f;P.res=0.3f;
    P.aA=0.005f;P.aD=0.2f;P.aS=0.7f;P.aR=0.2f;
    P.fA=0.005f;P.fD=0.2f;P.fS=0.3f;P.fR=0.2f;
}
static void genBass(Patch&P){
    initPatch(P);P.nOsc=10;
    float det=rr(4.f,9.f);
    P.osc[0]={W_SAW,0,0,0.36f,0.5f,-1,0.f,-1};
    P.osc[1]={W_SIN,0,rr(5.f,9.f),0.24f,0.5f,-1,0.f,0};
    P.osc[2]={W_SQR,0,det*0.5f,0.20f,rr(0.3f,0.45f),-1,0.f,-1};
    P.osc[3]={W_SIN,-12,0,rr(0.34f,0.44f),0.5f,-1,0.f,-1};
    P.osc[4]={W_SIN,-24,0,rr(0.12f,0.2f),0.5f,-1,0.f,-1};
    P.osc[5]={W_SIN,0,0,0.08f,0.5f,0,rr(1.4f,2.6f),-1};
    P.osc[6]={W_SAW,12,0,0.05f,0.5f,-1,0.f,-1};
    P.osc[7]={W_SIN,-12,2,0.05f,0.5f,-1,0.f,2};
    P.osc[8]={W_NOISE,0,0,0.03f,0.5f,-1,0.f,-1};
    P.osc[9]={W_TRI,7,0,0.06f,0.5f,-1,0.f,-1};
    P.cutoff=rr(75.f,110.f);P.res=rr(0.72f,0.88f);
    P.fenvAmt=rr(0.8f,0.92f);P.keytrack=0.6f;P.drive=rr(1.7f,2.4f);
    P.aA=0.003f;P.aD=rr(0.12f,0.18f);P.aS=rr(0.4f,0.52f);P.aR=0.13f;
    P.fA=0.001f;P.fD=rr(0.08f,0.14f);P.fS=0.06f;P.fR=0.1f;
    P.glide=rr(0.03f,0.05f);
    P.chRate=rr(0.4f,0.7f);P.chDepth=rr(0.004f,0.007f);P.chMix=rr(0.32f,0.42f);
    P.rvSize=rr(0.72f,0.88f);P.rvWet=rr(0.18f,0.26f);
}
static void genLead(Patch&P){
    initPatch(P);P.nOsc=10;
    float ix=rr(0.4f,0.9f);
    P.osc[0]={W_SIN,0,0,0.30f,0.5f,-1,0.f,-1};
    P.osc[1]={W_SIN,0,rr(3.f,6.f),0.24f,0.5f,0,ix,-1};
    P.osc[2]={W_SIN,0,-rr(3.f,6.f),0.24f,0.5f,0,ix*rr(1.2f,1.6f),-1};
    P.osc[3]={W_SIN,0,2,0.11f,0.5f,-1,0.f,2};
    P.osc[4]={W_SIN,12,0,0.16f,0.5f,0,rr(1.4f,2.2f),-1};
    P.osc[5]={W_SIN,12,-rr(4.f,8.f),0.13f,0.5f,0,rr(1.8f,2.8f),-1};
    P.osc[6]={W_SIN,12,3,0.07f,0.5f,-1,0.f,4};
    P.osc[7]={W_SIN,-12,0,0.14f,0.5f,0,0.3f,-1};
    P.osc[8]={W_SIN,7,0,0.08f,0.5f,-1,0.f,0};
    P.osc[9]={W_SIN,19,0,0.04f,0.5f,1,rr(4.f,7.f),-1};
    P.lfoRate=rr(5.2f,6.1f);P.lfoWave=L_SIN;
    P.lfoToPitch=rr(0.03f,0.07f);P.lfoToCut=0.04f;
    P.shRate=rr(6.f,9.f);P.shToCut=rr(0.08f,0.16f);
    P.cutoff=rr(1500.f,2400.f);P.res=rr(0.35f,0.55f);
    P.fenvAmt=rr(0.4f,0.6f);P.keytrack=0.7f;P.drive=rr(1.2f,1.6f);
    P.aA=0.006f;P.aD=0.3f;P.aS=0.72f;P.aR=rr(0.35f,0.55f);
    P.fA=0.004f;P.fD=0.22f;P.fS=0.3f;P.fR=0.35f;
    P.glide=rr(0.04f,0.07f);
}
static void genRingPad(Patch&P){
    initPatch(P);P.nOsc=10;
    float c1=rr(60.f,120.f);
    P.osc[0]={W_SIN,0,0,0.32f,0.5f,-1,0.f,-1};
    P.osc[1]={W_SIN,0,c1,0.32f,0.5f,-1,0.f,0};
    P.osc[2]={W_SIN,12,0,0.22f,0.5f,-1,0.f,-1};
    P.osc[3]={W_SIN,12,rr(4.f,10.f),0.22f,0.5f,-1,0.f,2};
    P.osc[4]={W_SIN,7,-rr(2.f,5.f),0.18f,0.5f,-1,0.f,-1};
    P.osc[5]={W_SIN,7,rr(2.f,5.f),0.18f,0.5f,-1,0.f,4};
    P.osc[6]={W_SIN,(float)(rnd()<0.5f?3:4),0,0.12f,0.5f,-1,0.f,-1};
    P.osc[7]={W_SIN,(float)(rnd()<0.5f?3:4),rr(3.f,7.f),0.12f,0.5f,-1,0.f,6};
    P.osc[8]={W_SIN,-12,0,0.20f,0.5f,-1,0.f,-1};
    P.osc[9]={W_NOISE,0,0,0.018f,0.5f,-1,0.f,-1};
    P.lfoRate=rr(3.2f,4.4f);P.lfoWave=L_TRI;
    P.lfoToPitch=0.02f;P.lfoToCut=rr(0.04f,0.08f);
    P.tremRate=P.lfoRate+rr(0.05f,0.2f);P.tremDepth=rr(0.08f,0.14f);
    P.cutoff=rr(750.f,1100.f);P.res=0.28f;
    P.fenvAmt=0.4f;P.keytrack=0.55f;P.drive=1.1f;
    P.aA=rr(0.3f,0.5f);P.aD=0.6f;P.aS=0.85f;P.aR=rr(1.0f,1.5f);
    P.fA=0.5f;P.fD=0.8f;P.fS=0.6f;P.fR=1.0f;
}
static void genStrings(Patch&P){
    initPatch(P);P.nOsc=10;
    float sp=rr(8.f,13.f);
    P.osc[0]={W_SAW,0,-sp,0.17f,0.5f,-1,0.f,-1};
    P.osc[1]={W_SAW,0,-sp*0.4f,0.18f,0.5f,-1,0.f,-1};
    P.osc[2]={W_SAW,0,0,0.18f,0.5f,-1,0.f,-1};
    P.osc[3]={W_SAW,0,sp*0.4f,0.18f,0.5f,-1,0.f,-1};
    P.osc[4]={W_SAW,0,sp,0.17f,0.5f,-1,0.f,-1};
    P.osc[5]={W_SAW,12,-rr(5.f,8.f),0.09f,0.5f,-1,0.f,-1};
    P.osc[6]={W_SAW,12,rr(5.f,8.f),0.09f,0.5f,-1,0.f,-1};
    P.osc[7]={W_SIN,-12,0,0.13f,0.5f,-1,0.f,-1};
    P.osc[8]={W_TRI,0,3,0.07f,0.5f,-1,0.f,-1};
    P.osc[9]={W_NOISE,0,0,0.022f,0.5f,-1,0.f,-1};
    P.shRate=rr(2.6f,3.6f);P.shToCut=rr(0.05f,0.09f);
    P.lfoRate=5.3f;P.lfoWave=L_SIN;P.lfoToPitch=0.018f;P.lfoToCut=0.02f;
    P.cutoff=rr(2000.f,2800.f);P.res=0.2f;
    P.fenvAmt=0.3f;P.keytrack=0.55f;P.drive=1.05f;
    P.aA=rr(0.24f,0.34f);P.aD=0.55f;P.aS=0.84f;P.aR=rr(0.85f,1.1f);
    P.fA=0.4f;P.fD=0.65f;P.fS=0.6f;P.fR=0.9f;
}
/* genPerc — builds the percussion patch, shaped by kickBoost (0.1–3.0).
 * kickBoost=1.0 is the neutral sound; higher values increase:
 *   body amplitude (osc 0 level), sub-octave level (osc 8),
 *   FM pitch-drop index (osc 1 → osc 0) for a harder snap transient,
 *   amplitude decay time (longer thump), filter saturation drive,
 *   and lower the filter cutoff (more low-end passes through).
 * The snare (osc 3-4) and hat (osc 5-9) are unchanged by kickBoost.  */
static void genPerc(Patch&P, float kickBoost=1.0f){
    initPatch(P);P.nOsc=10;
    float bodyAmp = fminf(1.0f, 0.72f*kickBoost);
    float subAmp  = fminf(0.9f, 0.38f*kickBoost);
    float fmIdx   = fminf(14.f, 8.0f*kickBoost);
    float bodyDec = fmaxf(0.08f, fminf(0.35f, 0.08f+0.10f*(kickBoost-1.0f)));
    float bodyRel = bodyDec*1.5f;
    P.osc[0]={W_SIN,13,0,bodyAmp,0.5f,-1,0.f,-1};
    P.osc[1]={W_SIN,13,0,0.55f,  0.5f, 0,fmIdx,-1};
    P.osc[2]={W_NOISE,0,0,0.30f, 0.5f,-1,0.f,-1};
    P.osc[3]={W_TRI,0,0,0.46f,   0.5f,-1,0.f,-1};
    P.osc[4]={W_NOISE,0,0,0.6f,  0.5f,-1,0.f,3};
    P.osc[5]={W_NOISE,0,0,0.55f, 0.5f,-1,0.f,-1};
    P.osc[6]={W_SIN,0,0,0.20f,   0.5f,-1,0.f,-1};
    P.osc[7]={W_SIN,7,0,0.12f,   0.5f,-1,0.f,-1};
    P.osc[8]={W_SIN,1,0,subAmp,  0.5f,-1,0.f,-1};
    P.osc[9]={W_NOISE,0,2,0.26f, 0.5f,-1,0.f,-1};
    P.cutoff=fmaxf(200.f,6000.f-1500.f*(kickBoost-1.f));
    P.res=0.28f;P.fenvAmt=0.45f;P.keytrack=0.f;
    P.drive=fminf(2.5f,1.0f+0.5f*(kickBoost-1.f));
    P.aA=0.001f;P.aD=bodyDec;P.aS=0.0f;P.aR=bodyRel;
    P.fA=0.001f;P.fD=bodyDec*0.6f;P.fS=0.0f;P.fR=bodyRel*0.5f;
}

/* ── song assembly ───────────────────────────────────────────────────
 * compose() writes all notes into C.notes[].
 *   length   1-10 scales every section's bar count (1=short, 10=long).
 *            Length 5 is the baseline; sections scale by 0.4–2.0×.
 *            Songs of length ≥8 gain an extra chorus reprise.
 *   kickBoost passed through to percBar() to scale kick velocity;
 *            the patch itself is shaped by genPerc() before this call. */
struct Section{const char*name;int bars;float energy;int kind;};
enum{S_INTRO,S_VERSE,S_PRE,S_CHORUS,S_BREAK,S_OUTRO};

static int compose(Composer&C,Note*notes,char*report,
                   int length=5,float kickBoost=1.0f)
{
    C.notes=notes;C.n=0;
    for(int i=0;i<5;i++)C.lastMidi[i]=48;
    memset(C.strVoice,0,sizeof(C.strVoice));
    memset(C.padVoice,0,sizeof(C.padVoice));
    C.key=48+ri(0,11);
    C.tempo=ri(112,126);
    C.beat=60.f/(float)C.tempo;

    if(length<1)length=1;if(length>10)length=10;
    float sc=0.4f+(length-1)*(1.6f/9.f);
    auto bars=[&](int n)->int{return std::max(2,(int)roundf(n*sc));};
    auto rbars=[&](int lo,int hi)->int{
        int slo=std::max(2,(int)roundf(lo*sc));
        int shi=std::max(slo,(int)roundf(hi*sc));
        return ri(slo,shi);
    };

    Section form[12];int nf=0;
    form[nf++]={"INTRO", rbars(4,8), 0.30f,S_INTRO};
    form[nf++]={"VERSE", bars(16),   0.55f,S_VERSE};
    form[nf++]={"PRE",   bars(8),    0.72f,S_PRE};
    form[nf++]={"CHORUS",bars(16),   0.95f,S_CHORUS};
    form[nf++]={"BREAK", bars(8),    0.42f,S_BREAK};
    if(rnd()<0.6f)form[nf++]={"VERSE",bars(8),0.62f,S_VERSE};
    form[nf++]={"CHORUS",bars(16),   1.00f,S_CHORUS};
    if(length>=8)form[nf++]={"CHORUS",bars(8),1.00f,S_CHORUS};
    form[nf++]={"OUTRO", rbars(6,10),0.35f,S_OUTRO};

    Chord vProg[4],cProg[4],bProg[4];
    makeProg(vProg,0);
    makeProg(cProg,(rnd()<0.5f)?3:0);
    bProg[0]={CH_ROOT[3],CH_TYPE[3]};bProg[1]={CH_ROOT[4],CH_TYPE[4]};
    bProg[2]={CH_ROOT[0],CH_TYPE[0]};bProg[3]={CH_ROOT[2],CH_TYPE[2]};
    makeMotif(C.vMotifDeg,C.vMotifT,C.vMotifD,&C.vMotifN,5);
    makeMotif(C.cMotifDeg,C.cMotifT,C.cMotifD,&C.cMotifN,6);
    int melBase=C.key+24;if(melBase<66)melBase+=12;

    float bar=0.f;
    for(int s=0;s<nf;s++){
        Section&S=form[s];
        Chord*prog=(S.kind==S_CHORUS)?cProg:(S.kind==S_BREAK)?bProg:vProg;
        for(int b=0;b<S.bars;b++){
            float bb=(bar+b)*4.f;
            Chord ch=prog[b%4],nx=prog[(b+1)%4];
            float E=S.energy;
            if(S.kind!=S_INTRO&&!(S.kind==S_OUTRO&&b>=S.bars-2))
                percBar(C,bb,E,(b%8==7&&E>0.5f),kickBoost);
            if(S.kind!=S_INTRO||b>=S.bars/2)
                bassBar(C,bb,ch,nx,E);
            if(S.kind!=S_BREAK||b>=S.bars-2){
                voiceLead(C,ch,C.strVoice,C.key+7);
                float sv=0.35f+0.4f*E;
                if(S.kind==S_OUTRO)sv*=1.f-(float)b/S.bars;
                for(int v2=0;v2<4;v2++)emit(C,bb,3.7f,C.strVoice[v2],sv,3);
            }
            if(b%2==0&&S.kind!=S_INTRO){
                voiceLead(C,ch,C.padVoice,C.key+12);
                float pv=0.3f+0.28f*E;
                if(S.kind==S_OUTRO)pv*=1.f-(float)b/S.bars;
                for(int v2=0;v2<4;v2++)emit(C,bb,7.4f,C.padVoice[v2],pv,2);
            }
            if(S.kind==S_VERSE||S.kind==S_CHORUS||S.kind==S_PRE||
               (S.kind==S_BREAK&&b>=2)||(S.kind==S_INTRO&&b>=S.bars/2)||
               S.kind==S_OUTRO){
                if(b%2==0){
                    int dev=DEV_NONE,ph=(b/2)%4;
                    if(ph==1)dev=(rnd()<0.5f)?DEV_SEQ_UP:DEV_SHIFT;
                    if(ph==2)dev=(rnd()<0.4f)?DEV_INV:DEV_SEQ_DN;
                    if(ph==3)dev=DEV_SHIFT;
                    int reg=melBase;
                    if(S.kind==S_CHORUS)reg+=3;
                    if(S.kind==S_PRE)   reg+=(b/2);
                    if(S.kind==S_OUTRO||S.kind==S_INTRO)reg-=2;
                    float mv=0.55f+0.4f*E;
                    const int*md=(S.kind==S_CHORUS)?C.cMotifDeg:C.vMotifDeg;
                    const float*mt=(S.kind==S_CHORUS)?C.cMotifT:C.vMotifT;
                    const float*mdur=(S.kind==S_CHORUS)?C.cMotifD:C.vMotifD;
                    int mn=(S.kind==S_CHORUS)?C.cMotifN:C.vMotifN;
                    playMotif(C,md,mt,mdur,mn,bb,ch,reg,dev,mv,1);
                    if(S.kind==S_CHORUS)
                        playMotif(C,md,mt,mdur,mn,bb,ch,reg-9,DEV_INV,mv*0.55f,2);
                }
            }
        }
        bar+=S.bars;
    }
    float endB=bar*4.f;
    int t[4];Chord tonic={0,0};chordTones(C,tonic,0,t);
    emit(C,endB,10.f,C.key-12,0.9f,0);
    for(int v2=0;v2<4;v2++){
        emit(C,endB,10.f,t[v2]+12,0.55f,3);
        emit(C,endB,10.f,t[v2]+24,0.4f,2);
    }
    emit(C,endB+1,8.f,C.key+36,0.6f,1);

    static const char*NOTE_N[12]={"C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"};
    static const char*RN[7]={"i","ii","III","iv","v","VI","VII"};
    char*p=report; char*re=report+512;
    p+=snprintf(p,(size_t)(re-p),
        "Key: %s minor   Tempo: %d BPM   Length: %d/10   Kick: %.1fx\nForm: ",
        NOTE_N[C.key%12],C.tempo,length,kickBoost);
    for(int s=0;s<nf;s++)p+=snprintf(p,(size_t)(re-p),"%s(%d) ",form[s].name,form[s].bars);
    p+=snprintf(p,(size_t)(re-p),"\nVerse:  ");
    for(int i=0;i<4;i++)p+=snprintf(p,(size_t)(re-p),"%s ",RN[vProg[i].deg]);
    p+=snprintf(p,(size_t)(re-p),"\nChorus: ");
    for(int i=0;i<4;i++)p+=snprintf(p,(size_t)(re-p),"%s ",RN[cProg[i].deg]);
    p+=snprintf(p,(size_t)(re-p),"\n");
    return C.n;
}

/* ══════════════════════════════════════════════════════════════════════
   FREEWAY LAYER  (CPU — runs after GPU render, mixes into stereo bus)
   ══════════════════════════════════════════════════════════════════════ */
static float fw_rnd(){
    g_rng^=g_rng<<13;g_rng^=g_rng>>17;g_rng^=g_rng<<5;
    return (float)(g_rng&0xFFFFFF)/16777216.f;
}
static float fw_rr(float a,float b){return a+(b-a)*fw_rnd();}
static int   fw_ri(int a,int b){
    return a+(int)(fw_rnd()*(float)(b-a+1)*0.9999f);
}

struct FWCar{
    float tPass,v,d,loud,fFire;
    bool  halfOrder;
    float hAmp[14],roarAmp,roarHz;
    bool  hasHorn;
    float hornF1,hornF2,hornStart,hornDur;
    bool  hornAngry;
};

static void fw_buildCar(FWCar&C,float tPass,
                        float trafficLevel,float honkLevel,bool isTailgater)
{
    C.tPass=tPass;
    static const float laneD[4]={5.f,9.f,14.f,18.f};
    static const float laneDir[4]={1.f,1.f,-1.f,-1.f};
    int lane=fw_ri(0,3);
    C.d=laneD[lane];
    C.v=laneDir[lane]*fw_rr(22.f,36.f);
    float nr=fw_rnd();
    int nation=(nr<0.55f)?0:(nr<0.75f)?2:1;
    bool era60=fw_rnd()<0.25f;
    bool era80=!era60&&fw_rnd()<0.40f;
    bool isV8=(nation==0)?(fw_rnd()<0.78f):(fw_rnd()<0.08f);
    float rpm=isV8?fw_rr(2000.f,2800.f):(nation==1?fw_rr(2800.f,4000.f):fw_rr(2400.f,3400.f));
    C.fFire=rpm/60.f*(isV8?4.f:2.f);
    C.halfOrder=isV8;
    float tilt=isV8?0.70f:(nation==1?0.55f:1.10f);
    float ratioStep=isV8?0.5f:1.0f;
    for(int k=0;k<14;k++){
        float ratio=ratioStep*(float)(k+1);
        float a=powf(1.f/ratio,tilt);
        if(isV8&&(k%2==0))a*=1.45f;
        C.hAmp[k]=a;
    }
    float baseLoud=isV8?1.05f:(nation==1?0.90f:0.62f);
    if(era60)baseLoud*=1.28f;
    if(era80){baseLoud*=0.60f;for(int k=0;k<14;k++)C.hAmp[k]*=expf(-0.13f*(float)(k+1));}
    C.loud=baseLoud*fw_rr(0.82f,1.18f)*(0.55f+0.45f*trafficLevel);
    C.roarAmp=isV8?0.45f:(nation==1?0.38f:0.20f);
    C.roarHz =isV8?340.f:(nation==1?850.f:240.f);
    float pHorn=isTailgater?0.55f*honkLevel:0.06f*honkLevel;
    C.hasHorn=(fw_rnd()<fminf(0.95f,pHorn));
    if(C.hasHorn){
        if(nation==0){C.hornF1=fw_rr(335.f,370.f);C.hornF2=fw_rr(425.f,460.f);}
        else if(nation==1){C.hornF1=fw_rr(315.f,345.f);C.hornF2=fw_rr(395.f,425.f);}
        else{C.hornF1=fw_rr(440.f,500.f);C.hornF2=0.f;}
        C.hornAngry=(fw_rnd()<0.25f*honkLevel);
        if(C.hornAngry){C.hornStart=fw_rr(2.5f,5.0f);C.hornDur=fw_rr(1.2f,2.8f);}
        else           {C.hornStart=fw_rr(1.0f,3.0f);C.hornDur=fw_rr(0.12f,0.35f);}
    }
}

static inline float fw_lp1(float x,float&z,float c){z=z*c+x*(1.f-c);return z;}

static void fw_renderCar(const FWCar&C,float*L,float*R,
                          int N,float sr,unsigned noiseSeed)
{
    const float SPEED_C=343.f;
    float invSr=1.f/sr;
    float halfWin=300.f/fabsf(C.v);
    int s0=std::max(0,(int)((C.tPass-halfWin)*sr));
    int s1=std::min(N, (int)((C.tPass+halfWin)*sr));
    if(s0>=s1)return;
    float zRoar=0.f;
    float cRoar=expf(-2.f*(float)M_PI*C.roarHz*invSr);
    auto hn=[&](unsigned i)->float{
        unsigned h=i*0x9E3779B9u^noiseSeed*0x85EBCA6Bu;
        h^=h>>13;h*=0xC2B2AE35u;h^=h>>16;
        return (float)(int)h*4.6566129e-10f;
    };
    for(int si=s0;si<s1;si++){
        float tg=si*invSr;
        float x0=C.v*(tg-C.tPass);
        float r0=sqrtf(x0*x0+C.d*C.d);
        /* two Newton steps for retarded time */
        float tau=tg-r0/SPEED_C;
        for(int it=0;it<2;it++){
            float xt=C.v*(tau-C.tPass);
            float rt=sqrtf(xt*xt+C.d*C.d);
            tau=tg-rt/SPEED_C;
        }
        float xtau=C.v*(tau-C.tPass);
        float rtau=sqrtf(xtau*xtau+C.d*C.d);

        /* engine harmonics */
        float eng=0.f;
        float ratioStep=C.halfOrder?0.5f:1.0f;
        for(int k=0;k<14;k++){
            float ratio=ratioStep*(float)(k+1);
            float f=C.fFire*ratio;
            float ab=expf(-5.5e-10f*f*f*rtau);
            float ph=C.fFire*tau*ratio; ph-=floorf(ph);
            eng+=C.hAmp[k]*ab*sinf(2.f*(float)M_PI*ph);
        }
        /* roar */
        float nz=hn((unsigned)si);
        float roar=fw_lp1(nz,zRoar,cRoar)*C.roarAmp*3.2f;
        /* ground reflection */
        float rGnd=sqrtf(xtau*xtau+(C.d+1.2f)*(C.d+1.2f));
        float tauG=tau-(rGnd-rtau)/SPEED_C;
        float engG=0.f;
        for(int k=0;k<6;k++){
            float ratio=ratioStep*(float)(k+1);
            float f=C.fFire*ratio;
            float ab=expf(-5.5e-10f*f*f*rGnd);
            float ph=C.fFire*tauG*ratio;ph-=floorf(ph);
            engG+=C.hAmp[k]*ab*sinf(2.f*(float)M_PI*ph)*0.35f;
        }
        float amp=C.loud/fmaxf(rtau,2.5f);
        float s=(eng+engG+roar)*amp;
        /* horn */
        if(C.hasHorn){
            float tauRel=tau-(C.tPass-C.hornStart);
            if(tauRel>=0.f&&tauRel<C.hornDur){
                float sgnV=(C.v>=0.f)?1.f:-1.f;
                float fw2=0.5f*(1.f-(xtau*sgnV)/sqrtf(xtau*xtau+36.f));
                float dir=0.18f+0.82f*fw2;
                float envH=fminf(1.f,tauRel/0.006f)*fminf(1.f,(C.hornDur-tauRel)/0.04f);
                float horn=0.f;
                float freqs[2]={C.hornF1,C.hornF2};
                for(int q=0;q<2;q++){
                    if(freqs[q]<=0.f)continue;
                    float gains[3]={1.f,0.50f,0.22f};
                    for(int h=1;h<=3;h++){
                        float f=freqs[q]*(float)h;
                        float ab=expf(-5.5e-10f*f*f*rtau);
                        float ph=f*tau;ph-=floorf(ph);
                        horn+=gains[h-1]*ab*sinf(2.f*(float)M_PI*ph);
                    }
                }
                s+=horn*envH*dir*amp*1.8f;
            }
        }
        /* pan */
        float sinAz=xtau/fmaxf(rtau,0.01f);
        sinAz=fmaxf(-1.f,fminf(1.f,sinAz));
        float panAng=(sinAz+1.f)*0.25f*(float)M_PI;
        float gL=cosf(panAng),gR=sinf(panAng);
        int itd=(int)(fabsf(sinAz)*0.00066f*sr);
        int iL=si,iR=si;
        if(sinAz<0.f)iR=std::min(N-1,si+itd);
        else         iL=std::min(N-1,si+itd);
        L[iL]+=s*gL;R[iR]+=s*gR;
    }
}

static void fw_renderWash(float*L,float*R,int N,float sr,
                           float trafficLevel,unsigned seed2)
{
    if(trafficLevel<0.05f)return;
    float amp=0.018f*trafficLevel;
    float cLo=expf(-2.f*(float)M_PI*180.f/sr);
    float cHi=expf(-2.f*(float)M_PI*55.f/sr);
    float zL=0,zR=0,zhL=0,zhR=0;
    unsigned rs=seed2^0xABCD1234u;
    for(int i=0;i<N;i++){
        unsigned h=((unsigned)i)*0x9E3779B9u^rs;h^=h>>13;h*=0xC2B2AE35u;h^=h>>16;
        float n=(float)(int)h*4.6566129e-10f;
        float vL=fw_lp1(n,zL,cLo);vL=vL-fw_lp1(vL,zhL,cHi);
        float vR=fw_lp1(n,zR,cLo);vR=vR-fw_lp1(vR,zhR,cHi);
        float g=amp*(0.5f+0.5f*sinf(2.f*(float)M_PI*0.07f*i/sr));
        L[i]+=vL*g;R[i]+=vR*g*0.9f;
    }
}

/* renderFreeway — synthesises traffic into L/R and returns.
 * Called from main after the music FX bus; main then blends the result
 * into the music buffers weighted by (mix) vs (1-mix).
 *   trafficLevel  controls car density: 0=none 0.5=~8 cars 1.0=~20 2.0=~50
 *   honkLevel     controls horn probability per tailgating/random car
 *   seed2         independent RNG seed so traffic is reproducible but
 *                 does not disturb the music composer's RNG stream      */
static void renderFreeway(float*L,float*R,int N,float sr,
                           float trafficLevel,float honkLevel,unsigned seed2)
{
    if(trafficLevel<=0.f&&honkLevel<=0.f)return;
    float dur=N/sr;
    int nCars=(int)(trafficLevel*trafficLevel*25.f+trafficLevel*8.f);
    nCars=std::max(0,std::min(120,nCars));
    printf("Freeway: %d cars, traffic=%.1f honk=%.1f\n",nCars,trafficLevel,honkLevel);
    unsigned savedRng=g_rng;
    g_rng=seed2^0xDEADBEEFu;
    fw_renderWash(L,R,N,sr,trafficLevel,seed2);
    float t=fw_rr(1.f,5.f);
    for(int c=0;c<nCars&&t<dur-3.f;c++){
        FWCar car;
        bool isTail=(c>0)&&(fw_rnd()<0.40f*trafficLevel);
        float gap=isTail?fw_rr(0.5f,2.2f):fw_rr(2.f,12.f/fmaxf(trafficLevel,0.3f));
        t+=gap;if(t>=dur-2.f)break;
        fw_buildCar(car,t,trafficLevel,honkLevel,isTail);
        fw_renderCar(car,L,R,N,sr,seed2^(unsigned)(c*2654435761u));
    }
    g_rng=savedRng;
}

/* ══════════════════════════════════════════════════════════════════════
   FX BUS  (CPU)
   ══════════════════════════════════════════════════════════════════════ */
static void applyChorus(const float*mono,float*L,float*R,int N,float sr,
                        float rate,float depth,float mix)
{
    int maxD=(int)(0.06f*sr);
    float*dl=(float*)calloc(maxD,sizeof(float));
    int wp=0;
    float base=0.018f*sr,dep=depth*sr;
    for(int i=0;i<N;i++){
        dl[wp]=mono[i];
        float lph=rate*i/sr;
        float dL=base+dep*(0.5f+0.5f*sinf(TWO_PI*lph));
        float dR=base+dep*(0.5f+0.5f*sinf(TWO_PI*lph+2.1f));
        auto tap=[&](float d)->float{
            float rp=(float)wp-d;while(rp<0)rp+=maxD;
            int i0=(int)rp;float fr=rp-i0;int i1=(i0+1)%maxD;
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
    for(int i=0;i<NC;i++){
        csz[i]=(int)(cd[i]*sr*(0.8f+0.4f*room));
        cbL[i]=(float*)calloc(csz[i],sizeof(float));
        cbR[i]=(float*)calloc(csz[i]+7,sizeof(float));
    }
    for(int i=0;i<NA;i++){
        asz[i]=(int)(ad[i]*sr);
        abL[i]=(float*)calloc(asz[i],sizeof(float));
        abR[i]=(float*)calloc(asz[i],sizeof(float));
    }
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
        b[i*2]  =(int16_t)(fmaxf(-1.f,fminf(1.f,L[i]*g+d))*32767.f);
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

/* ══════════════════════════════════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════════════════════════════════ */
int main(int argc,char**argv)
{
    const char*out  =(argc>1)?argv[1]:"musetraffic01.wav"; /* output file      */
    int   sr        =(argc>2)?atoi(argv[2]):SR_DEFAULT;     /* sample rate Hz   */
    unsigned seed;                                          /* 0 = random       */
    auto makeSeed=[&]()->unsigned{
        return (unsigned)time(NULL)
              ^((unsigned)clock()*1000003u)
              ^((unsigned)(uintptr_t)(void*)&seed>>4);
    };
    if(argc>3){ seed=(unsigned)atoi(argv[3]); if(seed==0) seed=makeSeed(); }
    else seed=makeSeed();
    int   length    =(argc>4)?atoi(argv[4]):5;              /* 1-10 duration    */
    if(length<1)length=1;if(length>10)length=10;
    float kickBoost =(argc>5)?(float)atof(argv[5]):1.0f;   /* 0.1-3.0 kick     */
    if(kickBoost<0.1f)kickBoost=0.1f;if(kickBoost>3.0f)kickBoost=3.0f;
    float trafficLevel=(argc>6)?(float)atof(argv[6]):0.0f; /* 0-2 car density  */
    if(trafficLevel<0.f)trafficLevel=0.f;if(trafficLevel>2.f)trafficLevel=2.f;
    float honkLevel =(argc>7)?(float)atof(argv[7]):0.0f;   /* 0-2 horn rate    */
    if(honkLevel<0.f)honkLevel=0.f;if(honkLevel>2.f)honkLevel=2.f;
    /* mix: 0.0=music only  0.5=equal blend  1.0=traffic only  [default 0.2]
       writeWav normalises the final result so loudness is consistent.  */
    float mix=(argc>8)?(float)atof(argv[8]):0.2f;          /* 0-1 music/traffic*/
    if(mix<0.f)mix=0.f;if(mix>1.f)mix=1.f;
    sr_seed(seed); srand(g_rng);
    seed=g_rng;    /* capture actual seed (clock-derived if 0 was passed) */

    printf("══ MUSETRAFFIC01 — GPU Disco + CPU Freeway ══\n");
    printf("SEED: %u  LENGTH: %d/10  KICK: %.1fx  TRAFFIC: %.1f  HONK: %.1f  MIX: %.2f\n",
           seed,length,kickBoost,trafficLevel,honkLevel,mix);
    printf("Reproduce: musetraffic01 out.wav %d %u %d %.1f %.1f %.1f %.2f\n\n",
           sr,seed,length,kickBoost,trafficLevel,honkLevel,mix);
    printf("Sample rate: %d Hz\n",sr);
    fflush(stdout);

    cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("GPU: %s (SM %d.%d)\n\n",prop.name,prop.major,prop.minor);
    fflush(stdout);

    /* — compose — */
    printf("Composing...\n"); fflush(stdout);
    Patch*Ps=(Patch*)calloc(MAX_PATCH,sizeof(Patch));
    genBass(Ps[0]);genLead(Ps[1]);genRingPad(Ps[2]);
    genStrings(Ps[3]);genPerc(Ps[4],kickBoost);
    Note*notes=(Note*)calloc(MAX_NOTES,sizeof(Note));
    Composer C={};
    char report[512];
    int nNotes=compose(C,notes,report,length,kickBoost);
    printf("%s\nNotes: %d\n\n",report,nNotes);
    fflush(stdout);

    float maxR=0.f;
    for(int i=0;i<5;i++)maxR=fmaxf(maxR,Ps[i].aR);
    float endT=0.f;
    for(int i=0;i<nNotes;i++)
        endT=fmaxf(endT,notes[i].t+notes[i].dur+maxR);
    endT+=3.f;
    int N=(int)(endT*(float)sr);
    printf("Duration: %.1f s  (%d samples at %d Hz)\n",endT,N,sr);
    fflush(stdout);
    if(N<=0){ fprintf(stderr,"ERROR: N=%d, nothing to render\n",N); return 1; }

    /* — GPU allocations — */
    printf("Allocating GPU buffers (%.1f MB)...\n",(float)N*4/1e6f);
    fflush(stdout);

    /* — GPU synthesis — */
    Patch*dP;Note*dN;float*dBus;
    CUDA_CHECK(cudaMalloc(&dP,MAX_PATCH*sizeof(Patch)));
    CUDA_CHECK(cudaMalloc(&dN,MAX_NOTES*sizeof(Note)));
    CUDA_CHECK(cudaMalloc(&dBus,(size_t)N*4));
    CUDA_CHECK(cudaMemcpy(dP,Ps,MAX_PATCH*sizeof(Patch),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dN,notes,MAX_NOTES*sizeof(Note),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dBus,0,(size_t)N*4));

    cudaEvent_t e0,e1;
    CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    int nb=(nNotes+63)/64;
    voiceKernel<<<nb,64>>>(dP,dN,nNotes,dBus,N,(float)sr,seed);
    CUDA_CHECK(cudaEventRecord(e1));CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    printf("GPU kernel: %.0f ms (%.0fx real-time)\n",ms,endT*1000.f/ms);
    fflush(stdout);

    float*mono=(float*)malloc((size_t)N*4);
    CUDA_CHECK(cudaMemcpy(mono,dBus,(size_t)N*4,cudaMemcpyDeviceToHost));

    /* — stereo FX bus — */
    float*L=(float*)calloc(N,sizeof(float));
    float*R=(float*)calloc(N,sizeof(float));
    printf("FX bus: chorus + reverb...\n");
    applyChorus(mono,L,R,N,(float)sr,Ps[0].chRate,Ps[0].chDepth,Ps[0].chMix);
    applyReverb(L,R,N,(float)sr,Ps[0].rvSize,Ps[0].rvWet);

    /* — apply music/traffic gain balance —
       music gain  = 1 - mix  (1.0 at mix=0, 0.0 at mix=1)
       traffic gain = mix      (0.0 at mix=0, 1.0 at mix=1)
       writeWav normalises the final result, so only the ratio matters. */
    float musicGain   = 1.0f - mix;
    float trafficGain = (trafficLevel>0.f||honkLevel>0.f) ? mix : 0.f;
    if(trafficGain < 1e-4f && musicGain < 1e-4f) musicGain = 1.0f; /* safety */
    for(int i=0;i<N;i++){ L[i]*=musicGain; R[i]*=musicGain; }

    /* — freeway layer — */
    if(trafficLevel>0.f||honkLevel>0.f){
        /* render traffic into a temporary buffer then blend in          */
        float*tL=(float*)calloc(N,sizeof(float));
        float*tR=(float*)calloc(N,sizeof(float));
        renderFreeway(tL,tR,N,(float)sr,trafficLevel,honkLevel,seed^0xF0F0F0F0u);
        for(int i=0;i<N;i++){ L[i]+=tL[i]*trafficGain; R[i]+=tR[i]*trafficGain; }
        free(tL); free(tR);
    }

    printf("Writing WAV...\n"); fflush(stdout);
    writeWav(out,L,R,N,sr);

    free(Ps);free(notes);free(mono);free(L);free(R);
    cudaFree(dP);cudaFree(dN);cudaFree(dBus);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 0;
}
