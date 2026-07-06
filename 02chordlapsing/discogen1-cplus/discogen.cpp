/*
 * discogen_mac.cpp — Generative Disco Machine (macOS / CPU port)
 * ===============================================================
 * Identical musical output to discogen.cu; GPU replaced with a
 * simple per-note render loop on the CPU.
 *
 * BUILD (macOS, Xcode CLT or Homebrew clang):
 *   c++ -std=c++14 -O2 -o discogen discogen_mac.cpp -lm
 *
 * RUN:
 *   ./discogen                         new original song every run
 *   ./discogen out.wav 48000 SEED      reproduce a song you liked
 *
 * THE COMPOSITION MODEL — unchanged from the GPU version:
 *
 *  HARMONY  Markov chain over 6 functional chord degrees in a
 *           randomly chosen minor key; weighted transitions from
 *           disco practice; cadential forcing at phrase positions.
 *
 *  MELODY   2-bar seed motif developed by diatonic sequence,
 *           inversion, rhythmic displacement (phase-shifted off-beat
 *           entries).  Strong beats snap to chord tones; weak beats
 *           snap to scale tones.  Chorus hook derived separately.
 *
 *  VOICING  Nearest-inversion voice leading for strings / pad.
 *
 *  FORM     INTRO VERSE PRE CHORUS BREAK [VERSE] CHORUS OUTRO,
 *           driven by an energy curve (velocities, register, density).
 *
 *  TIMBRE   Five patches generated fresh each run: bass, phase-FM
 *           lead, ring-mod pad, strings, drums.  Detune spreads,
 *           FM indices, ring-pair intervals, filter shapes all vary.
 */

/** modeled after the nVidia 4060TI version of the app that runs in a second!
 is a plain C++ function called once per note in a sequential loop. The synthesis math inside is identical — same nonlinear Moog ladder (Padé tanh, 2× oversampled), same polyBLEP anti-aliasing, same oscillator routing, same ADSR 
*/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define TWO_PI  6.28318530717958647692f
#define SR_DEFAULT 48000
#define MAX_OSC    10
#define MAX_NOTES  6144
#define MAX_PATCH  8

/* ══════════════════════════════════════════════════════════════════════
   SYNTH DATA STRUCTURES  (identical to .cu)
   ══════════════════════════════════════════════════════════════════════ */
enum Wave  { W_SAW=0, W_SQR, W_TRI, W_SIN, W_NOISE };
enum LWave { L_SIN=0, L_TRI, L_SQR };

struct OscP {
    int   wave;
    float semi, cents, level, pw;
    int   fmSrc;   float fmAmt;
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
   CPU SYNTH HELPERS  (replacing __device__ functions)
   ══════════════════════════════════════════════════════════════════════ */
static inline float hnoise(unsigned i, unsigned s) {
    unsigned h = i*0x9E3779B9u ^ s*0x85EBCA6Bu;
    h ^= h>>13; h *= 0xC2B2AE35u; h ^= h>>16;
    return (float)(int)h * 4.6566129e-10f;
}
static inline float ftanh(float x) {
    float x2 = x*x;
    return x*(27.f+x2)/(27.f+9.f*x2);
}
static inline float polyblep(float t, float dt) {
    if (t < dt)       { t /= dt;       return  t+t - t*t - 1.f; }
    if (t > 1.f - dt) { t = (t-1.f)/dt; return t*t + t+t + 1.f; }
    return 0.f;
}
static float adsr(float t, float dur, float A, float D, float S, float R) {
    if (t < 0.f) return 0.f;
    float e;
    if      (t < A)     e = t / std::max(A, 1e-4f);
    else if (t < A+D)   e = 1.f - (t-A)/std::max(D,1e-4f)*(1.f-S);
    else if (t < dur)   e = S;
    else { float tr = t-dur; if (tr > R) return 0.f; e = S*(1.f-tr/std::max(R,1e-4f)); }
    return std::max(0.f, std::min(1.f, e));
}

/* ── render one note into an accumulation buffer (mono) ─────────────── */
static void renderNote(const Patch*Ps, const Note&nt,
                       float*bus, int N, float sr, unsigned seed)
{
    const Patch& P = Ps[nt.patch];
    float invSr  = 1.f / sr;
    int   startS = (int)(nt.t * sr);
    float totDur = nt.dur + P.aR + 0.05f;
    int   nSamp  = (int)(totDur * sr);

    float ph[MAX_OSC], oscOut[MAX_OSC];
    /* use hnoise for initial phases, seeded per note like the GPU did  */
    int vi = startS;   /* close enough as a unique-per-note index       */
    for (int i = 0; i < MAX_OSC; i++) {
        ph[i]     = hnoise((unsigned)(vi*17+i), seed)*0.5f + 0.5f;
        oscOut[i] = 0.f;
    }
    float s1=0,s2=0,s3=0,s4=0;

    float baseHz = 440.f * powf(2.f, ((float)nt.midi     - 69.f)/12.f);
    float prevHz = 440.f * powf(2.f, ((float)nt.prevMidi - 69.f)/12.f);

    for (int i = 0; i < nSamp; i++) {
        float t    = (float)i * invSr;
        float tAbs = nt.t + t;

        /* glide */
        float gl     = (P.glide > 1e-4f) ? std::min(1.f, t/P.glide) : 1.f;
        float noteHz = prevHz * powf(baseHz/prevHz, gl);

        /* LFO */
        float lph = P.lfoRate * tAbs; lph -= floorf(lph);
        float lfo;
        if      (P.lfoWave == L_SIN) lfo = sinf(TWO_PI*lph);
        else if (P.lfoWave == L_TRI) lfo = 4.f*fabsf(lph-0.5f) - 1.f;
        else                          lfo = (lph < 0.5f) ? 1.f : -1.f;

        /* sample & hold */
        float sh = 0.f;
        if (P.shRate > 0.01f)
            sh = hnoise((unsigned)floorf(tAbs*P.shRate), seed^0x5AAD);

        float pitchMod = exp2f((P.lfoToPitch*lfo + P.shToPitch*sh)/12.f);
        float f0 = noteHz * pitchMod;

        /* oscillators */
        float mix = 0.f;
        for (int o = 0; o < P.nOsc; o++) {
            const OscP& op = P.osc[o];
            float f  = f0 * exp2f((op.semi + op.cents*0.01f)/12.f);
            float dt = f * invSr;
            float pmod = 0.f;
            if (op.fmSrc >= 0 && op.fmSrc < o)
                pmod = op.fmAmt * oscOut[op.fmSrc] * 0.5f;
            ph[o] += dt; ph[o] -= floorf(ph[o]);
            float p = ph[o] + pmod; p -= floorf(p); if (p < 0.f) p += 1.f;
            float v;
            if (op.wave == W_SAW) {
                v = 2.f*p - 1.f - polyblep(p, dt);
            } else if (op.wave == W_SQR) {
                float pw = op.pw;
                v = (p < pw ? 1.f : -1.f);
                v += polyblep(p, dt);
                float p2 = p - pw; if (p2 < 0.f) p2 += 1.f;
                v -= polyblep(p2, dt);
            } else if (op.wave == W_TRI) {
                v = 4.f*fabsf(p - 0.5f) - 1.f;
            } else if (op.wave == W_SIN) {
                v = sinf(TWO_PI * p);
            } else {
                /* noise: hash of sample index, unique per oscillator   */
                v = hnoise((unsigned)(i*MAX_OSC + o), seed ^ (unsigned)(vi*31));
            }
            if (op.ringSrc >= 0 && op.ringSrc < o) v *= oscOut[op.ringSrc];
            oscOut[o] = v;
            mix += v * op.level;
        }

        /* filter envelope + modulated cutoff */
        float fenv = adsr(t, nt.dur, P.fA, P.fD, P.fS, P.fR);
        float cut  = P.cutoff
            * exp2f( P.fenvAmt * fenv * 5.f
                   + P.lfoToCut * lfo * 2.f
                   + P.shToCut  * sh  * 2.f
                   + P.keytrack * log2f(noteHz/261.63f));
        cut = std::max(20.f, std::min(0.45f*sr, cut));

        /* nonlinear Moog ladder (Huovilainen, 2× oversampled)         */
        float g = 1.f - expf(-TWO_PI * cut * invSr * 0.5f);
        float k = 4.f * P.res;
        float x = mix * P.drive;
        for (int os = 0; os < 2; os++) {
            float in = ftanh(x - k*s4);
            s1 += g*(in   - s1);
            s2 += g*(ftanh(s1) - ftanh(s2));
            s3 += g*(ftanh(s2) - ftanh(s3));
            s4 += g*(ftanh(s3) - ftanh(s4));
        }
        float out = s4 / std::max(0.3f, P.drive*0.7f);

        /* VCA */
        float aenv = adsr(t, nt.dur, P.aA, P.aD, P.aS, P.aR);
        if (aenv <= 0.f && t > nt.dur) break;
        float trem = 1.f;
        if (P.tremDepth > 0.001f) {
            float tp = P.tremRate * tAbs; tp -= floorf(tp);
            trem = 1.f - P.tremDepth*(0.5f + 0.5f*sinf(TWO_PI*tp));
        }
        float s = out * aenv * nt.vel * trem * 0.5f;

        int gI = startS + i;
        if (gI >= 0 && gI < N) bus[gI] += s;
    }
}

/* ══════════════════════════════════════════════════════════════════════
   COMPOSER  (CPU — identical logic to discogen.cu)
   ══════════════════════════════════════════════════════════════════════ */
static unsigned g_rng;
static void   sr_seed(unsigned s){ g_rng = s ? s : 1; }
static float  rnd(){ g_rng^=g_rng<<13; g_rng^=g_rng>>17; g_rng^=g_rng<<5;
                     return (float)(g_rng&0xFFFFFF)/16777216.f; }
static float  rr(float a,float b){ return a+(b-a)*rnd(); }
static int    ri(int a,int b){ return a+(int)(rnd()*(float)(b-a+1)*0.9999f); }
static int    pick(const float*w,int n){
    float s=0; for(int i=0;i<n;i++) s+=w[i];
    float x=rnd()*s;
    for(int i=0;i<n;i++){ x-=w[i]; if(x<=0) return i; }
    return n-1;
}

static const int MINOR[7] = {0,2,3,5,7,8,10};
struct Chord { int deg, type; };
static const int CH_ROOT[6] = {0,3,4,5,6,2};
static const int CH_TYPE[6] = {0,0,2,1,1,1};
static const float MARKOV[6][6] = {
    {0.10f,0.22f,0.16f,0.26f,0.20f,0.06f},
    {0.30f,0.06f,0.30f,0.12f,0.16f,0.06f},
    {0.55f,0.08f,0.05f,0.20f,0.08f,0.04f},
    {0.16f,0.14f,0.16f,0.06f,0.42f,0.06f},
    {0.48f,0.08f,0.12f,0.16f,0.06f,0.10f},
    {0.16f,0.20f,0.14f,0.30f,0.14f,0.06f},
};

struct Composer {
    int   key, tempo;
    Note* notes; int n;
    int   lastMidi[5];
    int   vMotifDeg[8]; float vMotifT[8], vMotifD[8]; int vMotifN;
    int   cMotifDeg[8]; float cMotifT[8], cMotifD[8]; int cMotifN;
    int   strVoice[4], padVoice[4];
    float beat;
};

static void emit(Composer&C,float beat,float durB,int midi,float vel,int patch){
    if(C.n>=MAX_NOTES) return;
    if(midi<12||midi>108) return;
    C.notes[C.n]={beat*C.beat, durB*C.beat, vel, midi, C.lastMidi[patch], patch};
    C.lastMidi[patch]=midi;
    C.n++;
}
static void chordTones(const Composer&C,Chord c,int oct,int*out){
    int root=C.key+MINOR[c.deg]+12*oct;
    out[0]=root; out[1]=root+((c.type==0)?3:4);
    out[2]=root+7; out[3]=(c.type==2)?root+10:root+12;
}
static int snapChord(const Composer&C,Chord c,int midi){
    int t[4]; chordTones(C,c,0,t);
    int best=midi,bd=99;
    for(int o=-24;o<=48;o+=12) for(int i=0;i<4;i++){
        int d=abs(midi-(t[i]+o)); if(d<bd){bd=d;best=t[i]+o;}
    } return best;
}
static int snapScale(const Composer&C,int midi){
    int best=midi,bd=99;
    for(int o=-24;o<=60;o+=12) for(int i=0;i<7;i++){
        int c=C.key+MINOR[i]+o, d=abs(midi-c); if(d<bd){bd=d;best=c;}
    } return best;
}
static void makeProg(Chord*prog,int start){
    int cur=start;
    for(int i=0;i<4;i++){
        prog[i]={CH_ROOT[cur],CH_TYPE[cur]};
        if(i==2){ float w[6]; memcpy(w,MARKOV[cur],sizeof(w));
                  w[2]*=3.f; w[4]*=2.f; cur=pick(w,6); }
        else cur=pick(MARKOV[cur],6);
    }
}

struct RCell{ int n; float t[6],d[6]; };
static const RCell RHY[]={
    {4,{0,1,2,3},            {0.9f,0.9f,0.9f,0.9f}},
    {5,{0.5f,1,1.5f,2.5f,3}, {0.4f,0.4f,0.9f,0.4f,0.9f}},
    {5,{0,0.5f,1.5f,2,3},    {0.4f,0.9f,0.4f,0.9f,0.9f}},
    {6,{0,0.5f,1,1.5f,2.5f,3},{0.4f,0.4f,0.4f,0.9f,0.4f,0.9f}},
    {4,{0.5f,1.5f,2,3},      {0.9f,0.4f,0.9f,0.9f}},
    {3,{0,1.5f,2.5f},        {1.4f,0.9f,1.4f}},
    {5,{0,1,1.5f,2,2.5f},    {0.9f,0.4f,0.4f,0.4f,1.4f}},
};
#define NRHY (int)(sizeof(RHY)/sizeof(RHY[0]))

static void makeMotif(int*deg,float*tt,float*dd,int*nn,int span){
    int n=0, curDeg=ri(0,6);
    for(int bar=0;bar<2;bar++){
        const RCell&rc=RHY[ri(0,NRHY-1)];
        for(int i=0;i<rc.n&&n<8;i++){
            tt[n]=bar*4+rc.t[i]; dd[n]=rc.d[i]; deg[n]=curDeg;
            float r=rnd();
            int step=(r<0.42f)?1:(r<0.78f)?-1:(r<0.9f)?ri(2,3):-ri(2,3);
            curDeg+=step;
            if(curDeg>span) curDeg=span-ri(1,2);
            if(curDeg<-1)   curDeg=ri(0,1);
            n++;
        }
    }
    *nn=n;
}

enum Dev { DEV_NONE=0, DEV_SEQ_UP, DEV_SEQ_DN, DEV_INV, DEV_SHIFT };
static void playMotif(Composer&C,const int*deg,const float*tt,const float*dd,
                      int n,float barBeat,Chord ch,int baseMidi,
                      int dev,float vel,int patch)
{
    float shift=(dev==DEV_SHIFT)?0.5f:0.f;
    for(int i=0;i<n;i++){
        int d=deg[i];
        if(dev==DEV_SEQ_UP) d+=1;
        if(dev==DEV_SEQ_DN) d-=1;
        if(dev==DEV_INV)    d=-d+2;
        int midi=baseMidi+((d>=0)? MINOR[d%7]+12*(d/7)
                                 : MINOR[(d%7+7)%7]-12*((-d+6)/7));
        float pos=tt[i];
        bool strong=(fmodf(pos,1.f)<0.01f);
        midi=strong? snapChord(C,ch,midi):snapScale(C,midi);
        float v=vel*(strong?1.f:0.9f)*rr(0.93f,1.f);
        emit(C,barBeat+pos+shift,dd[i],midi,v,patch);
    }
}

static void voiceLead(Composer&C,Chord ch,int*voice,int lowAnchor){
    int t[4]; chordTones(C,ch,0,t);
    for(int v2=0;v2<4;v2++){
        int want=t[v2%4];
        int prev=voice[v2]? voice[v2]:lowAnchor+v2*4;
        int best=want,bd=99;
        for(int o=-24;o<=36;o+=12){
            int cand=want+o;
            if(cand<lowAnchor-6||cand>lowAnchor+34) continue;
            int d2=abs(cand-prev); if(d2<bd){bd=d2;best=cand;}
        }
        voice[v2]=best;
    }
}

static void bassBar(Composer&C,float barBeat,Chord ch,Chord next,float energy){
    int t[4]; chordTones(C,ch,0,t);
    int root=t[0];
    while(root>C.key+MINOR[ch.deg]-12+24) root-=12;
    root=snapScale(C,root); if(root<28) root+=12;
    int nt2[4]; chordTones(C,next,0,nt2);
    int nroot=nt2[0];
    while(nroot>root+8) nroot-=12;
    while(nroot<root-8) nroot+=12;
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

static void percBar(Composer&C,float barBeat,float energy,int fill){
    for(int b=0;b<4;b++){
        emit(C,barBeat+b,0.35f,24,0.95f*energy+0.2f,4);
        if(b%2==1) emit(C,barBeat+b,0.22f,38,0.8f*energy+0.15f,4);
    }
    if(fill) for(int i=0;i<8;i++)
        emit(C,barBeat+2+i*0.25f,0.15f,38,0.5f+0.06f*i,4);
}

/* ── patch generators (identical to .cu) ───────────────────────────── */
static void initPatch(Patch&P){
    memset(&P,0,sizeof(P));
    for(int i=0;i<MAX_OSC;i++){P.osc[i].fmSrc=-1;P.osc[i].ringSrc=-1;P.osc[i].pw=0.5f;}
    P.drive=1.f; P.cutoff=1000.f; P.res=0.3f;
    P.aA=0.005f;P.aD=0.2f;P.aS=0.7f;P.aR=0.2f;
    P.fA=0.005f;P.fD=0.2f;P.fS=0.3f;P.fR=0.2f;
}
static void genBass(Patch&P){
    initPatch(P); P.nOsc=10;
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
    P.cutoff=rr(75.f,110.f); P.res=rr(0.72f,0.88f);
    P.fenvAmt=rr(0.8f,0.92f); P.keytrack=0.6f; P.drive=rr(1.7f,2.4f);
    P.aA=0.003f; P.aD=rr(0.12f,0.18f); P.aS=rr(0.4f,0.52f); P.aR=0.13f;
    P.fA=0.001f; P.fD=rr(0.08f,0.14f); P.fS=0.06f; P.fR=0.1f;
    P.glide=rr(0.03f,0.05f);
    P.chRate=rr(0.4f,0.7f); P.chDepth=rr(0.004f,0.007f); P.chMix=rr(0.32f,0.42f);
    P.rvSize=rr(0.72f,0.88f); P.rvWet=rr(0.18f,0.26f);
}
static void genLead(Patch&P){
    initPatch(P); P.nOsc=10;
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
    P.lfoRate=rr(5.2f,6.1f); P.lfoWave=L_SIN;
    P.lfoToPitch=rr(0.03f,0.07f); P.lfoToCut=0.04f;
    P.shRate=rr(6.f,9.f); P.shToCut=rr(0.08f,0.16f);
    P.cutoff=rr(1500.f,2400.f); P.res=rr(0.35f,0.55f);
    P.fenvAmt=rr(0.4f,0.6f); P.keytrack=0.7f; P.drive=rr(1.2f,1.6f);
    P.aA=0.006f; P.aD=0.3f; P.aS=0.72f; P.aR=rr(0.35f,0.55f);
    P.fA=0.004f; P.fD=0.22f; P.fS=0.3f; P.fR=0.35f;
    P.glide=rr(0.04f,0.07f);
}
static void genRingPad(Patch&P){
    initPatch(P); P.nOsc=10;
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
    P.lfoRate=rr(3.2f,4.4f); P.lfoWave=L_TRI;
    P.lfoToPitch=0.02f; P.lfoToCut=rr(0.04f,0.08f);
    P.tremRate=P.lfoRate+rr(0.05f,0.2f); P.tremDepth=rr(0.08f,0.14f);
    P.cutoff=rr(750.f,1100.f); P.res=0.28f;
    P.fenvAmt=0.4f; P.keytrack=0.55f; P.drive=1.1f;
    P.aA=rr(0.3f,0.5f); P.aD=0.6f; P.aS=0.85f; P.aR=rr(1.0f,1.5f);
    P.fA=0.5f; P.fD=0.8f; P.fS=0.6f; P.fR=1.0f;
}
static void genStrings(Patch&P){
    initPatch(P); P.nOsc=10;
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
    P.shRate=rr(2.6f,3.6f); P.shToCut=rr(0.05f,0.09f);
    P.lfoRate=5.3f; P.lfoWave=L_SIN; P.lfoToPitch=0.018f; P.lfoToCut=0.02f;
    P.cutoff=rr(2000.f,2800.f); P.res=0.2f;
    P.fenvAmt=0.3f; P.keytrack=0.55f; P.drive=1.05f;
    P.aA=rr(0.24f,0.34f); P.aD=0.55f; P.aS=0.84f; P.aR=rr(0.85f,1.1f);
    P.fA=0.4f; P.fD=0.65f; P.fS=0.6f; P.fR=0.9f;
}
static void genPerc(Patch&P){
    initPatch(P); P.nOsc=10;
    P.osc[0]={W_SIN,0,0,0.72f,0.5f,-1,0.f,-1};
    P.osc[1]={W_SIN,24,0,0.55f,0.5f,0,rr(7.f,9.5f),-1};
    P.osc[2]={W_NOISE,0,0,0.26f,0.5f,-1,0.f,-1};
    P.osc[3]={W_TRI,0,0,0.46f,0.5f,-1,0.f,-1};
    P.osc[4]={W_NOISE,0,0,0.6f,0.5f,-1,0.f,3};
    P.osc[5]={W_NOISE,0,0,0.55f,0.5f,-1,0.f,-1};
    P.osc[6]={W_SIN,0,0,0.2f,0.5f,-1,0.f,-1};
    P.osc[7]={W_SIN,7,0,0.12f,0.5f,-1,0.f,-1};
    P.osc[8]={W_SIN,-12,0,0.12f,0.5f,-1,0.f,-1};
    P.osc[9]={W_NOISE,0,2,0.26f,0.5f,-1,0.f,-1};
    P.cutoff=rr(7500.f,9500.f); P.res=0.32f; P.fenvAmt=0.5f;
    P.keytrack=0.f; P.drive=1.f;
    P.aA=0.001f; P.aD=rr(0.05f,0.07f); P.aS=0.04f; P.aR=0.07f;
    P.fA=0.001f; P.fD=0.04f; P.fS=0.f; P.fR=0.05f;
}

/* ── song assembly (identical to .cu) ──────────────────────────────── */
struct Section { const char*name; int bars; float energy; int kind; };
enum { S_INTRO,S_VERSE,S_PRE,S_CHORUS,S_BREAK,S_OUTRO };

static int compose(Composer&C,Note*notes,char*report)
{
    C.notes=notes; C.n=0;
    for(int i=0;i<5;i++) C.lastMidi[i]=48;
    memset(C.strVoice,0,sizeof(C.strVoice));
    memset(C.padVoice,0,sizeof(C.padVoice));
    C.key=48+ri(0,11);
    C.tempo=ri(112,126);
    C.beat=60.f/(float)C.tempo;

    Section form[10]; int nf=0;
    form[nf++]={"INTRO", ri(4,8),  0.30f,S_INTRO};
    form[nf++]={"VERSE", 16,       0.55f,S_VERSE};	/* change '16' to '32' to make the piece twice as long.
    form[nf++]={"PRE",   8,        0.72f,S_PRE};
    form[nf++]={"CHORUS",16,       0.95f,S_CHORUS};
    form[nf++]={"BREAK", 8,        0.42f,S_BREAK};
    if(rnd()<0.6f) form[nf++]={"VERSE",8,0.62f,S_VERSE};
    form[nf++]={"CHORUS",16,       1.00f,S_CHORUS};
    form[nf++]={"OUTRO", ri(6,10), 0.35f,S_OUTRO};

    Chord vProg[4],cProg[4],bProg[4];
    makeProg(vProg,0);
    makeProg(cProg,(rnd()<0.5f)?3:0);
    bProg[0]={CH_ROOT[3],CH_TYPE[3]}; bProg[1]={CH_ROOT[4],CH_TYPE[4]};
    bProg[2]={CH_ROOT[0],CH_TYPE[0]}; bProg[3]={CH_ROOT[2],CH_TYPE[2]};

    makeMotif(C.vMotifDeg,C.vMotifT,C.vMotifD,&C.vMotifN,5);
    makeMotif(C.cMotifDeg,C.cMotifT,C.cMotifD,&C.cMotifN,6);

    int melBase=C.key+24; if(melBase<66) melBase+=12;

    float bar=0.f;
    for(int s=0;s<nf;s++){
        Section&S=form[s];
        Chord*prog=(S.kind==S_CHORUS)?cProg:(S.kind==S_BREAK)?bProg:vProg;
        for(int b=0;b<S.bars;b++){
            float bb=(bar+b)*4.f;
            Chord ch=prog[b%4], nx=prog[(b+1)%4];
            float E=S.energy;

            if(S.kind!=S_INTRO&&!(S.kind==S_OUTRO&&b>=S.bars-2))
                percBar(C,bb,E,(b%8==7&&E>0.5f));
            if(S.kind!=S_INTRO||b>=S.bars/2)
                bassBar(C,bb,ch,nx,E);

            if(S.kind!=S_BREAK||b>=S.bars-2){
                voiceLead(C,ch,C.strVoice,C.key+7);
                float sv=0.35f+0.4f*E;
                if(S.kind==S_OUTRO) sv*=1.f-(float)b/S.bars;
                for(int v2=0;v2<4;v2++) emit(C,bb,3.7f,C.strVoice[v2],sv,3);
            }
            if(b%2==0&&S.kind!=S_INTRO){
                voiceLead(C,ch,C.padVoice,C.key+12);
                float pv=0.3f+0.28f*E;
                if(S.kind==S_OUTRO) pv*=1.f-(float)b/S.bars;
                for(int v2=0;v2<4;v2++) emit(C,bb,7.4f,C.padVoice[v2],pv,2);
            }

            if(S.kind==S_VERSE||S.kind==S_CHORUS||
               S.kind==S_PRE||(S.kind==S_BREAK&&b>=2)||
               (S.kind==S_INTRO&&b>=S.bars/2)||S.kind==S_OUTRO){
                if(b%2==0){
                    int dev=DEV_NONE;
                    int ph=(b/2)%4;
                    if(ph==1) dev=(rnd()<0.5f)?DEV_SEQ_UP:DEV_SHIFT;
                    if(ph==2) dev=(rnd()<0.4f)?DEV_INV:DEV_SEQ_DN;
                    if(ph==3) dev=DEV_SHIFT;
                    int reg=melBase;
                    if(S.kind==S_CHORUS) reg+=3;
                    if(S.kind==S_PRE)    reg+=(b/2);
                    if(S.kind==S_OUTRO||S.kind==S_INTRO) reg-=2;
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
    int t[4]; Chord tonic={0,0}; chordTones(C,tonic,0,t);
    emit(C,endB,10.f,C.key-12,0.9f,0);
    for(int v2=0;v2<4;v2++){
        emit(C,endB,10.f,t[v2]+12,0.55f,3);
        emit(C,endB,10.f,t[v2]+24,0.4f,2);
    }
    emit(C,endB+1,8.f,C.key+36,0.6f,1);

    static const char*NOTE_N[12]={"C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"};
    static const char*RN[7]={"i","ii","III","iv","v","VI","VII"};
    char*p=report;
    p+=sprintf(p,"Key: %s minor   Tempo: %d BPM\nForm: ",NOTE_N[C.key%12],C.tempo);
    for(int s=0;s<nf;s++) p+=sprintf(p,"%s(%d) ",form[s].name,form[s].bars);
    p+=sprintf(p,"\nVerse prog:  "); for(int i=0;i<4;i++) p+=sprintf(p,"%s ",RN[vProg[i].deg]);
    p+=sprintf(p,"\nChorus prog: "); for(int i=0;i<4;i++) p+=sprintf(p,"%s ",RN[cProg[i].deg]);
    p+=sprintf(p,"\n");
    return C.n;
}

/* ══════════════════════════════════════════════════════════════════════
   FX BUS + WAV  (identical to .cu CPU side)
   ══════════════════════════════════════════════════════════════════════ */
static void applyChorus(const float*mono,float*L,float*R,int N,float sr,
                        float rate,float depth,float mix)
{
    int maxD=(int)(0.06f*sr);
    float*dl=(float*)calloc(maxD,sizeof(float));
    int wp=0;
    float base=0.018f*sr, dep=depth*sr;
    for(int i=0;i<N;i++){
        dl[wp]=mono[i];
        float lph=rate*i/sr;
        float dL=base+dep*(0.5f+0.5f*sinf(TWO_PI*lph));
        float dR=base+dep*(0.5f+0.5f*sinf(TWO_PI*lph+2.1f));
        auto tap=[&](float d)->float{
            float rp=(float)wp-d; while(rp<0)rp+=maxD;
            int i0=(int)rp; float fr=rp-i0; int i1=(i0+1)%maxD;
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
    if(wet<0.005f) return;
    static const float cd[]={0.02972f,0.03168f,0.03396f,0.03668f,
                              0.03856f,0.04024f,0.04228f,0.04384f};
    static const float ad[]={0.0053f,0.0071f,0.0102f,0.0123f};
    int NC=8,NA=4; float fb=0.78f*room+0.1f, damp=0.32f;
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
            cpL[i]=(cpL[i]+1==csz[i])?0:cpL[i]+1; oL+=dL;
            float dR=cbR[i][cpR[i]];
            lpR[i]=dR*(1.f-damp)+lpR[i]*damp;
            cbR[i][cpR[i]]=iR+lpR[i]*fb;
            cpR[i]=(cpR[i]+1==csz[i])?0:cpR[i]+1; oR+=dR;
        }
        for(int i=0;i<NA;i++){
            float bL=abL[i][apL[i]]; float vL=oL+bL*0.5f;
            abL[i][apL[i]]=vL; apL[i]=(apL[i]+1==asz[i])?0:apL[i]+1; oL=bL-vL;
            float bR=abR[i][apR[i]]; float vR=oR+bR*0.5f;
            abR[i][apR[i]]=vR; apR[i]=(apR[i]+1==asz[i])?0:apR[i]+1; oR=bR-vR;
        }
        L[s]=iL*(1.f-wet)+oL*wet/(float)NC;
        R[s]=iR*(1.f-wet)+oR*wet/(float)NC;
    }
    for(int i=0;i<NC;i++){free(cbL[i]);free(cbR[i]);}
    for(int i=0;i<NA;i++){free(abL[i]);free(abR[i]);}
}

static void writeWav(const char*path,const float*L,const float*R,int N,int sr){
    FILE*f=fopen(path,"wb");
    if(!f){fprintf(stderr,"Cannot open %s\n",path);return;}
    int16_t*b=(int16_t*)malloc((size_t)N*4);
    float pk=1e-9f;
    for(int i=0;i<N;i++){pk=std::max(pk,fabsf(L[i]));pk=std::max(pk,fabsf(R[i]));}
    float g=0.93f/pk;
    for(int i=0;i<N;i++){
        float d=(((float)(rand()&0xFFFF)+(float)(rand()&0xFFFF))/65536.f-1.f)/32768.f;
        b[i*2  ]=(int16_t)(std::max(-1.f,std::min(1.f,L[i]*g+d))*32767.f);
        b[i*2+1]=(int16_t)(std::max(-1.f,std::min(1.f,R[i]*g+d))*32767.f);
    }
    uint32_t dS=(uint32_t)N*4, cS=36+dS, bR=(uint32_t)sr*4;
    uint16_t bA=4,bp=16,fm=1,ch=2; uint32_t sc=16;
    fwrite("RIFF",1,4,f); fwrite(&cS,4,1,f); fwrite("WAVEfmt ",1,8,f);
    fwrite(&sc,4,1,f); fwrite(&fm,2,1,f); fwrite(&ch,2,1,f);
    fwrite(&sr,4,1,f); fwrite(&bR,4,1,f); fwrite(&bA,2,1,f); fwrite(&bp,2,1,f);
    fwrite("data",1,4,f); fwrite(&dS,4,1,f);
    fwrite(b,2,(size_t)N*2,f);
    free(b); fclose(f);
    printf("Wrote %s  (%.1f s, %d Hz stereo)\n",path,(double)N/sr,sr);
}

/* ══════════════════════════════════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════════════════════════════════ */
int main(int argc,char**argv)
{
    const char*out =(argc>1)?argv[1]:"disco_gen.wav";
    int   sr       =(argc>2)?atoi(argv[2]):SR_DEFAULT;
    unsigned seed  =(argc>3)?(unsigned)atoi(argv[3]):(unsigned)time(NULL);
    sr_seed(seed); srand(seed);

    printf("═══ GENERATIVE DISCO MACHINE (macOS CPU) ═══\n");
    printf("SEED: %u  (rerun './discogen out.wav %d %u')\n\n",seed,sr,seed);

    /* — compose — */
    Patch*Ps=(Patch*)calloc(MAX_PATCH,sizeof(Patch));
    genBass(Ps[0]); genLead(Ps[1]); genRingPad(Ps[2]);
    genStrings(Ps[3]); genPerc(Ps[4]);

    Note*notes=(Note*)calloc(MAX_NOTES,sizeof(Note));
    Composer C={};
    char report[512];
    int nNotes=compose(C,notes,report);
    printf("%s",report);
    printf("Notes: %d\n\n",nNotes);

    float maxR=0.f;
    for(int i=0;i<5;i++) maxR=std::max(maxR,Ps[i].aR);
    float endT=0.f;
    for(int i=0;i<nNotes;i++)
        endT=std::max(endT,notes[i].t+notes[i].dur+maxR);
    endT+=3.f;
    int N=(int)(endT*(float)sr);
    printf("Duration: %.1f s   Notes: %d\n",endT,nNotes);

    /* — render all notes into mono accumulation buffer — */
    float*mono=(float*)calloc(N,sizeof(float));
    printf("Rendering");
    for(int i=0;i<nNotes;i++){
        renderNote(Ps,notes[i],mono,N,(float)sr,seed);
        if(i%(nNotes/20+1)==0){printf(".");fflush(stdout);}
    }
    printf("\n");

    /* — stereo FX bus — */
    float*L=(float*)calloc(N,sizeof(float));
    float*R=(float*)calloc(N,sizeof(float));
    printf("FX bus: chorus + reverb...\n");
    applyChorus(mono,L,R,N,(float)sr,Ps[0].chRate,Ps[0].chDepth,Ps[0].chMix);
    applyReverb(L,R,N,(float)sr,Ps[0].rvSize,Ps[0].rvWet);

    writeWav(out,L,R,N,sr);

    free(Ps); free(notes); free(mono); free(L); free(R);
    return 0;
}

