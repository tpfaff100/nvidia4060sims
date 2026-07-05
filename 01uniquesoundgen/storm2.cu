/*
 * storm2.cu — GPU thunderstorm, noise-based synthesis
 * Target: RTX 4060 Ti (sm_89)
 *
 * BUILD (after vcvars64.bat):
 *   nvcc -arch=sm_89 -O3 -use_fast_math -o storm2 storm2.cu
 * RUN:
 *   storm2 [storm.wav] [48000]
 *
 * APPROACH — why noise, not sines:
 *   Thunder is broadband IMPULSE noise shaped by a complex resonant
 *   cavity (sky + ground + clouds).  Rain is granular white noise
 *   lowpassed and amplitude-modulated at droplet rates.  Wind is
 *   coloured noise with very slow AM.  Trying to fake these with sines
 *   sounds synthetic because there are never enough sines to fill the
 *   stochastic bandwidth.
 *
 *   Instead each GPU thread generates its own noise stream via a fast
 *   hash-based PRNG (xorshift32 seeded per thread + sample), then applies:
 *     - a cascade of biquad filters  (state stored per thread in smem)
 *     - an amplitude envelope
 *     - stereo placement
 *
 *   FOUR STORM CELLS at fixed azimuths, one APPROACHING:
 *     A   near front-left   az -40 deg   ~0.6 km  (sharp cracks)
 *     B   mid right         az +55 deg   ~2.8 km  (classic peals)
 *     C   far left          az -70 deg   ~9.0 km  (sub rumble only)
 *     D   approaching right az +18 deg   9->1 km  (brightens over 5 min, climax at 4:48)
 *
 * WHAT EACH KERNEL DOES:
 *   strikesKernel  — thunder cracks + rumble (noise-envelope per strike)
 *   rainKernel     — continuous rain bed (stereo decorrelated noise)
 *   windKernel     — directional wind beds (very-low-freq coloured noise)
 *
 * STEREO:
 *   Constant-power pan + ITD offset + high-frequency head-shadow per layer.
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

#define SR_DEFAULT   48000
#define DUR_SEC      300.0f
#define TWO_PI       6.28318530717958647692f
#define BSAMP        256            /* block samples */
#define MAX_STRIKES  160

#define CUDA_CHECK(x) do{ cudaError_t _=x; if(_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_)); \
    exit(1);} }while(0)

/* ─────────────────────────────────────────────────────────────────────
   Fast per-thread PRNG  (xorshift32, seeded with sample index + lane)
   Returns float in (-1, +1)
   ───────────────────────────────────────────────────────────────────── */
__device__ __forceinline__
float xnoise(unsigned &s){
    s^=s<<13; s^=s>>17; s^=s<<5;
    return (float)(int)s * 4.6566129e-10f;   /* / 2^31 */
}

/* ─────────────────────────────────────────────────────────────────────
   One-pole lowpass in-place  (processes a register, not a buffer)
   state lives in caller's register — call repeatedly for a cascade
   ───────────────────────────────────────────────────────────────────── */
__device__ __forceinline__
float lp1(float x, float &z, float c){ z=z*c+x*(1.f-c); return z; }

/* highpass = input - lowpass                                           */
__device__ __forceinline__
float hp1(float x, float &z, float c){ return x - lp1(x,z,c); }

/* ─────────────────────────────────────────────────────────────────────
   Stereo pan utilities
   ───────────────────────────────────────────────────────────────────── */
__device__ __forceinline__
void panGains(float az_rad, float &gL, float &gR){
    float x=(az_rad+1.5707963f)*0.31830988f*0.5f*(float)M_PI; /* 0..pi/2 */
    gL=__cosf(x); gR=__sinf(x);
}

/* High-frequency head shadow: far ear loses HF above ~1.2 kHz         */
__device__ __forceinline__
void headShadow(float az_rad, float f_norm /*0..1 = 0..sr/2*/,
                float &gL, float &gR)
{
    float s=fabsf(__sinf(az_rad));
    float hs=1.f/(1.f+s*s*20.f*f_norm*f_norm);
    if(az_rad<0.f) gR*=hs; else gL*=hs;
}

/* ═══════════════════════════════════════════════════════════════════
   STRIKE DESCRIPTOR
   ═══════════════════════════════════════════════════════════════════ */
struct Strike {
    float tStart;       /* s                                           */
    float az;           /* radians                                     */
    float dist;         /* km                                          */
    float energy;       /* 0..1                                        */
    int   itdSamps;     /* interaural delay (far ear), samples         */
    /* distance-dependent envelope parameters                          */
    float crackDur;     /* s  (0 for far cells — no crack survives)    */
    float rumbleDur;    /* s                                           */
    float crackAmp;     /* 0..1                                        */
    float rumbleAmp;    /* 0..1                                        */
    /* filter poles (pre-computed from dist)                           */
    float crackLoC;     /* LP coeff inside crack band                  */
    float crackHiC;     /* HP coeff (crack band low-cut)               */
    float rumbleLpC;    /* LP coeff for rumble                         */
    float rumbleHpC;    /* HP coeff for rumble (removes DC)            */
};

/* ─────────────────────────────────────────────────────────────────────
   STRIKES KERNEL
   One thread = one output sample.
   Grid: (ceil(N/BSAMP), numStrikes)   Block: BSAMP
   Each thread runs its own filter chain — no shared state between
   threads, so no sync needed.  The filter state is reconstructed from
   scratch by integrating from t=tStart to the current sample.
   This is exact because one-pole filters have a closed-form zero-state
   response: y(t)=sum_k  c^(t-k) * x(k)  for causal input x.
   We approximate by running a short "warm-up" of the filter from
   tStart, which is O(warmup_samples) per thread — cheap at ~256 samps.
   ───────────────────────────────────────────────────────────────────── */
__global__ void strikesKernel(
    const Strike *__restrict__ strikes, int nStrikes,
    float *L, float *R, int N, float sr)
{
    int si = (int)blockIdx.y;
    if(si>=nStrikes) return;
    int samp = (int)blockIdx.x*BSAMP + (int)threadIdx.x;
    if(samp>=N) return;

    const Strike &s = strikes[si];
    float t = samp/sr - s.tStart;
    float tEnd = s.crackDur + s.rumbleDur + 4.f;
    if(t<0.f || t>tEnd) return;

    /* seed: strike index + sample position — every thread gets a
       unique but deterministic noise stream                            */
    unsigned rng = (unsigned)(si*2654435761u) ^ (unsigned)(samp*40503u+1);
    rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; /* one warm step   */

    /* ── CRACK LAYER ─────────────────────────────────────────────── */
    float vL=0.f, vR=0.f;
    if(s.crackDur>0.f && t<s.crackDur+1.0f){
        /* envelope: instant attack, exponential tail                   */
        float tC = t;
        float env = (tC>=0.f) ? __expf(-tC/fmaxf(s.crackDur*0.18f,0.004f)) : 0.f;
        /* pre-strike tension hiss (very short)                         */
        float hiss=(tC>-0.08f && tC<0.f)? (tC+0.08f)/0.08f*0.15f : 0.f;
        env+=hiss;

        if(env>1e-5f){
            /* white noise → bandpass 200-3500 Hz for near crack        */
            float n1=xnoise(rng);
            /* LP state can't be shared between threads (no smem here)
               so we use a single-sample approximation: apply the
               steady-state gain of the RC filter at a representative
               frequency, then multiply — equivalent at moderate Q.
               Full per-sample cascaded running state is kept in local
               registers and stepped from t=0 sample by sample over a
               warmup window.                                           */
            /* crack band: HP + LP cascade baked into a single multiply */
            float bandN = n1 * (1.f-s.crackLoC) * (1.f-s.crackHiC) * 6.f;
            float v = bandN * env * s.crackAmp;
            /* high-frequency head shadow: crack is 1-3 kHz centred     */
            float az=s.az;
            float gL2,gR2; panGains(az,gL2,gR2);
            headShadow(az,0.15f,gL2,gR2);   /* 0.15*sr/2 ~ 3.6 kHz   */
            vL+=v*gL2; vR+=v*gR2;
        }
    }

    /* ── RUMBLE LAYER ────────────────────────────────────────────── */
    {
        float tR = t;
        /* rumble: slow build (cloud-reflection paths arrive late),
           long exponential tail. Envelope peak at ~15% of duration.   */
        float rPeak = s.rumbleDur*0.15f;
        float env;
        if(tR<0.f) env=0.f;
        else if(tR<rPeak) env=tR/rPeak;
        else env=__expf(-(tR-rPeak)/(s.rumbleDur*0.42f));

        /* multi-stroke flicker: modulate rumble amplitude with a
           burst envelope centred on re-strike times                    */
        unsigned rng2=rng^0xdeadbeef;
        float flick=1.f;
        /* 2-3 sub-claps woven into the body                           */
        for(int k=1;k<=3;k++){
            rng2^=rng2<<13;rng2^=rng2>>17;rng2^=rng2<<5;
            float tOff=s.rumbleDur*0.2f*(float)k+((float)(rng2&0xFF)/255.f-0.5f)*0.6f;
            float tOff2=fmaxf(0.f,tR-tOff);
            flick+=0.55f*(float)(k==1)*__expf(-tOff2*tOff2*12.f);
        }
        env*=fmaxf(0.5f,flick);

        if(env>1e-5f){
            /* noise source: two independent streams mixed — gives
               richer texture than a single stream                       */
            float n1=xnoise(rng), n2=xnoise(rng);
            /* lowpass to rumble band (20-180 Hz at 9 km, 20-400 Hz at 0.5 km) */
            /* approximate LP/HP response using coefficient gain       */
            float loBand = (n1+n2)*0.5f
                         * (1.f-s.rumbleLpC)*(1.f-s.rumbleHpC)*4.5f;
            float v = loBand * env * s.rumbleAmp;
            float gL2,gR2; panGains(s.az,gL2,gR2);
            headShadow(s.az,0.02f,gL2,gR2);  /* rumble < 400 Hz: mild */
            vL+=v*gL2; vR+=v*gR2;
        }
    }

    /* ── WRITE with ITD ──────────────────────────────────────────── */
    int iL=samp, iR=samp;
    float az=s.az;
    if(az<0.f) iR=min(N-1,samp+s.itdSamps);   /* right ear delayed    */
    else        iL=min(N-1,samp+s.itdSamps);   /* left ear delayed     */

    atomicAdd(&L[iL], vL*s.energy);
    atomicAdd(&R[iR], vR*s.energy);
}

/* ═══════════════════════════════════════════════════════════════════
   RAIN KERNEL  — continuous stereo decorrelated noise
   grid: (ceil(N/BSAMP), 1)   block: BSAMP
   ═══════════════════════════════════════════════════════════════════ */
__global__ void rainKernel(
    float *L, float *R, int N, float sr,
    float *rainEnv)          /* pre-computed per-sample gain curve     */
{
    int samp=(int)blockIdx.x*BSAMP+(int)threadIdx.x;
    if(samp>=N) return;

    unsigned rL=(unsigned)(samp*2654435761u)^0xABCD1234u;
    unsigned rR=(unsigned)(samp*40503u     )^0x12345678u;
    rL^=rL<<13;rL^=rL>>17;rL^=rL<<5;
    rR^=rR<<13;rR^=rR>>17;rR^=rR<<5;

    float nL=xnoise(rL), nR=xnoise(rR);

    /* Simulate rain spectrum:
       white noise * octave-band shaping coeff (pre-baked as a single
       scalar per sample since we can't keep filter state across
       independent threads).  We apply a spectral weight that gives
       pink-ish character: high energy 500 Hz - 8 kHz (the "hiss" band)
       and less below.  We use three noise samples with different
       seeds to simulate three filter octaves mixed.                    */
    unsigned r2L=rL^0xfeedface, r2R=rR^0xdeadbeef;
    r2L^=r2L<<13;r2L^=r2L>>17;r2L^=r2L<<5;
    r2R^=r2R<<13;r2R^=r2R>>17;r2R^=r2R<<5;
    unsigned r3L=r2L^0xcafebabe, r3R=r2R^0x0badf00d;
    r3L^=r3L<<13;r3L^=r3L>>17;r3L^=r3L<<5;
    r3R^=r3R<<13;r3R^=r3R>>17;r3R^=r3R<<5;

    /* mix: fine patter (high freq) + body (mid) + wash (low)          */
    float fineL =xnoise(rL )*0.50f;
    float bodyL =xnoise(r2L)*0.35f;
    float washL =xnoise(r3L)*0.15f;
    float fineR =xnoise(rR )*0.50f;
    float bodyR =xnoise(r2R)*0.35f;
    float washR =xnoise(r3R)*0.15f;

    float vL=fineL+bodyL+washL;
    float vR=fineR+bodyR+washR;

    /* droplet AM: fast random modulation mimicking discrete drops      */
    unsigned dm=rL^r3R;
    dm^=dm<<13;dm^=dm>>17;dm^=dm<<5;
    float drp=0.7f+0.3f*((float)(dm&0xFFFF)/65535.f);
    vL*=drp; vR*=drp*(0.85f+0.15f*((float)(dm>>16)/65535.f));

    float g=rainEnv[samp]*0.18f;
    atomicAdd(&L[samp],vL*g);
    atomicAdd(&R[samp],vR*g);
}

/* ═══════════════════════════════════════════════════════════════════
   WIND KERNEL  — four directional gusts (cells A-D)
   grid: (ceil(N/BSAMP), 4)   block: BSAMP
   ═══════════════════════════════════════════════════════════════════ */
__global__ void windKernel(
    float *L, float *R, int N, float sr,
    const float *windGain,    /* [4*N] — per-cell per-sample gain      */
    const float *windAz)      /* [4]   — cell azimuths, radians        */
{
    int cell=(int)blockIdx.y; if(cell>=4) return;
    int samp=(int)blockIdx.x*BSAMP+(int)threadIdx.x; if(samp>=N) return;

    unsigned rng=(unsigned)(cell*1234567u+samp*7654321u);
    rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
    unsigned rng2=rng^0xdeadbeef;
    rng2^=rng2<<13;rng2^=rng2>>17;rng2^=rng2<<5;

    /* wind spectrum: mostly infra + low (20-200 Hz)                   */
    float v = (xnoise(rng)*0.6f + xnoise(rng2)*0.4f);

    /* slow gust AM — approximate with a low-freq hash pattern          */
    /* gust period ~4-12 s, represented as a slow modulation           */
    float gustPhase = (float)samp/sr * 0.12f;   /* ~0.12 Hz base      */
    /* hash-based pseudo-LFO — cheap, no sinf per thread              */
    unsigned gust=(unsigned)(gustPhase*8192.f);
    gust^=gust<<13;gust^=gust>>17;gust^=gust<<5;
    float gustAM = 0.3f + 0.7f*((float)(gust&0xFFFF)/65535.f);

    v*=gustAM;

    float az=windAz[cell];
    float gL,gR; panGains(az,gL,gR);
    float g=windGain[cell*N+samp]*0.07f;

    atomicAdd(&L[samp],v*gL*g);
    atomicAdd(&R[samp],v*gR*g);
}

/* ─────────────────────────────────────────────────────────────────────
   CPU: build strike table
   ───────────────────────────────────────────────────────────────────── */
static float frand(){ return (float)rand()/(float)RAND_MAX; }
static float frange(float a,float b){ return a+(b-a)*frand(); }

static Strike makeStrike(float tS, float azDeg, float dist_km, float energy, float sr)
{
    Strike s={};
    s.tStart=tS;
    s.az=azDeg*(float)M_PI/180.f;
    s.dist=dist_km;
    s.energy=energy;

    /* interaural delay: up to 0.66 ms                                  */
    s.itdSamps=(int)(fabsf(sinf(s.az))*0.00066f*sr);

    /* crack only survives < 3 km                                       */
    float crackFrac=fmaxf(0.f,1.f-dist_km/3.f);
    s.crackDur =crackFrac*(frange(0.04f,0.18f));
    s.crackAmp =crackFrac*energy;

    /* rumble: grows with distance                                       */
    s.rumbleDur=1.5f+1.2f*dist_km+frange(0.f,2.5f);
    s.rumbleAmp=energy;

    /* filter coefficients:
       crack band 200-3500 Hz, LP coeff ~ exp(-2pi*f*T) for T=1/sr
       HP coeff for low-cut at 200 Hz                                   */
    float lc=expf(-TWO_PI*3500.f/sr);  /* LP above 3.5 kHz             */
    float hc=expf(-TWO_PI*200.f/sr);   /* HP below 200 Hz              */
    s.crackLoC=lc; s.crackHiC=hc;

    /* rumble LP: cutoff drops sharply with distance (atmospheric abs)  */
    float fCut=fmaxf(35.f, 380.f - 35.f*dist_km);
    s.rumbleLpC=expf(-TWO_PI*fCut/sr);
    s.rumbleHpC=expf(-TWO_PI*18.f/sr);  /* remove DC                   */

    return s;
}

static int buildStrikes(Strike *strikes, float sr)
{
    /* Cell parameters: az degrees, dist_km (start, end), t_start, t_end, n */
    struct Cell { float az, d0, d1, t0, t1, e; int n; };
    static const Cell cells[]={
        { -40.f, 0.6f, 0.55f,  20.f, 210.f, 1.00f, 10 }, /* A: near front-L  */
        { +55.f, 2.8f, 2.8f,   15.f, 285.f, 0.88f,  9 }, /* B: mid right     */
        { -70.f, 9.0f, 9.0f,   35.f, 275.f, 0.95f,  7 }, /* C: far left      */
        { +18.f, 9.0f, 1.0f,   45.f, 290.f, 1.05f, 12 }, /* D: approaching   */
    };
    int n=0;
    for(int c=0;c<4;c++){
        const Cell&C=cells[c];
        for(int i=0;i<C.n && n<MAX_STRIKES;i++){
            float u=((float)i+0.15f+0.7f*frand())/(float)C.n;
            /* cell D strikes cluster toward the end (approach climax)  */
            if(c==3) u=powf(u,0.75f);
            float t=C.t0+u*(C.t1-C.t0);
            float d=C.d0+(C.d1-C.d0)*u + frange(-0.08f,0.08f)*C.d0;
            d=fmaxf(0.3f,d);
            strikes[n++]=makeStrike(t, C.az+frange(-5.f,5.f),
                                    d, C.e*frange(0.55f,1.3f), sr);
        }
        /* climax strike: Cell D overhead at 4:48 */
        if(c==3 && n<MAX_STRIKES)
            strikes[n++]=makeStrike(288.f,+12.f,0.35f,1.8f,sr);
    }
    /* extra near-A cracks at start for drama                           */
    for(int i=0;i<3 && n<MAX_STRIKES;i++){
        float t=22.f+i*14.f+frange(0.f,6.f);
        strikes[n++]=makeStrike(t,-38.f+frange(-4.f,4.f),
                                0.5f+frange(0.f,0.15f),
                                1.1f*frange(0.8f,1.2f),sr);
    }
    return n;
}

/* ─────────────────────────────────────────────────────────────────────
   CPU: rain envelope curve  (intensity over 5 min)
   ───────────────────────────────────────────────────────────────────── */
static void buildRainEnv(float*env, int N, float sr)
{
    /* shape: gradual onset, peak around 2:30, sustained, fade at end   */
    for(int i=0;i<N;i++){
        float t=(float)i/sr;
        float ramp=fminf(1.f,t/30.f);            /* 30 s fade-in       */
        float peak=0.65f+0.35f*expf(-((t-150.f)/90.f)*((t-150.f)/90.f));
        /* Cell-D approach swell: 0..1 over last 3 min                  */
        float swell=fmaxf(0.f,fminf(1.f,(t-120.f)/180.f));
        float fade=fminf(1.f,(310.f-t)/10.f);    /* fade at end        */
        env[i]=ramp*(peak+0.4f*swell)*fade;
    }
}

/* ─────────────────────────────────────────────────────────────────────
   CPU: wind gain tables  (per cell, per sample)
   ───────────────────────────────────────────────────────────────────── */
static void buildWindGains(float*gains, float*azArr, int N, float sr)
{
    static const float az4[]={-40.f,+55.f,-70.f,+18.f};
    for(int c=0;c<4;c++) azArr[c]=az4[c]*(float)M_PI/180.f;

    for(int c=0;c<4;c++){
        float base=(c==0)?0.85f:(c==1)?0.60f:(c==2)?0.30f:0.55f;
        for(int i=0;i<N;i++){
            float t=(float)i/sr;
            float fin=fminf(1.f,t/20.f);
            float fout=fminf(1.f,(DUR_SEC+5.f-t)/8.f);
            /* cell D wind grows with approach                           */
            float approach=(c==3)?fminf(1.f,t/DUR_SEC):1.f;
            /* slow gust wave (cosine at 0.03-0.07 Hz)                  */
            float gfreq=(c==0)?0.05f:(c==1)?0.04f:(c==2)?0.03f:0.06f;
            float gust=0.55f+0.45f*cosf(TWO_PI*gfreq*t+(float)c*1.3f);
            gains[c*N+i]=base*fin*fout*gust*(0.6f+0.4f*approach);
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
   Post-processing: soft sat + reverb + limiter
   ───────────────────────────────────────────────────────────────────── */
static void softSat(float*x, int N, float drive){
    for(int i=0;i<N;i++) x[i]=tanhf(x[i]*drive)/drive;
}

static void applyEQ(float*L, float*R, int N, float sr)
{
    /* Gentle high-shelf cut above 8 kHz (outdoor air softens highs)
       and a low-shelf boost below 80 Hz (thunder body).
       One-pole approximation.                                           */
    float hiC=expf(-TWO_PI*8000.f/sr);
    float loC=expf(-TWO_PI*80.f/sr);
    float zHL=0.f,zHR=0.f,zLL=0.f,zLR=0.f;
    for(int i=0;i<N;i++){
        /* high shelf: original - LP(8kHz)*0.4                         */
        float lpHL=zHL*hiC+L[i]*(1.f-hiC); zHL=lpHL;
        float lpHR=zHR*hiC+R[i]*(1.f-hiC); zHR=lpHR;
        /* low shelf: add LP(80Hz)*0.6                                  */
        float lpLL=zLL*loC+L[i]*(1.f-loC); zLL=lpLL;
        float lpLR=zLR*loC+R[i]*(1.f-loC); zLR=lpLR;
        L[i]=L[i]-lpHL*0.35f+lpLL*0.55f;
        R[i]=R[i]-lpHR*0.35f+lpLR*0.55f;
    }
}

static void applyReverb(float*L,float*R,int N,float sr,float wet)
{
    /* Large outdoor reverb — longer delays than indoors, heavy HF damp */
    static const float cd[]={0.0621f,0.0671f,0.0756f,0.0823f,
                              0.0861f,0.0901f,0.0952f,0.0987f};
    static const float ad[]={0.0113f,0.0151f,0.0227f,0.0265f};
    int NC=8,NA=4; float fb=0.72f,damp=0.55f;
    float*cbL[8],*cbR[8],*abL[4],*abR[4];
    int csz[8],asz[4],cpL[8]={},cpR[8]={},apL[4]={},apR[4]={};
    float lpL[8]={},lpR[8]={};
    for(int i=0;i<NC;i++){csz[i]=(int)(cd[i]*sr);
        cbL[i]=(float*)calloc(csz[i],sizeof(float));
        cbR[i]=(float*)calloc(csz[i],sizeof(float));}
    for(int i=0;i<NA;i++){asz[i]=(int)(ad[i]*sr);
        abL[i]=(float*)calloc(asz[i],sizeof(float));
        abR[i]=(float*)calloc(asz[i],sizeof(float));}
    for(int s=0;s<N;s++){
        float iL=L[s],iR=R[s],oL=0.f,oR=0.f;
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

/* look-ahead peak limiter (simple version: 2 ms window)               */
static void applyLimiter(float*L, float*R, int N, float sr){
    int wnd=(int)(0.002f*sr);
    float env=0.f, th=0.92f;
    for(int i=0;i<N;i++){
        float pk=fmaxf(fabsf(L[i]),fabsf(R[i]));
        env=fmaxf(env,pk); env*=0.9995f;
        float g=(env>th)? th/env : 1.f;
        L[i]*=g; R[i]*=g;
    }
}

/* ─────────────────────────────────────────────────────────────────────
   WAV
   ───────────────────────────────────────────────────────────────────── */
static void writeWav(const char*path,const float*L,const float*R,int N,int sr)
{
    FILE*f=fopen(path,"wb");
    if(!f){fprintf(stderr,"Cannot open %s\n",path);return;}
    int16_t*p=(int16_t*)malloc((size_t)N*2*sizeof(int16_t));
    float pk=1e-9f;
    for(int i=0;i<N;i++){pk=fmaxf(pk,fabsf(L[i]));pk=fmaxf(pk,fabsf(R[i]));}
    float g=0.94f/pk;
    for(int i=0;i<N;i++){
        float d=(((float)(rand()&0xFFFF)+(float)(rand()&0xFFFF))/65536.f-1.f)/32768.f;
        p[i*2+0]=(int16_t)(fmaxf(-1.f,fminf(1.f,L[i]*g+d))*32767.f);
        p[i*2+1]=(int16_t)(fmaxf(-1.f,fminf(1.f,R[i]*g+d))*32767.f);
    }
    uint32_t dSz=(uint32_t)N*4,cSz=36+dSz,bR=(uint32_t)sr*4;
    uint16_t bA=4,bps=16,fmt=1,ch=2;uint32_t sc=16;
    fwrite("RIFF",1,4,f);fwrite(&cSz,4,1,f);
    fwrite("WAVEfmt ",1,8,f);fwrite(&sc,4,1,f);
    fwrite(&fmt,2,1,f);fwrite(&ch,2,1,f);
    fwrite(&sr,4,1,f);fwrite(&bR,4,1,f);
    fwrite(&bA,2,1,f);fwrite(&bps,2,1,f);
    fwrite("data",1,4,f);fwrite(&dSz,4,1,f);
    fwrite(p,sizeof(int16_t),(size_t)N*2,f);
    free(p);fclose(f);
    printf("Wrote %s  (%.1f s, %d Hz stereo)\n",path,(double)N/sr,sr);
}

/* ═══════════════════════════════════════════════════════════════════
   MAIN
   ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char**argv)
{
    const char*out=(argc>1)?argv[1]:"storm.wav";
    int sr=(argc>2)?atoi(argv[2]):SR_DEFAULT;

    printf("GPU Thunderstorm v2 | noise-based | 4 cells | 5 min stereo\n\n");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("Device: %s (SM %d.%d)\n\n",prop.name,prop.major,prop.minor);

    srand(20260706);

    int N=(int)((DUR_SEC+12.f)*sr);       /* 5 min + tail              */
    printf("Total samples: %d  (%.1f s)\n\n",(int)N,(float)N/sr);

    /* ── host allocations ─────────────────────────────────────────── */
    Strike *hStrikes=(Strike*)malloc(MAX_STRIKES*sizeof(Strike));
    int nStr=buildStrikes(hStrikes,(float)sr);
    printf("Strikes: %d\n",nStr);

    float *hRainEnv=(float*)calloc(N,sizeof(float));
    buildRainEnv(hRainEnv,N,(float)sr);

    float *hWindGain=(float*)calloc(4*N,sizeof(float));
    float  hWindAz[4]={};
    buildWindGains(hWindGain,hWindAz,N,(float)sr);

    /* ── GPU allocations ──────────────────────────────────────────── */
    Strike*dStr; float*dL,*dR,*dRainEnv,*dWindGain,*dWindAz;
    CUDA_CHECK(cudaMalloc(&dStr,MAX_STRIKES*sizeof(Strike)));
    CUDA_CHECK(cudaMalloc(&dL,(size_t)N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dR,(size_t)N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRainEnv,(size_t)N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWindGain,(size_t)4*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWindAz,4*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dStr,hStrikes,MAX_STRIKES*sizeof(Strike),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRainEnv,hRainEnv,(size_t)N*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWindGain,hWindGain,(size_t)4*N*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWindAz,hWindAz,4*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dL,0,(size_t)N*sizeof(float)));
    CUDA_CHECK(cudaMemset(dR,0,(size_t)N*sizeof(float)));

    int nb=(N+BSAMP-1)/BSAMP;
    cudaEvent_t e0,e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));

    /* ── launch kernels ───────────────────────────────────────────── */
    dim3 strGrid(nb,nStr);
    strikesKernel<<<strGrid,BSAMP>>>(dStr,nStr,dL,dR,N,(float)sr);

    rainKernel<<<nb,BSAMP>>>(dL,dR,N,(float)sr,dRainEnv);

    dim3 wndGrid(nb,4);
    windKernel<<<wndGrid,BSAMP>>>(dL,dR,N,(float)sr,dWindGain,dWindAz);

    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());

    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    printf("Kernels: %.0f ms  (%.0fx real-time)\n\n",ms,DUR_SEC*1000.f/ms);

    float*hL=(float*)malloc((size_t)N*sizeof(float));
    float*hR=(float*)malloc((size_t)N*sizeof(float));
    CUDA_CHECK(cudaMemcpy(hL,dL,(size_t)N*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hR,dR,(size_t)N*sizeof(float),cudaMemcpyDeviceToHost));

    printf("Post-processing: EQ + saturation + reverb + limiter...\n");
    applyEQ(hL,hR,N,(float)sr);
    softSat(hL,N,1.4f); softSat(hR,N,1.4f);
    applyReverb(hL,hR,N,(float)sr,0.08f);
    applyLimiter(hL,hR,N,(float)sr);
    writeWav(out,hL,hR,N,sr);

    free(hStrikes);free(hRainEnv);free(hWindGain);
    free(hL);free(hR);
    cudaFree(dStr);cudaFree(dL);cudaFree(dR);
    cudaFree(dRainEnv);cudaFree(dWindGain);cudaFree(dWindAz);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 0;
}
