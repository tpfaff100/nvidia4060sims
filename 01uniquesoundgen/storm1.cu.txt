/*
 * storm_synth.cu — GPU procedural thunderstorm field, single file
 * Target: RTX 4060 Ti (sm_89, Ada Lovelace)
 *
 * BUILD (after vcvars64.bat):
 *   nvcc -arch=sm_89 -O3 -use_fast_math -o storm_synth storm_synth.cu
 * RUN:
 *   storm_synth [storm.wav] [samplerate]        (default 48000)
 *
 * WHAT IT RENDERS
 *   5 minutes, stereo, four independent storm cells around the listener:
 *
 *     Cell A  near,  front-left  (az -35 deg, ~0.8 km)  sharp cracks + rumble
 *     Cell B  mid,   right       (az +55 deg,  3.2 km)  classic rolling peals
 *     Cell C  far,   hard-left   (az -72 deg,  8.5 km)  deep delayed rumbles only
 *     Cell D  APPROACHING, right-of-centre: 9 km -> 1.4 km over the 5 minutes.
 *             Its thunder grows louder, brighter and more frequent, ending in
 *             a close climax strike at 4:45.
 *
 *   Plus: wide decorrelated rain bed whose intensity follows storm activity,
 *   and four directional gusting wind beds (one per cell).
 *
 * PHYSICS BAKED PER SINUSOID (why the GPU spectral approach wins):
 *   • Atmospheric absorption  amp *= exp(-k * f^2 * r)  — exact per component.
 *     This alone creates the near-crack vs far-rumble difference: at 8 km
 *     nothing above ~300 Hz survives; at 800 m the 3 kHz crack tears through.
 *   • 1/r spreading loss.
 *   • Rumble duration grows with distance (path-length spread across the
 *     tortuous lightning channel): sub-clap onsets are scattered over
 *     T ~ 1 + 1.1*r_km seconds.
 *   • Multi-stroke flicker: 60% of flashes get 1-3 re-strikes 60-250 ms later.
 *
 * STEREO / SPATIAL (the "nice channel separation"):
 *   • Constant-power pan per component with ±8 deg jitter around the cell
 *     azimuth (a lightning channel is kilometres long — it isn't a point).
 *   • ITD: the far ear receives the event up to 0.66 ms late (event-level
 *     sample offset).
 *   • Head shadow: the far ear's high frequencies are attenuated
 *     1/(1+(f/1200)^2 * shadow) — baked into per-component ear gains.
 *   • Rain components alternate hard-left / hard-right with independent
 *     LFOs -> wide decorrelated wash between the ears.
 *
 * PRECISION NOTE: all phases use component-local time (t - onset) and are
 * reduced mod 1 cycle before the SFU sin, so float32 stays clean over the
 * full 5-minute render. Long beds are chopped into 22 s segments with 2 s
 * crossfades (phase re-randomisation is inaudible in noise).
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
#include <cassert>

#define SAMPLE_RATE_DEFAULT 48000
#define DURATION_SEC        300.f
#define MAX_COMP            192     /* per event */
#define MAX_EVENTS          224
#define BLOCK_SAMPLES       256
#define TWO_PI              6.28318530717958647692f
#define FMAX                10000.f

#define CUDA_CHECK(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1);} }while(0)

/* ─────────────────────────────────────────────────────────────────────
   Data
   ───────────────────────────────────────────────────────────────────── */
struct Comp {                /* one sinusoidal component               */
    float freq;              /* Hz                                     */
    float ampL, ampR;        /* per-ear gains: pan+shadow+absorption   */
    float phase;             /* cycles (0..1)                          */
    float onset;             /* s after event start                    */
    float attackTau;         /* s                                      */
    float sigma;             /* decay rate s^-1 (0 = sustained bed)    */
    float lfoRate, lfoDepth, lfoPhase;   /* amplitude modulation       */
};

struct Event {
    int   numComps;
    float startSec, durSec;
    float fadeIn, fadeOut;   /* event-level linear fades               */
    float ramp0, ramp1;      /* linear gain ramp across the event      */
    float gain;
    int   itdL, itdR;        /* interaural delay, samples              */
};

/* ─────────────────────────────────────────────────────────────────────
   KERNEL — grid (sampleBlocks, numEvents), thread = one sample
   ───────────────────────────────────────────────────────────────────── */
__global__ void stormKernel(
    const Event *__restrict__ events,
    const Comp  *__restrict__ comps,
    int numEvents, float *L, float *R, int N, float sr)
{
    int ei = (int)blockIdx.y;
    if (ei >= numEvents) return;
    int si = (int)blockIdx.x * BLOCK_SAMPLES + (int)threadIdx.x;
    if (si >= N) return;

    const Event e = events[ei];
    float t = si / sr - e.startSec;
    if (t < 0.f || t > e.durSec) return;

    float fade = fminf(1.f, t / e.fadeIn) *
                 fminf(1.f, (e.durSec - t) / e.fadeOut);
    if (fade <= 0.f) return;
    float ramp = e.ramp0 + (e.ramp1 - e.ramp0) * (t / e.durSec);

    const Comp *c = comps + ei * MAX_COMP;
    float sL = 0.f, sR = 0.f;

    for (int k = 0; k < e.numComps; k++) {
        float to = t - c[k].onset;
        if (to <= 0.f) continue;

        float env = (1.f - __expf(-to / c[k].attackTau))
                  *        __expf(-c[k].sigma * to);
        if (env < 1e-5f) continue;

        /* amplitude LFO (rain patter / wind gusts)                     */
        float lfo = 1.f + c[k].lfoDepth *
                    __sinf(TWO_PI * (c[k].lfoRate * t) + c[k].lfoPhase);

        /* phase in cycles, reduced before the SFU — float32-safe       */
        float cyc = c[k].freq * to + c[k].phase;
        cyc -= floorf(cyc);
        float v = env * lfo * __sinf(TWO_PI * cyc);

        sL += c[k].ampL * v;
        sR += c[k].ampR * v;
    }

    float g = e.gain * fade * ramp;
    int iL = si + e.itdL; if (iL >= N) iL = N - 1;
    int iR = si + e.itdR; if (iR >= N) iR = N - 1;
    atomicAdd(&L[iL], sL * g);
    atomicAdd(&R[iR], sR * g);
}

/* ─────────────────────────────────────────────────────────────────────
   CPU event construction
   ───────────────────────────────────────────────────────────────────── */
static float frand(){ return (float)rand()/(float)RAND_MAX; }
static float frange(float a,float b){ return a+(b-a)*frand(); }
static Comp* ec(Comp*buf,int ei){ return buf+ei*MAX_COMP; }

/* atmospheric absorption: ~5 dB/km at 1 kHz, scaling with f^2          */
static float airAbs(float f, float r_m){
    return expf(-5.75e-10f * f*f * r_m);
}

/* per-component ear gains: constant-power pan + head shadow            */
static void earGains(float azDeg, float f, float*gL, float*gR){
    float az = azDeg * (float)M_PI/180.f;             /* -90..+90       */
    float x  = (sinf(az)+1.f)*0.25f*(float)M_PI;      /* 0..pi/2        */
    float pL = cosf(x), pR = sinf(x);
    float shadow = 0.75f * fabsf(sinf(az));
    float hf = 1.f/(1.f+(f/1200.f)*(f/1200.f)*shadow);
    if (az > 0.f) pL *= hf; else pR *= hf;            /* far ear duller */
    *gL=pL; *gR=pR;
}

static void eventSpatial(Event&e, float azDeg, float sr){
    float az = azDeg*(float)M_PI/180.f;
    int d = (int)(fabsf(sinf(az)) * 0.00066f * sr);   /* up to ~32 smp  */
    if (az < 0.f){ e.itdL=0; e.itdR=d; } else { e.itdL=d; e.itdR=0; }
}

/* ══════════════════════ THUNDER ═════════════════════════════════════
   One flash = crack cluster (near cells only) + 6..10 low sub-claps
   scattered over a distance-dependent rumble time + infrasonic
   afterglow + optional re-strikes.                                     */
static int makeThunder(Event*ev, Comp*cb, int idx,
                       float tStart, float r_m, float azDeg,
                       float energy, float sr)
{
    Event&e = ev[idx]; e = {};
    Comp *c = ec(cb,idx);
    int k = 0;

    float r_km  = r_m/1000.f;
    float T     = 1.0f + 1.1f*r_km + frange(0.f,1.5f);  /* rumble length */
    float dist  = powf(400.f/r_m, 0.9f);                /* 1/r-ish       */
    float near_ = fmaxf(0.f,(2500.f-r_m)/2500.f);       /* crack factor  */

    /* — crack: the tearing HF transient, only survives short paths —   */
    if (near_ > 0.f){
        int nC = 24;
        for(int i=0;i<nC && k<MAX_COMP;i++){
            float f = 350.f*powf(3800.f/350.f, frand());  /* log-random  */
            if (f>FMAX) continue;
            float a = airAbs(f,r_m)*powf(f/1000.f,-0.35f)*near_*1.6f;
            earGains(azDeg+frange(-8.f,8.f), f, &c[k].ampL,&c[k].ampR);
            c[k].ampL*=a; c[k].ampR*=a;
            c[k].freq=f; c[k].phase=frand();
            c[k].onset=frange(0.f,0.030f);
            c[k].attackTau=frange(0.0004f,0.0015f);
            c[k].sigma=frange(5.f,12.f);
            c[k].lfoRate=frange(20.f,60.f); c[k].lfoDepth=0.5f;
            c[k].lfoPhase=frand()*TWO_PI;
            k++;
        }
    }

    /* — rolling body: clustered sub-claps, denser early —              */
    int nSub = 6 + (int)fminf(4.f, T*0.8f);
    for(int j=0;j<nSub;j++){
        float tj = powf(frand(),1.4f)*T;                 /* front-loaded */
        float aj = powf(1.f - tj/T, 0.7f)*frange(0.55f,1.15f);
        int nB = 12;
        for(int i=0;i<nB && k<MAX_COMP;i++){
            float f = 16.f*powf(280.f/16.f, powf(frand(),1.3f));
            float a = airAbs(f,r_m)*aj*powf(60.f/fmaxf(f,25.f),0.25f);
            earGains(azDeg+frange(-8.f,8.f), f, &c[k].ampL,&c[k].ampR);
            c[k].ampL*=a; c[k].ampR*=a;
            c[k].freq=f; c[k].phase=frand();
            c[k].onset=tj+frange(0.f,0.045f);
            c[k].attackTau=0.004f+0.028f*r_km/8.f+frange(0.f,0.01f);
            c[k].sigma=frange(0.7f,2.9f);
            c[k].lfoRate=frange(4.f,14.f); c[k].lfoDepth=0.35f;
            c[k].lfoPhase=frand()*TWO_PI;
            k++;
        }
    }

    /* — re-strikes: lightning flickers —                               */
    if (frand()<0.6f && near_>0.05f){
        int nRe = 1+(int)(frand()*2.f);
        for(int rs=0;rs<nRe;rs++){
            float tOff = frange(0.06f,0.25f)*(rs+1);
            for(int i=0;i<8 && k<MAX_COMP;i++){
                float f = 300.f*powf(3000.f/300.f,frand());
                if (f>FMAX) continue;
                float a = airAbs(f,r_m)*near_*0.7f*powf(0.6f,(float)rs);
                earGains(azDeg+frange(-6.f,6.f),f,&c[k].ampL,&c[k].ampR);
                c[k].ampL*=a; c[k].ampR*=a;
                c[k].freq=f; c[k].phase=frand();
                c[k].onset=tOff+frange(0.f,0.02f);
                c[k].attackTau=0.0008f; c[k].sigma=frange(6.f,11.f);
                c[k].lfoRate=30.f; c[k].lfoDepth=0.4f;
                c[k].lfoPhase=frand()*TWO_PI;
                k++;
            }
        }
    }

    /* — infrasonic afterglow —                                         */
    for(int i=0;i<8 && k<MAX_COMP;i++){
        float f=frange(12.f,45.f);
        float a=airAbs(f,r_m)*0.5f;
        earGains(azDeg,f,&c[k].ampL,&c[k].ampR);
        c[k].ampL*=a; c[k].ampR*=a;
        c[k].freq=f; c[k].phase=frand();
        c[k].onset=T*frange(0.25f,0.5f);
        c[k].attackTau=0.3f; c[k].sigma=frange(0.25f,0.5f);
        c[k].lfoRate=frange(0.5f,2.f); c[k].lfoDepth=0.3f;
        c[k].lfoPhase=frand()*TWO_PI;
        k++;
    }

    e.numComps=k;
    e.startSec=tStart; e.durSec=T+8.f;
    e.fadeIn=0.001f; e.fadeOut=2.f;
    e.ramp0=1.f; e.ramp1=1.f;
    e.gain=energy*dist*2.4f;
    eventSpatial(e,azDeg,sr);
    return idx+1;
}

/* ══════════════════════ RAIN BED (wide stereo) ══════════════════════
   144 components, log-spaced 300 Hz..9 kHz, alternating hard-L/hard-R
   with independent patter LFOs -> broad decorrelated wash.             */
static int makeRainSeg(Event*ev, Comp*cb, int idx,
                       float t0, float dur, float fin, float fout,
                       float g0, float g1)
{
    Event&e=ev[idx]; e={};
    Comp*c=ec(cb,idx);
    int k=0;
    for(int i=0;i<144 && k<MAX_COMP;i++){
        float u=(float)i/143.f;
        float f=300.f*powf(9000.f/300.f,u)*frange(0.96f,1.04f);
        /* broadband hiss with gentle 2-5 kHz emphasis                  */
        float a=powf(f/2000.f,0.30f)/(1.f+powf(f/6500.f,4.f));
        a/= (1.f+0.85f);                       /* LFO normalisation     */
        int side=i&1;
        float az=side? frange(35.f,80.f):frange(-80.f,-35.f);
        earGains(az,f,&c[k].ampL,&c[k].ampR);
        c[k].ampL*=a; c[k].ampR*=a;
        c[k].freq=f; c[k].phase=frand();
        c[k].onset=0.f; c[k].attackTau=0.05f; c[k].sigma=0.f;
        c[k].lfoRate=frange(7.f,30.f);         /* droplet granularity   */
        c[k].lfoDepth=0.85f;
        c[k].lfoPhase=frand()*TWO_PI;
        k++;
    }
    /* low "wash" of water on ground                                    */
    for(int i=0;i<12 && k<MAX_COMP;i++){
        float f=frange(90.f,280.f);
        earGains(frange(-30.f,30.f),f,&c[k].ampL,&c[k].ampR);
        float a=0.35f/(1.f+0.6f);
        c[k].ampL*=a; c[k].ampR*=a;
        c[k].freq=f; c[k].phase=frand();
        c[k].onset=0.f; c[k].attackTau=0.1f; c[k].sigma=0.f;
        c[k].lfoRate=frange(0.3f,1.2f); c[k].lfoDepth=0.6f;
        c[k].lfoPhase=frand()*TWO_PI;
        k++;
    }
    e.numComps=k;
    e.startSec=t0; e.durSec=dur;
    e.fadeIn=fin; e.fadeOut=fout;
    e.ramp0=g0; e.ramp1=g1;
    e.gain=0.22f; e.itdL=0; e.itdR=0;
    return idx+1;
}

/* ══════════════════════ WIND BED (directional) ══════════════════════ */
static int makeWindSeg(Event*ev, Comp*cb, int idx,
                       float azDeg, float t0, float dur,
                       float fin, float fout, float g0, float g1,
                       float sr)
{
    Event&e=ev[idx]; e={};
    Comp*c=ec(cb,idx);
    int k=0;
    for(int i=0;i<40 && k<MAX_COMP;i++){
        float u=(float)i/39.f;
        float f=18.f*powf(420.f/18.f,u)*frange(0.94f,1.06f);
        float a=powf(60.f/fmaxf(f,30.f),0.5f)/(1.f+0.95f);
        earGains(azDeg+frange(-15.f,15.f),f,&c[k].ampL,&c[k].ampR);
        c[k].ampL*=a; c[k].ampR*=a;
        c[k].freq=f; c[k].phase=frand();
        c[k].onset=0.f; c[k].attackTau=0.3f; c[k].sigma=0.f;
        c[k].lfoRate=frange(0.05f,0.35f);      /* slow gusting          */
        c[k].lfoDepth=0.95f;
        c[k].lfoPhase=frand()*TWO_PI;
        k++;
    }
    e.numComps=k;
    e.startSec=t0; e.durSec=dur;
    e.fadeIn=fin; e.fadeOut=fout;
    e.ramp0=g0; e.ramp1=g1;
    e.gain=0.10f; eventSpatial(e,azDeg,sr);
    return idx+1;
}

/* ══════════════════════ STORM FIELD SCORE ═══════════════════════════ */
struct Cell { float az, r0, r1, tBegin, tEnd, energy, wind; int nTh; };

static float cellR(const Cell&s, float t){
    float u=(t-s.tBegin)/(s.tEnd-s.tBegin);
    u=fmaxf(0.f,fminf(1.f,u));
    return s.r0+(s.r1-s.r0)*u;
}

/* local rain intensity: cell A peak early-mid, cell D swells at end    */
static float rainCurve(float t){
    float a=expf(-((t-110.f)/80.f)*((t-110.f)/80.f));
    float d=fmaxf(0.f,fminf(1.f,(t-120.f)/170.f));
    float g=0.30f+0.55f*a+0.6f*d;
    return fminf(1.25f,g);
}

static int buildStormField(Event*ev, Comp*cb, float sr)
{
    static const Cell cells[4]={
        /*      az     r0     r1    t0    t1   energy wind nTh */
        { -35.f,  900.f,  700.f,  18.f, 205.f, 1.00f, 0.85f,  9 }, /* A */
        { +55.f, 3200.f, 3200.f,  10.f, 290.f, 0.90f, 0.60f,  8 }, /* B */
        { -72.f, 8500.f, 8500.f,  30.f, 280.f, 1.05f, 0.35f,  7 }, /* C */
        { +20.f, 9000.f, 1400.f,  40.f, 292.f, 1.10f, 0.75f, 13 }, /* D */
    };
    int idx=0;

    /* — thunder schedules —                                            */
    for(int s=0;s<4;s++){
        const Cell&S=cells[s];
        for(int i=0;i<S.nTh;i++){
            float u=((float)i+0.2f+0.6f*frand())/(float)S.nTh;
            /* cell D strikes bunch up as it closes in                   */
            if (s==3) u=powf(u,0.8f);
            float t=S.tBegin+u*(S.tEnd-S.tBegin);
            float r=cellR(S,t)*frange(0.85f,1.15f);
            idx=makeThunder(ev,cb,idx,t,r,S.az+frange(-6.f,6.f),
                            S.energy*frange(0.6f,1.3f),sr);
        }
    }
    /* climax: cell D nearly overhead at 4:45                            */
    idx=makeThunder(ev,cb,idx,285.f,1100.f,+15.f,1.7f,sr);

    /* — rain bed: 22 s segments, 2 s crossfades, intensity curve —      */
    for(float t0=0.f;t0<DURATION_SEC;t0+=20.f){
        float dur=fminf(22.f,DURATION_SEC+3.f-t0);
        float fin=(t0<1.f)?6.f:2.f;
        float g0=rainCurve(t0), g1=rainCurve(t0+dur);
        idx=makeRainSeg(ev,cb,idx,t0,dur,fin,2.f,g0,g1);
    }

    /* — wind beds, one per cell; D's wind grows with approach —         */
    for(int s=0;s<4;s++){
        const Cell&S=cells[s];
        for(float t0=0.f;t0<DURATION_SEC;t0+=20.f){
            float dur=fminf(22.f,DURATION_SEC+3.f-t0);
            float fin=(t0<1.f)?8.f:2.f;
            float base=S.wind*frange(0.75f,1.15f);
            float g0=base,g1=base*frange(0.8f,1.2f);
            if(s==3){ g0*=0.3f+0.7f*t0/DURATION_SEC;
                      g1*=0.3f+0.7f*(t0+dur)/DURATION_SEC; }
            idx=makeWindSeg(ev,cb,idx,S.az,t0,dur,fin,2.f,g0,g1,sr);
        }
    }

    assert(idx<=MAX_EVENTS);
    return idx;
}

/* ─────────────────────────────────────────────────────────────────────
   Gentle saturation (tames thunder crest factor so rain stays present)
   + outdoor-scaled Schroeder tail + WAV out
   ───────────────────────────────────────────────────────────────────── */
static void softSat(float*x,int N){
    for(int i=0;i<N;i++) x[i]=x[i]/(1.f+0.30f*fabsf(x[i]));
}

static void applyReverb(float*L,float*R,int N,float sr,float wet)
{
    static const float cd[]={0.0437f,0.0471f,0.0526f,0.0573f,
                              0.0605f,0.0633f,0.0668f,0.0694f};
    static const float ad[]={0.0097f,0.0130f,0.0187f,0.0218f};
    int NC=8,NA=4; float fb=0.78f,damp=0.45f;
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

static void writeWav(const char*path,const float*L,const float*R,int N,int sr)
{
    FILE*f=fopen(path,"wb");
    if(!f){fprintf(stderr,"Cannot open %s\n",path);return;}
    int16_t*pcm=(int16_t*)malloc((size_t)N*2*sizeof(int16_t));
    float peak=1e-9f;
    for(int i=0;i<N;i++){peak=fmaxf(peak,fabsf(L[i]));peak=fmaxf(peak,fabsf(R[i]));}
    float gain=0.95f/peak;
    for(int i=0;i<N;i++){
        float d=(((float)(rand()&0xFFFF)+(float)(rand()&0xFFFF))/65536.f-1.f)/32768.f;
        pcm[i*2+0]=(int16_t)(fmaxf(-1.f,fminf(1.f,L[i]*gain+d))*32767.f);
        pcm[i*2+1]=(int16_t)(fmaxf(-1.f,fminf(1.f,R[i]*gain+d))*32767.f);
    }
    uint32_t dSz=(uint32_t)N*4,cSz=36+dSz,bRate=(uint32_t)sr*4;
    uint16_t bAl=4,bps=16,fmt=1,ch=2;uint32_t sc=16;
    fwrite("RIFF",1,4,f);fwrite(&cSz,4,1,f);
    fwrite("WAVEfmt ",1,8,f);fwrite(&sc,4,1,f);
    fwrite(&fmt,2,1,f);fwrite(&ch,2,1,f);
    fwrite(&sr,4,1,f);fwrite(&bRate,4,1,f);
    fwrite(&bAl,2,1,f);fwrite(&bps,2,1,f);
    fwrite("data",1,4,f);fwrite(&dSz,4,1,f);
    fwrite(pcm,sizeof(int16_t),(size_t)N*2,f);
    free(pcm);fclose(f);
    printf("Wrote %s  (%.1f s, %d Hz, stereo)\n",path,(double)N/sr,sr);
}

/* ═════════════════════════════════════════════════════════════════════ */
int main(int argc,char**argv)
{
    const char*out=(argc>1)?argv[1]:"storm.wav";
    int sr=(argc>2)?atoi(argv[2]):SAMPLE_RATE_DEFAULT;

    printf("GPU Thunderstorm Field | 4 cells | %.0f s stereo | sm_89\n\n",
           DURATION_SEC);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("Device: %s (SM %d.%d, %d SMs)\n\n",
           prop.name,prop.major,prop.minor,prop.multiProcessorCount);

    srand(20260705);

    Event*he=(Event*)calloc(MAX_EVENTS,sizeof(Event));
    Comp *hc=(Comp* )calloc((size_t)MAX_EVENTS*MAX_COMP,sizeof(Comp));
    int ne=buildStormField(he,hc,(float)sr);

    long totalComps=0;
    for(int i=0;i<ne;i++) totalComps+=he[i].numComps;
    int N=(int)((DURATION_SEC+10.f)*sr);
    printf("Events: %d   Components: %ld   Samples: %d\n\n",
           ne,totalComps,N);

    Event*de; Comp*dc; float*dL,*dR;
    CUDA_CHECK(cudaMalloc(&de,MAX_EVENTS*sizeof(Event)));
    CUDA_CHECK(cudaMalloc(&dc,(size_t)MAX_EVENTS*MAX_COMP*sizeof(Comp)));
    CUDA_CHECK(cudaMalloc(&dL,(size_t)N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dR,(size_t)N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(de,he,MAX_EVENTS*sizeof(Event),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc,hc,(size_t)MAX_EVENTS*MAX_COMP*sizeof(Comp),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dL,0,(size_t)N*sizeof(float)));
    CUDA_CHECK(cudaMemset(dR,0,(size_t)N*sizeof(float)));

    int nb=(N+BLOCK_SAMPLES-1)/BLOCK_SAMPLES;
    dim3 grid(nb,ne),block(BLOCK_SAMPLES);
    printf("Kernel grid=(%d,%d)x%d — rendering...\n",nb,ne,BLOCK_SAMPLES);

    cudaEvent_t e0,e1;
    CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    stormKernel<<<grid,block>>>(de,dc,ne,dL,dR,N,(float)sr);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    printf("Kernel: %.0f ms  (%.1fx real-time)\n\n",ms,DURATION_SEC*1000.f/ms);

    float*hL=(float*)malloc((size_t)N*sizeof(float));
    float*hR=(float*)malloc((size_t)N*sizeof(float));
    CUDA_CHECK(cudaMemcpy(hL,dL,(size_t)N*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hR,dR,(size_t)N*sizeof(float),cudaMemcpyDeviceToHost));

    printf("Saturation + outdoor tail...\n");
    softSat(hL,N); softSat(hR,N);
    applyReverb(hL,hR,N,(float)sr,0.10f);
    writeWav(out,hL,hR,N,sr);

    free(he);free(hc);free(hL);free(hR);
    cudaFree(de);cudaFree(dc);cudaFree(dL);cudaFree(dR);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 0;
}
