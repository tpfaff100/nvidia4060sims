/*
 * disco10k.cu — 10,000-oscillator GPU disco synthesizer + algorithmic composer
 * Target: RTX 4060 Ti (sm_89 / Ada Lovelace)
 *
 * BUILD (after vcvars64.bat):
 *   nvcc -arch=sm_89 -O3 -use_fast_math -o disco10k disco10k.cu
 *
 * RUN:
 *   disco10k                         new original song every run
 *   disco10k out.wav 48000 SEED      reproduce a favourite (seed printed)
 *
 * ARCHITECTURE — why this saturates the GPU where the Moog engine didn't:
 *
 *   The Moog engine was VOICE-PER-THREAD: one thread ran the complete
 *   signal chain for one note. With ~400 notes per song and threads that
 *   each ran for seconds, the GPU was mostly idle.
 *
 *   This engine is OSCILLATOR-PER-THREAD: all 10,000 oscillators run
 *   simultaneously. Each thread is assigned exactly one oscillator for
 *   the entire song — it loops over every sample, computing ONE sinusoid,
 *   accumulating the phase in a register, and atomicAdd-ing its contribution
 *   into the shared float32 stereo buffer. Within each warp of 32 threads,
 *   we do a warp-reduce before the atomic write, cutting memory traffic 32×.
 *
 *   10,000 threads × 48,000 Hz × 120 s = 57.6 billion __sinf evaluations.
 *   The 4060 Ti has 544 SFUs running at 2.5 GHz ≈ 1.38 TFLOP/s for trig.
 *   Estimated render time: under 1 minute for a 2-minute song. A CPU with
 *   8 cores doing the same work at ~1 Gsin/s would take ~60 seconds just
 *   for one second of audio.
 *
 * OSCILLATOR ALLOCATION (10,000 total):
 *   Instrument family         Count   Character
 *   ─────────────────────────────────────────────────────────
 *   Bass (Moog-style)          200    fundamental + 14 harmonics + FM cloud
 *   Lead melody               1000    phase-FM sine cluster + full harmonic series
 *   String ensemble           2400    360 detuned saws × 6 harmonics each → lush wall
 *   Choir (3 formant layers)  1200    sine partials shaped by Gaussian formant bands
 *   Pad (ring-mod shimmer)     600    ring-modulated pairs + harmonics
 *   Brass stabs                800    FM-driven harmonic stack, 120 voices × 6 harm
 *   Percussion                 400    inharmonic Bessel-mode partials (kick/snare/hat)
 *   Spatial shimmer            600    room modes + diffuse reverb pre-delay sines
 *   Sub-bass cloud             200    20–55 Hz infrasonic bed
 *   Texture / air              600    barely-audible upper partials, breath noise sines
 *   Harmonic richness fill    2000    additional harmonics on all instruments
 *   TOTAL                    10000
 *
 * COMPOSER — same functional-harmony Markov chain + motif development as
 * discogen.cu, but now the "patch" for each section is a SET OF OSCILLATOR
 * DESCRIPTORS rather than a handful of OscP structs. Each note event maps
 * to a GROUP of oscillators that are activated for its duration; between
 * notes, those oscillators fade via a shared amplitude envelope computed
 * in the kernel (no inter-thread communication needed — each osc knows its
 * own group's envelope parameters).
 *
 * AUDIO QUALITY features unlocked by 10k oscillators:
 *   • String ensemble: 360 slightly detuned saws → emergent chorus/ensemble
 *     beating, impossible to fake with <16 oscillators
 *   • Choir: 200 sines per formant band → smooth Gaussian formant shape
 *     (like convolution) rather than a few peaked resonators
 *   • Percussion: 80–120 inharmonic Bessel-zero partials per hit → membrane
 *     physics rather than simple exponential decays
 *   • Spatial shimmer: 600 room-mode sines with randomised decay rates →
 *     a diffuse reverb tail baked directly into the synthesis, zero CPU cost
 *   • Sub-bass cloud: 200 oscillators 20–55 Hz, each independently amplitude-
 *     modulated → visceral low-end that speakers reproduce physically
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
#include <cassert>

/* ── compile-time constants ─────────────────────────────────────────── */
#define SR          48000
#define N_OSC       10000
#define MAX_EVENTS  4096
#define MAX_SONG_S  180.0f       /* 3 min max                           */
#define TWO_PI      6.28318530717958647692f
#define BLOCK       256          /* threads per CUDA block              */

#define CUDA_CHECK(x) do{ cudaError_t _=x; if(_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_)); \
    exit(1);} }while(0)

/* ══════════════════════════════════════════════════════════════════════
   OSCILLATOR DESCRIPTOR TABLE
   One row per oscillator, laid out for coalesced GPU access.
   All values baked by the CPU composer; the GPU kernel is stateless.
   ══════════════════════════════════════════════════════════════════════ */
struct OscDesc {
    float freq;          /* base Hz (before Doppler/vibrato = 0 here)   */
    float amp;           /* peak amplitude (normalised so sum ~ 1)      */
    float phase0;        /* initial phase, cycles (0..1)                */
    float panL, panR;    /* constant-power stereo gains                 */
    /* envelope: ADSR in seconds, then per-section on/off times         */
    float aA, aD, aS, aR;
    /* FM: freq += fmAmt * sin(2pi * fmFreq * t)  (0 = off)            */
    float fmFreq, fmAmt;
    /* ring-mod: multiply output by sin(2pi * rmFreq * t) (0 = off)    */
    float rmFreq;        /* absolute Hz ring mod                        */
    float rmRatio;       /* pitch-TRACKING ring mod: rm = rmRatio*noteHz*/
    /* vibrato: freq *= (1 + vibDepth * sin(2pi * vibRate * t + vibPh)) */
    float vibRate, vibDepth, vibPh;
    /* amplitude LFO: amp *= (1 + lfoDepth * sin(2pi * lfoRate * t))   */
    float lfoRate, lfoDepth;
    /* group ID: used to look up note events for this oscillator         */
    int   group;
};

/* Note event for one oscillator group                                   */
struct Event {
    float tOn, tOff;     /* absolute time in seconds                    */
    float vel;           /* 0..1 velocity                              */
    float noteHz;        /* pitch ratio multiplier (1.0 for perc/abs)  */
    float hpf;           /* spectral high-pass cutoff Hz (0 = off)     */
    int   group;
};

/* ══════════════════════════════════════════════════════════════════════
   10,000-OSCILLATOR SYNTHESIS KERNEL
   Grid: (N_OSC / BLOCK, 1)   Block: BLOCK threads
   Each thread owns exactly one oscillator for the whole song.
   ══════════════════════════════════════════════════════════════════════ */
__global__ void synthKernel(
    const OscDesc *__restrict__ oscs,
    const Event   *__restrict__ events,
    const int     *__restrict__ evStart,  /* first event for group g   */
    const int     *__restrict__ evCount,  /* count for group g         */
    float         *__restrict__ bufL,
    float         *__restrict__ bufR,
    int totalSamples,
    float sr,
    unsigned seed)
{
    int oi = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (oi >= N_OSC) return;

    const OscDesc o = oscs[oi];
    float invSr = 1.f / sr;

    /* scan events for this group once at the start — kept in registers */
    int eg = o.group;
    int eS = evStart[eg];
    int eC = evCount[eg];

    float phase = o.phase0;
    /* suppress unused-variable warnings — these were scaffolding for a
       future stateful inter-sample optimisation that was simplified away */
    (void)seed;

    for (int si = 0; si < totalSamples; si++) {
        float t = si * invSr;

        /* ── find current event for this group ─────────────────────── */
        bool gated = false;
        float noteHz = o.freq;
        float vel    = 0.f;
        float tRel   = 0.f;
        float dur    = 0.f;
        float hpf    = 0.f;

        for (int ei = eS; ei < eS + eC; ei++) {
            if (t >= events[ei].tOn && t < events[ei].tOff + o.aR + 0.05f) {
                /* o.freq is a RATIO for pitched groups (harmonic number
                   × detune) and an ABSOLUTE Hz for perc/shimmer groups
                   whose events carry noteHz = 1.0. Either way:          */
                noteHz = o.freq * events[ei].noteHz;
                vel    = events[ei].vel;
                tRel   = t - events[ei].tOn;
                dur    = events[ei].tOff - events[ei].tOn;
                hpf    = events[ei].hpf;
                gated  = true;
                break;
            }
        }
        /* ── EVERY thread must reach the warp shuffle below (full-mask
           __shfl_down_sync requires all 32 lanes to participate), so
           silent oscillators fall through with sample = 0 instead of
           `continue`-ing past the reduction.                           */
        float env = 0.f, vib = 1.f, fmOffset = 0.f;
        if (gated) {
            /* ── ADSR envelope ──────────────────────────────────────── */
            if (tRel < o.aA)
                env = tRel / fmaxf(o.aA, 1e-5f);
            else if (tRel < o.aA + o.aD)
                env = 1.f - (tRel - o.aA) / fmaxf(o.aD, 1e-5f) * (1.f - o.aS);
            else if (tRel < dur)
                env = o.aS;
            else {
                float tr = tRel - dur;
                env = (tr < o.aR) ? o.aS * (1.f - tr / fmaxf(o.aR, 1e-5f)) : 0.f;
            }
            env = fmaxf(0.f, fminf(1.f, env));
            if (env > 1e-6f) {
                if (o.vibDepth > 0.f)
                    vib = 1.f + o.vibDepth * __sinf(TWO_PI * o.vibRate * t + o.vibPh);
                if (o.fmAmt > 0.f)
                    fmOffset = o.fmAmt * __sinf(TWO_PI * o.fmFreq * t);
            }
        }

        /* ── advance phase (always, gated or not) ───────────────────── */
        float f = (gated ? noteHz : o.freq) * vib;
        phase += f * invSr;
        phase -= floorf(phase);

        float sample = 0.f;
        /* audibility & anti-alias guard: partials above 18.5 kHz or past
           Nyquist add nothing pleasant — they contribute silence.       */
        if (gated && env > 1e-6f && f <= 18500.f && f <= 0.45f * sr) {
            /* ── SPECTRAL HIGH-PASS: exact 24 dB/oct per-partial gain ──
               g = r^4/(1+r^4), r = f/fc. Zero phase distortion — the
               additive-synthesis superpower over analog filters.        */
            float hpGain = 1.f;
            if (hpf > 0.f) {
                float r  = f / hpf;
                float r4 = r * r; r4 *= r4;
                hpGain = r4 / (1.f + r4);
            }
            if (hpGain >= 1e-4f) {
                float p = phase + fmOffset;
                p -= floorf(p);
                float s = __sinf(TWO_PI * p);

                /* ring modulation: pitch-tracking (musical) or absolute */
                if (o.rmRatio > 0.f)
                    s *= __sinf(TWO_PI * (o.rmRatio * noteHz) * t);
                else if (o.rmFreq > 0.f)
                    s *= __sinf(TWO_PI * o.rmFreq * t);

                float amod = 1.f;
                if (o.lfoDepth > 0.f)
                    amod = 1.f + o.lfoDepth * __sinf(TWO_PI * o.lfoRate * t);

                sample = s * env * vel * o.amp * amod * hpGain;
            }
        }

        /* ── warp-level reduction before global write ───────────────── */
        /* Add contributions from all 32 threads in the warp into lane 0
           then only lane 0 does atomicAdd. This cuts atomic ops 32×.   */
        float sL = sample * o.panL;
        float sR = sample * o.panR;

        /* warp shuffle reduction */
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            sL += __shfl_down_sync(0xFFFFFFFF, sL, offset);
            sR += __shfl_down_sync(0xFFFFFFFF, sR, offset);
        }

        if ((threadIdx.x & 31) == 0 && (sL != 0.f || sR != 0.f)) {
            atomicAdd(&bufL[si], sL);
            atomicAdd(&bufR[si], sR);
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
   CPU COMPOSER — same Markov + motif engine as discogen.cu
   (abbreviated here; full logic identical)
   ══════════════════════════════════════════════════════════════════════ */
static unsigned g_rng = 1;
static float rnd(){ g_rng^=g_rng<<13;g_rng^=g_rng>>17;g_rng^=g_rng<<5;
                    return (float)(g_rng&0xFFFFFF)/16777216.f; }
static float rr(float a,float b){ return a+(b-a)*rnd(); }
static int   ri(int a,int b){ return a+(int)(rnd()*(float)(b-a+1)*0.9999f); }
static int   pick(const float*w,int n){
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
static void makeProg(Chord*prog,int start){
    int cur=start;
    for(int i=0;i<4;i++){
        prog[i]={CH_ROOT[cur],CH_TYPE[cur]};
        if(i==2){float w[6];memcpy(w,MARKOV[cur],24);w[2]*=3;w[4]*=2;cur=pick(w,6);}
        else cur=pick(MARKOV[cur],6);
    }
}
static void chordTones(int key,Chord c,int oct,int*out){
    int r=key+MINOR[c.deg]+12*oct;
    out[0]=r; out[1]=r+((c.type==0)?3:4); out[2]=r+7;
    out[3]=(c.type==2)?r+10:r+12;
}
static float midiHz(int m){ return 440.f*powf(2.f,((float)m-69.f)/12.f); }
static int snapChord(int key,Chord c,int midi){
    int t[4];chordTones(key,c,0,t);
    int best=midi,bd=99;
    for(int o=-24;o<=48;o+=12)for(int i=0;i<4;i++){
        int d=abs(midi-(t[i]+o));if(d<bd){bd=d;best=t[i]+o;}
    }return best;
}
static int snapScale(int key,int midi){
    int best=midi,bd=99;
    for(int o=-24;o<=60;o+=12)for(int i=0;i<7;i++){
        int c=key+MINOR[i]+o,d=abs(midi-c);if(d<bd){bd=d;best=c;}
    }return best;
}

/* rhythm lexicon (off-beat syncopation) */
struct RCell{int n;float t[6],d[6];};
static const RCell RHY[]={
  {4,{0,1,2,3},{0.9f,0.9f,0.9f,0.9f}},
  {5,{0.5f,1,1.5f,2.5f,3},{0.4f,0.4f,0.9f,0.4f,0.9f}},
  {5,{0,0.5f,1.5f,2,3},{0.4f,0.9f,0.4f,0.9f,0.9f}},
  {6,{0,0.5f,1,1.5f,2.5f,3},{0.4f,0.4f,0.4f,0.9f,0.4f,0.9f}},
  {4,{0.5f,1.5f,2,3},{0.9f,0.4f,0.9f,0.9f}},
  {3,{0,1.5f,2.5f},{1.4f,0.9f,1.4f}},
};
#define NRHY 6

/* ══════════════════════════════════════════════════════════════════════
   OSCILLATOR LAYER DEFINITIONS
   Each oscillator is part of a named GROUP.  Groups map to musical roles.
   ══════════════════════════════════════════════════════════════════════ */
enum GroupID {
    G_BASS=0, G_LEAD, G_STRING_A, G_STRING_B,
    G_CHOIR_A, G_CHOIR_B, G_CHOIR_C,
    G_PAD, G_BRASS, G_KICK, G_SNARE, G_HAT,
    G_SHIMMER, G_SUBBASS, G_TEXTURE, G_ARP,
    N_GROUPS
};
/* oscillator count per group — must sum to N_OSC */
static const int G_COUNT[N_GROUPS]={
    200,  /* BASS          */
    800,  /* LEAD          */
    900,  /* STRING_A      */
    900,  /* STRING_B      */
    360,  /* CHOIR_A       */
    360,  /* CHOIR_B       */
    360,  /* CHOIR_C       */
    500,  /* PAD           */
    700,  /* BRASS         */
    220,  /* KICK          */
    200,  /* SNARE         */
    180,  /* HAT           */
    600,  /* SHIMMER       */
    200,  /* SUBBASS       */
   3120,  /* TEXTURE (harmonic-series air, fills to 10,000 exactly)   */
    400,  /* ARP (arpeggiator: 50 voices × 8 harmonics, half ring-mod) */
};
/* verify at compile-time: use a runtime check */

/* ── build oscillator descriptor table ─────────────────────────────── */
static void buildOscillators(OscDesc*D, int key)
{
    int idx=0;
    /* helper: constant-power pan from angle (-1..+1)                   */
    auto pan=[](float a,float&L,float&R){
        float x=(a+1.f)*0.25f*(float)M_PI;
        L=cosf(x);R=sinf(x);
    };

    /* ── GROUP 0: BASS (200 osc) ─────────────────────────────────────
       Layers: fundamental harmonic series 1..14 × 14 = 196, + 4 FM osc */
    {
        for(int h=1;h<=14&&idx<200;h++){
            for(int v=0;v<14&&idx<200;v++){
                OscDesc&d=D[idx++];
                /* harmonic h, voice v: amplitude = 1/h^0.85,
                   freq will be multiplied by note root                */
                d.freq   = (float)h;        /* ratio — scaled by noteHz */
                d.amp    = powf(1.f/(float)h,0.85f)*0.012f;
                d.phase0 = rnd();
                pan(rr(-0.12f,0.12f),d.panL,d.panR);
                d.aA=0.004f;d.aD=0.15f;d.aS=0.50f;d.aR=0.14f;
                d.fmFreq=(h==1)?rr(2.f,3.f):0.f;
                d.fmAmt =(h==1)?rr(0.08f,0.18f):0.f;
                d.rmFreq=0.f;
                d.vibRate=0.f;d.vibDepth=0.f;d.vibPh=0.f;
                d.lfoRate=0.f;d.lfoDepth=0.f;
                d.group=G_BASS;
            }
        }
        /* + 4 FM growl oscillators: sub-fundamental with strong slow FM,
           the "throat" of the bass when the note digs in                */
        for(int v=0;v<4;v++){
            OscDesc&d=D[idx++];
            d.freq   = 0.5f;                 /* sub-octave ratio         */
            d.amp    = 0.010f*(0.8f+0.4f*rnd());
            d.phase0 = rnd();
            pan(rr(-0.06f,0.06f),d.panL,d.panR);
            d.aA=0.005f;d.aD=0.18f;d.aS=0.55f;d.aR=0.15f;
            d.fmFreq=rr(15.f,40.f);d.fmAmt=rr(0.5f,1.2f);
            d.rmFreq=0.f;d.rmRatio=0.f;
            d.vibRate=0.f;d.vibDepth=0.f;d.vibPh=0.f;
            d.lfoRate=0.f;d.lfoDepth=0.f;
            d.group=G_BASS;
        }
    }

    /* ── GROUP 1: LEAD MELODY (800 osc) ──────────────────────────────
       Layers: 12-osc phase-FM core × 10 note-harmonics = 120
               + 400 shimmer sines (random detune, low amp)
               + 280 harmonic fill                                     */
    {
        /* core: 10 harmonics × 80 detuned voices each                  */
        for(int h=1;h<=10;h++){
            float hAmp=powf(1.f/(float)h,0.65f)*0.004f;
            for(int v=0;v<80;v++){
                OscDesc&d=D[idx++];
                d.freq  =(float)h;
                d.amp   =hAmp*(0.8f+0.4f*rnd());
                d.phase0=rnd();
                pan(rr(-0.55f,0.55f),d.panL,d.panR);
                d.aA=0.007f;d.aD=0.28f;d.aS=0.70f;d.aR=0.45f;
                float ix=rr(0.f,0.5f)*(h==1?1.f:0.3f);
                d.fmFreq=(ix>0.01f)?rr(0.9f,1.1f)*(float)(h+(h<3?1:0)):0.f;
                d.fmAmt=ix;
                d.rmFreq=0.f;
                d.vibRate=rr(5.0f,6.2f);d.vibDepth=rr(0.004f,0.009f);
                d.vibPh=rnd()*TWO_PI;
                d.lfoRate=rr(0.3f,0.8f);d.lfoDepth=rr(0.01f,0.04f);
                d.group=G_LEAD;
            }
        }
    }

    /* ── GROUPS 2-3: STRING ENSEMBLE (900 + 900 = 1800 osc) ─────────
       900 saws approximated as harmonic sums up to H=6.
       150 voices × 6 harmonics = 900 per ensemble.                   */
    for(int grp=0;grp<2;grp++){
        int gid=(grp==0)?G_STRING_A:G_STRING_B;
        float octShift=(grp==0)?0.f:1.f;
        for(int v=0;v<150;v++){
            float detune=rr(-22.f,22.f);          /* cents              */
            float vPan=rr(-0.75f,0.75f);
            for(int h=1;h<=6;h++){
                OscDesc&d=D[idx++];
                /* sawtooth = sum 1/h; detune applied to the whole voice*/
                d.freq  = (float)h * powf(2.f,detune/1200.f+octShift);
                d.amp   = (1.f/(float)h)*0.0022f*(0.85f+0.3f*rnd());
                d.phase0= rnd();
                pan(vPan,d.panL,d.panR);
                d.aA=0.30f;d.aD=0.55f;d.aS=0.84f;d.aR=1.0f;
                d.fmFreq=0.f;d.fmAmt=0.f;d.rmFreq=0.f;
                d.vibRate=5.3f;d.vibDepth=0.012f;d.vibPh=rnd()*TWO_PI;
                d.lfoRate=rr(2.8f,3.6f);d.lfoDepth=rr(0.03f,0.06f);
                d.group=gid;
            }
        }
    }

    /* ── GROUPS 4-6: CHOIR (360 × 3 = 1080 osc) ─────────────────────
       200 sines per formant band, shaped by Gaussian envelope.
       Three vowel formant configurations (Aah, Oh, Ee).              */
    struct Formant{float f1,f2,f3,bw;};
    Formant vowels[3]={
        {800.f,1200.f,2500.f,120.f},    /* Aah  */
        {500.f, 850.f,2500.f, 90.f},    /* Oh   */
        {270.f,2200.f,3000.f, 80.f},    /* Ee   */
    };
    for(int vow=0;vow<3;vow++){
        int gid=G_CHOIR_A+vow;
        Formant&F=vowels[vow];
        for(int i=0;i<360;i++){
            OscDesc&d=D[idx++];
            /* distribute harmonics 1..120, shaped by formant Gaussians */
            float h=(float)(i%120)+1.f;
            /* pick which formant this partial belongs to               */
            /* Gaussian weight per formant                              */
            float baseHz=261.63f;       /* middle C placeholder; scaled by noteHz */
            float fHz=baseHz*h;
            auto gauss=[](float x,float mu,float sig)->float{
                float d=(x-mu)/sig;return expf(-0.5f*d*d);
            };
            float g1=gauss(fHz,F.f1,F.bw*1.5f);
            float g2=gauss(fHz,F.f2,F.bw*2.0f)*0.7f;
            float g3=gauss(fHz,F.f3,F.bw*2.5f)*0.4f;
            float gTotal=g1+g2+g3+0.01f;
            d.freq  =h;
            d.amp   =gTotal*0.003f*(0.8f+0.4f*rnd());
            d.phase0=rnd();
            pan(rr(-0.65f,0.65f),d.panL,d.panR);
            d.aA=0.3f;d.aD=0.5f;d.aS=0.82f;d.aR=0.9f;
            d.fmFreq=0.f;d.fmAmt=0.f;d.rmFreq=0.f;
            d.vibRate=rr(4.8f,5.6f);d.vibDepth=rr(0.015f,0.028f);
            d.vibPh=rnd()*TWO_PI;
            d.lfoRate=rr(0.4f,0.9f);d.lfoDepth=rr(0.02f,0.05f);
            d.group=gid;
        }
    }

    /* ── GROUP 7: PAD ring-mod shimmer (500 osc) ─────────────────────
       500 pitch-tracking ring-modulated oscillators. Each is ring-modded
       against a frequency 60-130 cents above its own — the sum and
       difference sidebands beat slowly against neighbouring pairs,
       producing the Tibetan-bowl shimmer. rmRatio (not rmFreq) makes
       the ring mod TRACK the note pitch, so the shimmer stays musical
       in every chord.                                                  */
    {
        for(int p2=0;p2<500;p2++){
            float baseRatio=(float)(1+p2%6);
            float splitCents=rr(60.f,130.f);
            OscDesc&dA=D[idx++];
            dA.freq  =baseRatio;
            dA.amp   =0.0035f*(0.7f+0.5f*rnd());
            dA.phase0=rnd();
            pan(rr(-0.8f,0.8f),dA.panL,dA.panR);
            dA.aA=0.35f;dA.aD=0.6f;dA.aS=0.85f;dA.aR=1.2f;
            dA.fmFreq=0.f;dA.fmAmt=0.f;
            dA.rmFreq=0.f;
            /* pitch-tracking ring mod, slightly detuned from itself     */
            dA.rmRatio=baseRatio*(1.f+splitCents/1200.f);
            dA.vibRate=rr(3.5f,4.5f);dA.vibDepth=0.02f;dA.vibPh=rnd()*TWO_PI;
            dA.lfoRate=rr(3.6f,4.6f);dA.lfoDepth=0.10f;
            dA.group=G_PAD;
        }
    }

    /* ── GROUP 8: BRASS (700 osc) ────────────────────────────────────
       100 voices × 7 harmonics; odd harmonics emphasised (brass body) */
    {
        for(int v=0;v<100;v++){
            float det=rr(-8.f,8.f);
            float vPan=rr(-0.6f,0.6f);
            float pres=rr(0.7f,1.2f);   /* pressure: shapes spectrum   */
            for(int h=1;h<=7;h++){
                OscDesc&d=D[idx++];
                float brassA=(1.f-expf(-h*pres*0.4f))*powf((float)h,-1.1f);
                d.freq  =(float)h*powf(2.f,det/1200.f);
                d.amp   =brassA*0.005f*(0.8f+0.4f*rnd());
                d.phase0=rnd();
                pan(vPan,d.panL,d.panR);
                d.aA=0.004f;d.aD=0.12f;d.aS=0.6f;d.aR=0.15f;
                /* FM inharmonicity: slight index on each               */
                d.fmFreq=(h<=2)?rr(0.5f,1.5f)*(float)h:0.f;
                d.fmAmt =(h<=2)?rr(0.3f,0.7f)*pres:0.f;
                d.rmFreq=0.f;
                d.vibRate=rr(5.8f,6.5f);d.vibDepth=rr(0.005f,0.012f);
                d.vibPh=rnd()*TWO_PI;
                d.lfoRate=0.f;d.lfoDepth=0.f;
                d.group=G_BRASS;
            }
        }
    }

    /* ── GROUPS 9-11: PERCUSSION (220+200+180 = 600 osc) ─────────────
       Bessel-zero-spaced inharmonic partials, each with individual decay */
    static const float BZ[]={1.f,2.295f,3.598f,4.903f,6.209f,7.516f,
                              8.823f,10.13f,11.44f,12.75f,14.06f,15.36f,
                              16.67f,17.98f,19.29f,20.6f,21.9f,23.21f};
    int NBZ=18;
    /* kick: low, fast decay */
    for(int i=0;i<220;i++){
        OscDesc&d=D[idx++];
        int bi=i%NBZ;
        float r=BZ[bi]*(0.9f+0.2f*rnd());
        d.freq  =r*rr(55.f,75.f);       /* absolute Hz for perc        */
        d.amp   =powf(r,-1.2f)*0.008f*(rnd()*0.4f+0.6f);
        d.phase0=rnd();
        pan(rr(-0.15f,0.15f),d.panL,d.panR);
        d.aA=0.002f;d.aD=0.f;d.aS=0.f;
        d.aR=rr(0.04f,0.12f)*(1.f+0.05f*bi);
        d.fmFreq=rr(80.f,180.f);d.fmAmt=rr(2.f,6.f)*(1.f-bi*0.04f);
        d.fmAmt=fmaxf(0.f,d.fmAmt);
        d.rmFreq=0.f;d.vibRate=0.f;d.vibDepth=0.f;d.vibPh=0.f;
        d.lfoRate=0.f;d.lfoDepth=0.f;
        d.group=G_KICK;
        /* NOTE for perc groups: freq is ABSOLUTE, not a ratio.
           The event noteHz will be set to 1.0 so freq stays unchanged */
    }
    /* snare: mid inharmonic */
    for(int i=0;i<200;i++){
        OscDesc&d=D[idx++];
        int bi=i%NBZ;
        d.freq  =BZ[bi]*rr(180.f,280.f);
        d.amp   =powf(BZ[bi],-0.9f)*0.006f*(rnd()*0.4f+0.6f);
        d.phase0=rnd();
        pan(rr(-0.08f,0.08f),d.panL,d.panR);
        d.aA=0.001f;d.aD=0.f;d.aS=0.f;
        d.aR=rr(0.06f,0.18f)*(1.f+0.04f*bi);
        d.fmFreq=0.f;d.fmAmt=0.f;d.rmFreq=0.f;
        d.vibRate=0.f;d.vibDepth=0.f;d.vibPh=0.f;
        d.lfoRate=rr(15.f,40.f);d.lfoDepth=0.8f;/* noise-like AM       */
        d.group=G_SNARE;
    }
    /* hat: high, fast */
    for(int i=0;i<180;i++){
        OscDesc&d=D[idx++];
        d.freq  =rr(3000.f,14000.f);
        d.amp   =0.003f*(rnd()*0.4f+0.6f);
        d.phase0=rnd();
        pan(rr(-0.05f,0.05f),d.panL,d.panR);
        d.aA=0.001f;d.aD=0.f;d.aS=0.f;d.aR=rr(0.015f,0.055f);
        d.fmFreq=0.f;d.fmAmt=0.f;d.rmFreq=0.f;
        d.vibRate=0.f;d.vibDepth=0.f;d.vibPh=0.f;
        d.lfoRate=rr(25.f,80.f);d.lfoDepth=0.9f;
        d.group=G_HAT;
    }

    /* ── GROUP 12: SHIMMER / reverb sines (600 osc) ──────────────────
       Random-frequency sines with very slow, independent AM:
       produces a diffuse shimmering reverb-like tail baked into synthesis */
    {
        for(int i=0;i<600;i++){
            OscDesc&d=D[idx++];
            d.freq  =rr(200.f,8000.f);
            d.amp   =0.0006f*rnd();
            d.phase0=rnd();
            pan(rr(-1.f,1.f),d.panL,d.panR);
            d.aA=rr(0.5f,2.f);d.aD=rr(0.5f,1.5f);d.aS=0.5f;d.aR=rr(1.f,4.f);
            d.fmFreq=0.f;d.fmAmt=0.f;d.rmFreq=0.f;
            d.vibRate=0.f;d.vibDepth=0.f;d.vibPh=0.f;
            d.lfoRate=rr(0.05f,0.4f);d.lfoDepth=0.9f;
            d.group=G_SHIMMER;
        }
    }

    /* ── GROUP 13: SUB-BASS CLOUD (200 osc) ──────────────────────────
       Infrasonic bed 20-55 Hz — felt rather than heard.
       All locked to chord root, slowly AM'd                           */
    {
        for(int i=0;i<200;i++){
            OscDesc&d=D[idx++];
            float harm=(float)(1+i%3);
            d.freq  =harm;            /* ratio: ×noteHz in kernel       */
            d.amp   =0.0065f/harm*(0.7f+0.6f*rnd());
            d.phase0=rnd();
            pan(rr(-0.25f,0.25f),d.panL,d.panR);
            d.aA=0.4f;d.aD=0.6f;d.aS=0.9f;d.aR=1.5f;
            d.fmFreq=0.f;d.fmAmt=0.f;d.rmFreq=0.f;
            d.vibRate=rr(0.3f,0.8f);d.vibDepth=rr(0.02f,0.05f);
            d.vibPh=rnd()*TWO_PI;
            d.lfoRate=rr(0.2f,0.6f);d.lfoDepth=rr(0.15f,0.40f);
            d.group=G_SUBBASS;
        }
    }

    /* ── GROUP 14: TEXTURE / AIR (3120 osc) ─────────────────────────
       Tuned to the HARMONIC SERIES of the chord root (freq = ratio),
       so all 3120 partials are consonant with whatever chord is
       playing — "attractive to human hearing" by construction.
       Consonance-biased harmonic selection: low harmonics and
       octave/fifth-related ones (2,3,4,6,8,12,16...) are favoured;
       amplitude tapers as 1/h^1.15 so the series sums to a warm,
       natural rolloff. A quarter get consonant pitch-tracking ring
       mod (ratios 1.5 / 2 / 3 = perfect fifth / octave / twelfth,
       whose sum & difference tones stay inside the harmonic series). */
    {
        static const int CONS[]={1,2,3,4,5,6,8,10,12,16,20,24,32,48};
        int NC2=14;
        for(int i=0;i<3120;i++){
            OscDesc&d=D[idx++];
            int h;
            if(rnd()<0.6f) h=CONS[ri(0,NC2-1)];        /* consonant set */
            else           h=1+ri(0,47);               /* general series*/
            float det=rr(-6.f,6.f);                    /* gentle cents  */
            d.freq  =(float)h*powf(2.f,det/1200.f);    /* RATIO         */
            /* natural harmonic rolloff, mid-band presence lift         */
            float roll=powf(1.f/(float)h,1.15f);
            d.amp   =roll*0.0022f*(0.6f+0.8f*rnd());
            d.phase0=rnd();
            pan(rr(-1.f,1.f),d.panL,d.panR);
            d.aA=rr(0.05f,0.6f);d.aD=0.4f;d.aS=0.6f;d.aR=rr(0.2f,1.2f);
            d.fmFreq=0.f;d.fmAmt=0.f;
            d.rmFreq=0.f;
            /* consonant ring mod on 25%: fifth/octave/twelfth ratios   */
            if(rnd()<0.25f){
                float rma[3]={1.5f,2.0f,3.0f};
                d.rmRatio=rma[ri(0,2)];
            }
            d.vibRate=(rnd()<0.3f)?rr(0.5f,8.f):0.f;
            d.vibDepth=(d.vibRate>0.f)?rr(0.005f,0.03f):0.f;
            d.vibPh=rnd()*TWO_PI;
            d.lfoRate=rr(0.8f,18.f);d.lfoDepth=rr(0.2f,0.8f);
            d.group=G_TEXTURE;
        }
    }

    /* ── GROUP 15: ARPEGGIATOR (400 osc) ─────────────────────────────
       50 voices × 8 harmonics. Fast plucky envelope for 16th-note
       broken-chord runs. HALF the voices carry pitch-tracking ring
       modulation at consonant ratios — the arpeggio alternates between
       pure and ring-modulated timbres as the composer round-robins
       through voices, giving that sparkling metallic disco-sequencer
       character (think Moroder's "Chase").                            */
    {
        for(int v=0;v<50;v++){
            float det=rr(-4.f,4.f);
            float vPan=((v&1)?1.f:-1.f)*rr(0.3f,0.8f);  /* L/R ping-pong */
            int   ringy=(v%2==0);
            float rma[3]={1.5f,2.0f,3.0f};
            float rmr=ringy? rma[ri(0,2)]:0.f;
            for(int h=1;h<=8;h++){
                OscDesc&d=D[idx++];
                d.freq  =(float)h*powf(2.f,det/1200.f);
                d.amp   =powf(1.f/(float)h,0.9f)*0.006f*(0.8f+0.4f*rnd());
                d.phase0=rnd();
                pan(vPan,d.panL,d.panR);
                d.aA=0.002f;d.aD=0.06f;d.aS=0.25f;d.aR=0.09f;
                d.fmFreq=0.f;d.fmAmt=0.f;
                d.rmFreq=0.f;
                d.rmRatio=(h<=4)?rmr:0.f;   /* ring the body, not the air */
                d.vibRate=0.f;d.vibDepth=0.f;d.vibPh=0.f;
                d.lfoRate=0.f;d.lfoDepth=0.f;
                d.group=G_ARP;
            }
        }
    }

    /* pad fill — should be exactly N_OSC now                           */
    assert(idx==N_OSC);
}

/* ══════════════════════════════════════════════════════════════════════
   EVENT GENERATOR — maps composition output to group events
   ══════════════════════════════════════════════════════════════════════ */
struct EvBuf{
    Event ev[MAX_EVENTS]; int n;
    void add(float ton,float toff,float vel,float hz,int g,float hpf=0.f){
        if(n>=MAX_EVENTS)return;
        ev[n++]={ton,toff,vel,hz,hpf,g};
    }
};

/* add events for all pitched instruments for one bar-chord.
   hpf: optional spectral high-pass (2000/5000/7000 Hz palette) applied
   to the harmonic bed (strings/choir/pad/texture/shimmer) — the bass
   and sub-bass stay full-range so the groove keeps its body.           */
static void chordEvent(EvBuf&E,float tOn,float tOff,float vel,
                       int key,Chord c,int oct,float hpf=0.f)
{
    int t[4];chordTones(key,c,oct,t);
    float root =midiHz(t[0]);
    float third=midiHz(t[1]);
    float fifth=midiHz(t[2]);
    /* strings: A + B octave up — root of chord                         */
    E.add(tOn,tOff,vel*0.7f, root,     G_STRING_A,hpf);
    E.add(tOn,tOff,vel*0.6f, root*2.f, G_STRING_B,hpf);
    /* choir: three vowels on root / fifth / third                      */
    E.add(tOn,tOff,vel*0.55f,root, G_CHOIR_A,hpf);
    E.add(tOn,tOff,vel*0.50f,fifth,G_CHOIR_B,hpf);
    E.add(tOn,tOff,vel*0.45f,third,G_CHOIR_C,hpf);
    /* pad shimmer: root                                                 */
    E.add(tOn,tOff,vel*0.45f,root,G_PAD,hpf);
    /* sub bass: root one octave lower — never high-passed              */
    E.add(tOn,tOff,vel*0.9f, root*0.5f,G_SUBBASS);
    /* shimmer: ABSOLUTE-frequency group -> pitch ratio must be 1.0     */
    E.add(tOn,tOff,vel*0.35f,1.f,G_SHIMMER,hpf);
    /* texture: harmonic-series air locked to chord root                */
    E.add(tOn,tOff,vel*0.25f,root,G_TEXTURE,hpf);
}
static void bassEvent(EvBuf&E,float tOn,float tOff,float vel,int midi){
    E.add(tOn,tOff,vel,midiHz(midi),G_BASS);
}
static void leadEvent(EvBuf&E,float tOn,float tOff,float vel,int midi){
    E.add(tOn,tOff,vel,midiHz(midi),G_LEAD);
}
static void brassEvent(EvBuf&E,float tOn,float tOff,float vel,int midi){
    E.add(tOn,tOff,vel,midiHz(midi),G_BRASS);
}
static void kickEvent(EvBuf&E,float t){
    E.add(t,t+0.35f,1.0f,1.f,G_KICK);  /* freq=1 (absolute in desc)   */
}
static void snareEvent(EvBuf&E,float t){
    E.add(t,t+0.22f,0.85f,1.f,G_SNARE);
}
static void hatEvent(EvBuf&E,float t){
    E.add(t,t+0.12f,0.55f,1.f,G_HAT);
}

/* spectral high-pass palette — the three requested cutoffs             */
static const float HPF_SEL[3]={2000.f,5000.f,7000.f};

/* ── full song composer ─────────────────────────────────────────────── */
static int composeSong(EvBuf&E,int key,float bps,char*report)
{
    float beat=1.f/bps;
    Chord vProg[4],cProg[4];
    makeProg(vProg,0);
    makeProg(cProg,(rnd()<0.5f)?3:0);

    /* motif storage */
    int   mDeg[8];float mT[8],mD[8];int mN=0;
    /* rhythm motif */
    const RCell&rc=RHY[ri(0,NRHY-1)];
    mN=rc.n;
    for(int i=0;i<mN;i++){mT[i]=rc.t[i];mD[i]=rc.d[i];mDeg[i]=ri(-1,5);}

    /* form */
    struct Sec{const char*name;int bars;float en;};
    Sec form[8];int nf=0;
    form[nf++]={"INTRO",ri(4,8),0.28f};
    form[nf++]={"VERSE",16,0.55f};
    form[nf++]={"PRE",8,0.72f};
    form[nf++]={"CHORUS",16,0.96f};
    form[nf++]={"BREAK",8,0.40f};
    if(rnd()<0.55f)form[nf++]={"VERSE",8,0.60f};
    form[nf++]={"CHORUS",16,1.0f};
    form[nf++]={"OUTRO",ri(6,10),0.32f};

    /* melody base register */
    int melBase=key+24; if(melBase<66)melBase+=12;

    float bar=0.f;
    int nArp=0,nHpf=0;
    for(int s=0;s<nf;s++){
        Sec&S=form[s];
        bool isChorus=(strcmp(S.name,"CHORUS")==0);
        bool isPre   =(strcmp(S.name,"PRE")==0);
        bool isBreak =(strcmp(S.name,"BREAK")==0);
        Chord*prog=isChorus?cProg:vProg;
        for(int b=0;b<S.bars;b++){
            float bb=(bar+b)*4.f*beat;
            float be=bb+4.f*beat;
            Chord ch=prog[b%4];
            float E2=S.en;

            /* percussion */
            bool isIntro=(s==0), isOutro=(s==nf-1);
            if(!isIntro||b>=S.bars/2){
                for(int q=0;q<4;q++){
                    kickEvent(E,bb+q*beat);
                    if(q%2==1)snareEvent(E,bb+q*beat);
                    hatEvent(E,bb+q*beat);
                    hatEvent(E,bb+q*beat+beat*0.5f);
                }
                if(b%8==7&&E2>0.5f)
                    for(int q=0;q<8;q++)snareEvent(E,bb+2*beat+q*beat*0.25f);
            }

            /* chord bed — with the spectral HPF palette:
               BREAK: harmonic bed thinned by a random 2k/5k/7k cutoff
                      (bass & sub stay full — the classic disco
                      "everything drops out but the groove" trick)
               PRE:   opening build — 7 kHz for the first third,
                      5 kHz for the middle, 2 kHz for the last, so the
                      spectrum "descends into" the chorus              */
            float cv=isOutro? E2*(1.f-(float)b/S.bars):E2;
            float hp=0.f;
            if(isBreak){ hp=HPF_SEL[ri(0,2)]; nHpf++; }
            else if(isPre){
                int ph3=(b*3)/S.bars; if(ph3>2)ph3=2;
                hp=HPF_SEL[2-ph3]; nHpf++;
            }
            chordEvent(E,bb,be,cv,key,ch,1,hp);

            /* brass stabs: chorus only, on 2-beat marks */
            if(isChorus){
                int ct[4];chordTones(key,ch,2,ct);
                for(int q=0;q<2;q++){
                    float ts=bb+q*2*beat;
                    for(int v2=0;v2<4;v2++)
                        brassEvent(E,ts,ts+beat*0.3f,E2*0.8f,ct[v2]);
                }
            }

            /* bass */
            if(!isIntro||b>=S.bars/2){
                /* generate bass notes with funk pattern                 */
                int t4[4];chordTones(key,ch,0,t4);
                int root=t4[0]; while(root>key+12+12)root-=12;
                if(root<28)root+=12;
                float r=rnd();
                if(r<0.4f){
                    bassEvent(E,bb,bb+beat*1.8f,0.95f,root);
                    bassEvent(E,bb+beat*2,bb+beat*3.8f,0.90f,root+7);
                }else{
                    bassEvent(E,bb,bb+beat*0.9f,0.98f,root);
                    bassEvent(E,bb+beat*0.5f,bb+beat*1.4f,0.82f,root);
                    bassEvent(E,bb+beat,bb+beat*1.9f,0.90f,root+3);
                    bassEvent(E,bb+beat*1.5f,bb+beat*2.4f,0.82f,root);
                    bassEvent(E,bb+beat*2,bb+beat*2.9f,0.90f,root+7);
                    bassEvent(E,bb+beat*2.5f,bb+beat*3.4f,0.82f,root+10);
                    bassEvent(E,bb+beat*3,be,0.88f,root+(rnd()<0.3f?12:5));
                }
            }

            /* melody: 2-bar motif development                          */
            if(b%2==0){
                bool hasMel=(!isIntro||(b>=S.bars/2))&&!isOutro;
                if(!hasMel&&isOutro&&b<2)hasMel=true;
                if(hasMel){
                    int reg=melBase;
                    if(isChorus)reg+=4;
                    if(isPre)reg+=b;
                    /* pick development op */
                    int dev=ri(0,3);
                    float mv=0.55f+0.4f*E2;
                    float shift=(dev==3)?beat*0.5f:0.f;
                    for(int i=0;i<mN;i++){
                        int deg=mDeg[i];
                        if(dev==1)deg+=1;
                        if(dev==2)deg=-deg+2;
                        int mdeg=(deg%7+7)%7;
                        int oct2=deg/7+(deg<0?-1:0);
                        int midi=key+MINOR[mdeg]+12*(4+oct2);
                        bool strong=(fmodf(mT[i],1.f)<0.01f);
                        midi=strong?snapChord(key,ch,midi):snapScale(key,midi);
                        if(midi<48)midi+=12;if(midi>92)midi-=12;
                        float ton=bb+mT[i]*beat+shift;
                        float toff=ton+mD[i]*beat;
                        leadEvent(E,ton,toff,mv*rr(0.92f,1.f),midi);
                    }
                }
            }

            /* ── OCCASIONAL ARPEGGIOS: 16th-note broken-chord runs ────
               35% chance per 2-bar phrase in PRE and CHORUS sections.
               Three patterns: up (2 octaves), down, up-down triangle.
               Half the runs get a random HPF from the 2k/5k/7k palette
               (airy sparkle); the ARP voices themselves alternate pure
               and ring-modulated timbres as the run climbs.            */
            if((isChorus||isPre)&&b%2==0&&rnd()<0.35f){
                int ct[4];chordTones(key,ch,2,ct);
                int pat=ri(0,2);
                float ahp=(rnd()<0.5f)?HPF_SEL[ri(0,2)]:0.f;
                if(ahp>0.f)nHpf++;
                for(int q=0;q<32;q++){          /* 16ths across 2 bars */
                    int pi2;
                    if(pat==0)      pi2=q%8;              /* up          */
                    else if(pat==1) pi2=7-(q%8);          /* down        */
                    else{int m8=q%14;pi2=(m8<8)?m8:14-m8;}/* up-down     */
                    int midi=ct[pi2%4]+12*(pi2/4);
                    float ton=bb+q*0.25f*beat;
                    E.add(ton,ton+0.22f*beat,
                          E2*rr(0.5f,0.68f),midiHz(midi),G_ARP,ahp);
                }
                nArp++;
            }
        }
        bar+=S.bars;
    }

    /* final tonic hold */
    float endB=bar*4.f*beat;
    int t4[4];Chord tonic={0,0};chordTones(key,tonic,1,t4);
    for(int v2=0;v2<4;v2++){
        bassEvent(E,endB,endB+8.f,0.9f,t4[0]-12);
        chordEvent(E,endB,endB+8.f,0.7f,key,tonic,1);
        leadEvent(E,endB+1.f,endB+8.f,0.65f,t4[3]+12);
    }

    static const char*NM[12]={"C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"};
    static const char*RN[7]={"i","ii","III","iv","v","VI","VII"};
    char*p=report;
    p+=sprintf(p,"Key: %s minor  Tempo: %.0f BPM\nForm: ",NM[key%12],bps*60.f);
    for(int s=0;s<nf;s++)p+=sprintf(p,"%s(%d) ",form[s].name,form[s].bars);
    p+=sprintf(p,"\nVerse:  "); for(int i=0;i<4;i++)p+=sprintf(p,"%s ",RN[vProg[i].deg]);
    p+=sprintf(p,"\nChorus: "); for(int i=0;i<4;i++)p+=sprintf(p,"%s ",RN[cProg[i].deg]);
    p+=sprintf(p,"\nArpeggio runs: %d   Spectral HPF events (2k/5k/7k): %d\n",
               nArp,nHpf);
    return E.n;
}

/* ── sort events and build per-group index ──────────────────────────── */
static void indexEvents(const EvBuf&E,
                        int*evStart,int*evCount,
                        Event*sortedEv)
{
    memset(evStart,0,N_GROUPS*sizeof(int));
    memset(evCount,0,N_GROUPS*sizeof(int));
    for(int i=0;i<E.n;i++)
        if(E.ev[i].group>=0&&E.ev[i].group<N_GROUPS)
            evCount[E.ev[i].group]++;
    evStart[0]=0;
    for(int g=1;g<N_GROUPS;g++)evStart[g]=evStart[g-1]+evCount[g-1];
    int cur[N_GROUPS];memcpy(cur,evStart,N_GROUPS*sizeof(int));
    for(int i=0;i<E.n;i++){
        int g=E.ev[i].group;
        if(g>=0&&g<N_GROUPS)sortedEv[cur[g]++]=E.ev[i];
    }
}

/* ── CPU post-processing ─────────────────────────────────────────────── */
static void softSat(float*x,int N,float d){
    for(int i=0;i<N;i++)x[i]=tanhf(x[i]*d)/d;
}
static void applyEQ(float*L,float*R,int N,float sr){
    /* gentle high-shelf rolloff above 14 kHz + low-shelf boost <60 Hz  */
    float cH=expf(-TWO_PI*14000.f/sr), cL=expf(-TWO_PI*60.f/sr);
    float zHL=0,zHR=0,zLL=0,zLR=0;
    for(int i=0;i<N;i++){
        float lpHL=zHL*cH+L[i]*(1-cH);zHL=lpHL;
        float lpHR=zHR*cH+R[i]*(1-cH);zHR=lpHR;
        float lpLL=zLL*cL+L[i]*(1-cL);zLL=lpLL;
        float lpLR=zLR*cL+R[i]*(1-cL);zLR=lpLR;
        L[i]=L[i]-lpHL*0.30f+lpLL*0.45f;
        R[i]=R[i]-lpHR*0.30f+lpLR*0.45f;
    }
}
static void writeWav(const char*p,const float*L,const float*R,int N,int sr){
    FILE*f=fopen(p,"wb");if(!f){fprintf(stderr,"open fail\n");return;}
    int16_t*b=(int16_t*)malloc((size_t)N*4);
    float pk=1e-9f;
    for(int i=0;i<N;i++){pk=fmaxf(pk,fabsf(L[i]));pk=fmaxf(pk,fabsf(R[i]));}
    float g=0.93f/pk;
    for(int i=0;i<N;i++){
        float d=(((float)(rand()&0xFFFF)+(float)(rand()&0xFFFF))/65536.f-1.f)/32768.f;
        b[i*2  ]=(int16_t)(fmaxf(-1.f,fminf(1.f,L[i]*g+d))*32767.f);
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
int main(int argc,char**argv){
    const char*out=(argc>1)?argv[1]:"disco10k.wav";
    int sr=(argc>2)?atoi(argv[2]):SR;
    unsigned seed=(argc>3)?(unsigned)atoi(argv[3]):(unsigned)time(NULL);
    g_rng=seed?seed:1; srand(seed);

    printf("═══ 10,000-OSCILLATOR GENERATIVE DISCO ═══\n");
    printf("SEED: %u  (rerun 'disco10k out.wav %d %u')\n\n",seed,sr,seed);

    cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("Device: %s (SM %d.%d, %d SMs, %lld MB VRAM)\n\n",
           prop.name,prop.major,prop.minor,prop.multiProcessorCount,
           (long long)prop.totalGlobalMem/1048576);

    /* — verify group counts sum to N_OSC — */
    int gtot=0;for(int i=0;i<N_GROUPS;i++)gtot+=G_COUNT[i];
    if(gtot!=N_OSC){fprintf(stderr,"G_COUNT sum %d != %d\n",gtot,N_OSC);return 1;}

    /* — compose — */
    int key=48+ri(0,11);
    float bps=rr(112.f,126.f)/60.f;
    EvBuf*EB=(EvBuf*)calloc(1,sizeof(EvBuf));
    char report[512];
    composeSong(*EB,key,bps,report);
    printf("%s",report);
    printf("Events: %d\n",EB->n);

    /* — duration — */
    float endT=0.f;
    for(int i=0;i<EB->n;i++)endT=fmaxf(endT,EB->ev[i].tOff+4.f);
    endT=fminf(endT,MAX_SONG_S);
    int totalSamples=(int)(endT*(float)sr);
    printf("Duration: %.1f s (%d samples)\n",endT,totalSamples);
    printf("Oscillators: %d × %d samples = %.2fB sin evals\n\n",
           N_OSC,totalSamples,(double)N_OSC*totalSamples/1e9);

    /* — build oscillator table — */
    OscDesc*hOsc=(OscDesc*)calloc(N_OSC,sizeof(OscDesc));
    buildOscillators(hOsc,key);

    /* — build event index — */
    Event*sortedEv=(Event*)calloc(MAX_EVENTS,sizeof(Event));
    int evStart[N_GROUPS],evCount[N_GROUPS];
    indexEvents(*EB,evStart,evCount,sortedEv);

    /* — GPU allocations — */
    OscDesc*dOsc;Event*dEv;
    int*dEvStart,*dEvCount;
    float*dL,*dR;
    CUDA_CHECK(cudaMalloc(&dOsc,N_OSC*sizeof(OscDesc)));
    CUDA_CHECK(cudaMalloc(&dEv,MAX_EVENTS*sizeof(Event)));
    CUDA_CHECK(cudaMalloc(&dEvStart,N_GROUPS*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dEvCount,N_GROUPS*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dL,(size_t)totalSamples*4));
    CUDA_CHECK(cudaMalloc(&dR,(size_t)totalSamples*4));
    CUDA_CHECK(cudaMemcpy(dOsc,hOsc,N_OSC*sizeof(OscDesc),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEv,sortedEv,MAX_EVENTS*sizeof(Event),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEvStart,evStart,N_GROUPS*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEvCount,evCount,N_GROUPS*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dL,0,(size_t)totalSamples*4));
    CUDA_CHECK(cudaMemset(dR,0,(size_t)totalSamples*4));

    /* — launch — */
    int nBlocks=(N_OSC+BLOCK-1)/BLOCK;
    printf("Kernel: %d blocks × %d threads = %d oscillators\n",
           nBlocks,BLOCK,nBlocks*BLOCK);

    cudaEvent_t e0,e1;
    CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));

    synthKernel<<<nBlocks,BLOCK>>>(
        dOsc,dEv,dEvStart,dEvCount,
        dL,dR,totalSamples,(float)sr,seed);

    CUDA_CHECK(cudaEventRecord(e1));CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    printf("Kernel: %.0f ms  (%.0fx real-time)\n\n",ms,endT*1000.f/ms);

    /* — copy & post-process — */
    float*hL=(float*)malloc((size_t)totalSamples*4);
    float*hR=(float*)malloc((size_t)totalSamples*4);
    CUDA_CHECK(cudaMemcpy(hL,dL,(size_t)totalSamples*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hR,dR,(size_t)totalSamples*4,cudaMemcpyDeviceToHost));

    printf("Post: EQ + saturation...\n");
    applyEQ(hL,hR,totalSamples,(float)sr);
    softSat(hL,totalSamples,1.25f);
    softSat(hR,totalSamples,1.25f);
    writeWav(out,hL,hR,totalSamples,sr);

    free(hOsc);free(EB);free(sortedEv);free(hL);free(hR);
    cudaFree(dOsc);cudaFree(dEv);cudaFree(dEvStart);cudaFree(dEvCount);
    cudaFree(dL);cudaFree(dR);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 0;
}
