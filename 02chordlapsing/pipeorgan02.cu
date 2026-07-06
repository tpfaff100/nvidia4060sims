/*
 * pipeorgan01.cu — 10,000-oscillator GPU pipe organ
 * ==================================================
 * Target: RTX 4060 Ti (sm_89 / Ada Lovelace)
 *
 * BUILD (after vcvars64.bat):
 *   nvcc -arch=sm_89 -O3 -use_fast_math -o pipeorgan01 pipeorgan01.cu
 *
 * RUN:
 *   pipeorgan01 [out.wav] [sr] [seed] [length 1-10] [room 0-2] [tremulant 0-1]
 *
 *   pipeorgan01                            new piece every run
 *   pipeorgan01 out.wav 48000 0 5 1.0 0.6  default room, moderate tremulant
 *   pipeorgan01 out.wav 48000 42 7 1.5 0.8 reproduce seed 42, larger room
 *
 * PARAMETERS
 *   seed        0 = different every run (default), else reproducible
 *   length      1=~1:15   5=~3:00   10=~6:15
 *   room        0=dry chamber  1.0=cathedral  2.0=vast gothic  [1.0]
 *   tremulant   0=off  0.5=gentle  1.0=full church tremulant   [0.6]
 *
 * PIPE ORGAN PHYSICS — what makes this different from a sine synthesizer
 *
 *  PIPE FAMILIES (10 groups, 10,000 oscillators total):
 *
 *  PRINCIPAL chorus (2000 osc)
 *    The backbone of the organ.  Four ranks: 8' (unison), 4' (octave),
 *    2' (superoctave), 1' (larigot).  Each rank = 500 oscillators with
 *    per-pipe random detune ±2 cents (pipe scaling variation) and
 *    slightly different harmonic rolloff per pipe.
 *    Open flue pipe spectrum: all harmonics, amplitude ≈ 1/h^0.85,
 *    with pipe inharmonicity: f_n = n*f1*sqrt(1 + B*n^2), B≈0.00012
 *    (wall stiffness stretches upper partials slightly sharp).
 *
 *  FLUTE ranks (1200 osc)
 *    Bourdon 8' (stopped pipe, odd harmonics only — the "hooty" quality)
 *    + Flute 4' (open, very low harmonic count, fundamental dominant).
 *    Stopped pipe physically: closed end is a pressure antinode, so
 *    wavelength = 4L and only odd modes fit: 1st, 3rd, 5th...
 *
 *  STRING ranks (800 osc)
 *    Salicional 8': narrow-scaled pipes produce strong upper harmonics.
 *    Rich spectrum, amplitude rises toward middle harmonics before
 *    falling — the "stringy" brightness.  Very slight mouth cutup
 *    (narrow flue) modelled as reduced fundamental level.
 *
 *  CELESTE (800 osc)
 *    Voix celeste 8': two string ranks, one sharp by 6-10 cents.
 *    The beating between detuned pairs produces the shimmering celeste.
 *    Beat frequency = f1*(2^(cents/1200)-1) ≈ 1-2 Hz at 8' pitch.
 *    Each celeste oscillator is ring-modulated against its paired
 *    string oscillator for additional sideband shimmer.
 *
 *  REED ranks (1000 osc)
 *    Trumpet 8' + Oboe 8'.  Reed pipes have a shallot (metal tongue)
 *    that vibrates against the boot, producing a sawtooth-rich spectrum
 *    with a formant-like peak around 800-1200 Hz (boot resonance).
 *    Modelled as: harmonic amplitude * 1/h^0.5 * formant(h*f1),
 *    where formant is a Gaussian peak centered on the boot resonance.
 *    Oboe has a narrower, more nasal formant; trumpet has a broader one.
 *
 *  MIXTURE stops (1200 osc)
 *    Mixture = multiple pipes per key, each tuned to an upper partial.
 *    A four-rank mixture at 2⅔'+2'+1⅗'+1' gives the third, fourth,
 *    fifth, and eighth harmonic of the 8' fundamental.
 *    Creates the "brilliant" upper-work brightness, not audible as
 *    separate pitches but perceived as timbral richness.
 *    Tuned to historical temperament (Werckmeister III) so some keys
 *    have slightly different mixture colors.
 *
 *  PEDAL division (1200 osc)
 *    16' Open Diapason + 8' Octave.  16' pipes at 32 Hz (low C) have
 *    wavelengths of 10.7m — nearly infrasonic.  Modelled with very few
 *    harmonics (fundamental + 2nd + 3rd) but huge amplitude.
 *    Pipe speech is slow (50+ ms) due to large air column.
 *
 *  CHIFF / pipe speech (400 osc)
 *    The attack transient of flue pipes: a burst of broadband noise
 *    (the "chiff" or "tchh") before the tone stabilizes.  Duration
 *    and spectral content vary by pipe family and scaling.
 *    Modelled as fast-decaying bandpassed noise per group.
 *
 *  ROOM MODES / cathedral acoustics (800 osc)
 *    Actual room resonance modes of a cathedral-sized space.
 *    Each oscillator is one room mode with frequency from the
 *    rectangular room equation: f_nml = c/2 * sqrt((n/Lx)^2+(m/Ly)^2+(l/Lz)^2)
 *    with room dimensions randomised per run.  Each mode has its own
 *    slow amplitude envelope (the reverberant decay, RT60 3-5s).
 *
 *  WIND / mechanical noise (600 osc)
 *    Wind chest rumble, key action click, blower noise.
 *    Adds the physical presence of a real instrument — the organ
 *    "breathes" even when no notes are playing.  Also models the
 *    wind tremulant: a low-frequency flutter that modulates all
 *    speaking pipes simultaneously in pitch and amplitude.
 *
 *  TREMULANT
 *    Unlike a simple LFO, the organ tremulant modulates wind pressure,
 *    affecting pitch AND amplitude of every speaking pipe.  Each pipe
 *    gets a slightly different tremulant phase (they don't all flutter
 *    in perfect unison) producing the characteristic "wobble" of a
 *    well-regulated church organ.  Depth scales with the tremulant param.
 *
 *  INHARMONICITY
 *    Real pipes are not perfectly cylindrical: wall stiffness causes
 *    upper harmonics to be slightly sharp of their ideal integer ratios.
 *    B-factor (stiffness) varies by pipe family:
 *      Principal (metal, thick wall): B ≈ 0.00012
 *      Flute (wood, wide bore):       B ≈ 0.00008
 *      String (narrow, metal):        B ≈ 0.00020
 *    f_n = n*f1*sqrt(1 + B*n^2)
 *
 * COMPOSER
 *    Reuses the Markov-chain harmony + motif development from the disco
 *    engine, adapted for four-part organ texture:
 *    - Manual I (right hand): principal / reed melody
 *    - Manual II (left hand): flute / string accompaniment
 *    - Pedal: 16' bass line
 *    - Mixture: doubles melody at upper partials in strong sections
 *    Form: Prelude → Chorale → Development → Recapitulation → Postlude
 *
 * GPU ARCHITECTURE (identical to disco10k.cu)
 *    Oscillator-per-thread.  10,000 threads, each owns one oscillator
 *    for the whole piece, phase in register, loops all samples.
 *    Warp-level __shfl_down_sync reduction → one atomicAdd per warp.
 *    Padded to PAD_OSC (multiple of BLOCK=256) so no thread exits early
 *    before the full-mask shuffle.  Chunked launches for TDR safety.
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
#include <algorithm>

#define SR          48000
#define N_OSC       10000
#define MAX_EVENTS  12288
#define MAX_SONG_S  420.0f      /* 7 min max                           */
#define TWO_PI      6.28318530717958647692f
#define BLOCK       256
#define PAD_OSC     (((N_OSC+BLOCK-1)/BLOCK)*BLOCK)
#define CHUNK       131072      /* samples per kernel launch (TDR safe) */

#define CUDA_CHECK(x) do{ cudaError_t _=x; if(_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_)); \
    exit(1);} }while(0)

/* ══════════════════════════════════════════════════════════════════════
   OSCILLATOR DESCRIPTOR
   All pipe physics baked in at build time by the CPU.
   ══════════════════════════════════════════════════════════════════════ */
struct OscDesc {
    /* Pitch */
    float freqRatio;    /* ratio to noteHz: 1.0=unison, 2.0=octave, etc. */
    float detuneCents;  /* per-pipe random tuning variation              */
    float inharmonB;    /* stiffness coefficient B for f_n stretch        */
    int   harmonic;     /* which harmonic of the pipe fundamental (1-16)  */
    /* Amplitude */
    float amp;          /* peak amplitude                                 */
    float panL, panR;
    float phase0;       /* initial phase (0..1), randomised per pipe      */
    /* Pipe speech (chiff) */
    float chiffDur;     /* duration of attack noise burst (s), 0=none    */
    float chiffBandHz;  /* centre frequency of attack noise               */
    /* Envelope: ADSR */
    float aA, aD, aS, aR;
    /* Tremulant coupling */
    float tremPitchDepth;  /* cents peak deviation                        */
    float tremAmpDepth;    /* fractional amplitude modulation             */
    float tremPhase;       /* oscillator's phase in tremulant cycle       */
    /* Wind noise coupling (for wind group) */
    float windAmp;      /* how much this osc contributes to wind noise   */
    /* Pipe family flags */
    int   group;
    int   stoppedPipe;  /* 1 = odd harmonics only (bourdon)              */
    /* Room mode decay (room group only) */
    float roomDecay;    /* 1/e decay time in seconds                     */
};

/* ══════════════════════════════════════════════════════════════════════
   EVENT — one note/chord for one organ group
   ══════════════════════════════════════════════════════════════════════ */
struct Event {
    float tOn, tOff, vel, noteHz, hpf;
    int   group;
};

/* ══════════════════════════════════════════════════════════════════════
   ORGAN GROUPS
   ══════════════════════════════════════════════════════════════════════ */
enum GID {
    G_PRINCIPAL=0, G_FLUTE, G_STRING, G_CELESTE,
    G_REED, G_MIXTURE, G_PEDAL, G_CHIFF,
    G_ROOM, G_WIND,
    N_GROUPS
};
/* Must sum to N_OSC = 10000 */
static const int G_CNT[N_GROUPS] = {
    2000,   /* PRINCIPAL  */
    1200,   /* FLUTE      */
     800,   /* STRING     */
     800,   /* CELESTE    */
    1000,   /* REED       */
    1200,   /* MIXTURE    */
    1200,   /* PEDAL      */
     400,   /* CHIFF      */
     800,   /* ROOM       */
     600,   /* WIND       */
};

/* ══════════════════════════════════════════════════════════════════════
   GPU SYNTHESIS KERNEL
   ══════════════════════════════════════════════════════════════════════ */
__global__ void organKernel(
    const OscDesc *__restrict__ D,
    const Event   *__restrict__ EV,
    const int     *__restrict__ evStart,
    const int     *__restrict__ evCount,
    float         *__restrict__ phBuf,
    int           *__restrict__ eiBuf,
    float         *__restrict__ bufL,
    float         *__restrict__ bufR,
    int s0, int s1, float sr,
    float tremRate,          /* tremulant Hz                            */
    float tremDepth)         /* 0..1 depth scale                        */
{
    int oi = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    const OscDesc o = D[oi];
    float invSr = 1.f/sr;
    int eg = o.group;
    int eS = evStart[eg];
    int eEnd = eS + evCount[eg];

    float phase   = phBuf[oi];
    int   ei      = eiBuf[oi];
    if(ei < eS) ei = eS;

    /* chiff state: decaying noise envelope */
    float chiffEnv = 0.f;

    for(int si = s0; si < s1; si++){
        float t = si * invSr;

        /* monotonic event pointer */
        while(ei+1 < eEnd && t >= EV[ei+1].tOn) ei++;

        bool gated = false;
        float noteHz = o.freqRatio;
        float vel=0.f, tRel=0.f, dur=0.f;
        if(eS < eEnd && t >= EV[ei].tOn && t < EV[ei].tOff + o.aR + 0.1f){
            /* Actual pipe frequency: apply inharmonicity stretch
               f_n = n*f1*sqrt(1 + B*n^2), where f1 = noteHz*freqRatio/harmonic
               freqRatio already encodes n*f1/f1, so:                  */
            float f1base = EV[ei].noteHz * (o.freqRatio / (float)o.harmonic);
            float fn = f1base * (float)o.harmonic
                     * __fsqrt_rn(1.f + o.inharmonB*(float)(o.harmonic*o.harmonic));
            /* add per-pipe detune */
            fn *= exp2f(o.detuneCents / 1200.f);
            noteHz = fn;
            vel    = EV[ei].vel;
            tRel   = t - EV[ei].tOn;
            dur    = EV[ei].tOff - EV[ei].tOn;
            gated  = true;
            /* reset chiff on new note */
            if(tRel < invSr*2.f) chiffEnv = o.chiffDur > 0.f ? 1.f : 0.f;
        }

        /* ── ADSR envelope ──────────────────────────────────────────── */
        float env = 0.f;
        if(gated){
            if     (tRel < o.aA)       env = tRel / fmaxf(o.aA, 1e-5f);
            else if(tRel < o.aA+o.aD)  env = 1.f-(tRel-o.aA)/fmaxf(o.aD,1e-5f)*(1.f-o.aS);
            else if(tRel < dur)         env = o.aS;
            else { float tr=tRel-dur;
                   env=(tr<o.aR)?o.aS*(1.f-tr/fmaxf(o.aR,1e-5f)):0.f; }
            env = fmaxf(0.f, fminf(1.f, env));
        }

        /* ── TREMULANT ──────────────────────────────────────────────── */
        /* Organ tremulant: sinusoidal wind pressure fluctuation
           modulates both pitch (via reed tongue / flue slot width)
           and amplitude (via wind pressure).  Each pipe has its own
           phase so they don't all flutter in unison.                   */
        float tremPh = tremRate*t + o.tremPhase;
        tremPh -= floorf(tremPh);
        float tremSin = __sinf(TWO_PI*tremPh);
        float pitchMod = 1.f + tremDepth*(o.tremPitchDepth/1200.f)*tremSin;
        float ampMod   = 1.f + tremDepth*o.tremAmpDepth*tremSin*0.7f;

        /* ── advance phase ──────────────────────────────────────────── */
        float f = noteHz * pitchMod;
        phase += f * invSr;
        phase -= floorf(phase);

        float sample = 0.f;
        if(gated && env > 1e-6f && f > 1.f && f < 0.45f*sr){

        if(o.group == G_WIND && o.windAmp > 0.f){
            /* Wind chest: low-frequency rumble only, NOT white noise.
               A real organ's wind noise is almost entirely below 80 Hz
               — the blower motor fundamental and its harmonics, plus
               wind chest resonance.  Model as a sine at the oscillator's
               own frequency (already set to 20-180 Hz in the builder)
               with slow random amplitude variation (hash AM).
               This produces organ "breath" without audible hiss.       */
            float p = phase; p -= floorf(p);
            float windSine = __sinf(TWO_PI * p);
            /* slow AM: hash changes only every ~256 samples for smoothness */
            unsigned hAM = ((unsigned)(si>>8)) * 0x9E3779B9u
                         ^ (unsigned)oi * 2654435761u;
            hAM ^= hAM>>13; hAM *= 0xC2B2AE35u; hAM ^= hAM>>16;
            float amDepth = (float)(hAM & 0xFFFF) / 65536.f; /* 0-1   */
            sample = windSine * (0.4f + 0.6f * amDepth) * o.windAmp * env;

            } else if(o.group == G_ROOM){
                /* Room modes: pure sine at modal frequency with
                   independent slow decay (convolution-reverb-like)     */
                float p = phase; p -= floorf(p);
                float roomSin = __sinf(TWO_PI*p);
                /* room modes have their own long decay regardless of note*/
                float roomEnv = gated ? env : 0.f;
                sample = roomSin * roomEnv * o.amp * ampMod;

            } else {
                /* ── Pipe tone: sine at this harmonic's frequency ───── */
                float p = phase; p -= floorf(p);
                float tone = __sinf(TWO_PI*p);

                /* ── Chiff (pipe speech noise burst on attack) ──────── */
                float chiff = 0.f;
                if(o.chiffDur > 0.f && chiffEnv > 0.f){
                    unsigned hh = (unsigned)si*0xC2B2AE35u^(unsigned)oi;
                    hh^=hh>>13;hh*=0x9E3779B9u;hh^=hh>>17;
                    float nz = (float)(int)hh*4.6566129e-10f;
                    /* bandpass around chiffBandHz: two one-pole filters */
                    /* (stateless approximation: shaped by harmonic)     */
                    float band = __sinf(TWO_PI*o.chiffBandHz*t);
                    chiff = nz * band * chiffEnv * 0.35f;
                    chiffEnv *= (1.f - invSr/fmaxf(o.chiffDur, 1e-5f));
                    if(chiffEnv < 0.f) chiffEnv = 0.f;
                }

                sample = (tone + chiff) * env * vel * o.amp * ampMod;
            }
        }

        /* ── warp-level reduction (full mask: all threads always here) */
        float sL = sample * o.panL;
        float sR = sample * o.panR;
        #pragma unroll
        for(int off=16; off>=1; off>>=1){
            sL += __shfl_down_sync(0xFFFFFFFF, sL, off);
            sR += __shfl_down_sync(0xFFFFFFFF, sR, off);
        }
        if((threadIdx.x & 31)==0 && (sL!=0.f||sR!=0.f)){
            atomicAdd(&bufL[si], sL);
            atomicAdd(&bufR[si], sR);
        }
    }
    phBuf[oi] = phase;
    eiBuf[oi] = ei;
}

/* ══════════════════════════════════════════════════════════════════════
   CPU COMPOSER  (Markov harmony + motif development, adapted for organ)
   ══════════════════════════════════════════════════════════════════════ */
static unsigned G_RNG = 1;
static void  sr_seed(unsigned s){ G_RNG=(s!=0)?s:(unsigned)time(NULL); }
static float rnd(){ G_RNG^=G_RNG<<13;G_RNG^=G_RNG>>17;G_RNG^=G_RNG<<5;
                    return (float)(G_RNG&0xFFFFFF)/16777216.f; }
static float rr(float a,float b){ return a+(b-a)*rnd(); }
static int   ri(int a,int b){
    return a+(int)(rnd()*(float)(b-a+1)*0.9999f); }
static int   pick(const float*w,int n){
    float s=0;for(int i=0;i<n;i++)s+=w[i];
    float x=rnd()*s;
    for(int i=0;i<n;i++){x-=w[i];if(x<=0)return i;}
    return n-1;
}

static const int MINOR[7]={0,2,3,5,7,8,10};
struct Chord{ int deg,type; };
static const int CH_ROOT[6]={0,3,4,5,6,2};
static const int CH_TYPE[6]={0,0,2,1,1,1};
static const float MARKOV[6][6]={
    {0.12f,0.20f,0.18f,0.24f,0.18f,0.08f},
    {0.32f,0.06f,0.28f,0.12f,0.14f,0.08f},
    {0.58f,0.08f,0.04f,0.18f,0.08f,0.04f},
    {0.18f,0.14f,0.16f,0.06f,0.38f,0.08f},
    {0.50f,0.08f,0.14f,0.14f,0.06f,0.08f},
    {0.18f,0.18f,0.16f,0.28f,0.12f,0.08f},
};

static float midiHz(int m){ return 440.f*powf(2.f,(m-69)/12.f); }
static void chordTones(int key,Chord c,int oct,int*out){
    int r=key+MINOR[c.deg]+12*oct;
    out[0]=r;out[1]=r+((c.type==0)?3:4);
    out[2]=r+7;out[3]=(c.type==2)?r+10:r+12;
}
static int snapChord(int key,Chord c,int midi,int lo,int hi){
    int t[4];chordTones(key,c,0,t);
    int best=midi,bd=99;
    for(int o=-24;o<=36;o+=12)for(int i=0;i<4;i++){
        int cand=t[i]+o;
        if(cand<lo||cand>hi)continue;
        int d=abs(midi-cand);if(d<bd){bd=d;best=cand;}
    }return best;
}
static int snapScale(int key,int midi,int lo,int hi){
    int best=midi,bd=99;
    for(int o=-24;o<=48;o+=12)for(int i=0;i<7;i++){
        int c=key+MINOR[i]+o;
        if(c<lo||c>hi)continue;
        int d=abs(midi-c);if(d<bd){bd=d;best=c;}
    }return best;
}
static void makeProg(Chord*prog,int start){
    int cur=start;
    for(int i=0;i<4;i++){
        prog[i]={CH_ROOT[cur],CH_TYPE[cur]};
        if(i==2){float w[6];memcpy(w,MARKOV[cur],24);
                 w[2]*=3.f;w[4]*=2.f;cur=pick(w,6);}
        else cur=pick(MARKOV[cur],6);
    }
}

/* ══════════════════════════════════════════════════════════════════════
   EVENT BUFFER
   ══════════════════════════════════════════════════════════════════════ */
struct EvBuf{
    Event ev[MAX_EVENTS]; int n;
    void add(float on,float off,float vel,float hz,int g,float hpf=0.f){
        if(n>=MAX_EVENTS)return;
        ev[n++]={on,off,vel,hz,hpf,g};
    }
};

/* ══════════════════════════════════════════════════════════════════════
   SONG ASSEMBLY  — Disco Organ composer
   Disco sensibility through a pipe organ:
   • Pedal = four-on-the-floor kick (16' diapason), heavy low register
   • Manual II (low) = held minor/dom7 chords on flute+string+celeste
   • Manual I (high) = syncopated reed+principal melody in upper octaves
   • Mixture only on climax bars for brilliance
   • Disco progressions: Am-G-F-E7, Dm-Am-Bb-A7, etc.
   • Church reverb post-process (Schroeder, long RT60)
   ══════════════════════════════════════════════════════════════════════ */

/* ── Disco bass groove on pedal: 4-on-floor + octave pumps ─────────── */
static void discoPedal(EvBuf&E,float bb,float beat,int root,int nextRoot,
                       float vel,float E2)
{
    /* 16' + 8' always — stress the low register                        */
    /* beat 1: long root                                                 */
    E.add(bb,          bb+beat*0.88f,vel,       midiHz(root),  G_PEDAL);
    /* beat 1-and: octave pump (disco signature)                         */
    E.add(bb+beat*0.5f,bb+beat*0.92f,vel*0.82f, midiHz(root+12),G_PEDAL);
    /* beat 2: fifth or approach                                         */
    int fifth=root+7;
    while(fifth>43)fifth-=12;
    E.add(bb+beat,     bb+beat*1.88f,vel*0.90f, midiHz(fifth), G_PEDAL);
    /* beat 3: root again                                                */
    E.add(bb+beat*2,   bb+beat*2.88f,vel,        midiHz(root),  G_PEDAL);
    /* beat 3-and: octave pump                                           */
    E.add(bb+beat*2.5f,bb+beat*2.92f,vel*0.78f, midiHz(root+12),G_PEDAL);
    /* beat 4: chromatic walk to next root                               */
    int walk=(nextRoot>root)?root+1:root-1;
    while(walk>43)walk-=12;
    while(walk<24)walk+=12;
    E.add(bb+beat*3,   bb+beat*3.88f,vel*0.88f, midiHz(walk),  G_PEDAL);
    /* sub-bass reinforcement: 16' octave below for depth               */
    int subRoot=root-12;
    while(subRoot<12)subRoot+=12;
    E.add(bb,          bb+beat*1.8f, vel*0.70f,  midiHz(subRoot),G_PEDAL);
    (void)E2;
}

/* ── Disco chord bed: voiced very low for church room excitation ─────
   Cathedral rooms bloom most below 200 Hz — C2 (65 Hz) and below.
   The chord is voiced in octaves 0-1 (C1-C2 region) so the room modes
   are strongly excited and ring for several seconds after the note.   */
static void discoChordLow(EvBuf&E,float tOn,float tOff,
                          int key,Chord c,float vel)
{
    int ct[4]; chordTones(key,c,0,ct);
    /* root voicing: C1-C2 range (midi 24-36) — maximum room excitation */
    for(int i=0;i<4;i++){
        int midi=ct[i];
        while(midi>40)midi-=12;      /* pull into C1-C2 band           */
        while(midi<24)midi+=12;
        E.add(tOn,tOff,vel,          midiHz(midi),   G_FLUTE);
        E.add(tOn,tOff,vel*0.65f,    midiHz(midi),   G_STRING);
        E.add(tOn,tOff,vel*0.45f,    midiHz(midi),   G_CELESTE);
        /* double at the octave above for presence */
        E.add(tOn,tOff,vel*0.55f,    midiHz(midi+12),G_FLUTE);
        E.add(tOn,tOff,vel*0.40f,    midiHz(midi+12),G_STRING);
    }
    /* heavy room feed on the root — excites the longest-ringing modes */
    E.add(tOn,tOff+1.5f,vel*0.90f,  midiHz(ct[0]-12<12?ct[0]:ct[0]-12),G_ROOM);
}

/* ── Syncopated melody on Manual I (high: principal+reed) ───────────── */
/* Disco rhythm cells — off-beat entries */
struct DCell{int n;float t[8],d[8];};
static const DCell DISCO_RHY[]={
    {6,{0.5f,1,1.5f,2.5f,3,3.5f},{0.45f,0.45f,0.9f,0.45f,0.45f,0.9f}},
    {5,{0,0.5f,1.5f,2,3},        {0.45f,0.9f,0.45f,0.9f,0.9f}},
    {7,{0,0.5f,1,1.5f,2,2.5f,3}, {0.4f,0.4f,0.4f,0.4f,0.4f,0.4f,1.8f}},
    {5,{0.5f,1.5f,2,2.5f,3.5f},  {0.9f,0.45f,0.45f,0.9f,0.45f}},
    {4,{0,1,2,3},                 {0.9f,1.8f,0.9f,1.8f}},        /* slower*/
};
#define NDISCO_RHY 5

static void discoMelody(EvBuf&E,float bb,float beat,
                        int key,Chord ch,int melBase,
                        const int*motifDeg,const float*motifT,
                        const float*motifD,int motifN,
                        int dev,float vel,bool withMixture)
{
    float shift=(dev==4)?beat*0.5f:0.f;
    for(int i=0;i<motifN;i++){
        int d=motifDeg[i];
        if(dev==1)d+=1;
        if(dev==2)d-=1;
        if(dev==3)d=-d+2;
        int midi=melBase+((d>=0)?MINOR[d%7]+12*(d/7)
                                :MINOR[(d%7+7)%7]-12*((-d+6)/7));
        bool strong=(fmodf(motifT[i],1.f)<0.01f);
        if(strong) midi=snapChord(key,ch,midi,melBase-3,melBase+20);
        else       midi=snapScale(key,midi,melBase-3,melBase+20);
        float ton=bb+motifT[i]*beat+shift;
        float toff=ton+motifD[i]*beat;
        if(ton<0.f)ton=0.f;
        if(toff>ton+0.01f){
            /* LOW-MID register: C3-C4 for cathedral bloom (midi 48-64) */
            while(midi<48)midi+=12;
            while(midi>64)midi-=12;
            /* primary colour: flute — blooms richly in stone acoustics */
            E.add(ton,toff,vel*rr(0.88f,1.f), midiHz(midi),  G_FLUTE);
            E.add(ton,toff,vel*0.60f,          midiHz(midi),  G_STRING);
            /* reed/principal only on climax sections for brightness    */
            if(withMixture){
                E.add(ton,toff,vel*0.55f,      midiHz(midi),  G_REED);
                E.add(ton,toff,vel*0.40f,      midiHz(midi+12),G_PRINCIPAL);
            }
        }
    }
}

/* ── Counter-melody: deep low register on bourdon flute ─────────────── */
static void discoCounter(EvBuf&E,float bb,float beat,
                         int key,Chord ch,
                         const int*motifDeg,const float*motifT,
                         const float*motifD,int motifN,float vel)
{
    for(int i=0;i<motifN;i++){
        int d=-motifDeg[i]+2;
        int midi=key+MINOR[((d%7)+7)%7]+36;   /* C2 area — deep bourdon */
        midi=snapChord(key,ch,midi,24,48);
        float ton=bb+motifT[i]*beat;
        float toff=ton+motifD[i]*beat;
        if(ton<0.f)ton=0.f;
        if(toff>ton+0.01f){
            E.add(ton,toff,vel*0.70f,midiHz(midi),  G_FLUTE);  /* bourdon 8' */
            E.add(ton,toff,vel*0.45f,midiHz(midi+12),G_PEDAL); /* octave up  */
        }
    }
}

static void composeSong(EvBuf&E,int key,float beat,int length,char*report)
{
    if(length<1)length=1;
    if(length>10)length=10;
    float sc=0.4f+(length-1)*(1.6f/9.f);
    auto bars=[&](int n)->int{return std::max(2,(int)roundf(n*sc));};
    auto rbars=[&](int lo,int hi)->int{
        int slo=std::max(2,(int)roundf(lo*sc));
        int shi=std::max(slo,(int)roundf(hi*sc));
        return ri(slo,shi);
    };

    /* Disco form with organ character                                  */
    struct Sec{const char*name;int bars;float en;};
    Sec form[10];int nf=0;
    form[nf++]={"INTRO",    rbars(4,8),  0.35f};  /* pads swell in    */
    form[nf++]={"VERSE",    bars(16),    0.62f};
    form[nf++]={"PRE",      bars(8),     0.78f};
    form[nf++]={"CHORUS",   bars(16),    1.00f};
    form[nf++]={"BREAK",    bars(8),     0.40f};  /* low only: pedal+flute */
    if(rnd()<0.6f)form[nf++]={"VERSE2", bars(8),  0.68f};
    form[nf++]={"CHORUS2",  bars(16),    1.00f};
    if(length>=7)form[nf++]={"BIG_CLIMAX",bars(8),1.00f};
    form[nf++]={"OUTRO",    rbars(6,10), 0.38f};

    /* Disco progressions — picked randomly per run */
    struct Prog{const char*name;int semis[4];int type[4];};
    static const Prog DISCO_PROGS[]={
        {"Donna Summer",  {0,-2,-4,-5}, {0,1,1,2}},  /* Am G F E7     */
        {"Moroder",       {3, 0, 1, 0}, {0,0,1,2}},  /* Dm Am Bb A7   */
        {"Philly",        {0,-4,-2, 3}, {0,1,1,0}},  /* Am F G Dm     */
        {"Eurodisco",     {0, 3,-2,-4}, {0,0,1,1}},  /* Am Dm G F     */
    };
    int pi=ri(0,3);
    Chord vProg[4],cProg[4];
    /* build from the selected disco progression                        */
    for(int i=0;i<4;i++){
        /* find which CH_ROOT matches semis offset, or use direct       */
        vProg[i]={0,DISCO_PROGS[pi].type[i]};
        /* encode semitone offset as a degree (approximate)            */
        int sem=DISCO_PROGS[pi].semis[i];
        /* find closest MINOR degree                                    */
        int bestDeg=0,bestDist=99;
        for(int j=0;j<7;j++){
            int d=abs(((MINOR[j]+sem)%12+12)%12);
            if(d<bestDist){bestDist=d;bestDeg=j;}
        }
        /* store raw semitone in deg field for use in chordTones       */
        vProg[i].deg=bestDeg;
        cProg[i]=vProg[i];  /* chorus uses same prog, different energy */
    }
    /* chorus: shift up one degree for lift                            */
    makeProg(cProg,(ri(0,2)));

    /* Motifs: use disco rhythm cells                                   */
    int vMotifDeg[8];float vMotifT[8],vMotifD[8];int vMotifN=0;
    int cMotifDeg[8];float cMotifT[8],cMotifD[8];int cMotifN=0;
    /* build from disco rhythm cell */
    const DCell&vCell=DISCO_RHY[ri(0,NDISCO_RHY-1)];
    const DCell&cCell=DISCO_RHY[ri(0,NDISCO_RHY-1)];
    vMotifN=vCell.n;cMotifN=cCell.n;
    int curDeg=ri(0,4);
    for(int i=0;i<vMotifN;i++){
        vMotifT[i]=vCell.t[i];vMotifD[i]=vCell.d[i];vMotifDeg[i]=curDeg;
        curDeg+=((rnd()<0.5f)?1:-1);
        if(curDeg>5)curDeg=4;
        if(curDeg<0)curDeg=0;
    }
    curDeg=ri(2,5);  /* chorus starts higher                           */
    for(int i=0;i<cMotifN;i++){
        cMotifT[i]=cCell.t[i];cMotifD[i]=cCell.d[i];cMotifDeg[i]=curDeg;
        curDeg+=((rnd()<0.5f)?1:-1);
        if(curDeg>6)curDeg=5;
        if(curDeg<1)curDeg=1;
    }

    int pedalNote=key;
    while(pedalNote>43)pedalNote-=12;
    if(pedalNote<24)pedalNote+=12;

    /* MELODY register: C3-C4 (midi 48-64) — where cathedral rooms bloom.
       The G_ROOM oscillators are tuned to actual room modes in this
       frequency band; notes here ring for seconds after the key lifts. */
    int melBase=key+36;
    if(melBase<48)melBase+=12;
    if(melBase>55)melBase-=12;

    float bar=0.f;
    for(int s=0;s<nf;s++){
        bool isChorus=(strncmp(form[s].name,"CHORUS",6)==0||
                       strncmp(form[s].name,"BIG_C",5)==0);
        bool isBreak =(strncmp(form[s].name,"BREAK",5)==0);
        bool isIntro =(strncmp(form[s].name,"INTRO",5)==0);
        bool isOutro =(strncmp(form[s].name,"OUTRO",5)==0);
        bool isPre   =(strncmp(form[s].name,"PRE",3)==0);
        Chord*prog=isChorus?cProg:vProg;

        for(int b=0;b<form[s].bars;b++){
            float bb=(bar+b)*4.f*beat;
            float be=bb+4.f*beat;
            Chord ch=prog[b%4],nx=prog[(b+1)%4];
            float E2=form[s].en;
            if(isOutro)E2*=1.f-(float)b/form[s].bars;

            /* ── PEDAL: four-on-the-floor disco bass ──────────────── */
            if(!isIntro||b>=form[s].bars/2){
                int ct2[4];chordTones(key,ch,0,ct2);
                int root=ct2[0];while(root>43)root-=12;if(root<24)root+=12;
                int nct2[4];chordTones(key,nx,0,nct2);
                int nroot=nct2[0];while(nroot>43)nroot-=12;if(nroot<24)nroot+=12;
                float pvel=0.88f+0.12f*E2;   /* pedal always loud      */
                if(!isBreak){
                    discoPedal(E,bb,beat,root,nroot,pvel,E2);
                    /* extra sub-bass room feed locked to pedal pitch —
                       excites the longest-ringing cathedral modes      */
                    int subPitch=root-12; while(subPitch<12)subPitch+=12;
                    E.add(bb,be+2.5f,0.95f,midiHz(subPitch),   G_ROOM);
                    E.add(bb,be+2.5f,0.75f,midiHz(subPitch+7), G_ROOM);
                    E.add(bb,be+2.5f,0.60f,midiHz(subPitch+12),G_ROOM);
                }else{
                    /* break: sparse half-note pedal only               */
                    E.add(bb,      bb+beat*1.9f,pvel,midiHz(root),G_PEDAL);
                    E.add(bb+beat*2,be-0.05f,   pvel*0.88f,midiHz(root),G_PEDAL);
                    E.add(bb,be+1.5f,0.70f,midiHz(root-12<12?root:root-12),G_ROOM);
                }
                pedalNote=nroot;
            }

            /* ── LOW MANUAL: chord bed (stressed low register) ─────── */
            if(!isBreak){
                float cv=0.45f+0.42f*E2;
                discoChordLow(E,bb,be-0.02f,key,ch,cv);
            }else{
                /* break: just flute sustain                            */
                int ct3[4];chordTones(key,ch,0,ct3);
                for(int v=0;v<2;v++)
                    E.add(bb,be-0.02f,0.45f*E2,midiHz(ct3[v]),G_FLUTE);
            }

            /* ── ROOM: heavy feed — excite cathedral modes strongly ─── */
            int ct4[4];chordTones(key,ch,0,ct4);
            /* feed the lowest available root to maximise room bloom    */
            int deepRoot=ct4[0]; while(deepRoot>36)deepRoot-=12;
            if(deepRoot<12)deepRoot+=12;
            E.add(bb,be+2.0f,E2*0.80f,midiHz(deepRoot),   G_ROOM);
            E.add(bb,be+2.0f,E2*0.60f,midiHz(deepRoot+7), G_ROOM);
            E.add(bb,be+2.0f,E2*0.40f,midiHz(deepRoot+12),G_ROOM);
            E.add(bb,be+1.0f,0.12f,1.f,               G_WIND);

            /* ── HIGH MELODY: syncopated disco on principal+reed ───── */
            if(!isIntro||b>=form[s].bars/2){
                if(b%2==0&&!isBreak){
                    int*md=isChorus?cMotifDeg:vMotifDeg;
                    float*mt=isChorus?cMotifT:vMotifT;
                    float*mdu=isChorus?cMotifD:vMotifD;
                    int mn=isChorus?cMotifN:vMotifN;
                    int dev=0;int ph=(b/2)%4;
                    if(ph==1)dev=(rnd()<0.5f)?1:4;
                    if(ph==2)dev=(rnd()<0.5f)?3:2;
                    if(ph==3)dev=4;
                    int reg=melBase;
                    if(isChorus)reg+=5;      /* chorus: up a fourth, still low */
                    if(isPre)   reg+=b/4;    /* gentle crescendo upward         */
                    float mv=0.58f+0.40f*E2;
                    bool withMix=isChorus||
                                 strncmp(form[s].name,"BIG_C",5)==0;
                    discoMelody(E,bb,beat,key,ch,reg,md,mt,mdu,mn,
                                dev,mv,withMix);
                    /* counter-melody in low register on chorus         */
                    if(isChorus&&b%4==0)
                        discoCounter(E,bb,beat,key,ch,md,mt,mdu,mn,mv*0.55f);
                }
            }
        }
        bar+=form[s].bars;
    }

    /* Final cadence: low voicing — let the church ring for 14 seconds  */
    float endB=bar*4.f*beat;
    int ct[4];Chord tonic={0,0};chordTones(key,tonic,0,ct);
    /* chord bed: C1-C2 band for maximum room excitation               */
    for(int v=0;v<4;v++){
        int lo=ct[v]; while(lo>36)lo-=12; while(lo<24)lo+=12;
        E.add(endB,endB+14.f,0.92f,midiHz(lo),   G_FLUTE);
        E.add(endB,endB+14.f,0.72f,midiHz(lo),   G_STRING);
        E.add(endB,endB+14.f,0.65f,midiHz(lo+12),G_FLUTE);
    }
    /* pedal: root at 16' and 8'                                        */
    int deepPedal=ct[0]; while(deepPedal>28)deepPedal-=12;
    E.add(endB,endB+14.f,0.98f,midiHz(deepPedal),   G_PEDAL);
    E.add(endB,endB+14.f,0.90f,midiHz(deepPedal+12),G_PEDAL);
    /* room: saturate the modal space for a cathedral-length decay      */
    E.add(endB,endB+16.f,1.00f,midiHz(deepPedal),   G_ROOM);
    E.add(endB,endB+16.f,0.85f,midiHz(deepPedal+7), G_ROOM);
    E.add(endB,endB+16.f,0.70f,midiHz(deepPedal+12),G_ROOM);
    E.add(endB,endB+16.f,0.12f,1.f,                 G_WIND);

    static const char*NM[12]={"C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"};
    static const char*RN[7]={"i","ii","III","iv","v","VI","VII"};
    char*p=report;char*re=report+512;
    p+=snprintf(p,(size_t)(re-p),
        "Style: Disco Organ  Prog: %s  Key: %s minor  Length: %d/10\nForm: ",
        DISCO_PROGS[pi].name,NM[key%12],length);
    for(int s=0;s<nf;s++)
        p+=snprintf(p,(size_t)(re-p),"%s(%d) ",form[s].name,form[s].bars);
    p+=snprintf(p,(size_t)(re-p),"\nVerse: ");
    for(int i=0;i<4;i++)p+=snprintf(p,(size_t)(re-p),"%s ",RN[vProg[i].deg]);
    p+=snprintf(p,(size_t)(re-p),"\nChorus: ");
    for(int i=0;i<4;i++)p+=snprintf(p,(size_t)(re-p),"%s ",RN[cProg[i].deg]);
    p+=snprintf(p,(size_t)(re-p),"\n");
}

/* ══════════════════════════════════════════════════════════════════════
   OSCILLATOR TABLE BUILDER  — all pipe physics baked here
   ══════════════════════════════════════════════════════════════════════ */
static void buildOscillators(OscDesc*D, float roomScale)
{
    int idx=0;
    auto pan=[](float a,float&L,float&R){
        float x=(a+1.f)*0.25f*(float)M_PI;
        L=cosf(x);R=sinf(x);
    };

    /* ── PRINCIPAL chorus (2000 osc) ────────────────────────────────
       Four ranks: 8'(×1), 4'(×2), 2'(×4), 1'(×8) pitch ratios.
       Each rank: 125 voices × 4 harmonics = 500 osc/rank × 4 = 2000.
       Open metal flue pipe: all harmonics, 1/h^0.85 rolloff.
       Inharmonicity B=0.00012 (thin metal wall).                      */
    {
        float rankRatios[4]={1.f,2.f,4.f,8.f};
        float rankVols[4]  ={1.f,0.72f,0.55f,0.38f};
        for(int rank=0;rank<4;rank++){
            for(int v=0;v<125;v++){
                float det=rr(-2.0f,2.0f);      /* pipe-to-pipe variation*/
                float vPan=rr(-0.75f,0.75f);
                for(int h=1;h<=4;h++){
                    OscDesc&d=D[idx++];
                    d.freqRatio  =rankRatios[rank]*(float)h;
                    d.harmonic   =h;
                    d.detuneCents=det;
                    d.inharmonB  =0.00012f*(1.f+rr(-0.2f,0.2f));
                    d.amp        =rankVols[rank]*powf(1.f/(float)h,0.85f)*0.009f
                                  *(0.85f+0.30f*rnd());
                    d.phase0     =rnd();
                    pan(vPan,d.panL,d.panR);
                    d.aA=0.018f;d.aD=0.05f;d.aS=0.98f;d.aR=0.12f;
                    /* pipe speech: short chiff on attack                */
                    d.chiffDur   =(h==1)?rr(0.010f,0.018f):0.f;
                    d.chiffBandHz=(h==1)?rr(800.f,1400.f):0.f;
                    /* tremulant: principal responds moderately          */
                    d.tremPitchDepth=rr(3.f,6.f);
                    d.tremAmpDepth  =rr(0.04f,0.09f);
                    d.tremPhase     =rnd();
                    d.windAmp=0.f;d.stoppedPipe=0;d.roomDecay=0.f;
                    d.group=G_PRINCIPAL;
                }
            }
        }
    }

    /* ── FLUTE ranks (1200 osc) ─────────────────────────────────────
       Bourdon 8' (stopped pipe, odd harmonics): 150 voices × 4 h = 600
       Flute 4' (open, few harmonics):           150 voices × 4 h = 600
       Stopped pipe: only odd harmonics fit the resonator.
       Wide bore → large mouth → fundamental dominates heavily.
       Inharmonicity lower (wood, wider bore): B=0.00008               */
    {
        for(int section=0;section<2;section++){
            float ratio=(section==0)?1.f:2.f;   /* 8' vs 4'            */
            int stopped=(section==0)?1:0;
            float bFactor=(section==0)?0.00008f:0.00010f;
            for(int v=0;v<150;v++){
                float det=rr(-1.5f,1.5f);
                float vPan=rr(-0.60f,0.60f);
                for(int h=1;h<=4;h++){
                    int actualH=(stopped)?(2*h-1):h; /* odd only if stopped */
                    OscDesc&d=D[idx++];
                    d.freqRatio  =ratio*(float)actualH;
                    d.harmonic   =actualH;
                    d.detuneCents=det;
                    d.inharmonB  =bFactor;
                    /* flute: fundamental strong, harmonics fall fast   */
                    d.amp=powf(1.f/(float)actualH,1.4f)*0.010f*(0.85f+0.30f*rnd());
                    if(stopped&&h>2)d.amp*=0.4f;
                    d.phase0=rnd();
                    pan(vPan,d.panL,d.panR);
                    d.aA=0.025f;d.aD=0.06f;d.aS=0.97f;d.aR=0.18f;
                    /* flute chiff: longer, lower frequency             */
                    d.chiffDur   =(h==1)?rr(0.018f,0.028f):0.f;
                    d.chiffBandHz=(h==1)?rr(400.f,700.f):0.f;
                    d.tremPitchDepth=rr(4.f,8.f);
                    d.tremAmpDepth  =rr(0.06f,0.12f);
                    d.tremPhase=rnd();
                    d.windAmp=0.f;d.stoppedPipe=stopped;d.roomDecay=0.f;
                    d.group=G_FLUTE;
                }
            }
        }
    }

    /* ── STRING ranks (800 osc) ─────────────────────────────────────
       Salicional 8': narrow-scaled pipe → strong upper harmonics.
       100 voices × 8 harmonics = 800 osc.
       Amplitude rises toward h=4-6 then falls (formant-like peak).
       Narrow mouth → slow speech, very slight chiff.
       Higher inharmonicity (narrow metal): B=0.00020                  */
    {
        for(int v=0;v<100;v++){
            float det=rr(-3.0f,3.0f);
            float vPan=rr(-0.80f,0.80f);
            for(int h=1;h<=8;h++){
                OscDesc&d=D[idx++];
                d.freqRatio  =(float)h;
                d.harmonic   =h;
                d.detuneCents=det;
                d.inharmonB  =0.00020f*(1.f+rr(-0.2f,0.2f));
                /* string spectrum: rises to peak around h=4-5 */
                float peak=expf(-0.5f*powf((float)h-4.5f,2.f)/2.5f);
                float roll=powf(1.f/(float)h,0.7f);
                d.amp=(peak*0.5f+roll*0.5f)*0.007f*(0.85f+0.30f*rnd());
                d.phase0=rnd();
                pan(vPan,d.panL,d.panR);
                d.aA=0.035f;d.aD=0.08f;d.aS=0.95f;d.aR=0.20f;
                d.chiffDur   =(h<=2)?rr(0.006f,0.010f):0.f;
                d.chiffBandHz=(h<=2)?rr(1200.f,2000.f):0.f;
                d.tremPitchDepth=rr(2.f,5.f);
                d.tremAmpDepth  =rr(0.03f,0.07f);
                d.tremPhase=rnd();
                d.windAmp=0.f;d.stoppedPipe=0;d.roomDecay=0.f;
                d.group=G_STRING;
            }
        }
    }

    /* ── CELESTE (800 osc) ──────────────────────────────────────────
       Voix celeste 8': two string ranks, one tuned +7 cents sharp.
       The beat frequency between the paired ranks creates shimmer.
       100 voices × 8 harmonics = 800 osc.
       Per-pipe beat = f*(2^(7/1200)-1) ≈ 0.40% of fundamental Hz.   */
    {
        for(int v=0;v<100;v++){
            float det=7.0f+rr(-1.f,1.f);  /* sharp string of the pair */
            float vPan=rr(-0.80f,0.80f);
            for(int h=1;h<=8;h++){
                OscDesc&d=D[idx++];
                d.freqRatio  =(float)h;
                d.harmonic   =h;
                d.detuneCents=det;
                d.inharmonB  =0.00020f*(1.f+rr(-0.2f,0.2f));
                float peak=expf(-0.5f*powf((float)h-4.5f,2.f)/2.5f);
                float roll=powf(1.f/(float)h,0.7f);
                d.amp=(peak*0.5f+roll*0.5f)*0.006f*(0.85f+0.30f*rnd());
                d.phase0=rnd();
                pan(vPan,d.panL,d.panR);
                d.aA=0.035f;d.aD=0.08f;d.aS=0.95f;d.aR=0.20f;
                d.chiffDur=(h<=2)?rr(0.006f,0.010f):0.f;
                d.chiffBandHz=(h<=2)?rr(1200.f,2000.f):0.f;
                d.tremPitchDepth=rr(2.f,5.f);
                d.tremAmpDepth  =rr(0.03f,0.07f);
                d.tremPhase=rnd();
                d.windAmp=0.f;d.stoppedPipe=0;d.roomDecay=0.f;
                d.group=G_CELESTE;
            }
        }
    }

    /* ── REED ranks (1000 osc) ──────────────────────────────────────
       Trumpet 8' + Oboe 8'.  Reed pipe: shallot vibrates to produce
       a nearly-sawtooth pressure wave at the foot; the resonator
       (cone or cylinder) shapes a formant peak.
       Trumpet: broad conical resonator → formant peak ~800-1200 Hz.
       Oboe: narrow cylindrical resonator → narrower peak ~600-900 Hz.
       100 voices × 5 harmonics per reed = 500 per rank = 1000.
       Inharmonicity low (reed is frequency-determining): B=0.00005    */
    {
        for(int reedType=0;reedType<2;reedType++){
            float formantHz=(reedType==0)?rr(900.f,1200.f):rr(600.f,850.f);
            float formantBW=(reedType==0)?500.f:280.f;
            for(int v=0;v<100;v++){
                float det=rr(-1.5f,1.5f);
                float vPan=rr(-0.65f,0.65f);
                for(int h=1;h<=5;h++){
                    OscDesc&d=D[idx++];
                    d.freqRatio  =(float)h;
                    d.harmonic   =h;
                    d.detuneCents=det;
                    d.inharmonB  =0.00005f;
                    /* Reed spectrum: sawtooth-like modified by formant */
                    float saw=1.f/(float)h;
                    /* formant peak: assuming C4=261Hz reference        */
                    float fHz=261.63f*(float)h;
                    float formant=expf(-0.5f*powf((fHz-formantHz)/formantBW,2.f));
                    d.amp=(saw*(0.6f+0.8f*formant))*0.010f*(0.85f+0.30f*rnd());
                    d.phase0=rnd();
                    pan(vPan,d.panL,d.panR);
                    /* Reed: slower attack than flue (tongue must start vibrating)*/
                    d.aA=0.030f;d.aD=0.06f;d.aS=0.96f;d.aR=0.10f;
                    d.chiffDur   =(h==1)?rr(0.025f,0.040f):0.f;
                    d.chiffBandHz=(h==1)?formantHz*0.8f:0.f;
                    /* Reed tremulant: strong response */
                    d.tremPitchDepth=rr(5.f,9.f);
                    d.tremAmpDepth  =rr(0.07f,0.14f);
                    d.tremPhase=rnd();
                    d.windAmp=0.f;d.stoppedPipe=0;d.roomDecay=0.f;
                    d.group=G_REED;
                }
            }
        }
    }

    /* ── MIXTURE stops (1200 osc) ───────────────────────────────────
       Mixture = multiple pipes per key at upper partial pitches.
       A four-rank mixture: 2⅔'(3×), 2'(4×), 1⅗'(5×), 1'(8×).
       150 voices × 4 harmonic pitches = 600 per "side" × 2 = 1200.
       These are bright, high-pitched — individually inaudible, but
       together they add "brilliance" to the full organ sound.
       Historical temperament: slight tuning variation by key.         */
    {
        float mixtureRatios[4]={3.f,4.f,5.f,8.f};  /* partial numbers  */
        float mixtureVols[4]  ={0.55f,0.50f,0.45f,0.35f};
        for(int rank=0;rank<4;rank++){
            for(int v=0;v<75;v++){
                float det=rr(-1.5f,1.5f);
                /* slight tuning variation — historical temperament colour*/
                float tempAdj=rr(-3.f,3.f);
                float vPan=rr(-0.85f,0.85f);
                for(int h=1;h<=4;h++){
                    OscDesc&d=D[idx++];
                    d.freqRatio  =mixtureRatios[rank]*(float)h;
                    d.harmonic   =h;
                    d.detuneCents=det+tempAdj;
                    d.inharmonB  =0.00012f;
                    d.amp=mixtureVols[rank]*powf(1.f/(float)h,1.0f)*0.006f
                          *(0.85f+0.30f*rnd());
                    d.phase0=rnd();
                    pan(vPan,d.panL,d.panR);
                    d.aA=0.012f;d.aD=0.04f;d.aS=0.97f;d.aR=0.10f;
                    d.chiffDur=(h==1)?rr(0.008f,0.014f):0.f;
                    d.chiffBandHz=(h==1)?rr(2000.f,4000.f):0.f;
                    d.tremPitchDepth=rr(3.f,7.f);
                    d.tremAmpDepth  =rr(0.05f,0.10f);
                    d.tremPhase=rnd();
                    d.windAmp=0.f;d.stoppedPipe=0;d.roomDecay=0.f;
                    d.group=G_MIXTURE;
                }
            }
        }
    }

    /* ── PEDAL division (1200 osc) ──────────────────────────────────
       16' Open Diapason + 8' Octave.
       16' at low C = 32.7 Hz — felt as much as heard.
       150 voices × 4 harmonics = 600 per rank × 2 = 1200.
       The fundamental must dominate absolutely; upper harmonics
       provide the "thud" on attack then fall away, leaving pure
       sub-bass that excites cathedral room modes for seconds.         */
    {
        float pedalRatios[2]={0.5f,1.f};   /* 16'=0.5× vs 8'=1× of note */
        float pedalVols[2]  ={2.20f,1.40f};  /* significantly louder      */
        for(int rank=0;rank<2;rank++){
            for(int v=0;v<150;v++){
                float det=rr(-1.5f,1.5f);
                float vPan=rr(-0.20f,0.20f);  /* near-mono: sub-bass is omni*/
                for(int h=1;h<=4;h++){
                    OscDesc&d=D[idx++];
                    d.freqRatio  =pedalRatios[rank]*(float)h;
                    d.harmonic   =h;
                    d.detuneCents=det;
                    d.inharmonB  =0.00008f;
                    /* steep rolloff: fundamental overwhelms harmonics  */
                    d.amp=pedalVols[rank]*powf(1.f/(float)h,1.8f)*0.022f
                          *(0.85f+0.30f*rnd());
                    d.phase0=rnd();
                    pan(vPan,d.panL,d.panR);
                    /* Slow speech, very long release for sub-bass ring */
                    d.aA=0.065f;d.aD=0.18f;d.aS=0.97f;
                    d.aR=rr(1.2f,1.8f);   /* 1.2-1.8s release — long ring */
                    d.chiffDur   =(h==1)?rr(0.040f,0.065f):0.f;
                    d.chiffBandHz=(h==1)?rr(80.f,200.f):0.f;
                    d.tremPitchDepth=rr(0.5f,1.5f);  /* minimal trem on pedal */
                    d.tremAmpDepth  =rr(0.01f,0.03f);
                    d.tremPhase=rnd();
                    d.windAmp=0.f;d.stoppedPipe=0;d.roomDecay=0.f;
                    d.group=G_PEDAL;
                }
            }
        }
    }

    /* ── CHIFF / pipe speech (400 osc) ─────────────────────────────
       Dedicated attack transient oscillators: broadband noise in
       specific frequency bands, fast decay.  Not pitch-locked.
       50 voices × 8 "harmonics" (spectral bands) = 400.              */
    {
        float chiffBands[8]={300.f,600.f,900.f,1400.f,2000.f,3000.f,5000.f,8000.f};
        for(int v=0;v<50;v++){
            float vPan=rr(-0.60f,0.60f);
            for(int h=0;h<8;h++){
                OscDesc&d=D[idx++];
                d.freqRatio   =1.f;
                d.harmonic    =1;
                d.detuneCents =rr(-5.f,5.f);
                d.inharmonB   =0.f;
                d.amp         =0.008f*(0.6f+0.6f*rnd())
                               *expf(-0.5f*powf((float)h-2.f,2.f)/4.f);
                d.phase0=rnd();
                pan(vPan,d.panL,d.panR);
                d.aA=0.001f;d.aD=rr(0.012f,0.028f);d.aS=0.f;d.aR=0.02f;
                d.chiffDur   =rr(0.015f,0.030f);
                d.chiffBandHz=chiffBands[h]*(0.8f+0.4f*rnd());
                d.tremPitchDepth=0.f;d.tremAmpDepth=0.f;d.tremPhase=0.f;
                d.windAmp=0.f;d.stoppedPipe=0;d.roomDecay=0.f;
                d.group=G_CHIFF;
            }
        }
    }

    /* ── ROOM MODES / cathedral acoustics (800 osc) ─────────────────
       Actual rectangular room resonance modes.
       f_nml = c/2 * sqrt((n/Lx)^2+(m/Ly)^2+(l/Lz)^2)
       Room dimensions randomised per run within cathedral range.
       Each mode has its own RT60-based amplitude decay.               */
    {
        const float SPEED_SOUND=343.f;
        /* Cathedral dimensions: 12-20m wide, 40-70m long, 18-30m high */
        float Lx=rr(12.f,20.f)*roomScale;
        float Ly=rr(40.f,70.f)*roomScale;
        float Lz=rr(18.f,30.f)*roomScale;
        /* RT60 (Sabine): T60 = 0.161*V/A ≈ 3-6s for cathedral        */
        float rt60=rr(2.5f+1.5f*roomScale, 4.0f+2.0f*roomScale);
        float decayRate=6.908f/rt60;  /* ln(1000000)/rt60              */

        /* enumerate modes, pick lowest 800 by frequency               */
        struct Mode{float f;int nx,ny,nz;};
        static Mode modes[8192];int nm=0;
        for(int nx=0;nx<=5&&nm<8192;nx++)
        for(int ny=0;ny<=8&&nm<8192;ny++)
        for(int nz=0;nz<=5&&nm<8192;nz++){
            if(nx==0&&ny==0&&nz==0)continue;
            float f=SPEED_SOUND/2.f*sqrtf(
                (float)(nx*nx)/(Lx*Lx)+
                (float)(ny*ny)/(Ly*Ly)+
                (float)(nz*nz)/(Lz*Lz));
            if(f<0.45f*SR && f>10.f)
                modes[nm++]={f,nx,ny,nz};
        }
        /* sort by frequency, take lowest 800 */
        std::sort(modes,modes+nm,[](const Mode&a,const Mode&b){return a.f<b.f;});
        int nRoom=std::min(nm,800);
        for(int i=0;i<nRoom;i++){
            OscDesc&d=D[idx++];
            d.freqRatio   =modes[i].f;  /* absolute Hz — noteHz×1.0  */
            d.harmonic    =1;
            d.detuneCents =0.f;
            d.inharmonB   =0.f;
            d.amp         =0.0008f*(0.4f+0.8f*rnd())
                           *expf(-modes[i].f/2000.f); /* high modes quieter*/
            d.phase0=rnd();
            /* room modes spread across the stereo field               */
            pan(rr(-1.f,1.f),d.panL,d.panR);
            d.aA=rr(0.5f,2.0f);d.aD=0.f;d.aS=0.8f;d.aR=rt60*rr(0.6f,1.0f);
            d.chiffDur=0.f;d.chiffBandHz=0.f;
            d.tremPitchDepth=0.f;d.tremAmpDepth=0.f;d.tremPhase=0.f;
            d.windAmp=0.f;d.stoppedPipe=0;
            d.roomDecay=decayRate*(0.8f+0.4f*rnd());
            d.group=G_ROOM;
        }
        /* fill any remaining slots if nm < 800 */
        while(idx < (int)(D - D) + G_CNT[G_PRINCIPAL]+G_CNT[G_FLUTE]+
              G_CNT[G_STRING]+G_CNT[G_CELESTE]+G_CNT[G_REED]+
              G_CNT[G_MIXTURE]+G_CNT[G_PEDAL]+G_CNT[G_CHIFF]+G_CNT[G_ROOM]){
            OscDesc&d=D[idx++];
            d.freqRatio=100.f;d.amp=0.f;d.panL=0.5f;d.panR=0.5f;
            d.aA=0.01f;d.aR=0.01f;d.group=G_ROOM;d.harmonic=1;
        }
    }

    /* ── WIND / mechanical noise (600 osc) ──────────────────────────
       Wind chest rumble and blower hum — purely subliminal.
       Each oscillator is a sine at 20-180 Hz (blower harmonics, chest
       resonances) with slow AM.  Total level is intentionally very low:
       you should feel the organ breathe, not hear discrete noise.      */
    {
        for(int i=0;i<600;i++){
            OscDesc&d=D[idx++];
            d.freqRatio   =rr(20.f,180.f);   /* blower/chest frequencies */
            d.harmonic    =1;
            d.detuneCents =0.f;
            d.inharmonB   =0.f;
            d.amp         =0.f;              /* amp unused for wind group  */
            d.phase0=rnd();
            pan(rr(-0.3f,0.3f),d.panL,d.panR);
            /* slow attack so wind rises naturally with the organ        */
            d.aA=2.5f;d.aD=0.f;d.aS=1.0f;d.aR=3.0f;
            d.chiffDur=0.f;d.chiffBandHz=0.f;
            d.tremPitchDepth=0.f;d.tremAmpDepth=0.f;d.tremPhase=0.f;
            /* windAmp: much lower — subliminal presence only           */
            d.windAmp=0.000035f*(0.5f+0.8f*rnd());
            d.stoppedPipe=0;d.roomDecay=0.f;
            d.group=G_WIND;
        }
    }

    assert(idx==N_OSC);
}

/* ══════════════════════════════════════════════════════════════════════
   EVENT INDEXING
   ══════════════════════════════════════════════════════════════════════ */
static int cmpEv(const void*a,const void*b){
    float d=((Event*)a)->tOn-((Event*)b)->tOn;
    return (d<0.f)?-1:(d>0.f)?1:0;
}
static void indexEvents(const EvBuf&E,int*evS,int*evC,Event*sorted){
    memset(evS,0,(N_GROUPS+1)*sizeof(int));
    memset(evC,0,(N_GROUPS+1)*sizeof(int));
    for(int i=0;i<E.n;i++)if(E.ev[i].group<N_GROUPS)evC[E.ev[i].group]++;
    for(int g=1;g<=N_GROUPS;g++)evS[g]=evS[g-1]+evC[g-1];
    int cur[N_GROUPS+1];memcpy(cur,evS,(N_GROUPS+1)*sizeof(int));
    for(int i=0;i<E.n;i++){int g=E.ev[i].group;
        if(g<N_GROUPS)sorted[cur[g]++]=E.ev[i];}
    for(int g=0;g<N_GROUPS;g++)if(evC[g]>1)
        qsort(sorted+evS[g],(size_t)evC[g],sizeof(Event),cmpEv);
}

/* ══════════════════════════════════════════════════════════════════════
   POST-PROCESSING
   ══════════════════════════════════════════════════════════════════════ */
/* ── Church reverb: Schroeder + all-pass network, long RT60 ─────────
   Tuned for a large stone cathedral: RT60 3-5 seconds, pre-delay
   40-60 ms (listener far from organ loft).
   Schroeder 1962: 8 parallel comb filters → 4 series all-pass filters.
   Comb delays chosen to be mutually prime (no common factors) to avoid
   metallic colouration.  All-pass adds diffusion.
   roomScale 0=dry chamber (RT60~1.5s), 1=cathedral (~3.5s), 2=gothic (~5.5s) */
/* ── Select reverb mode at compile time ──────────────────────────────
   0 = Schroeder algorithmic reverb (fast, always reliable)
   1 = Convolution reverb with synthesised cathedral IR (~30-80s extra)
   Change this value and rebuild to switch.                            */
#define USE_CONV_REVERB 1

#if !USE_CONV_REVERB
/* ── Schroeder church reverb: only compiled when USE_CONV_REVERB=0 ── */
static void applyChurchReverb(float*L,float*R,int N,float sr,float roomScale)
{
    if(roomScale<0.01f)return;
    /* comb filter delays (ms → samples), prime-ish separations       */
    static const float CD_MS[]={
        29.7f,37.1f,41.1f,43.7f,47.3f,53.5f,59.3f,61.4f};
    static const float AD_MS[]={5.0f,1.7f,12.7f,9.8f};
    int NC=8,NA=4;
    /* feedback and damping derived directly from roomScale             */
    float fb=0.84f+0.06f*fminf(roomScale,1.5f);  /* feedback < 1.0    */
    float damp=0.18f+0.10f*roomScale;             /* HF damping        */
    /* pre-delay: organ in a loft, listener mid-nave → 45-65ms        */
    float preDelayMs=40.f+12.f*roomScale;
    int preD=(int)(preDelayMs*0.001f*sr);

    /* allocate comb and all-pass buffers                              */
    float*cbL[8],*cbR[8],*abL[4],*abR[4];
    int csz[8],asz[4],cpL[8]={},cpR[8]={},apL[4]={},apR[4]={};
    float lpL[8]={},lpR[8]={};
    for(int i=0;i<NC;i++){
        csz[i]=(int)(CD_MS[i]*0.001f*sr*(0.9f+0.2f*roomScale));
        cbL[i]=(float*)calloc(csz[i]+1,sizeof(float));
        cbR[i]=(float*)calloc(csz[i]+8,sizeof(float));/* slight L/R asymm*/
    }
    for(int i=0;i<NA;i++){
        asz[i]=(int)(AD_MS[i]*0.001f*sr);
        abL[i]=(float*)calloc(asz[i]+1,sizeof(float));
        abR[i]=(float*)calloc(asz[i]+1,sizeof(float));
    }
    /* pre-delay buffer */
    float*pdL=(float*)calloc(preD+2,sizeof(float));
    float*pdR=(float*)calloc(preD+2,sizeof(float));
    int pdPos=0;

    float wet=0.28f+0.22f*roomScale;  /* wet mix: 0.28 at dry, 0.50 at max */
    float dry=1.0f - wet*0.35f;       /* dry stays present            */

    for(int s=0;s<N;s++){
        /* pre-delay tap                                               */
        float xL=pdL[pdPos],xR=pdR[pdPos];
        pdL[pdPos]=L[s];pdR[pdPos]=R[s];
        pdPos=(pdPos+1>preD)?0:pdPos+1;

        /* 8 parallel comb filters                                     */
        float oL=0,oR=0;
        for(int i=0;i<NC;i++){
            float dL=cbL[i][cpL[i]];
            lpL[i]=dL*(1.f-damp)+lpL[i]*damp;
            cbL[i][cpL[i]]=xL+lpL[i]*fb;
            cpL[i]=(cpL[i]+1>=csz[i])?0:cpL[i]+1; oL+=dL;
            float dR=cbR[i][cpR[i]];
            lpR[i]=dR*(1.f-damp)+lpR[i]*damp;
            cbR[i][cpR[i]]=xR+lpR[i]*fb;
            cpR[i]=(cpR[i]+1>=csz[i])?0:cpR[i]+1; oR+=dR;
        }
        oL/=(float)NC; oR/=(float)NC;

        /* 4 series all-pass filters                                   */
        for(int i=0;i<NA;i++){
            float bL=abL[i][apL[i]];float vL=oL+bL*0.5f;
            abL[i][apL[i]]=vL;apL[i]=(apL[i]+1>=asz[i])?0:apL[i]+1;oL=bL-vL;
            float bR=abR[i][apR[i]];float vR=oR+bR*0.5f;
            abR[i][apR[i]]=vR;apR[i]=(apR[i]+1>=asz[i])?0:apR[i]+1;oR=bR-vR;
        }

        L[s]=L[s]*dry + oL*wet;
        R[s]=R[s]*dry + oR*wet;
    }
    for(int i=0;i<NC;i++){free(cbL[i]);free(cbR[i]);}
    for(int i=0;i<NA;i++){free(abL[i]);free(abR[i]);}
    free(pdL);free(pdR);
}
#endif /* !USE_CONV_REVERB */

/* ══════════════════════════════════════════════════════════════════════
   CONVOLUTION REVERB  — FFT partitioned overlap-add
   Uses a physically-synthesised cathedral impulse response (IR):
     • Direct sound at t=0
     • 12 early reflections (floor/ceiling/side walls/rear) 2-80 ms
     • Pre-diffuse cluster 80-250 ms (dense comb of reflections)
     • Exponential diffuse tail, RT60 = roomScale×(2.5-4.5) s
   Partitioned convolution keeps memory and CPU cost manageable:
     partition size B=2048, FFT size 4096, ~94 partitions for a 4s IR.
   Total cost ≈ 6 s CPU for a 3-minute song — acceptable post-process.
   ══════════════════════════════════════════════════════════════════════ */

/* ── Minimal radix-2 Cooley-Tukey FFT ───────────────────────────────
   in-place complex FFT on interleaved float array (re,im,re,im,...).
   n must be a power of 2.  inverse=true for IFFT.                    */
static void fft_r2(float*x, int n, bool inverse)
{
    /* bit-reversal permutation */
    for(int i=1,j=0; i<n; i++){
        int bit=n>>1;
        for(; j&bit; bit>>=1) j^=bit;
        j^=bit;
        if(i<j){ std::swap(x[2*i],x[2*j]); std::swap(x[2*i+1],x[2*j+1]); }
    }
    /* Cooley-Tukey butterfly */
    for(int len=2; len<=n; len<<=1){
        float ang = (float)(2*M_PI/len) * (inverse?1.f:-1.f);
        float wRe=cosf(ang), wIm=sinf(ang);
        for(int i=0; i<n; i+=len){
            float curRe=1.f, curIm=0.f;
            for(int j=0; j<len/2; j++){
                int u=i+j, v=i+j+len/2;
                float uRe=x[2*u],   uIm=x[2*u+1];
                float vRe=x[2*v],   vIm=x[2*v+1];
                float tRe=curRe*vRe-curIm*vIm;
                float tIm=curRe*vIm+curIm*vRe;
                x[2*u]=uRe+tRe; x[2*u+1]=uIm+tIm;
                x[2*v]=uRe-tRe; x[2*v+1]=uIm-tIm;
                float nRe=curRe*wRe-curIm*wIm;
                float nIm=curRe*wIm+curIm*wRe;
                curRe=nRe; curIm=nIm;
            }
        }
    }
    if(inverse){
        float inv=(float)(1.0/n);
        for(int i=0;i<2*n;i++) x[i]*=inv;
    }
}

/* ── Synthesise cathedral impulse response ───────────────────────────
   Returns a heap-allocated mono IR of length irLen samples.
   Based on a rectangular room model (Lx×Ly×Lz) with randomised
   dimensions, plus a smooth exponential diffuse tail.
   seed2: independent RNG so IR doesn't affect the music seed.         */
static float* buildIR(int irLen, float sr, float roomScale, unsigned seed2)
{
    float*ir = (float*)calloc(irLen, sizeof(float));

    /* local RNG — doesn't disturb global G_RNG */
    auto xrng=[&]()->float{
        seed2^=seed2<<13;seed2^=seed2>>17;seed2^=seed2<<5;
        return (float)(seed2&0xFFFFFF)/16777216.f;
    };

    /* ── Direct sound ───────────────────────────────────────────────── */
    ir[0] = 1.0f;

    /* ── Early reflections: 12 image sources ───────────────────────── */
    /* Cathedral dimensions (scale with roomScale)                      */
    float Lx = (12.f+xrng()*8.f)*fmaxf(roomScale,0.5f);
    float Ly = (40.f+xrng()*30.f)*fmaxf(roomScale,0.5f);
    float Lz = (18.f+xrng()*12.f)*fmaxf(roomScale,0.5f);
    float c  = 343.f;
    /* listener position: mid-nave */
    float lx=Lx*0.5f, ly=Ly*0.4f, lz=Lz*0.45f;
    /* source position: organ loft (west end, high up) */
    float sx=Lx*0.5f, sy=Ly*0.05f, sz=Lz*0.75f;

    /* image source method: 6 first-order reflections */
    float walls[6][3]={
        {2*0   -sx,sy,sz},   /* front wall  */
        {2*Lx  -sx,sy,sz},   /* back wall   */
        {sx,2*0   -sy,sz},   /* left wall   */
        {sx,2*Ly  -sy,sz},   /* right wall  */
        {sx,sy,2*0   -sz},   /* floor       */
        {sx,sy,2*Lz  -sz},   /* ceiling     */
    };
    float wallAbsorb[6]={0.92f,0.90f,0.88f,0.88f,0.85f,0.94f};  /* stone/vault */

    for(int w=0;w<6;w++){
        float dx=walls[w][0]-lx, dy=walls[w][1]-ly, dz=walls[w][2]-lz;
        float dist=sqrtf(dx*dx+dy*dy+dz*dz);
        float delay_s=dist/c;
        int   delay_n=(int)(delay_s*sr);
        if(delay_n>=irLen)continue;
        /* amplitude: 1/r spreading × wall absorption coefficient     */
        float refDist=5.f;   /* reference distance for normalisation   */
        float amp=wallAbsorb[w]*(refDist/fmaxf(dist,refDist));
        /* HF air absorption: e^(-0.00008*f_avg*dist), f_avg~2kHz     */
        amp*=expf(-0.00008f*2000.f*delay_s);
        ir[delay_n]+=amp;
        /* second-order: reflect off opposite wall (approximate)       */
        float delay2_s=delay_s+Ly/c*(0.4f+0.3f*xrng());
        int d2=(int)(delay2_s*sr);
        if(d2<irLen) ir[d2]+=amp*wallAbsorb[w]*0.5f;
    }

    /* ── Pre-diffuse cluster: dense reflections 80-250 ms ──────────── */
    int preDiffStart=(int)(0.080f*sr);
    int preDiffEnd  =(int)(0.250f*sr);
    for(int i=0;i<40;i++){
        int pos=preDiffStart+(int)(xrng()*(preDiffEnd-preDiffStart));
        if(pos>=irLen)continue;
        float amp=(0.08f+xrng()*0.12f)*expf(-3.f*(float)(pos-preDiffStart)/(preDiffEnd-preDiffStart));
        ir[pos]+=(xrng()<0.5f?1.f:-1.f)*amp;  /* random polarity */
    }

    /* ── Diffuse exponential tail ───────────────────────────────────── */
    /* RT60 (time to decay 60 dB): 2.5s at roomScale=0.5, 4.5s at 2.0 */
    float rt60 = 2.5f + roomScale * 1.0f;
    float decayPerSample = powf(10.f,-3.f/(rt60*sr));  /* -60dB in rt60 s */
    int tailStart = (int)(0.080f * sr);  /* tail starts after early reflections */
    float env = 0.12f;  /* initial diffuse tail amplitude */
    /* noise seed for the tail */
    unsigned ts = seed2 ^ 0xCAFEBABEu;
    for(int i=tailStart; i<irLen; i++){
        ts^=ts<<13;ts^=ts>>17;ts^=ts<<5;
        float n=(float)(int)ts*4.6566129e-10f;
        /* slight HF rolloff over time (air absorption in long tail) */
        float hfRoll=expf(-0.0002f*(float)(i-tailStart));
        ir[i]+=n*env*hfRoll;
        env*=decayPerSample;
    }

    /* ── Normalise IR: peak-normalise then scale tail down ──────────
       The direct sound at ir[0]=1.0 sets the peak.  We want:
         direct sound:         amplitude 1.0  (reference)
         early reflections:    0.2 - 0.6      (correct 1/r spreading)
         diffuse tail peak:    ~0.12          (naturally from env=0.12)
       Scale so peak = 1.0 then the wet mix gain controls the blend.
       Don't use energy normalisation — the long tail dominates energy
       but should not suppress the direct sound.                       */
    float pk=0.f;
    for(int i=0;i<irLen;i++) pk=fmaxf(pk,fabsf(ir[i]));
    if(pk>1e-9f){float invPk=1.f/pk; for(int i=0;i<irLen;i++) ir[i]*=invPk;}
    /* The tail is now scaled relative to the direct impulse.
       Higher scale = more room character vs dry sound.
       0.55 gives a cathedral that dominates the mix on low notes,
       which is correct for sub-bass pipes in a stone building.        */
    for(int i=0;i<irLen;i++) ir[i]*=0.55f;

    /* sanity check */
    float irPeak=0.f,irRMS=0.f;
    for(int i=0;i<irLen;i++){irPeak=fmaxf(irPeak,fabsf(ir[i]));irRMS+=ir[i]*ir[i];}
    irRMS=sqrtf(irRMS/irLen);
    printf("  [IR] len=%.1fs  peak=%.4f  RMS=%.6f  RT60=%.1fs\n",
           (float)irLen/sr, irPeak, irRMS,
           2.5f+roomScale*1.0f);
    fflush(stdout);

    return ir;
}

/* ── Partitioned overlap-add convolution reverb ─────────────────────
   L/R are modified in place: output = dry_signal * (1-wet) + conv * wet
   ir: mono impulse response, irLen samples.
   B: partition size (must be power of 2).                            */
static void applyConvReverb(float*L, float*R, int N,
                             const float*ir, int irLen,
                             float wet, float dry)
{
    const int B      = 2048;
    const int FFTN   = B*2;        /* 4096 — FFT size for each partition  */
    const int P      = (irLen+B-1)/B;  /* number of IR partitions          */

    /* ── Pre-compute FFT of each IR partition ─────────────────────── */
    /* Store as complex interleaved: irFFT[p][0..FFTN*2-1]            */
    float**irFFT_L=(float**)malloc(P*sizeof(float*));
    for(int p=0;p<P;p++){
        irFFT_L[p]=(float*)calloc(FFTN*2,sizeof(float));
        int pStart=p*B;
        int pLen=std::min(B,irLen-pStart);
        for(int i=0;i<pLen;i++) irFFT_L[p][2*i]=ir[pStart+i];
        fft_r2(irFFT_L[p],FFTN,false);
    }
    /* For stereo we use the same IR but could differ; use same here   */

    /* ── Working FFT buffer for input and output ─────────────────── */
    float*xfftL=(float*)calloc(FFTN*2,sizeof(float));
    float*xfftR=(float*)calloc(FFTN*2,sizeof(float));
    float*yfftL=(float*)calloc(FFTN*2,sizeof(float));
    float*yfftR=(float*)calloc(FFTN*2,sizeof(float));

    int nBlocks=(N+B-1)/B;
    /* totalOut must cover the last block's last partition's full FFTN output */
    int totalOut=nBlocks*B + P*B + FFTN;

    /* Allocate full output buffers */
    float*outL=(float*)calloc(totalOut,sizeof(float));
    float*outR=(float*)calloc(totalOut,sizeof(float));
    if(!outL||!outR){fprintf(stderr,"conv: out of memory\n");return;}

    printf("  [conv] %d blocks x %d partitions x FFT(%d) ...\n",nBlocks,P,FFTN);
    fflush(stdout);
    clock_t tConvStart=clock();

    for(int blk=0; blk<nBlocks; blk++){
        int s0=blk*B;
        int blkLen=std::min(B,N-s0);

        /* ── FFT of this input block ─────────────────────────────── */
        memset(xfftL,0,FFTN*2*sizeof(float));
        memset(xfftR,0,FFTN*2*sizeof(float));
        for(int i=0;i<blkLen;i++){
            xfftL[2*i]=L[s0+i];
            xfftR[2*i]=R[s0+i];
        }
        fft_r2(xfftL,FFTN,false);
        fft_r2(xfftR,FFTN,false);

        /* ── Convolve with each IR partition via complex multiply ─── */
        for(int p=0;p<P;p++){
            memset(yfftL,0,FFTN*2*sizeof(float));
            memset(yfftR,0,FFTN*2*sizeof(float));
            const float*hp=irFFT_L[p];
            for(int k=0;k<FFTN;k++){
                /* complex multiply: (a+bi)(c+di)=(ac-bd)+(ad+bc)i   */
                float aL=xfftL[2*k],bL=xfftL[2*k+1];
                float aR=xfftR[2*k],bR=xfftR[2*k+1];
                float c=hp[2*k],d=hp[2*k+1];
                yfftL[2*k]  =aL*c-bL*d;
                yfftL[2*k+1]=aL*d+bL*c;
                yfftR[2*k]  =aR*c-bR*d;
                yfftR[2*k+1]=aR*d+bR*c;
            }
            fft_r2(yfftL,FFTN,true);
            fft_r2(yfftR,FFTN,true);

            /* accumulate FFTN real samples into output at offset blk+p */
            int outStart=(blk+p)*B;
            for(int i=0;i<FFTN&&outStart+i<totalOut;i++){
                outL[outStart+i]+=yfftL[2*i];
                outR[outStart+i]+=yfftR[2*i];
            }
        }

        if(blk%64==0){
            float pct=100.f*(blk+1)/nBlocks;
            printf("  [conv] %.0f%%  (%d/%d blocks)\r",pct,blk+1,nBlocks);
            fflush(stdout);
        }
    }
    float tConvSec=(float)(clock()-tConvStart)/CLOCKS_PER_SEC;
    printf("  [conv] done -- %.1f s CPU time                    \n",tConvSec);

    /* ── Mix back: original dry + convolved wet ─────────────────── */
    for(int i=0;i<N;i++){
        L[i]=L[i]*dry + outL[i]*wet;
        R[i]=R[i]*dry + outR[i]*wet;
    }

    /* cleanup */
    for(int p=0;p<P;p++) free(irFFT_L[p]);
    free(irFFT_L);
    free(xfftL);free(xfftR);free(yfftL);free(yfftR);
    free(outL);free(outR);
}

static void softSat(float*x,int N,float d){
    for(int i=0;i<N;i++)x[i]=tanhf(x[i]*d)/d;
}
static void writeWav(const char*p,const float*L,const float*R,int N,int sr){
    FILE*f=fopen(p,"wb");if(!f){fprintf(stderr,"open %s fail\n",p);return;}
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
int main(int argc,char**argv)
{
    const char*out   =(argc>1)?argv[1]:"pipeorgan01.wav";
    int   sr         =(argc>2)?atoi(argv[2]):SR;
    unsigned seed;
    if(argc>3){
        seed=(unsigned)atoi(argv[3]);
        if(seed==0){
            /* high-resolution random seed: combine wall time, CPU time,
               and a pointer address (ASLR adds entropy on each run)   */
            seed=(unsigned)time(NULL)
                ^((unsigned)clock()*1000003u)
                ^((unsigned)((uintptr_t)(void*)&seed ^ ((uintptr_t)(void*)&seed>>32)));
        }
    } else {
        seed=(unsigned)time(NULL)
            ^((unsigned)clock()*1000003u)
            ^((unsigned)((uintptr_t)(void*)&seed ^ ((uintptr_t)(void*)&seed>>32)));
    }
    int   length     =(argc>4)?atoi(argv[4]):5;
    if(length<1)length=1;
    if(length>10)length=10;
    float roomScale  =(argc>5)?(float)atof(argv[5]):1.0f;
    if(roomScale<0.f)roomScale=0.f;
    if(roomScale>2.f)roomScale=2.f;
    float tremulant  =(argc>6)?(float)atof(argv[6]):0.6f;
    if(tremulant<0.f)tremulant=0.f;
    if(tremulant>1.f)tremulant=1.f;

    sr_seed(seed);srand(G_RNG);seed=G_RNG;

    printf("=== PIPEORGAN01 -- 10,000-oscillator GPU pipe organ ===\n");
    printf("SEED: %u  LENGTH: %d/10  ROOM: %.1f  TREMULANT: %.2f\n",
           seed,length,roomScale,tremulant);
    printf("Reproduce: pipeorgan01 out.wav %d %u %d %.1f %.2f\n\n",
           sr,seed,length,roomScale,tremulant);
    fflush(stdout);

    cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("GPU: %s (SM %d.%d, %d SMs)\n\n",
           prop.name,prop.major,prop.minor,prop.multiProcessorCount);
    fflush(stdout);

    /* ── verify group counts ── */
    {int t=0;for(int i=0;i<N_GROUPS;i++)t+=G_CNT[i];
     if(t!=N_OSC){fprintf(stderr,"G_CNT sum %d != %d\n",t,N_OSC);return 1;}}

    /* ── compose ── */
    printf("Composing...\n");fflush(stdout);
    int key=48+ri(0,11);
    float tempo=rr(108.f,124.f);     /* disco organ: strong groove tempo */
    float beat=60.f/tempo;
    EvBuf*EB=(EvBuf*)calloc(1,sizeof(EvBuf));
    char report[512];
    composeSong(*EB,key,beat,length,report);
    printf("%s\nEvents: %d   Tempo: %.0f BPM\n\n",report,EB->n,(double)tempo);
    fflush(stdout);

    /* ── timing ── */
    float endT=0.f;
    for(int i=0;i<EB->n;i++)endT=fmaxf(endT,EB->ev[i].tOff+8.f);
    endT=fminf(endT,MAX_SONG_S);
    int totalSamples=(int)(endT*(float)sr);
    printf("Duration: %.1f s  (%d samples)\n",endT,totalSamples);
    printf("Sin evals: %.1fB\n\n",(double)N_OSC*totalSamples/1e9);
    fflush(stdout);

    /* ── tremulant parameters ── */
    float tremRate=rr(5.2f,6.8f);   /* Hz: slightly different each run */

    /* ── build oscillator table ── */
    printf("Building oscillator table...\n");fflush(stdout);
    OscDesc*hOsc=(OscDesc*)calloc(PAD_OSC,sizeof(OscDesc));
    buildOscillators(hOsc,fmaxf(0.5f,roomScale));
    /* padding oscillators: sentinel, contribute silence */
    for(int i=N_OSC;i<PAD_OSC;i++){
        hOsc[i].freqRatio=100.f;hOsc[i].amp=0.f;
        hOsc[i].aA=0.01f;hOsc[i].aR=0.01f;
        hOsc[i].panL=0.5f;hOsc[i].panR=0.5f;
        hOsc[i].group=N_GROUPS;hOsc[i].harmonic=1;
    }

    /* ── index events ── */
    Event*sorted=(Event*)calloc(MAX_EVENTS,sizeof(Event));
    int evS[N_GROUPS+1],evC[N_GROUPS+1];
    indexEvents(*EB,evS,evC,sorted);

    /* ── persistent state ── */
    float*hPh=(float*)malloc(PAD_OSC*sizeof(float));
    int  *hEI=(int*)  malloc(PAD_OSC*sizeof(int));
    for(int i=0;i<PAD_OSC;i++){
        hPh[i]=hOsc[i].phase0;
        hEI[i]=evS[hOsc[i].group<N_GROUPS?hOsc[i].group:N_GROUPS];
    }

    /* ── GPU allocations ── */
    printf("Allocating GPU buffers (%.1f MB)...\n",
           (float)totalSamples*8/1e6f);fflush(stdout);
    OscDesc*dO;Event*dE;int*dS,*dC,*dEI;float*dPh,*dL,*dR;
    CUDA_CHECK(cudaMalloc(&dO,PAD_OSC*sizeof(OscDesc)));
    CUDA_CHECK(cudaMalloc(&dE,MAX_EVENTS*sizeof(Event)));
    CUDA_CHECK(cudaMalloc(&dS,(N_GROUPS+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dC,(N_GROUPS+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dPh,PAD_OSC*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dEI,PAD_OSC*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dL,(size_t)totalSamples*4));
    CUDA_CHECK(cudaMalloc(&dR,(size_t)totalSamples*4));
    CUDA_CHECK(cudaMemcpy(dO,hOsc,PAD_OSC*sizeof(OscDesc),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dE,sorted,MAX_EVENTS*sizeof(Event),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS,evS,(N_GROUPS+1)*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC,evC,(N_GROUPS+1)*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPh,hPh,PAD_OSC*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEI,hEI,PAD_OSC*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dL,0,(size_t)totalSamples*4));
    CUDA_CHECK(cudaMemset(dR,0,(size_t)totalSamples*4));

    /* ── chunked GPU render ── */
    int nBlocks=PAD_OSC/BLOCK;
    int nChunks=(totalSamples+CHUNK-1)/CHUNK;
    printf("Rendering: %d blocks x %d threads, %d chunks\n",
           nBlocks,BLOCK,nChunks);fflush(stdout);

    cudaEvent_t e0,e1;
    CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
    float msTotal=0.f;
    for(int c=0;c<nChunks;c++){
        int s0=c*CHUNK,s1=std::min(s0+CHUNK,totalSamples);
        CUDA_CHECK(cudaEventRecord(e0));
        organKernel<<<nBlocks,BLOCK>>>(
            dO,dE,dS,dC,dPh,dEI,dL,dR,s0,s1,(float)sr,
            tremRate,tremulant);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaGetLastError());
        float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
        msTotal+=ms;
        printf("  chunk %3d/%d  %.1fs  [%.0f ms total]\r",
               c+1,nChunks,(float)s1/sr,msTotal);
        fflush(stdout);
    }
    printf("\nGPU total: %.0f ms  (%.0fx real-time)\n\n",
           msTotal,endT*1000.f/msTotal);fflush(stdout);

    /* ── copy and post-process ── */
    float*hL=(float*)malloc((size_t)totalSamples*4);
    float*hR=(float*)malloc((size_t)totalSamples*4);
    CUDA_CHECK(cudaMemcpy(hL,dL,(size_t)totalSamples*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hR,dR,(size_t)totalSamples*4,cudaMemcpyDeviceToHost));
    /* — Post-processing ———————————————————————————————————————————————
       Reverb mode is set by #define USE_CONV_REVERB at the top of the
       FX section.  Only one reverb compiles and runs at a time.       */
    #if USE_CONV_REVERB
    printf("Post: Convolution reverb (synthesised cathedral IR, roomScale=%.1f)...\n",
           roomScale);
    fflush(stdout);
    {
        int irLen=(int)(fminf(fmaxf(roomScale,0.5f)*3.5f+1.5f, 5.0f)*(float)sr);
        printf("  IR length: %.2f s (%d samples)\n", (float)irLen/sr, irLen);
        fflush(stdout);
        float*ir=buildIR(irLen,(float)sr,roomScale,seed^0x1234ABCDu);

        /* verify IR has non-trivial energy */
        float irPeak=0.f;
        for(int i=0;i<irLen;i++) irPeak=fmaxf(irPeak,fabsf(ir[i]));
        printf("  IR peak amplitude: %.6f\n", irPeak);

        /* snapshot a few output samples before reverb */
        float pre0=hL[totalSamples/2];

        clock_t t0=clock();
        applyConvReverb(hL,hR,totalSamples,ir,irLen,0.90f,1.0f);
        clock_t t1=clock();
        float convMs=(float)(t1-t0)/CLOCKS_PER_SEC*1000.f;

        /* verify output changed */
        float post0=hL[totalSamples/2];
        printf("  Sample[mid] before: %.6f  after: %.6f  (changed: %s)\n",
               pre0, post0, (fabsf(post0-pre0)>1e-6f)?"YES":"NO - REVERB DID NOTHING");
        printf("  Convolution time: %.0f ms  (partitions=%d, blocks=%d)\n",
               convMs, (irLen+2047)/2048, (totalSamples+2047)/2048);
        fflush(stdout);
        free(ir);
    }
    #else
    printf("Post: Schroeder church reverb (roomScale=%.1f)...\n",roomScale);
    fflush(stdout);
    applyChurchReverb(hL,hR,totalSamples,(float)sr,roomScale);
    #endif
    printf("Post: gentle peak limiting...\n");fflush(stdout);
    softSat(hL,totalSamples,1.05f);   /* very gentle — mostly lets writeWav normalise */
    softSat(hR,totalSamples,1.05f);
    printf("Writing WAV...\n");fflush(stdout);
    writeWav(out,hL,hR,totalSamples,sr);

    free(hOsc);free(EB);free(sorted);free(hPh);free(hEI);free(hL);free(hR);
    cudaFree(dO);cudaFree(dE);cudaFree(dS);cudaFree(dC);
    cudaFree(dPh);cudaFree(dEI);cudaFree(dL);cudaFree(dR);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 0;
}
