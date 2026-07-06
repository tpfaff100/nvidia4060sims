/*
 * tank2.cu — GPU tank soundscape v2: stochastic clatter, not bell chimes
 * Target: RTX 4060 Ti (sm_89)
 *
 * BUILD (after vcvars64.bat):
 *   nvcc -arch=sm_89 -O3 -use_fast_math -o tank2 tank2.cu
 * RUN:
 *   tank2 [tank.wav] [duration_sec] [samplerate] [seed] [battle 0..2]
 *
 * WHY v1 FAILED (honest post-mortem):
 *   v1 track  = periodic sequence of pitched 5-mode struck plates
 *               -> sounded like rhythmic BELL CHIMES.
 *   v1 diesel = clean stack of 6 sinusoids -> sounded like an ORGAN.
 *   Real tank sound is NOISE-DOMINATED: a dense irregular metallic
 *   CLATTER (hundreds of overlapping micro-impacts/second from links,
 *   pins, sprocket teeth, return rollers, track slapping the hull) over
 *   a rough broadband diesel where the firing harmonics are buried in
 *   combustion noise. Tonal content is a faint skeleton under texture.
 *
 * v2 SYNTHESIS MODEL:
 *
 *  TRACK CLATTER   Sparse random IMPULSES (heavy-tailed amplitudes,
 *                  ~150-900/s scaling with speed) driving THREE ringing
 *                  metal RESONATORS (2-pole bandpass ~1.1k/2.6k/4.6kHz,
 *                  Q 25-45) + a raw high-passed click path.  The impulse
 *                  DENSITY is modulated at the link-lay rate
 *                  (speed/PITCH): at crawl you hear rhythmic CLUSTERS of
 *                  rattle (chunk..chunk..chunk); at speed the clusters
 *                  fuse into continuous metallic roar.  This is how the
 *                  real mechanism produces its sound: many tiny impacts,
 *                  rhythmically bunched, never a clean pitched strike.
 *
 *  TRACK SLAP      Sparser, larger impulses (loose track hitting return
 *                  rollers / hull) into a low 240 Hz resonator + thud LP.
 *                  Rate and weight grow with speed.
 *
 *  PIN SQUEAK      INTERMITTENT (not continuous): short gliding squeaks
 *                  gated on when a slow random threshold trips; mostly
 *                  at low speed. Faint.
 *
 *  DIESEL V12      Noise-first: combustion = lowpassed broadband noise
 *                  amplitude-modulated at the firing rate with strong
 *                  cycle-to-cycle roughness; plus one low exhaust-pulse
 *                  RESONATOR (~95 Hz) kicked by an impulse every firing;
 *                  plus exhaust puff noise; plus a FAINT (0.12) harmonic
 *                  bed so it reads pitched under load. RPM follows the
 *                  mission (idle 700 -> labouring 2200 on the climb).
 *
 *  FINAL DRIVE     Faint gear-mesh whine that scales with speed
 *                  (f ~ 180*v Hz) — the "speed" cue. Very subtle.
 *
 *  SAND            Granular band-passed crunch, speed-scaled.
 *  JOLTS           Rocks/berms: hull thud + burst of extra clatter.
 *
 * PROPAGATION unchanged from v1 (it was fine): retarded-time Doppler,
 * 1/r, air absorption via distance-dependent LP, ridge OCCLUSION
 * (drops behind the dune -> muffles, re-emerges), pan + ITD.
 * Battle layer (guns / MG / missiles / explosions) retained, boom-first.
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

#define SR_DEFAULT  48000
#define DUR_DEFAULT 90.0f
#define TWO_PI      6.28318530717958647692f
#define SEG         1024
#define WARM        4096        /* covers 5*tau of the 95 Hz resonator  */
#define SPEED_C     343.0f
#define PITCH       0.16f
#define ENG_CYL     12
#define MAX_KEY     4096
#define MAX_JOLT    64
#define MAX_BATTLE  128

#define CUDA_CHECK(x) do{ cudaError_t _=x; if(_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_)); \
    exit(1);} }while(0)

__device__ __forceinline__ float hnoise(unsigned i,unsigned s){
    unsigned h=i*0x9E3779B9u^s*0x85EBCA6Bu;
    h^=h>>13; h*=0xC2B2AE35u; h^=h>>16;
    return (float)(int)h*4.6566129e-10f;
}
__device__ __forceinline__ float huniform(unsigned i,unsigned s){
    return hnoise(i,s)*0.5f+0.5f;
}
__device__ __forceinline__ float smooth(float u){return u*u*(3.f-2.f*u);}
__device__ __forceinline__ float wander(float t,float rate,unsigned s){
    float x=t*rate; int k=(int)floorf(x); float u=smooth(x-(float)k);
    float a=hnoise((unsigned)k,s), b=hnoise((unsigned)(k+1),s);
    return a+(b-a)*u;
}
__device__ __forceinline__ float lp1s(float x,float&z,float c){
    z=z*c+x*(1.f-c); return z;
}
/* 2-pole resonator: metallic ring. coefs from (f,Q).                    */
struct Reso { float a1,a2,z1,z2; };
__device__ __forceinline__ void resoInit(Reso&R,float f,float Q,float sr){
    float w=TWO_PI*f/sr;
    float r=__expf(-w/(2.f*Q));
    R.a1=2.f*r*__cosf(w); R.a2=-r*r; R.z1=0;R.z2=0;
}
__device__ __forceinline__ float resoStep(Reso&R,float x){
    float y=x+R.a1*R.z1+R.a2*R.z2;
    R.z2=R.z1; R.z1=y;
    return y;
}

struct Key {
    float t,x,y,z,speed,rpm,occl,cumLink,cumEng;
};
__device__ void keyEval(const Key*__restrict__ k,int nk,float t,
    float&x,float&y,float&z,float&speed,float&rpm,float&occl,
    float&cumLink,float&cumEng)
{
    if(t<=k[0].t){const Key&a=k[0];x=a.x;y=a.y;z=a.z;speed=a.speed;
        rpm=a.rpm;occl=a.occl;cumLink=a.cumLink;cumEng=a.cumEng;return;}
    if(t>=k[nk-1].t){const Key&a=k[nk-1];x=a.x;y=a.y;z=a.z;speed=a.speed;
        rpm=a.rpm;occl=a.occl;cumLink=a.cumLink;cumEng=a.cumEng;return;}
    int lo=0,hi=nk-1;
    while(hi-lo>1){int m=(lo+hi)>>1; if(k[m].t<=t)lo=m;else hi=m;}
    const Key&a=k[lo];const Key&b=k[lo+1];
    float dt=b.t-a.t; if(dt<1e-6f)dt=1e-6f;
    float u=(t-a.t)/dt;
    x=a.x+(b.x-a.x)*u; y=a.y+(b.y-a.y)*u; z=a.z+(b.z-a.z)*u;
    speed=a.speed+(b.speed-a.speed)*u;
    rpm=a.rpm+(b.rpm-a.rpm)*u;
    occl=a.occl+(b.occl-a.occl)*u;
    cumLink=a.cumLink+(b.cumLink-a.cumLink)*u;
    cumEng =a.cumEng +(b.cumEng -a.cumEng )*u;
}

struct Jolt { float t,intensity; };

/* ═══════════════ TANK KERNEL (v2 source model) ══════════════════════ */
__global__ void tankKernel(
    const Key*__restrict__ keys,int nk,
    const Jolt*__restrict__ jolts,int nj,
    float*L,float*R,int N,float sr,unsigned seed)
{
    int segIdx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int segStart=segIdx*SEG; if(segStart>=N)return;
    int segEnd=min(segStart+SEG,N);
    float invSr=1.f/sr;

    /* — metal resonators (globally fixed detune from seed) —            */
    float d1=1.f+0.10f*hnoise(1,seed), d2=1.f+0.10f*hnoise(2,seed),
          d3=1.f+0.10f*hnoise(3,seed);
    Reso rm1,rm2,rm3,rSlap,rFire1,rFire2;
    resoInit(rm1,1150.f*d1,28.f,sr);
    resoInit(rm2,2600.f*d2,38.f,sr);
    resoInit(rm3,4600.f*d3,45.f,sr);
    resoInit(rSlap,240.f,9.f,sr);
    resoInit(rFire1,95.f,6.f,sr);
    resoInit(rFire2,160.f,7.f,sr);

    float zClick=0;                    /* click HP                        */
    float zThud=0;                     /* slap thud LP                    */
    float zComb1=0,zComb2=0;           /* combustion noise LP             */
    float zPuff=0;                     /* exhaust puff LP                 */
    float zSand1=0,zSand2=0,zSandH=0;
    float zOcc=0;
    float prevEngFrac=0.f;

    float cClick=__expf(-TWO_PI*2500.f*invSr);
    float cThud =__expf(-TWO_PI*160.f*invSr);
    float cPuff =__expf(-TWO_PI*420.f*invSr);
    float cSand =__expf(-TWO_PI*4200.f*invSr);
    float cSandH=__expf(-TWO_PI*900.f*invSr);

    for(int l=segStart-WARM;l<segEnd;l++){
        if(l<0)continue;
        float tg=(float)l*invSr;

        float x,y,z,speed,rpm,occl,cumLink,cumEng;
        keyEval(keys,nk,tg,x,y,z,speed,rpm,occl,cumLink,cumEng);
        float r0=__fsqrt_rn(x*x+y*y+z*z+1e-3f);
        float tau=tg-r0/SPEED_C;
        keyEval(keys,nk,tau,x,y,z,speed,rpm,occl,cumLink,cumEng);
        float r=__fsqrt_rn(x*x+y*y+z*z+1e-3f);

        float mix=0.f;
        float moving=fminf(1.f,speed/0.8f);

        /* jolts: extra clatter burst + hull thud                         */
        float joltBoost=0.f, joltThud=0.f;
        for(int ji=0;ji<nj;ji++){
            float ja=tau-jolts[ji].t;
            if(ja<0.f||ja>0.5f)continue;
            joltBoost+=jolts[ji].intensity*__expf(-ja*8.f);
            if(ja<0.05f) joltThud+=jolts[ji].intensity*__expf(-ja*60.f);
        }

        /* ── TRACK CLATTER: modulated impulsive noise -> metal resos ── */
        {
            /* impulse probability per sample: scales with link rate,
               bunched at link-lay phase (rhythmic at crawl)             */
            float linkFrac=cumLink-floorf(cumLink);
            float bunch=0.30f+0.70f*__expf(-8.f*fminf(linkFrac,1.f-linkFrac)
                                            *fminf(linkFrac,1.f-linkFrac)*4.f);
            float linkRate=speed/PITCH;
            float density=(30.f+95.f*linkRate)*(1.f+2.5f*joltBoost);
            float p=density*invSr*bunch*moving;

            float u=huniform((unsigned)l,seed^0xC1A7);
            float imp=0.f;
            if(u<p){
                float a=hnoise((unsigned)l,seed^0x5EED);
                imp=a*a*a*8.f;               /* heavy-tailed              */
            }
            float m1=resoStep(rm1,imp);
            float m2=resoStep(rm2,imp*0.8f);
            float m3=resoStep(rm3,imp*0.6f);
            float click=imp-lp1s(imp,zClick,cClick);
            mix+=(m1*0.7f+m2*0.55f+m3*0.4f+click*1.2f)*0.85f;
        }

        /* ── TRACK SLAP: sparse heavy impulses -> low reso + thud ────── */
        {
            float slapRate=(1.5f+2.5f*speed)*(1.f+2.f*joltBoost);
            float p=slapRate*invSr*moving;
            float u=huniform((unsigned)(l*7+3),seed^0x51AB);
            float imp=(u<p)? hnoise((unsigned)l,seed^0x7777)*3.5f:0.f;
            imp+=joltThud*hnoise((unsigned)l,seed^0x8888)*2.f;
            float lowring=resoStep(rSlap,imp);
            float thud=lp1s(imp,zThud,cThud);
            mix+=(lowring*0.5f+thud*1.4f);
        }

        /* ── PIN SQUEAK: intermittent gliding squeaks, low speed only ── */
        {
            float gate=wander(tau,0.5f,seed^0x99);
            if(gate>0.55f && speed<4.f && speed>0.2f){
                float g=(gate-0.55f)/0.45f;
                float f=1900.f+700.f*wander(tau,2.5f,seed^0xAA);
                float ph=f*tau; ph-=floorf(ph);
                mix+=__sinf(TWO_PI*ph)*g*g*0.12f;
            }
        }

        /* ── DIESEL: noise-first combustion + exhaust-pulse resonator ─ */
        {
            float fireFrac=cumEng-floorf(cumEng);
            /* firing impulse on phase wrap                               */
            float fireImp=0.f;
            if(fireFrac<prevEngFrac){
                float cyc=hnoise((unsigned)floorf(cumEng),seed^0xD1E5);
                fireImp=(1.f+0.6f*cyc)*2.2f;   /* cycle-to-cycle rough   */
            }
            prevEngFrac=fireFrac;
            float ex1=resoStep(rFire1,fireImp);
            float ex2=resoStep(rFire2,fireImp*0.6f);

            /* combustion noise, AM at firing rate + roughness            */
            float n=hnoise((unsigned)(l*5+1),seed^0xC0DE);
            float rc=__expf(-TWO_PI*(240.f+0.09f*rpm)*invSr);
            float comb=lp1s(n,zComb1,rc); comb=lp1s(comb,zComb2,rc);
            float am=0.65f+0.35f*__cosf(TWO_PI*fireFrac);
            float rough=0.7f+0.5f*wander(tau,rpm/60.f*3.f,seed^0xF0);
            comb*=am*rough;

            /* exhaust puffs                                              */
            float puffN=hnoise((unsigned)(l*3+2),seed^0xE0);
            float puff=lp1s(puffN,zPuff,cPuff)
                      *(0.4f+0.6f*__expf(-fireFrac*6.f));

            /* faint pitched skeleton                                     */
            float fireHz=rpm/60.f*(ENG_CYL*0.5f);
            float p1=cumEng; p1-=floorf(p1);
            float p2=cumEng*2.f; p2-=floorf(p2);
            float tone=0.7f*__sinf(TWO_PI*p1)+0.4f*__sinf(TWO_PI*p2);

            float load=fminf(1.6f,rpm/1400.f);
            mix+=(ex1*1.1f+ex2*0.6f+comb*2.6f+puff*0.8f+tone*0.12f)
                 *0.55f*load;
            (void)fireHz;
        }

        /* ── FINAL-DRIVE WHINE: faint speed cue ───────────────────────*/
        {
            float f=180.f*speed;
            if(f>60.f){
                float ph=f*tau; ph-=floorf(ph);
                mix+=__sinf(TWO_PI*ph)*0.03f*moving;
            }
        }

        /* ── SAND ────────────────────────────────────────────────────── */
        {
            float n=hnoise((unsigned)(l*11+5),seed^0x9E9E);
            float s=lp1s(n,zSand1,cSand); s=lp1s(s,zSand2,cSand);
            s=s-lp1s(s,zSandH,cSandH);
            unsigned g=(unsigned)l*0x27d4eb2du^seed;
            g^=g>>15; float gr=0.4f+0.6f*((float)(g&0xFFFF)/65535.f);
            mix+=s*gr*0.5f*fminf(1.f,speed/3.f);
        }

        /* ── propagation ─────────────────────────────────────────────── */
        float amp=1.f/fmaxf(r,2.5f);
        float s=mix*amp;
        /* distance air absorption + occlusion as one LP                  */
        float cutHz=fmaxf(300.f, 9000.f*__expf(-r/220.f))*(1.f-0.85f*occl)
                    +250.f;
        float cOc=__expf(-TWO_PI*cutHz*invSr);
        float sLP=lp1s(s,zOcc,cOc);
        float wet=fminf(1.f,occl+r/400.f);
        s=s*(1.f-wet)+sLP*wet;
        s*=(1.f-0.7f*occl);

        if(l>=segStart){
            float sinAz=x/__fsqrt_rn(x*x+z*z+1e-3f);
            sinAz=fmaxf(-1.f,fminf(1.f,sinAz));
            float pa=(sinAz+1.f)*0.25f*(float)M_PI;
            float gL=__cosf(pa),gR=__sinf(pa);
            int itd=(int)(fabsf(sinAz)*0.00066f*sr);
            int iL=l,iR=l;
            if(sinAz<0.f)iR=min(N-1,l+itd); else iL=min(N-1,l+itd);
            atomicAdd(&L[iL],s*gL*0.7f);
            atomicAdd(&R[iR],s*gR*0.7f);
        }
    }
}

/* ═══════════════ BATTLE KERNEL (boom-first, from v1 with fixes) ═════ */
enum { B_MAINGUN=0, B_MG, B_MISSILE, B_EXPLOSION };
struct Battle { int type; float t,az,dist,energy; unsigned seed; };

__global__ void battleKernel(
    const Battle*__restrict__ ev,int nev,
    float*L,float*R,int N,float sr)
{
    int bi=(int)blockIdx.y; if(bi>=nev)return;
    const Battle b=ev[bi];
    int segIdx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int segStart=segIdx*SEG;
    float winLen=(b.type==B_MAINGUN)?4.f:(b.type==B_MG)?3.f:
                 (b.type==B_MISSILE)?4.5f:6.f;
    int winSamps=(int)(winLen*sr);
    if(segStart>=winSamps)return;
    int segEnd=min(segStart+SEG,winSamps);
    int base=(int)(b.t*sr);
    float invSr=1.f/sr;
    float dkm=b.dist/1000.f;
    float loCut=fmaxf(45.f,900.f-90.f*dkm);
    float cLo=__expf(-TWO_PI*loCut*invSr);
    float cLo2=__expf(-TWO_PI*loCut*1.6f*invSr);
    float cHi=__expf(-TWO_PI*30.f*invSr);
    float z1=0,z2=0,zh=0,zc=0;
    float pa=(b.az+1.f)*0.25f*(float)M_PI;
    float gL=__cosf(pa),gR=__sinf(pa);
    float spread=1.f/fmaxf(b.dist,8.f);

    for(int l=segStart-WARM;l<segEnd;l++){
        float t=(float)l*invSr;
        if(t<0.f)continue;
        float v=0.f;

        if(b.type==B_MAINGUN){
            float n=hnoise((unsigned)l,b.seed);
            float crack=(t<0.02f&&dkm<1.f)? n*(1.f-t/0.02f)*3.5f*(1.f-dkm):0.f;
            float boomEnv=__expf(-t/0.5f)*(1.f-__expf(-t/0.004f));
            float boom=lp1s(n,z1,cLo); boom=lp1s(boom,z2,cLo);
            boom=(boom-lp1s(boom,zh,cHi))*boomEnv*3.2f;
            float tailEnv=__expf(-t/1.6f)*fmaxf(0.f,1.f-__expf(-(t-0.1f)/0.2f));
            float tail=lp1s(n,zc,cLo2)*tailEnv*1.3f
                       *(0.7f+0.3f*wander(t,7.f,b.seed^0x55));
            v=crack+boom+tail;
        }
        else if(b.type==B_MG){
            int nShots=12+(int)(b.energy*28.f);
            float rate=11.f+b.energy*5.f;
            float shot=0.f;
            for(int sIdx=0;sIdx<nShots;sIdx++){
                float st=(float)sIdx/rate+0.02f*hnoise((unsigned)sIdx,b.seed);
                float sa=t-st;
                if(sa>0.f&&sa<0.05f){
                    float n=hnoise((unsigned)(l+sIdx*7),b.seed^0x33);
                    shot+=n*__expf(-sa/0.008f)*1.4f;
                }
            }
            float body=lp1s(shot,z1,cLo);
            v=shot*0.6f+body*0.8f;
        }
        else if(b.type==B_MISSILE){
            float ign=0.6f;
            float n=hnoise((unsigned)l,b.seed);
            float pre=(t<ign)?(t/ign)*(t/ign):1.f;
            float roarEnv=(t<ign)?pre*0.5f:__expf(-(t-ign)/2.5f);
            float roar=lp1s(n,z1,cLo);roar=lp1s(roar,z2,cLo2);
            float hp=1200.f+2600.f*pre;
            float hph=hp*t;hph-=floorf(hph);
            v=roar*roarEnv*3.4f+0.13f*roarEnv*__sinf(TWO_PI*hph);
        }
        else{
            float n=hnoise((unsigned)l,b.seed);
            float nearF=fmaxf(0.f,1.f-dkm/3.f);
            float crack=(t<0.03f)?n*nearF*3.f*(1.f-t/0.03f):0.f;
            float boomEnv=__expf(-t/0.7f)*(1.f-__expf(-t/0.005f));
            float boom=lp1s(n,z1,cLo);boom=lp1s(boom,z2,cLo);
            boom=(boom-lp1s(boom,zh,cHi))*boomEnv*3.6f;
            float rumbleEnv=__expf(-t/2.4f);
            float rumble=lp1s(n,zc,cLo2)*rumbleEnv*1.5f
                        *(0.6f+0.4f*wander(t,5.f,b.seed^0x77));
            float deb=0.f;
            if(t>0.15f&&t<2.5f){
                unsigned d=(unsigned)l*0x2545F491u^b.seed;
                d^=d>>13;
                if((d&0x7FF)==0)deb=((float)(int)d*4.66e-10f)*nearF*2.f;
            }
            v=crack+boom+rumble+deb;
        }

        if(l>=segStart){
            int g=base+l; if(g<0||g>=N)continue;
            float s=v*spread*b.energy*22.f;
            atomicAdd(&L[g],s*gL);
            atomicAdd(&R[g],s*gR);
        }
    }
}

/* ═══════════════ DESERT WIND ════════════════════════════════════════ */
__global__ void ambienceKernel(float*L,float*R,int N,float sr,unsigned seed){
    int ch=(int)blockIdx.y;
    int segIdx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int segStart=segIdx*SEG; if(segStart>=N)return;
    int segEnd=min(segStart+SEG,N);
    unsigned salt=seed^(ch?0x1111u:0x2222u);
    float invSr=1.f/sr,z1=0,z2=0;
    for(int l=segStart-WARM;l<segEnd;l++){
        if(l<0)continue;
        float t=(float)l*invSr;
        float n=hnoise((unsigned)l,salt);
        float cut=120.f+220.f*wander(t,0.12f,salt^0x9);
        float c=__expf(-TWO_PI*cut*invSr);
        float v=lp1s(n,z1,c);v=lp1s(v,z2,c);
        float g=0.5f+0.5f*wander(t,0.07f,salt^0xA);
        if(l>=segStart){
            float o=v*g*0.05f;
            if(ch==0)atomicAdd(&L[l],o);else atomicAdd(&R[l],o);
        }
    }
}

/* ── CPU: trajectory & battle builders (same mission as v1) ──────────*/
static float frand(){return (float)rand()/(float)RAND_MAX;}
static float frange(float a,float b){return a+(b-a)*frand();}
static float clampf(float x,float a,float b){return x<a?a:x>b?b:x;}

struct Node { float t,x,z,speed,rpm,occl; };

static int buildTank(Key*keys,Jolt*jolts,float dur,int*nk_out,int*nj_out)
{
    Node nd[10]; int nn=0;
    float startSide=(frand()<0.5f)?-1.f:1.f;
    float x=startSide*frange(35.f,55.f), z=frange(45.f,75.f);
    nd[nn++]={0.f,x,z,0.f,700.f,0.f};
    nd[nn++]={frange(6.f,10.f),x*0.6f,z*0.7f,frange(4.f,6.f),1500.f,0.f};
    float passX=frange(-12.f,12.f),passZ=frange(10.f,22.f);
    nd[nn++]={dur*0.34f,passX,passZ,frange(5.f,7.f),1700.f,0.f};
    float rs=(frand()<0.5f)?-1.f:1.f;
    nd[nn++]={dur*0.52f,rs*frange(20.f,30.f),frange(30.f,45.f),
              frange(2.5f,3.5f),2150.f,0.15f};
    nd[nn++]={dur*0.62f,rs*frange(35.f,48.f),frange(45.f,60.f),
              frange(1.8f,2.6f),2300.f,0.85f};
    nd[nn++]={dur*0.72f,rs*frange(45.f,60.f),frange(60.f,80.f),
              frange(4.f,6.f),1400.f,0.55f};
    nd[nn++]={dur*0.85f,rs*frange(55.f,75.f),frange(70.f,95.f),
              frange(5.f,7.f),1650.f,0.35f};
    nd[nn++]={dur*0.96f,rs*frange(65.f,90.f),frange(85.f,110.f),
              frange(0.5f,1.2f),900.f,0.25f};
    nd[nn++]={dur,rs*frange(65.f,92.f),frange(88.f,115.f),0.f,700.f,0.25f};

    float dt=0.04f;
    int nk=0; float cumLink=0.f,cumEng=0.f;
    float prevLR=0.f,prevF=(700.f/60.f)*(ENG_CYL*0.5f);
    for(float t=0.f;t<=dur+1e-3f&&nk<MAX_KEY;t+=dt){
        int i=0;
        for(int j=1;j<nn;j++){if(nd[j].t>t){i=j-1;break;}i=nn-2;}
        float sd=nd[i+1].t-nd[i].t; if(sd<1e-4f)sd=1e-4f;
        float u=clampf((t-nd[i].t)/sd,0.f,1.f);
        float us=u*u*(3.f-2.f*u);
        float px=nd[i].x+(nd[i+1].x-nd[i].x)*us;
        float pz=nd[i].z+(nd[i+1].z-nd[i].z)*us;
        float sp=nd[i].speed+(nd[i+1].speed-nd[i].speed)*us;
        float rp=nd[i].rpm+(nd[i+1].rpm-nd[i].rpm)*us;
        float oc=nd[i].occl+(nd[i+1].occl-nd[i].occl)*us;
        sp*=1.f+0.08f*sinf(t*3.7f)+0.04f*sinf(t*11.f);
        sp=fmaxf(0.f,sp);
        float lr=sp/PITCH, fr=rp/60.f*(ENG_CYL*0.5f);
        cumLink+=0.5f*(prevLR+lr)*dt; cumEng+=0.5f*(prevF+fr)*dt;
        prevLR=lr;prevF=fr;
        keys[nk++]={t,px,0.f,pz,sp,rp,oc,cumLink,cumEng};
    }

    int nj=0;
    int nR=6+(int)(frand()*10.f);
    for(int i=0;i<nR&&nj<MAX_JOLT;i++){
        float jt=frange(dur*0.08f,dur*0.9f);
        int ki=(int)(jt/dt); if(ki>=nk)ki=nk-1;
        if(keys[ki].speed<0.6f)continue;
        jolts[nj++]={jt,frange(0.4f,1.2f)*fminf(1.5f,keys[ki].speed/4.f)};
    }
    *nk_out=nk;*nj_out=nj;
    return nk;
}

static int buildBattle(Battle*ev,float dur,float inten)
{
    if(inten<=0.f)return 0;
    int n=0;
    int nGun=(int)(3*inten)+(int)(frand()*3.f);
    for(int i=0;i<nGun&&n<MAX_BATTLE;i++)
        ev[n++]={B_MAINGUN,frange(3.f,dur-4.f),frange(-0.9f,0.9f),
                 frange(120.f,900.f),frange(0.7f,1.2f)*inten,(unsigned)rand()};
    int nMG=(int)(4*inten)+(int)(frand()*4.f);
    for(int i=0;i<nMG&&n<MAX_BATTLE;i++)
        ev[n++]={B_MG,frange(2.f,dur-3.f),frange(-1.f,1.f),
                 frange(80.f,600.f),frange(0.5f,1.f)*inten,(unsigned)rand()};
    int nMis=(int)(1.5f*inten)+(int)(frand()*2.f);
    for(int i=0;i<nMis&&n<MAX_BATTLE;i++)
        ev[n++]={B_MISSILE,frange(5.f,dur-5.f),frange(-1.f,1.f),
                 frange(150.f,700.f),frange(0.7f,1.1f)*inten,(unsigned)rand()};
    int nEx=(int)(5*inten)+(int)(frand()*4.f);
    for(int i=0;i<nEx&&n<MAX_BATTLE;i++)
        ev[n++]={B_EXPLOSION,frange(3.f,dur-6.f),frange(-1.f,1.f),
                 frange(200.f,2500.f),frange(0.8f,1.4f)*inten,(unsigned)rand()};
    return n;
}

static void softSat(float*x,int N,float d){for(int i=0;i<N;i++)x[i]=tanhf(x[i]*d)/d;}
static void writeWav(const char*p,const float*L,const float*R,int N,int sr){
    FILE*f=fopen(p,"wb");if(!f){fprintf(stderr,"open %s fail\n",p);return;}
    int16_t*b=(int16_t*)malloc((size_t)N*4);
    float pk=1e-9f;
    for(int i=0;i<N;i++){pk=fmaxf(pk,fabsf(L[i]));pk=fmaxf(pk,fabsf(R[i]));}
    float g=0.95f/pk;
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
    const char*out=(argc>1)?argv[1]:"tank.wav";
    float dur=(argc>2)?(float)atof(argv[2]):DUR_DEFAULT;
    int sr=(argc>3)?atoi(argv[3]):SR_DEFAULT;
    unsigned seed=(argc>4)?(unsigned)atoi(argv[4]):(unsigned)time(NULL);
    if(argc>4&&atoi(argv[4])==0)seed=(unsigned)time(NULL);
    float battle=(argc>5)?(float)atof(argv[5]):1.0f;

    printf("GPU Tank v2 (stochastic clatter) | %.0f s | battle=%.1f\n",dur,battle);
    printf("SEED: %u  (rerun 'tank2 out.wav %.0f %d %u %.1f')\n\n",
           seed,dur,sr,seed,battle);
    cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("Device: %s (SM %d.%d)\n\n",prop.name,prop.major,prop.minor);

    srand(seed);
    int N=(int)((dur+6.f)*sr);

    Key*hk=(Key*)calloc(MAX_KEY,sizeof(Key));
    Jolt*hj=(Jolt*)calloc(MAX_JOLT,sizeof(Jolt));
    int nk,nj;buildTank(hk,hj,dur,&nk,&nj);
    Battle*hb=(Battle*)calloc(MAX_BATTLE,sizeof(Battle));
    int nb=buildBattle(hb,dur,battle);
    printf("Keyframes: %d  Jolts: %d  Battle events: %d\n\n",nk,nj,nb);

    Key*dk;Jolt*dj;Battle*db;float*dL,*dR;
    CUDA_CHECK(cudaMalloc(&dk,MAX_KEY*sizeof(Key)));
    CUDA_CHECK(cudaMalloc(&dj,MAX_JOLT*sizeof(Jolt)));
    CUDA_CHECK(cudaMalloc(&db,MAX_BATTLE*sizeof(Battle)));
    CUDA_CHECK(cudaMalloc(&dL,(size_t)N*4));
    CUDA_CHECK(cudaMalloc(&dR,(size_t)N*4));
    CUDA_CHECK(cudaMemcpy(dk,hk,MAX_KEY*sizeof(Key),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dj,hj,MAX_JOLT*sizeof(Jolt),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,hb,MAX_BATTLE*sizeof(Battle),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dL,0,(size_t)N*4));
    CUDA_CHECK(cudaMemset(dR,0,(size_t)N*4));

    int segs=(N+SEG-1)/SEG;
    cudaEvent_t e0,e1;CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    dim3 tG((segs+63)/64,1);
    tankKernel<<<tG,64>>>(dk,nk,dj,nj,dL,dR,N,(float)sr,seed);
    if(nb>0){
        int bSegs=((int)(6.f*sr)+SEG-1)/SEG;
        dim3 bG((bSegs+63)/64,nb);
        battleKernel<<<bG,64>>>(db,nb,dL,dR,N,(float)sr);
    }
    dim3 aG((segs+63)/64,2);
    ambienceKernel<<<aG,64>>>(dL,dR,N,(float)sr,seed);
    CUDA_CHECK(cudaEventRecord(e1));CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    printf("Kernels: %.0f ms (%.0fx real-time)\n\n",ms,dur*1000.f/ms);

    float*hL=(float*)malloc((size_t)N*4);
    float*hR=(float*)malloc((size_t)N*4);
    CUDA_CHECK(cudaMemcpy(hL,dL,(size_t)N*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hR,dR,(size_t)N*4,cudaMemcpyDeviceToHost));
    printf("Post: saturation...\n");
    softSat(hL,N,1.3f);softSat(hR,N,1.3f);
    writeWav(out,hL,hR,N,sr);

    free(hk);free(hj);free(hb);free(hL);free(hR);
    cudaFree(dk);cudaFree(dj);cudaFree(db);cudaFree(dL);cudaFree(dR);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 0;
}
