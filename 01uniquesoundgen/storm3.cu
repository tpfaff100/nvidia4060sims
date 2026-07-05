/*
 * storm3.cu — GPU thunderstorm v3: REAL filtered noise, audible thunder,
 *             randomized every run, four quadrant cells.
 *
 * BUILD (after vcvars64.bat):
 *   nvcc -arch=sm_89 -O3 -use_fast_math -o storm3 storm3.cu
 * RUN:
 *   storm3 [storm.wav] [samplerate] [seed]
 *     - no seed given -> seeded from the clock: DIFFERENT STORM EVERY RUN
 *     - the seed is printed, pass it back in to reproduce a storm you liked
 *
 * WHY v2 HAD NO THUNDER (post-mortem):
 *   v2 "approximated" filters by multiplying noise with (1-coeff) factors.
 *   For the rumble band that evaluates to ~0.0005 = -66 dB. The strikes
 *   were rendered essentially at zero. This version runs REAL one-pole
 *   filter cascades with per-thread state.
 *
 * HOW STATEFUL FILTERS WORK ON GPU (segment + warm-up):
 *   IIR filters are sequential, but their transients die out. Each thread
 *   owns a 1024-sample output segment and starts its filter 1536 samples
 *   earlier with zero state, discarding the warm-up. Noise is a stateless
 *   hash of the absolute sample index, so overlapping threads see the
 *   exact same input stream -> segments join seamlessly.
 *
 * THE FOUR QUADRANTS (stereo can't render true rear, so rear cells are
 * cued the way the ear infers "behind": duller, more distant, softer
 * onsets — pinna filtering removes HF from rear sources):
 *   A  FRONT-LEFT   az -50   0.5-1.0 km  bright tearing cracks
 *   B  FRONT-RIGHT  az +50   ~2.5 km     classic crack+roll
 *   C  REAR-LEFT    az -75   6-9 km      dark delayed rumble (rear-cued)
 *   D  REAR-RIGHT   az +65   9 -> 1 km   APPROACHES over 5 min, climax 4:45
 *
 * THUNDER RECIPE (per strike, all filtered noise):
 *   crack   noise -> swept bandpass (4 kHz falling to ~900 Hz in 120 ms),
 *           instant attack; 1-3 flicker re-strikes; near cells only
 *   body    noise -> 2-pole LP 90-260 Hz, 20 ms attack, ~1.2 s decay
 *   rumble  noise -> 2-pole LP with WANDERING cutoff (28..distance-cap Hz),
 *           multiplied by a wandering "peal" envelope -> rolling peals;
 *           starts 50-300 ms after the crack; length grows with distance
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

#define SR_DEFAULT 48000
#define DUR_SEC    300.0f
#define TWO_PI     6.28318530717958647692f
#define SEG        1024      /* samples owned per thread                */
#define WARM       1536      /* filter warm-up discarded                */
#define MAX_STRIKES 96

#define CUDA_CHECK(x) do{ cudaError_t _=x; if(_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_)); \
    exit(1);} }while(0)

/* ── stateless hash noise: same value for same (index,salt) always ─── */
__device__ __forceinline__ float hnoise(unsigned i, unsigned salt){
    unsigned h = i*0x9E3779B9u ^ salt*0x85EBCA6Bu;
    h ^= h>>13; h *= 0xC2B2AE35u; h ^= h>>16;
    return (float)(int)h * 4.6566129e-10f;          /* -1..1 */
}
/* smooth wandering control curve, 0..1                                 */
__device__ __forceinline__ float wander(float t, float rate, unsigned salt){
    float x=t*rate; float fk=floorf(x); int k=(int)fk; float u=x-fk;
    u=u*u*(3.f-2.f*u);
    float a=hnoise((unsigned)k,salt)*0.5f+0.5f;
    float b=hnoise((unsigned)(k+1),salt)*0.5f+0.5f;
    return a+(b-a)*u;
}
__device__ __forceinline__ float lp1s(float x, float&z, float c){
    z = z*c + x*(1.f-c); return z;
}
__device__ __forceinline__ void panLR(float azRad, float&gL, float&gR){
    float x=(__sinf(azRad)+1.f)*0.25f*(float)M_PI;   /* const-power */
    gL=__cosf(x); gR=__sinf(x);
}

struct Strike {
    float tStart, az, distKm, energy;
    float dur;             /* total sounding time                       */
    float crackAmp, crackDur;
    float bodyAmp,  bodyLpHz;
    float rumbleAmp, rumbleDur, rumbleDelay, rumbleLpMax;
    float attackSoft;      /* distant strikes: slower onsets            */
    int   itd;             /* far-ear delay in samples                  */
    int   rear;            /* 1 = rear-cued (extra dullness)            */
    unsigned seed;
};

/* ═════════════════════ STRIKES ══════════════════════════════════════
   grid: ( ceil(strikeSamples/SEG) , nStrikes )  block: 1 warp x 8? ->
   simpler: block=64 threads, each thread one segment.
   thread's segment index = blockIdx.x*blockDim.x + threadIdx.x
   ═════════════════════════════════════════════════════════════════════ */
__global__ void strikesKernel(
    const Strike*__restrict__ st, int nSt,
    float*L, float*R, int N, float sr)
{
    int si=(int)blockIdx.y; if(si>=nSt) return;
    const Strike s=st[si];

    int strikeSamps=(int)(s.dur*sr);
    int segIdx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int segStart=segIdx*SEG;
    if(segStart>=strikeSamps) return;
    int segEnd=min(segStart+SEG, strikeSamps);
    int base=(int)(s.tStart*sr);

    /* filter states (per thread, warmed up)                            */
    float zc1=0,zc2=0,zch=0;          /* crack: LP,LP,HPstate           */
    float zb1=0,zb2=0;                /* body                           */
    float zr1=0,zr2=0;                /* rumble                         */

    /* flicker re-strike times (deterministic per strike)               */
    float f1=0.05f+0.14f*(hnoise(1,s.seed)*0.5f+0.5f);
    float f2=f1+0.07f+0.18f*(hnoise(2,s.seed)*0.5f+0.5f);

    float invSr=1.f/sr;
    float cBody=__expf(-TWO_PI*s.bodyLpHz*invSr);

    for(int l=segStart-WARM; l<segEnd; l++){
        if(l<0) continue;
        float t=(float)l*invSr;
        float n=hnoise((unsigned)l, s.seed);

        float outL=0.f, outR=0.f;

        /* ── CRACK: swept bandpass, instant attack, flicker ───────── */
        if(s.crackAmp>0.f){
            float sweep=900.f+3300.f*__expf(-t*9.f);
            float cHi=__expf(-TWO_PI*(sweep*1.6f)*invSr);
            float cLo=__expf(-TWO_PI*(sweep*0.5f)*invSr);
            float v=lp1s(n,zc1,cHi); v=lp1s(v,zc2,cHi);   /* 2p LP    */
            v=v-lp1s(v,zch,cLo);                           /* HP       */
            float tc=fmaxf(s.crackDur*0.30f,0.02f);
            float env=__expf(-t/tc)
                    +0.55f*((t>f1)?__expf(-(t-f1)/tc):0.f)
                    +0.30f*((t>f2)?__expf(-(t-f2)/tc):0.f);
            float c=v*env*s.crackAmp*3.2f;
            outL+=c; outR+=c;
        }

        /* ── BODY: the boom ───────────────────────────────────────── */
        {
            float v=lp1s(n,zb1,cBody); v=lp1s(v,zb2,cBody);
            float env=(1.f-__expf(-t/(0.02f+s.attackSoft)))
                     *__expf(-t/1.25f);
            float b=v*env*s.bodyAmp*7.f;
            outL+=b; outR+=b;
        }

        /* ── RUMBLE: rolling peals, wandering LP + peal envelope ──── */
        {
            float tr=t-s.rumbleDelay;
            if(tr>0.f){
                float cut=28.f+(s.rumbleLpMax-28.f)*wander(tr,0.9f,s.seed^0xAAAA);
                float cR=__expf(-TWO_PI*cut*invSr);
                float v=lp1s(n,zr1,cR); v=lp1s(v,zr2,cR);
                float peal=wander(tr,1.4f,s.seed^0x5555);
                peal*=peal;
                float env=(1.f-__expf(-tr/(0.12f+s.attackSoft*2.f)))
                         *__expf(-tr/(s.rumbleDur*0.45f))
                         *(0.35f+0.65f*peal);
                /* slight interaural decorrelation -> enveloping rumble */
                float nE=hnoise((unsigned)l,s.seed^0x1234)*0.3f;
                float r=v*env*s.rumbleAmp*9.f;
                outL+=r*(1.f+nE); outR+=r*(1.f-nE);
            }
        }

        if(l>=segStart){
            float gL,gR; panLR(s.az,gL,gR);
            /* rear cue: extra HF dullness handled in builder via
               crackAmp/rumbleLpMax; here rear just gets softer pan     */
            int g=base+l; if(g>=N) break;
            int gLidx=g, gRidx=g;
            if(s.az<0.f) gRidx=min(N-1,g+s.itd);
            else         gLidx=min(N-1,g+s.itd);
            atomicAdd(&L[gLidx], outL*gL*s.energy);
            atomicAdd(&R[gRidx], outR*gR*s.energy);
        }
    }
}

/* ═════════════════════ RAIN ═════════════════════════════════════════
   Per-channel independent streams: hiss band (HP300->LP6500) + droplet
   ticks (impulses through 2-5 kHz band) + low wash.
   grid: ( ceil(N/SEG/64), 2 )  block 64
   ═════════════════════════════════════════════════════════════════════ */
__global__ void rainKernel(
    float*L, float*R, int N, float sr,
    const float*__restrict__ env, unsigned seed)
{
    int ch=(int)blockIdx.y;
    int segIdx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int segStart=segIdx*SEG;
    if(segStart>=N) return;
    int segEnd=min(segStart+SEG,N);

    unsigned salt=seed ^ (ch? 0xBEEF5EEDu : 0xFACEFEEDu);
    float invSr=1.f/sr;
    float cLo=__expf(-TWO_PI*6500.f*invSr);
    float cHi=__expf(-TWO_PI*300.f*invSr);
    float cTk=__expf(-TWO_PI*3400.f*invSr);
    float cTkH=__expf(-TWO_PI*1500.f*invSr);
    float cW=__expf(-TWO_PI*260.f*invSr);

    float z1=0,z2=0,zh=0, t1=0,t2=0,th=0, zw=0;

    for(int l=segStart-WARM;l<segEnd;l++){
        if(l<0) continue;
        float n=hnoise((unsigned)l,salt);

        /* hiss band                                                    */
        float h=lp1s(n,z1,cLo); h=lp1s(h,z2,cLo);
        h=h-lp1s(h,zh,cHi);

        /* droplet ticks: rare hash-triggered impulses through a band   */
        unsigned u=(unsigned)l*0x27d4eb2du ^ salt;
        u^=u>>15; u*=0x2545F491u; u^=u>>13;
        float imp=((u&0x3FF)==0)? ((float)(int)u*4.66e-10f)*8.f : 0.f;
        float tk=lp1s(imp,t1,cTk); tk=lp1s(tk,t2,cTk);
        tk=tk-lp1s(tk,th,cTkH);

        /* ground wash                                                  */
        float w=lp1s(n,zw,cW);

        if(l>=segStart){
            float g=env[l];
            float v=(h*0.55f+tk*2.2f+w*0.35f)*g*0.55f;
            if(ch==0) atomicAdd(&L[l],v);
            else      atomicAdd(&R[l],v);
        }
    }
}

/* ═════════════════════ WIND ═════════════════════════════════════════
   4 directional cells: LP'd noise, wandering cutoff, gust gain table.
   grid: ( ceil(N/SEG/64), 4 )
   ═════════════════════════════════════════════════════════════════════ */
__global__ void windKernel(
    float*L, float*R, int N, float sr,
    const float*__restrict__ gain,   /* [4*N]  */
    const float*__restrict__ azArr,  /* [4]    */
    unsigned seed)
{
    int cell=(int)blockIdx.y; if(cell>=4) return;
    int segIdx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int segStart=segIdx*SEG;
    if(segStart>=N) return;
    int segEnd=min(segStart+SEG,N);

    unsigned salt=seed ^ (0xC0FFEEu+(unsigned)cell*0x9E37u);
    float invSr=1.f/sr;
    float z1=0,z2=0;
    float az=azArr[cell];
    float gL,gR; panLR(az,gL,gR);

    for(int l=segStart-WARM;l<segEnd;l++){
        if(l<0) continue;
        float t=(float)l*invSr;
        float n=hnoise((unsigned)l,salt);
        float cut=70.f+180.f*wander(t,0.25f,salt^0x77);
        float c=__expf(-TWO_PI*cut*invSr);
        float v=lp1s(n,z1,c); v=lp1s(v,z2,c);

        if(l>=segStart){
            float g=gain[cell*N+l];
            float o=v*g*2.4f;
            atomicAdd(&L[l],o*gL);
            atomicAdd(&R[l],o*gR);
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
   CPU: strike builder — the four quadrants
   ───────────────────────────────────────────────────────────────────── */
static float frand(){ return (float)rand()/(float)RAND_MAX; }
static float frange(float a,float b){ return a+(b-a)*frand(); }

static Strike makeStrike(float tS,float azDeg,float dKm,float en,
                         int rear,float sr)
{
    Strike s={};
    s.tStart=tS; s.az=azDeg*(float)M_PI/180.f;
    s.distKm=dKm; s.energy=en;
    s.itd=(int)(fabsf(sinf(s.az))*0.00066f*sr);
    s.rear=rear;
    s.seed=(unsigned)rand()*2654435761u+(unsigned)rand();

    float nearF=fmaxf(0.f,1.f-dKm/2.6f);
    if(rear) nearF*=0.45f;                    /* rear = pinna HF loss   */

    s.crackAmp=nearF;
    s.crackDur=frange(0.06f,0.20f);
    s.bodyAmp =0.65f+0.35f*nearF;
    s.bodyLpHz=(rear?70.f:95.f)+170.f*nearF;
    s.rumbleAmp=0.6f+0.4f*(1.f-nearF);
    s.rumbleDur=2.2f+1.25f*dKm+frange(0.f,2.f);
    s.rumbleDelay=frange(0.05f,0.30f)+dKm*0.02f;
    s.rumbleLpMax=(rear?0.6f:1.f)*(40.f+240.f*expf(-dKm/2.2f));
    s.rumbleLpMax=fmaxf(34.f,s.rumbleLpMax);
    s.attackSoft=0.05f*dKm/9.f;
    s.dur=s.rumbleDelay+s.rumbleDur+3.f;
    return s;
}

static int buildStrikes(Strike*out,float sr)
{
    struct Cell{float az,d0,d1,t0,t1,e;int n,rear;};
    Cell cells[4]={
        { -50.f, 0.55f, 0.9f,  15.f, 215.f, 1.05f, 9, 0 }, /* A FL near */
        { +50.f, 2.3f,  2.8f,  12.f, 288.f, 0.95f, 8, 0 }, /* B FR mid  */
        { -75.f, 6.5f,  9.0f,  30.f, 275.f, 1.00f, 7, 1 }, /* C RL far  */
        { +65.f, 9.0f,  1.0f,  40.f, 288.f, 1.05f,12, 1 }, /* D RR appr */
    };
    /* jitter cell geometry per run so storms differ structurally too   */
    for(int c=0;c<4;c++){
        cells[c].az+=frange(-8.f,8.f);
        cells[c].d0*=frange(0.85f,1.2f);
        cells[c].d1*=frange(0.85f,1.2f);
        cells[c].n +=(int)frange(-1.9f,2.9f);
        if(cells[c].n<4)cells[c].n=4;
    }
    int n=0;
    for(int c=0;c<4;c++){
        Cell&C=cells[c];
        for(int i=0;i<C.n&&n<MAX_STRIKES-2;i++){
            float u=((float)i+0.15f+0.7f*frand())/(float)C.n;
            if(c==3)u=powf(u,0.75f);          /* D bunches at the end   */
            float t=C.t0+u*(C.t1-C.t0);
            float d=C.d0+(C.d1-C.d0)*u;
            d*=frange(0.9f,1.1f);
            /* D approaching: it stops being "rear" once close          */
            int rear=C.rear&&(d>2.5f);
            out[n++]=makeStrike(t,C.az+frange(-6.f,6.f),fmaxf(0.3f,d),
                                C.e*frange(0.55f,1.35f),rear,sr);
        }
    }
    /* climax: D nearly overhead                                        */
    out[n++]=makeStrike(frange(283.f,289.f),frange(5.f,25.f),
                        frange(0.3f,0.5f),1.8f,0,sr);
    return n;
}

/* ── rain intensity & wind gain tables ───────────────────────────────*/
static void buildRainEnv(float*env,int N,float sr){
    float pk=frange(120.f,190.f), wd=frange(60.f,110.f);
    float sw0=frange(100.f,150.f);
    for(int i=0;i<N;i++){
        float t=(float)i/sr;
        float ramp=fminf(1.f,t/frange(24.f,26.f));
        float peak=0.6f+0.4f*expf(-((t-pk)/wd)*((t-pk)/wd));
        float swell=fmaxf(0.f,fminf(1.f,(t-sw0)/180.f));
        float fade=fmaxf(0.f,fminf(1.f,(DUR_SEC+8.f-t)/10.f));
        env[i]=ramp*(peak+0.35f*swell)*fade;
    }
}
static void buildWindGains(float*g,float*az,int N,float sr){
    float azd[4]={-50.f,50.f,-75.f,65.f};
    for(int c=0;c<4;c++)az[c]=azd[c]*(float)M_PI/180.f;
    for(int c=0;c<4;c++){
        float base=(c==0)?0.8f:(c==1)?0.55f:(c==2)?0.3f:0.55f;
        base*=frange(0.8f,1.25f);
        float gf=frange(0.03f,0.07f),ph=frange(0.f,6.28f);
        for(int i=0;i<N;i++){
            float t=(float)i/sr;
            float fin=fminf(1.f,t/18.f);
            float fout=fmaxf(0.f,fminf(1.f,(DUR_SEC+8.f-t)/8.f));
            float appr=(c==3)?(0.35f+0.65f*fminf(1.f,t/DUR_SEC)):1.f;
            float gust=0.5f+0.5f*cosf(TWO_PI*gf*t+ph);
            g[c*N+i]=base*fin*fout*gust*appr;
        }
    }
}

/* ── post chain ──────────────────────────────────────────────────────*/
static void softSat(float*x,int N,float d){
    for(int i=0;i<N;i++)x[i]=tanhf(x[i]*d)/d;
}
static void applyReverb(float*L,float*R,int N,float sr,float wet)
{
    static const float cd[]={0.0621f,0.0671f,0.0756f,0.0823f,
                              0.0861f,0.0901f,0.0952f,0.0987f};
    static const float ad[]={0.0113f,0.0151f,0.0227f,0.0265f};
    int NC=8,NA=4;float fb=0.74f,damp=0.5f;
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
    FILE*f=fopen(p,"wb");
    if(!f){fprintf(stderr,"Cannot open %s\n",p);return;}
    int16_t*b=(int16_t*)malloc((size_t)N*2*2);
    float pk=1e-9f;
    for(int i=0;i<N;i++){pk=fmaxf(pk,fabsf(L[i]));pk=fmaxf(pk,fabsf(R[i]));}
    float g=0.94f/pk;
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
    printf("Wrote %s  (%.1f s, %d Hz stereo)\n",p,(double)N/sr,sr);
}

/* ═════════════════════════════════════════════════════════════════════ */
int main(int argc,char**argv)
{
    const char*out=(argc>1)?argv[1]:"storm.wav";
    int sr=(argc>2)?atoi(argv[2]):SR_DEFAULT;
    unsigned seed=(argc>3)?(unsigned)atoi(argv[3]):(unsigned)time(NULL);
    srand(seed);

    printf("GPU Thunderstorm v3 | 4 quadrant cells | 5 min stereo\n");
    printf("SEED: %u   (re-run with 'storm3 out.wav %d %u' to reproduce)\n\n",
           seed,sr,seed);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("Device: %s (SM %d.%d)\n\n",prop.name,prop.major,prop.minor);

    int N=(int)((DUR_SEC+14.f)*sr);

    Strike*hs=(Strike*)malloc(MAX_STRIKES*sizeof(Strike));
    int ns=buildStrikes(hs,(float)sr);
    printf("Strikes: %d across 4 cells\n",ns);

    float*hRain=(float*)calloc(N,sizeof(float));
    buildRainEnv(hRain,N,(float)sr);
    float*hWind=(float*)calloc((size_t)4*N,sizeof(float));
    float hAz[4];
    buildWindGains(hWind,hAz,N,(float)sr);

    Strike*ds;float*dL,*dR,*dRain,*dWind,*dAz;
    CUDA_CHECK(cudaMalloc(&ds,MAX_STRIKES*sizeof(Strike)));
    CUDA_CHECK(cudaMalloc(&dL,(size_t)N*4));
    CUDA_CHECK(cudaMalloc(&dR,(size_t)N*4));
    CUDA_CHECK(cudaMalloc(&dRain,(size_t)N*4));
    CUDA_CHECK(cudaMalloc(&dWind,(size_t)4*N*4));
    CUDA_CHECK(cudaMalloc(&dAz,4*4));
    CUDA_CHECK(cudaMemcpy(ds,hs,MAX_STRIKES*sizeof(Strike),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRain,hRain,(size_t)N*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWind,hWind,(size_t)4*N*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAz,hAz,16,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dL,0,(size_t)N*4));
    CUDA_CHECK(cudaMemset(dR,0,(size_t)N*4));

    cudaEvent_t e0,e1;
    CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));

    /* longest strike determines segment grid                            */
    float maxDur=0.f;
    for(int i=0;i<ns;i++) if(hs[i].dur>maxDur) maxDur=hs[i].dur;
    int maxSegs=((int)(maxDur*sr)+SEG-1)/SEG;
    dim3 sGrid((maxSegs+63)/64, ns);
    strikesKernel<<<sGrid,64>>>(ds,ns,dL,dR,N,(float)sr);

    int rainSegs=(N+SEG-1)/SEG;
    dim3 rGrid((rainSegs+63)/64, 2);
    rainKernel<<<rGrid,64>>>(dL,dR,N,(float)sr,dRain,seed);

    dim3 wGrid((rainSegs+63)/64, 4);
    windKernel<<<wGrid,64>>>(dL,dR,N,(float)sr,dWind,dAz,seed);

    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    printf("Kernels: %.0f ms  (%.0fx real-time)\n\n",ms,DUR_SEC*1000.f/ms);

    float*hL=(float*)malloc((size_t)N*4);
    float*hR=(float*)malloc((size_t)N*4);
    CUDA_CHECK(cudaMemcpy(hL,dL,(size_t)N*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hR,dR,(size_t)N*4,cudaMemcpyDeviceToHost));

    printf("Post: saturation + outdoor reverb...\n");
    softSat(hL,N,1.3f);softSat(hR,N,1.3f);
    applyReverb(hL,hR,N,(float)sr,0.09f);
    writeWav(out,hL,hR,N,sr);

    free(hs);free(hRain);free(hWind);free(hL);free(hR);
    cudaFree(ds);cudaFree(dL);cudaFree(dR);
    cudaFree(dRain);cudaFree(dWind);cudaFree(dAz);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 0;
}
