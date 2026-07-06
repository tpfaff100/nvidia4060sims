/*
 * freeway2.cu — GPU roadside traffic v2: DENSE traffic + IMPATIENT DRIVERS
 * Target: RTX 4060 Ti (sm_89)
 *
 * BUILD (after vcvars64.bat):
 *   nvcc -arch=sm_89 -O3 -use_fast_math -o freeway2 freeway2.cu
 * RUN:
 *   freeway2 [freeway.wav] [samplerate] [seed]
 *     no seed -> clock-seeded, different rush hour every run (seed printed)
 *
 * NEW IN V2
 *   • 70-110 cars in 5 minutes (rush-hour density), frequent platoons
 *     and tailgating clusters
 *   • HORNS with era/nation-correct voicing:
 *       US  dual-tone ~350+440 Hz (the classic beat), full and low
 *       UK  dual-tone ~330+410 Hz, thinner
 *       JP  single high beep ~430-500 Hz
 *     All horns: brassy harmonic stack (1x, 2x, 3x), fast attack,
 *     Doppler-bent through the pass, FORWARD-DIRECTIVE (loud on
 *     approach, muffled after the car passes — horns face forward).
 *   • Impatience logic: tailgaters (gap < 2 s) honk at the car ahead
 *     65% of the time (2-4 short blasts on approach); a few drivers
 *     lean on the horn for 1.5-3 s angry blasts; 8% random single honks.
 *
 * Everything else from v1 retained: exact closed-form Doppler on every
 * component, half-order V8 burble, UK rasp + gear whine, muffled JP I4s,
 * era loudness/tilt (60s no-cat vs 80s catalytic), ground-image comb,
 * vehicle-to-vehicle moving reflections, time-warped roar/tire noise.
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
#define SEG        1024
#define WARM       1024
#define MAX_PASS   192
#define NHARM      18
#define MAX_BLAST  4
#define SPEED_C    343.0f

#define CUDA_CHECK(x) do{ cudaError_t _=x; if(_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_)); \
    exit(1);} }while(0)

__device__ __forceinline__ float hnoise(unsigned i,unsigned s){
    unsigned h=i*0x9E3779B9u^s*0x85EBCA6Bu;
    h^=h>>13; h*=0xC2B2AE35u; h^=h>>16;
    return (float)(int)h*4.6566129e-10f;
}
__device__ __forceinline__ float hlerp(float x,unsigned s){
    float fk=floorf(x); int k=(int)fk; float u=x-fk;
    float a=hnoise((unsigned)k,s), b=hnoise((unsigned)(k+1),s);
    return a+(b-a)*u;
}
__device__ __forceinline__ float wander(float t,float rate,unsigned s){
    float x=t*rate; float fk=floorf(x); int k=(int)fk; float u=x-fk;
    u=u*u*(3.f-2.f*u);
    float a=hnoise((unsigned)k,s)*0.5f+0.5f;
    float b=hnoise((unsigned)(k+1),s)*0.5f+0.5f;
    return a+(b-a)*u;
}
__device__ __forceinline__ float lp1s(float x,float&z,float c){
    z=z*c+x*(1.f-c); return z;
}
__device__ __forceinline__ void panLR(float sinAz,float&gL,float&gR){
    float x=(sinAz+1.f)*0.25f*(float)M_PI;
    gL=__cosf(x); gR=__sinf(x);
}

struct Pass {
    float tPass,v,d,t0,t1;
    /* engine                                                            */
    float fFire0,fSlope;
    int   halfOrders;
    float hAmp[NHARM];
    float loud,roarAmp,roarLpHz,tireAmp,whineHz,whineAmp;
    /* horn                                                              */
    int   hornN;                     /* number of blasts (0 = silent)   */
    float hornF1,hornF2;             /* dual tones (F2=0 -> single)     */
    float hornAmp;
    float hornStart[MAX_BLAST];      /* s relative to tPass (emission)  */
    float hornDur[MAX_BLAST];
    /* reflection                                                        */
    int   isRefl;
    float rv,rtPass,rd;
    unsigned seed;
};

/* ═══════════════ PASS KERNEL ════════════════════════════════════════ */
__global__ void passKernel(
    const Pass*__restrict__ ps,int nPass,
    float*L,float*R,int N,float sr)
{
    int pi=(int)blockIdx.y; if(pi>=nPass) return;
    const Pass p=ps[pi];

    int winSamps=(int)((p.t1-p.t0)*sr);
    int segIdx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int segStart=segIdx*SEG;
    if(segStart>=winSamps) return;
    int segEnd=min(segStart+SEG,winSamps);
    int base=(int)(p.t0*sr);

    float invSr=1.f/sr;
    float cRoar=__expf(-TWO_PI*p.roarLpHz*invSr);
    float cTire=__expf(-TWO_PI*2400.f*invSr);
    float cTireH=__expf(-TWO_PI*600.f*invSr);
    float zr=0,zt1=0,zt2=0,zth=0;

    for(int l=segStart-WARM;l<segEnd;l++){
        if(l<0) continue;
        float tg=p.t0+(float)l*invSr;
        float u=tg-p.tPass;

        float x=p.v*u;
        float rd2=x*x+p.d*p.d;
        float rDir=__fsqrt_rn(rd2+1.69f);
        float R1,R2,ampGeo,sinAz;
        if(!p.isRefl){
            R1=rDir; R2=__fsqrt_rn(rd2+4.41f);
            ampGeo=1.f/fmaxf(rDir,3.f);
            sinAz=x/__fsqrt_rn(rd2+1e-3f);
        }else{
            float xr=p.rv*(tg-p.rtPass);
            float dx=x-xr, dd=p.d-p.rd;
            float dij=__fsqrt_rn(dx*dx+dd*dd+1.f);
            float rjl=__fsqrt_rn(xr*xr+p.rd*p.rd+1.69f);
            R1=dij+rjl; R2=R1+0.7f;
            ampGeo=0.55f/(fmaxf(dij,2.f)*fmaxf(rjl,3.f))*8.f;
            sinAz=xr/__fsqrt_rn(xr*xr+p.rd*p.rd+1e-3f);
        }

        float tau1=u-R1/SPEED_C;
        float tau2=u-R2/SPEED_C;
        float rough=0.80f+0.40f*wander(tau1,6.5f,p.seed^0x99);

        /* engine harmonics (direct + ground)                            */
        float eng=0.f;
        float ratioStep=p.halfOrders?0.5f:1.0f;
        #pragma unroll
        for(int k=1;k<=NHARM;k++){
            float ratio=ratioStep*(float)k;
            float f=(p.fFire0+p.fSlope*tau1)*ratio;
            float ab=__expf(-5.75e-10f*f*f*rDir*(p.isRefl?2.f:1.f));
            float a=p.hAmp[k-1]*ab;
            if(a<1e-5f) continue;
            float ph1=ratio*(p.fFire0*tau1+0.5f*p.fSlope*tau1*tau1);
            float ph2=ratio*(p.fFire0*tau2+0.5f*p.fSlope*tau2*tau2);
            ph1-=floorf(ph1); ph2-=floorf(ph2);
            eng+=a*(__sinf(TWO_PI*ph1)+0.6f*__sinf(TWO_PI*ph2));
        }
        eng*=rough;

        /* roar + tires (time-warped noise)                              */
        float nr=hlerp(tau1*sr,p.seed^0xE0A5);
        float roar=lp1s(nr,zr,cRoar)*p.roarAmp*3.5f*rough;
        float nt=hlerp(tau1*sr,p.seed^0x71BE);
        float tv=lp1s(nt,zt1,cTire); tv=lp1s(tv,zt2,cTire);
        tv=tv-lp1s(tv,zth,cTireH);
        float tire=tv*p.tireAmp*2.2f;

        /* drivetrain whine                                              */
        float wh=0.f;
        if(p.whineAmp>0.f){
            float phw=p.whineHz*tau1; phw-=floorf(phw);
            wh=p.whineAmp*__expf(-5.75e-10f*p.whineHz*p.whineHz*rDir)
              *__sinf(TWO_PI*phw);
        }

        /* ── HORN: dual-tone + brassy harmonics, forward-directive ──── */
        float horn=0.f;
        if(p.hornN>0 && !p.isRefl){
            /* blast envelope                                            */
            float he=0.f;
            #pragma unroll
            for(int b=0;b<MAX_BLAST;b++){
                if(b>=p.hornN) break;
                float ts=tau1-p.hornStart[b];
                if(ts>0.f && ts<p.hornDur[b]){
                    float e=fminf(1.f,ts/0.006f)
                           *fminf(1.f,(p.hornDur[b]-ts)/0.05f);
                    he=fmaxf(he,fmaxf(0.f,e));
                }
            }
            if(he>0.f){
                /* horns face forward: loud approaching, muffled after  */
                float sgnv=(p.v>=0.f)?1.f:-1.f;
                float fw=0.5f*(1.f-(x*sgnv)/__fsqrt_rn(x*x+40.f));
                float dir=0.20f+0.80f*fw;
                float t2sum=0.f;
                float freqs[2]={p.hornF1,p.hornF2};
                #pragma unroll
                for(int q=0;q<2;q++){
                    float f0=freqs[q];
                    if(f0<=0.f) continue;
                    float hs[3]={1.f,0.5f,0.22f};
                    #pragma unroll
                    for(int h=1;h<=3;h++){
                        float f=f0*(float)h;
                        float ab=__expf(-5.75e-10f*f*f*rDir);
                        float ph=f*tau1; ph-=floorf(ph);
                        t2sum+=hs[h-1]*ab*__sinf(TWO_PI*ph);
                    }
                }
                horn=t2sum*he*dir*p.hornAmp;
            }
        }

        if(l>=segStart){
            int g=base+l; if(g>=N) break;
            float s=(eng+roar+tire+wh+horn)*p.loud*ampGeo;
            float gL,gR; panLR(fmaxf(-1.f,fminf(1.f,sinAz)),gL,gR);
            int itd=(int)(fabsf(sinAz)*0.00066f*sr);
            int iL=g,iR=g;
            if(sinAz<0.f) iR=min(N-1,g+itd); else iL=min(N-1,g+itd);
            atomicAdd(&L[iL],s*gL);
            atomicAdd(&R[iR],s*gR);
        }
    }
}

/* ═══════════════ AMBIENCE (denser: more distant wash) ═══════════════ */
__global__ void ambienceKernel(
    float*L,float*R,int N,float sr,unsigned seed)
{
    int ch=(int)blockIdx.y;
    int segIdx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int segStart=segIdx*SEG;
    if(segStart>=N) return;
    int segEnd=min(segStart+SEG,N);
    unsigned salt=seed^(ch?0xAB12u:0xCD34u);
    float invSr=1.f/sr;
    float z1=0,z2=0,zm=0;
    float cM=__expf(-TWO_PI*1100.f*invSr);
    for(int l=segStart-WARM;l<segEnd;l++){
        if(l<0)continue;
        float t=(float)l*invSr;
        float n=hnoise((unsigned)l,salt);
        float cut=100.f+140.f*wander(t,0.18f,salt^0x33);
        float c=__expf(-TWO_PI*cut*invSr);
        float v=lp1s(n,z1,c); v=lp1s(v,z2,c);
        float m=lp1s(n,zm,cM)*0.09f;
        if(l>=segStart){
            float g=0.6f+0.4f*wander(t,0.06f,salt^0x44);
            float o=(v*1.7f+m)*g*0.13f;
            if(ch==0)atomicAdd(&L[l],o); else atomicAdd(&R[l],o);
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
   CPU: car factory
   ───────────────────────────────────────────────────────────────────── */
static float frand(){return (float)rand()/(float)RAND_MAX;}
static float frange(float a,float b){return a+(b-a)*frand();}

enum Nation{US,UK,JP};

static void giveHorn(Pass&p,Nation nat,int angry)
{
    /* voicing                                                           */
    if(nat==US){ p.hornF1=frange(335.f,370.f); p.hornF2=frange(425.f,465.f);
                 p.hornAmp=frange(1.7f,2.3f); }
    else if(nat==UK){ p.hornF1=frange(315.f,345.f); p.hornF2=frange(395.f,430.f);
                 p.hornAmp=frange(1.4f,1.9f); }
    else       { p.hornF1=frange(430.f,505.f); p.hornF2=0.f;
                 p.hornAmp=frange(1.3f,1.8f); }

    if(angry){
        /* one long furious lean, starting on approach                   */
        p.hornN=1;
        p.hornStart[0]=frange(-4.5f,-2.0f);
        p.hornDur[0]=frange(1.5f,3.0f);
        p.hornAmp*=1.25f;
    }else{
        /* 2-4 short irritated blasts on approach                        */
        p.hornN=2+(int)(frand()*3.f); if(p.hornN>MAX_BLAST)p.hornN=MAX_BLAST;
        float t=frange(-4.0f,-1.5f);
        for(int b=0;b<p.hornN;b++){
            p.hornStart[b]=t;
            p.hornDur[b]=frange(0.12f,0.35f);
            t+=p.hornDur[b]+frange(0.10f,0.30f);
        }
    }
}

static Pass makeCar(float tPass,int lane,float sr)
{
    static const float laneD[4]={6.f,10.f,18.f,22.f};
    static const float laneDir[4]={+1.f,+1.f,-1.f,-1.f};

    Pass p={};
    p.tPass=tPass; p.d=laneD[lane];
    p.v=laneDir[lane]*frange(24.f,36.f);
    p.seed=(unsigned)rand()*2654435761u+(unsigned)rand();

    float rn=frand();
    Nation nat=(rn<0.5f)?US:(rn<0.75f)?JP:UK;
    int era=(frand()<0.33f)?60:(frand()<0.5f)?70:80;
    int isV8=(nat==US)?(frand()<0.8f):(frand()<0.08f);

    float rpm,loud,tilt;
    if(isV8){
        rpm=frange(2100.f,2800.f);
        p.halfOrders=1; p.fFire0=rpm/60.f*4.f;
        loud=1.10f; tilt=0.75f;
        p.roarAmp=0.5f; p.roarLpHz=350.f;
        for(int k=1;k<=NHARM;k++){
            float ratio=0.5f*k;
            float a=powf(1.f/ratio,tilt);
            if(k&1)a*=1.55f;
            p.hAmp[k-1]=a;
        }
    }else{
        rpm=(nat==UK)?frange(3000.f,4200.f):frange(2700.f,3600.f);
        p.halfOrders=0; p.fFire0=rpm/60.f*2.f;
        if(nat==UK){
            loud=0.95f; tilt=0.55f;
            p.roarAmp=0.42f; p.roarLpHz=900.f;
            for(int k=1;k<=NHARM;k++){
                float a=powf(1.f/(float)k,tilt);
                if(k>=4&&k<=10)a*=1.5f;
                p.hAmp[k-1]=a;
            }
            p.whineHz=frange(1100.f,2400.f);
            p.whineAmp=frange(0.02f,0.06f);
        }else{
            loud=0.58f; tilt=1.15f;
            p.roarAmp=0.20f; p.roarLpHz=260.f;
            for(int k=1;k<=NHARM;k++)
                p.hAmp[k-1]=powf(1.f/(float)k,tilt);
        }
    }
    if(era==60){ loud*=1.30f; }
    else if(era==80){
        loud*=0.62f;
        for(int k=1;k<=NHARM;k++)p.hAmp[k-1]*=expf(-0.14f*(float)k);
        p.roarAmp*=0.55f;
    }
    if(frand()<0.15f&&era!=80){
        p.fSlope=p.fFire0*frange(0.06f,0.16f);
        loud*=1.35f; p.roarAmp*=1.5f;
    }
    p.loud=loud*frange(0.85f,1.2f)*0.92f;      /* dense mix headroom   */
    p.tireAmp=0.20f*powf(fabsf(p.v)/30.f,3.f);
    if(!p.whineAmp&&frand()<0.25f){
        p.whineHz=frange(900.f,1800.f);
        p.whineAmp=frange(0.015f,0.04f);
    }
    /* random lone honker (8%)                                           */
    if(frand()<0.08f) giveHorn(p,nat,frand()<0.2f);

    float tw=350.f/fabsf(p.v);
    p.t0=fmaxf(0.f,tPass-tw); p.t1=fminf(DUR_SEC+4.f,tPass+tw);
    return p;
}

static Pass makeReflection(const Pass&em,const Pass&rf)
{
    Pass p=em;
    p.isRefl=1; p.hornN=0;
    p.rv=rf.v; p.rtPass=rf.tPass; p.rd=rf.d;
    p.tireAmp*=0.3f; p.whineAmp=0.f; p.loud*=0.8f;
    p.t0=fmaxf(em.t0,rf.t0); p.t1=fminf(em.t1,rf.t1);
    p.seed^=0xF1F1F1F1u;
    return p;
}

static int buildTraffic(Pass*out,float sr)
{
    int nTarget=70+(int)(frand()*41.f);        /* 70-110 cars           */
    int n=0, nHonks=0;
    float t=frange(2.f,6.f);
    while(n<MAX_PASS-40 && t<DUR_SEC-6.f){
        int lane=(int)(frand()*4.f); if(lane>3)lane=3;
        int lead=n;
        out[n++]=makeCar(t,lane,sr);

        /* tailgating cluster: 45% chance, 1-3 followers                 */
        if(frand()<0.45f){
            int nf=1+(int)(frand()*2.5f);
            float ft=t;
            for(int j=0;j<nf&&n<MAX_PASS-40;j++){
                float gap=frange(0.6f,2.4f);
                ft+=gap;
                int flane=(frand()<0.7f)?lane:((lane+1)&3);
                Pass f=makeCar(ft,flane,sr);
                /* IMPATIENCE: close follower honks at the car ahead     */
                if(gap<2.0f && frand()<0.65f && !f.hornN){
                    float rn2=frand();
                    Nation nat=(rn2<0.5f)?US:(rn2<0.75f)?JP:UK;
                    giveHorn(f,nat,frand()<0.25f);
                    nHonks++;
                }
                out[n++]=f;
            }
            (void)lead;
        }
        t+=frange(1.4f,7.5f)*(70.f/(float)nTarget);
    }
    int nDirect=n;
    int nRef=0;
    for(int i=0;i<nDirect&&nRef<32;i++)
      for(int j=0;j<nDirect&&nRef<32;j++){
        if(i==j||out[i].isRefl||out[j].isRefl)continue;
        if(fabsf(out[i].tPass-out[j].tPass)<3.f
           && fabsf(out[i].d-out[j].d)>3.f){
            out[n++]=makeReflection(out[i],out[j]);
            nRef++;
        }
      }
    int totalHonkers=0;
    for(int i=0;i<nDirect;i++) if(out[i].hornN) totalHonkers++;
    printf("Cars: %d   Honking drivers: %d   Reflections: %d\n",
           nDirect,totalHonkers,nRef);
    return n;
}

static void softSat(float*x,int N,float d){
    for(int i=0;i<N;i++)x[i]=tanhf(x[i]*d)/d;
}
static void writeWav(const char*pth,const float*L,const float*R,int N,int sr){
    FILE*f=fopen(pth,"wb");
    if(!f){fprintf(stderr,"Cannot open %s\n",pth);return;}
    int16_t*b=(int16_t*)malloc((size_t)N*4);
    float pk=1e-9f;
    for(int i=0;i<N;i++){pk=fmaxf(pk,fabsf(L[i]));pk=fmaxf(pk,fabsf(R[i]));}
    float g=0.94f/pk;
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
    printf("Wrote %s  (%.1f s, %d Hz stereo)\n",pth,(double)N/sr,sr);
}

int main(int argc,char**argv)
{
    const char*out=(argc>1)?argv[1]:"freeway.wav";
    int sr=(argc>2)?atoi(argv[2]):SR_DEFAULT;
    unsigned seed=(argc>3)?(unsigned)atoi(argv[3]):(unsigned)time(NULL);
    srand(seed);

    printf("GPU Freeway v2 | RUSH HOUR + impatient drivers | 5 min stereo\n");
    printf("SEED: %u  (rerun 'freeway2 out.wav %d %u' to reproduce)\n\n",
           seed,sr,seed);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("Device: %s (SM %d.%d)\n\n",prop.name,prop.major,prop.minor);

    int N=(int)((DUR_SEC+6.f)*sr);
    Pass*hp=(Pass*)calloc(MAX_PASS,sizeof(Pass));
    int np=buildTraffic(hp,(float)sr);

    Pass*dp;float*dL,*dR;
    CUDA_CHECK(cudaMalloc(&dp,MAX_PASS*sizeof(Pass)));
    CUDA_CHECK(cudaMalloc(&dL,(size_t)N*4));
    CUDA_CHECK(cudaMalloc(&dR,(size_t)N*4));
    CUDA_CHECK(cudaMemcpy(dp,hp,MAX_PASS*sizeof(Pass),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dL,0,(size_t)N*4));
    CUDA_CHECK(cudaMemset(dR,0,(size_t)N*4));

    float maxWin=0.f;
    for(int i=0;i<np;i++)maxWin=fmaxf(maxWin,hp[i].t1-hp[i].t0);
    int maxSegs=((int)(maxWin*sr)+SEG-1)/SEG;

    cudaEvent_t e0,e1;
    CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    dim3 pGrid((maxSegs+63)/64,np);
    passKernel<<<pGrid,64>>>(dp,np,dL,dR,N,(float)sr);
    int aSegs=(N+SEG-1)/SEG;
    dim3 aGrid((aSegs+63)/64,2);
    ambienceKernel<<<aGrid,64>>>(dL,dR,N,(float)sr,seed);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    printf("Kernels: %.0f ms (%.0fx real-time)\n\n",ms,DUR_SEC*1000.f/ms);

    float*hL=(float*)malloc((size_t)N*4);
    float*hR=(float*)malloc((size_t)N*4);
    CUDA_CHECK(cudaMemcpy(hL,dL,(size_t)N*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hR,dR,(size_t)N*4,cudaMemcpyDeviceToHost));
    printf("Post: saturation...\n");
    softSat(hL,N,1.35f);softSat(hR,N,1.35f);
    writeWav(out,hL,hR,N,sr);

    free(hp);free(hL);free(hR);
    cudaFree(dp);cudaFree(dL);cudaFree(dR);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 0;
}
