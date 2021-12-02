/*
 *=============================================================
 * %--------------------------------------------------
 * %  [ratemap, bm, cfs, signlen] = makeRateMap_IHC(x,fs,lowcf,highcf,numchans,frameshift,ti,compression,haircell,outermiddle,audible)
 * %
 * %  x             input signal
 * %  fs            sampling frequency in Hz
 * %  lowcf         centre frequency of lowest filter in Hz (100)
 * %  highcf        centre frequency of highest filter in Hz (7500)
 * %  numchans      number of channels in filterbank (34)
 * %  frameshift    interval between successive frames in ms (10)
 * %  ti            temporal integration in ms (8)
 * %  compression   type of compression ['cuberoot','log','none'] ('none')
 * %  haircell      inner hair cell model ['meddis', 'cooke', 'none'] ('none')
 * %  outermiddle   outermiddle transfer function ['iso', 'terhardt', 'none'] ('none')
 * %  audible       flag of audibility checing [true|fase] (false)
 *
 * %
 * %  Author:       Yan Tang
 * %  Created:      02/04/2012
 * %  Modified:     28/06/2017
 * %
 */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "mex.h"
#include "matrix.h"

#define TRUE    1
#define FALSE   0


/*=======================
 * Input arguments
 *=======================
 */
#define IN_x            prhs[0]
#define IN_fs           prhs[1]
#define IN_lowcf        prhs[2]
#define IN_highcf       prhs[3]
#define IN_numchans     prhs[4]
#define IN_frameshift   prhs[5]
#define IN_ti           prhs[6]
#define IN_compression  prhs[7]
#define IN_haircell     prhs[8]
#define IN_outermiddle  prhs[9]
#define IN_audible      prhs[10]


/*=======================
 * Output arguments
 *=======================
 */
#define OUT_ratemap     plhs[0]
#define OUT_bm          plhs[1]
#define OUT_cfs         plhs[2]
#define OUT_siglen      plhs[3]

/*=======================
 * Useful Const 9e-4
 *=======================
 */
#define BW_CORRECTION   1.019
#define VERY_SMALL_NUMBER  1e-200
#define ABSOLUTE_AMPLITUDE_THRESHOLD  1e-10
#define SPL_REF  2e-5
#define SPL_THRESHOLD  -100 //DB
#define haircellgain   10000


#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif

/*=======================
 * Meddis constants
 *=======================
 */
#define MED_Y    5.05
#define MED_G    2000.0
#define MED_L    2500.0
#define MED_R    6580.0
#define MED_X    66.31
#define MED_A    3.0
#define MED_B    300.0
#define MED_H    48000.0
#define MED_M    1.0


/*=======================
 * Cooke constants
 *=======================
 */
#define CANDFCONST   100.0
#define RELEASE      24.0
#define REFILL       6.0
#define SPONT        25.0
#define NORMSPIKE    1000.0


void meddisINI(int fs, double* status);
double meddisHC(double bm, double* status);

void cookeINI(int fs, double* status);
double cookeHC(double env, double* status);

double fay_threshold(double cf);
double terhardt_threshold(double cf);
double iso_threshold(double cf);

double db2amp(double dblevel);
void interp1(double* x, int size_x, double *y, double* xx, int size_xx, double* yy);
int findNearestNeighbourIndex(double value, double* x, int len );


/*=======================
 * Utility functions
 *=======================
 */
#define getMax(x,y)     ((x)>(y)?(x):(y))
#define getRound(x)     ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))

#define erb(x)          (24.7*(4.37e-3*(x)+1.0))
#define HzToErbRate(x)  (21.4*log10(4.37e-3*(x)+1.0))
#define ErbRateToHz(x)  ((pow(10.0,((x)/21.4))-1.0)/4.37e-3)

double fmin(double x, double y) { return(x < y ? x : y); }
double fmax(double x, double y) { return(x > y ? x : y); }

/*=======================
 * Meddis global variables
 *=======================
 */
double dt, kt;
double ymdt, xdt, ydt, rdt, gdt, hdt, lplusrdt;
double c, q, w;

/*=======================
 * Cooke global variables
 *=======================
 */
double vmin, k, l;


/*=======================
 * Main Function
 *=======================
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
   double *x, *ratemap, *senv, *bm, *cfs;
   int i, j, chan, fs, numchans;
   int nsamples, nsamples_padded, frameshift_samples, numframes;
   double lowcf, highcf, frameshift, ti, lowErb, highErb, spaceErb, cf;
   double a, tpt, tptbw, gain, intdecay, intgain, sumEnv, tmp, threshold;
   double p0r, p1r, p2r, p3r, p4r, p0i, p1i, p2i, p3i, p4i;
   double a1, a2, a3, a4, a5, cs, sn, u0r, u0i;
   double senv1, qcos, qsin, oldcs, coscf, sincf;
   double *pstatus, status[5],fire;
   int audible;
   char compression[20], outermiddle[20], haircell[10];
   
   pstatus = status;
   
   /*=========================================
    * input arguments
    *=========================================
    */
   
   if (nrhs < 1) { mexPrintf("??? Not enough input arguments.\n"); return; }
   
   if (nrhs < 2) fs = 8000;
   else fs = (int) mxGetScalar(IN_fs);
   
   if (nrhs < 3) lowcf = 100;
   else lowcf = mxGetScalar(IN_lowcf);
   
   if (nrhs < 4) highcf = 7500;
   else highcf = mxGetScalar(IN_highcf);
   
   if (nrhs < 5) numchans = 34;
   else numchans = (int) mxGetScalar(IN_numchans);
   
   if (nrhs < 6) frameshift = 10;
   else frameshift = mxGetScalar(IN_frameshift);
   
   if (nrhs < 7) ti = 8;
   else ti = mxGetScalar(IN_ti);
   
   if (nrhs < 8) strcpy(compression, "none");
   else mxGetString(IN_compression, compression, 19);
   
   if (nrhs < 9) strcpy(haircell, "none");
   else mxGetString(IN_haircell, haircell, 9);
   
   if (nrhs < 10) strcpy(outermiddle, "none");
   else mxGetString(IN_outermiddle, outermiddle, 19);
   
   if (nrhs < 11) audible = FALSE;
   else audible = (int) mxGetScalar(IN_audible);
   
   
   if (nrhs > 11) { mexPrintf("??? Too many input arguments.\n"); return; }
   if (nlhs > 4) { mexPrintf("??? Too many output arguments.\n"); return; }
   
   x = mxGetPr(IN_x);
   i = mxGetN(IN_x);
   j = mxGetM(IN_x);
   if (i>1 && j>1) { mexPrintf("??? Input x must be a vector.\n"); return; }
   
   nsamples = getMax(i,j);
   frameshift_samples = getRound(frameshift*fs/1000);
   numframes = (int)ceil((double)nsamples / (double)frameshift_samples);
   nsamples_padded = numframes*frameshift_samples;
   
   /*=========================================
    * output arguments
    *=========================================
    */
   OUT_ratemap = mxCreateDoubleMatrix(numchans, numframes, mxREAL);
   ratemap = mxGetPr(OUT_ratemap);
   
   if ( nlhs > 1 ) {
      OUT_bm = mxCreateDoubleMatrix (numchans, nsamples, mxREAL);
      bm = mxGetPr(OUT_bm);
   }
   
   if (nlhs > 2) {
      OUT_cfs = mxCreateDoubleMatrix(numchans, 1, mxREAL);
      cfs = mxGetPr(OUT_cfs);
   }
   
   if (nlhs > 3) {
      OUT_siglen = mxCreateDoubleScalar(nsamples_padded);
   }
   
   
   
   lowErb = HzToErbRate(lowcf);
   highErb = HzToErbRate(highcf);
   
   if (numchans > 1)  spaceErb = (highErb-lowErb)/(numchans-1);
   else  spaceErb = 0.0;
   
   /* Smoothed envelope */
   senv = (double*) mxCalloc(nsamples_padded, sizeof(double));
   
   tpt = 2 * M_PI / fs;
   intdecay = exp(-(1000.0/(fs*ti)));
   intgain = 1-intdecay;
   
   for (chan=0; chan<numchans; chan++){
      
      cf = ErbRateToHz(lowErb+chan*spaceErb);
      cfs[chan] = cf;
      tptbw = tpt * erb(cf) * BW_CORRECTION;
      a = exp(-tptbw);
      gain = (tptbw*tptbw*tptbw*tptbw)/3;
      
      
      // integrate outermiddle
      if(strcmp(outermiddle, "iso")==0){
         threshold = iso_threshold(cf);
      }else if(strcmp(outermiddle, "terhardt")==0){
         threshold = terhardt_threshold(cf);
      }else{
         threshold = 0;
      }
      gain *= db2amp(-threshold); //get amplitude for gain
      
      /* Update filter coefficiants */
      a1 = 4.0*a; a2 = -6.0*a*a; a3 = 4.0*a*a*a; a4 = -a*a*a*a; a5 = a*a;
      
      p0r=0.0; p1r=0.0; p2r=0.0; p3r=0.0; p4r=0.0;
      p0i=0.0; p1i=0.0; p2i=0.0; p3i=0.0; p4i=0.0;
      senv1=0.0;
      
      /*====================================================================================
       * complex z=x+j*y, exp(z) = exp(x)*(cos(y)+j*sin(y)) = exp(x)*cos(x)+j*exp(x)*sin(y).
       * z = -j * tpti * cf, exp(z) = cos(tpti*cf) - j * sin(tpti*cf)
       *====================================================================================
       */
      coscf = cos ( tpt * cf );
      sincf = sin ( tpt * cf );
      qcos = 1; qsin = 0;   /* t=0 & q = exp(-i*tpt*t*cf)*/
      
      
      if (strcmp(haircell,"meddis")==0){
         meddisINI(fs,pstatus);
      }else if(strcmp(haircell,"cooke")==0){
         vmin = SPONT / NORMSPIKE;
         k = RELEASE / fs;
         l = REFILL / fs;
         cookeINI(fs,pstatus);
      }
      
      for (i=0; i<nsamples; i++){
         
         p0r = qcos*x[i] + a1*p1r + a2*p2r + a3*p3r + a4*p4r;
         p0i = qsin*x[i] + a1*p1i + a2*p2i + a3*p3i + a4*p4i;
         
         /* Clip coefficients to stop them from becoming too close to zero */
         if (fabs(p0r) < VERY_SMALL_NUMBER)
            p0r = 0.0F;
         if (fabs(p0i) < VERY_SMALL_NUMBER)
            p0i = 0.0F;
         
         u0r = p0r + a1*p1r + a5*p2r;
         u0i = p0i + a1*p1i + a5*p2i;
         
         p4r = p3r; p3r = p2r; p2r = p1r; p1r = p0r;
         p4i = p3i; p3i = p2i; p2i = p1i; p1i = p0i;
         
         /*==========================================
          * Smoothed envelope by temporal integration
          *==========================================
          */
         if (strcmp(haircell,"meddis")==0){
            fire = haircellgain * ( u0r * qcos + u0i * qsin ) * gain;
            fire = meddisHC(fire,pstatus);
            fire = fmax(fire,0);
         }else{
            fire = sqrt(u0r*u0r+u0i*u0i) * gain;
            
            if (strcmp(haircell,"cooke")==0){
               fire = 20 * log10(fire/SPL_REF);
               fire = fmax(fire, SPL_THRESHOLD);
               fire = cookeHC(fire, pstatus);
            }
            
         }
         senv1 = senv[i]  = fire + intdecay*senv1;
         
         if ( nlhs > 1 ) {
            bm[chan+numchans*i] = ( u0r * qcos + u0i * qsin ) * gain;
         }
         
         
         /*=========================================
          * cs = cos(tpt*i*cf);
          * sn = -sin(tpt*i*cf);
          *=========================================
          */
         qcos = coscf * ( oldcs = qcos ) + sincf * qsin;
         qsin = coscf * qsin - sincf * oldcs;
      }
      
      
      for (i=nsamples; i<nsamples_padded; i++){
         
         p0r = a1*p1r + a2*p2r + a3*p3r + a4*p4r;
         p0i = a1*p1i + a2*p2i + a3*p3i + a4*p4i;
         
         u0r = p0r + a1*p1r + a5*p2r;
         u0i = p0i + a1*p1i + a5*p2i;
         
         p4r = p3r; p3r = p2r; p2r = p1r; p1r = p0r;
         p4i = p3i; p3i = p2i; p2i = p1i; p1i = p0i;
         
         /*==========================================
          * Envelope
          *==========================================
          * env0 = sqrt(u0r*u0r+u0i*u0i) * gain;
          */
         
         /*==========================================
          * Smoothed envelope by temporal integration
          *==========================================
          */
         if (strcmp(haircell,"meddis")==0){
            fire = haircellgain * ( u0r * qcos + u0i * qsin ) * gain;
            fire = meddisHC(fire,pstatus);
            fire = fmax(fire,0);
         }else{
            fire = sqrt(u0r*u0r+u0i*u0i) * gain;
            
            if (strcmp(haircell,"cooke")==0){
               fire = 20 * log10(fire/SPL_REF);
               fire = fmax(fire, SPL_THRESHOLD);
               fire = cookeHC(fire, pstatus);
            }
            
         }
         senv1 = senv[i]  = fire + intdecay*senv1;
         
//             senv1 = senv[i] = sqrt(u0r*u0r+u0i*u0i) * gain + intdecay*senv1;
         
      }
      
      /*==================================================================================
       * we take the mean of the smoothed envelope as the energy value in each frame
       * rather than simply sampling it.
       * ratemap(c,:) = intgain.*mean(reshape(smoothed_env,frameshift_samples,numframes));
       *==================================================================================
       */
      for (j=0; j<numframes; j++){
         sumEnv = 0.0;
         for (i=j*frameshift_samples; i<(j+1)*frameshift_samples; i++){
            sumEnv += senv[i];
         }
         tmp = intgain * sumEnv / frameshift_samples;
         if (audible){
            if (tmp < ABSOLUTE_AMPLITUDE_THRESHOLD){
               tmp = ABSOLUTE_AMPLITUDE_THRESHOLD;
            }
         }
         ratemap[chan+numchans*j] = tmp;
//             ratemap[chan+numchans*j] =  sumEnv / frameshift_samples;
      }
   }
   
   if (strcmp(compression, "cuberoot") == 0){
      for (i=0; i<numchans*numframes; i++)
         ratemap[i] = pow(ratemap[i], 0.3);
   }else if (strcmp(compression, "log") == 0){
      for (i=0; i<numchans*numframes; i++)
         ratemap[i] = 20 * log10(ratemap[i]); //(20*1e-6)
   }
   
   mxFree(senv);
   return;
}

void meddisINI(int fs, double* status){
   // parameter initialisation
   dt = 1.0 / fs;
   ymdt = MED_Y * MED_M * dt;
   xdt = MED_X * dt;
   ydt = MED_Y * dt;
   rdt = MED_R * dt;
   gdt = MED_G * dt;
   hdt = MED_H;
   kt = MED_G * MED_A / (MED_A + MED_B);
   c = MED_M * MED_Y * kt / (MED_L * kt + MED_Y * (MED_L + MED_R));
   q = c * (MED_L + MED_R) / kt;
   w = c * MED_R / MED_X;
   lplusrdt = (MED_L + MED_R) * dt;
   
   status[0] = c;
   status[1] = q;
   status[2] = w;
   
   return;
}


double meddisHC(double bm, double* status){
   double x, replenish, eject, reuptakeandloss, reuptake, reprocess;
   
   x = bm + MED_A;
   
   kt = 0.0;
   if (x > 0.0){
      kt = gdt * (x / (x+MED_B));
   }
   
   replenish = 0.0;
   if (status[1] < MED_M){
      replenish = ymdt - ydt * status[1];
   }
   
   
   eject = kt * status[1];
   reuptakeandloss = lplusrdt * status[0];
   reuptake  =  rdt * status[0];
   reprocess = xdt * status[2];
   
   status[1] += (replenish - eject + reprocess);
   if (status[1] < 0.0){
      status[1] = 0.0;
   }
   
   status[0] += (eject - reuptakeandloss);
   if (status[0] < 0.0){
      status[0] = 0.0;
   }
   
   status[2] += (reuptake - reprocess);
   if (status[2] < 0.0){
      status[2] = 0.0;
   }
   
   return (hdt * status[0] - 50.0);
}



void cookeINI(int fs, double *status){
   status[0] = vmin;  //vimm
   status[1] = 0.0;  // vrel;
   status[2] = 0.0;  // crel;
   status[3] = 1.0 - vmin; //vres;
   status[4] = l/(k*vmin + l); //cimm
   
   return;
}

double cookeHC(double env, double *status){
   double rp, rate, SPMin, vimm, vrel, crel, vres, cimm;
   double delta;
   vimm = status[0];
   vrel = status[1];
   crel = status[2];
   vres = status[3];
   cimm = status[4];
   
   rp = env/(env+CANDFCONST);
   SPMin = fmax(rp, vmin);
   
   if (SPMin > vimm){
      if(SPMin > (vimm+vrel)){
         delta = SPMin - (vrel+vimm);
         cimm = (cimm*vimm + crel*vrel + delta)/SPMin;
         vrel = 0;
      }else{
         delta = SPMin - vimm;
         cimm = (cimm*vimm + delta * crel)/SPMin;
         vrel -= delta;
      }
   }else if(vimm > SPMin){
      delta = vimm - SPMin;
      crel = (delta*cimm + vrel*crel)/(delta + vrel);
      vrel += delta;
   }
   
   vimm = SPMin;
   rate = k*cimm*vimm;
   cimm -= rate;
   cimm += l*(1-cimm);
   crel += l*(1-crel);
   
   status[0] = vimm;
   status[1] = vrel;
   status[2] = crel;
   status[3] = vres;
   status[4] = cimm;
   
   return rate;
}


double fay_threshold(double cf){
   double threshold, normcf = cf / 1000;
   threshold = 4.08 / normcf + 17.47 - 45.23 * normcf + 45.76 * pow(normcf,2) - 19.59 * pow(normcf,3) + 4.11 * pow(normcf,4) - 0.41 * pow(normcf,5) + 0.02 * pow(normcf,6);
   
   return threshold;
}

double terhardt_threshold(double cf){
   double threshold, normcf = cf / 1000;
   threshold = 3.64*pow(normcf, -0.8)-6.5*exp(-0.6*pow((normcf-3.3), 2))+ 10e-3*pow(normcf, 4);
   
   return threshold;
}

double iso_threshold(double cf){
   /*=======================
    * ISO threshold
    *=======================
    */
   double val_cfs[29] = {20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
   1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500};
   double val_threshold[29] = {74.3, 65, 56.3, 48.4, 41.7, 35.5, 29.8, 25.1, 20.7, 16.8, 13.8, 11.2, 8.9, 7.2, 6, 5,
   4.4, 4.2, 3.7, 2.6, 1, -1.2, -3.6, -3., -1.1, 6.6, 15.3, 16.4, 11.6};
   int size_cfs = sizeof(val_cfs) / sizeof(double);
   
   double threshold;
   interp1(val_cfs, size_cfs, val_threshold, &cf, 1, &threshold);
   
   return threshold;
}



double db2amp(double dblevel){
   return pow(10, ((dblevel)/20));
}


int findNearestNeighbourIndex(double value, double* x, int len ){
   double dist;
   int idx;
   int i;
   idx = -1;
   dist = DBL_MAX;
   for ( i = 0; i < len; i++ ) {
      double newDist = value - x[i];
      if ( newDist >= 0 && newDist < dist ) {
         dist = newDist;
         idx = i;
      }
   }
   return idx;
}

void interp1(double* x, int size_x, double *y, double* xx, int size_xx, double* yy){
   double dx, dy, *slope, *intercept;
   int i, indiceEnVector;
   
   slope=(double *)calloc(size_x, sizeof(double));
   intercept=(double *)calloc(size_x, sizeof(double));
   for(i = 0; i < size_x; i++){
      if(i<size_x-1){
         dx = x[i + 1] - x[i];
         dy = y[i + 1] - y[i];
         slope[i] = dy / dx;
         intercept[i] = y[i] - x[i] * slope[i];
      }else{
         slope[i]=slope[i-1];
         intercept[i]=intercept[i-1];
      }
   }
   for (i = 0; i < size_xx; i++) {
      indiceEnVector = findNearestNeighbourIndex(xx[i], x, size_x);
      if (indiceEnVector != -1)
         yy[i]=slope[indiceEnVector] * xx[i] + intercept[indiceEnVector];
      else
         yy[i]=DBL_MAX;
   }
   free(slope);
   free(intercept);
}
