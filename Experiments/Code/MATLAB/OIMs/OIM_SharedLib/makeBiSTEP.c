/*
 *=============================================================
 * %--------------------------------------------------
 * %  [STEP_L, STEP_R, bm_L, bm_R] = makeBiSTEP(x,fs, tranFn, lowcf,highcf,numchans,frameshift,ti,compression,outermiddle)
 * %
 * %  x             input signal
 * %  fs            sampling frequency in Hz
 * %  tranFn        binaural SPL transfer function
 * %  lowcf         centre frequency of lowest filter in Hz (100)
 * %  highcf        centre frequency of highest filter in Hz (7500)
 * %  numchans      number of channels in filterbank (34)
 * %  frameshift    interval between successive frames in ms (10)
 * %  ti            temporal integration in ms (8)
 * %  compression   type of compression ['cuberoot','log','none'] ('none')
 * %  outermiddle   outermiddle transfer function ['iso', 'terhardt', 'none'] ('none')
 * %
 * %  Author:       Yan Tang
 * %  Created:      31/10/2014
 * %  Modified:     28/06/2017
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
#define IN_tranFn       prhs[2]
#define IN_lowcf        prhs[3]
#define IN_highcf       prhs[4]
#define IN_numchans     prhs[5]
#define IN_frameshift   prhs[6]
#define IN_ti           prhs[7]
#define IN_compression  prhs[8]
#define IN_outermiddle  prhs[9]


/*=======================
 * Output arguments
 *=======================
 */
#define OUT_STEP_L     plhs[0]
#define OUT_STEP_R     plhs[1]
#define OUT_bm_L       plhs[2]
#define OUT_bm_R       plhs[3]

/*=======================
 * Useful Const 9e-4
 *=======================
 */
#define BW_CORRECTION   1.019
#define VERY_SMALL_NUMBER  1e-200
#define ABSOLUTE_AMPLITUDE_THRESHOLD  1e-10
#define SPL_REF  2e-5
#define SPL_THRESHOLD  -100 //dB
#define haircellgain   10000


#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif


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


/*=======================
 * Main Function
 *=======================
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
   double *x, *ratemap_l, *ratemap_r, *bm_l, *bm_r, *fn, fs;
   double *senv_l, *senv_r;
   int i, j, chan, numchans;
   int nsamples, nsamples_padded, frameshift_samples, numframes;
   double lowcf, highcf, frameshift, ti, lowErb, highErb, spaceErb, cf;
   double a, tpt, tptbw, gain, intdecay, intgain, sumEnv_l, sumEnv_r, tmp_l, tmp_r, threshold;
   double p0r, p1r, p2r, p3r, p4r, p0i, p1i, p2i, p3i, p4i;
   double a1, a2, a3, a4, a5, cs, sn, u0r, u0i;
   double senv1, qcos, qsin, oldcs, coscf, sincf;
   double *pstatus, status[5],fire;
   int audible;
   char compression[20], outermiddle[20], haircell[10];
   
   
   /*=========================================
    * input arguments
    *=========================================
    */
   
   if (nrhs < 3) { mexPrintf("??? Not enough input arguments.\n"); return; }
   
   
   if (nrhs < 4) lowcf = 100;
   else lowcf = mxGetScalar(IN_lowcf);
   
   if (nrhs < 5) highcf = 7500;
   else highcf = mxGetScalar(IN_highcf);
   
   if (nrhs < 6) numchans = 34;
   else numchans = (int) mxGetScalar(IN_numchans);
   
   if (nrhs < 7) frameshift = 10;
   else frameshift = mxGetScalar(IN_frameshift);
   
   if (nrhs < 8) ti = 8;
   else ti = mxGetScalar(IN_ti);
   
   if (nrhs < 9) strcpy(compression, "none");
   else mxGetString(IN_compression, compression, 19);
   
   if (nrhs < 10) strcpy(outermiddle, "none");
   else mxGetString(IN_outermiddle, outermiddle, 19);
   
   
   if (nrhs > 10) { mexPrintf("??? Too many input arguments.\n"); return; }
   if (nlhs > 4) { mexPrintf("??? Too many output arguments.\n"); return; }
   
   x = mxGetPr(IN_x);
   i = mxGetN(IN_x);
   j = mxGetM(IN_x);
   if (i>1 && j>1) { mexPrintf("??? Input x must be a vector.\n"); return; }
   fn = mxGetPr(IN_tranFn);
   fs = mxGetScalar(IN_fs);
   
   nsamples = getMax(i,j);
   frameshift_samples = getRound(frameshift*fs/1000);
   numframes = (int)ceil((double)nsamples / (double)frameshift_samples);
   nsamples_padded = numframes*frameshift_samples;
   
   /*=========================================
    * output arguments
    *=========================================
    */
   OUT_STEP_L = mxCreateDoubleMatrix(numchans, numframes, mxREAL);
   OUT_STEP_R = mxCreateDoubleMatrix(numchans, numframes, mxREAL);
   ratemap_l = mxGetPr(OUT_STEP_L);
   ratemap_r = mxGetPr(OUT_STEP_R);
   
   
   OUT_bm_L = mxCreateDoubleMatrix (numchans, nsamples, mxREAL);
   OUT_bm_R = mxCreateDoubleMatrix (numchans, nsamples, mxREAL);
   bm_l = mxGetPr(OUT_bm_L);
   bm_r = mxGetPr(OUT_bm_R);
   
   
   lowErb = HzToErbRate(lowcf);
   highErb = HzToErbRate(highcf);
   
   if (numchans > 1)  spaceErb = (highErb-lowErb)/(numchans-1);
   else  spaceErb = 0.0;
   
   /* Smoothed envelope */
   senv_l = (double*) mxCalloc(nsamples_padded, sizeof(double));
   senv_r = (double*) mxCalloc(nsamples_padded, sizeof(double));
   
   tpt = 2 * M_PI / fs;
   intdecay = exp(-(1000.0/(fs*ti)));
   intgain = 1-intdecay;
   
   for (chan=0; chan<numchans; chan++){
      
      cf = ErbRateToHz(lowErb+chan*spaceErb);
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
         senv1 = sqrt(u0r*u0r+u0i*u0i) * gain + intdecay*senv1;
         senv_l[i] = senv1 * fn[chan];
         senv_r[i] = senv1 * fn[numchans+chan];
         
         
         bm_l[chan+numchans*i] = ( u0r * qcos + u0i * qsin ) * gain * fn[chan];
         bm_r[chan+numchans*i] = ( u0r * qcos + u0i * qsin ) * gain * fn[numchans+chan];
         
         
         
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
         senv1 = sqrt(u0r*u0r+u0i*u0i) * gain + intdecay*senv1;
         senv_l[i] = senv1 * fn[chan];
         senv_r[i] = senv1 * fn[numchans+chan];
         
//             senv1 = senv[i] = sqrt(u0r*u0r+u0i*u0i) * gain + intdecay*senv1;
      }
      
      /*==================================================================================
       * we take the mean of the smoothed envelope as the energy value in each frame
       * rather than simply sampling it.
       * ratemap(c,:) = intgain.*mean(reshape(smoothed_env,frameshift_samples,numframes));
       *==================================================================================
       */
      for (j=0; j<numframes; j++){
         sumEnv_l = 0.0;
         sumEnv_r = 0.0;
         for (i=j*frameshift_samples; i<(j+1)*frameshift_samples; i++){
            sumEnv_l += senv_l[i];
            sumEnv_r += senv_r[i];
         }
         tmp_l = intgain * sumEnv_l / frameshift_samples;
         tmp_r = intgain * sumEnv_r / frameshift_samples;
         
         if (tmp_l < ABSOLUTE_AMPLITUDE_THRESHOLD){
            tmp_l = ABSOLUTE_AMPLITUDE_THRESHOLD;
         }
         if (tmp_r < ABSOLUTE_AMPLITUDE_THRESHOLD){
            tmp_r = ABSOLUTE_AMPLITUDE_THRESHOLD;
         }
         
         ratemap_l[chan+numchans*j] = tmp_l;
         ratemap_r[chan+numchans*j] = tmp_r;
      }
   }
   
   if (strcmp(compression, "cuberoot") == 0){
      for (i=0; i<numchans*numframes; i++){
         ratemap_l[i] = pow(ratemap_l[i], 0.3);
         ratemap_r[i] = pow(ratemap_r[i], 0.3);
      }
   }else if (strcmp(compression, "log") == 0){
      for (i=0; i<numchans*numframes; i++){
         ratemap_l[i] = 20 * log10(ratemap_l[i]); //(20*1e-6)
         ratemap_r[i] = 20 * log10(ratemap_r[i]); //(20*1e-6)
      }
   }
   
   mxFree(senv_l);
   mxFree(senv_r);
   return;
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