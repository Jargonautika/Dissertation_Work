function [PRout, a, AI] = Fai(Resp,FlagArr,N,Hnd,PRin)

if nargin~=5, error('Program error: Improper # of input arguments to FAI.m'); end;

SetupFlag = FlagArr(1);
% FlagArr(2,3) not used
bH        = FlagArr(4); 
extNoise  = FlagArr(5);

f = Resp(1,:);
R = Resp(2,:);

if extNoise,
	fB = N(1,:);
	Bx = N(2,:);
end;	

if SetupFlag,
	prm = PRin;
end;	

% edit p and bt here if GUI does not work with your version of MATLAB
if SetupFlag == 0 
 % proficiency factor
 p = 1.0;
 % Talking Level beta_T (see Fletcher 1953, p. 313)
 bt = 68;
else
 p  = str2num(char(prm(1)));
 bt = str2num(char(prm(2)));
end;

% definitions of constants in Charts 3,4, and 5
f0 = [200 300 400 500 600 700 800 900 1000 1250 1500 1750 2000,...
2500 3000 4000 5000 6000 7000 8000]';
f1 = [510 640 730 820 900 980 1070 1160 1250 1360 1470 1590 1710,...
1860 2020 2210 2430 2700 3100 3930]';
f2 = [320 500 630 750 870 990 1120 1260 1410 1570 1760 1960 2190,...
2440 2740 3090 3520 4080 4890 6600]';
f3 = [310 470 610 740 880 1020 1170 1330 1520 1720 1950 2200 2470,... 
2770 3090 3480 3920 4480 5210 6370]';
r = [1.33 1.06 1.02 1 1 1 1 1 1 1 1 1 1 1 1 1.03 1.06 1.1 1.17 1.39];
ZF = [46 59 62 63 63 63 63 63 63 63 63 63 63 63 63 61 59 57 53 44];
delta = 0.01*[-3 -1 1 3 4 4 3 2 0 -2 -4 -5 -6 -6 -5 -4 -2 0 1 3];
xV = [0 5 10 15 20 25 30 35 40 45 50 60 70 80 90 100 110 120];
Phi = [0 .22 .45 .65 .85 .93 .97 .99 1 1 1 1 1 1 1 1 1 1];
V = [0 .05 .11 .2 .3 .41 .54 .66 .75 .83 .89 .98 1 1 1 1 1 1];
bo_dash = [34 26 20.2 15.2 12.2 9.9 8.1 6.8 5.6 3.7 2.4 1.7 0.6,...
-0.6 -1.2 -0.1 2.8 7.1 12.1 15.1];           % Table 81
dkbo = [-16.8 -9 -3.2 1.8 5 7.4 9.5 11 12.4 14.8 16.6 17.7 19.3,...
21.4 22.7 23.1 21.6 18.6 14.7 12.6]';


% interpolate masking noise at frequencies f0
if extNoise
 B0 = interp1(log10(fB),Bx,log10(f0));
else
 B0 = -500*ones(size([1:20]'));
end;

% interpolate filter response at frequencies f0,f1,f2, and f3 
R0 = interp1(log10(f),R,log10(f0));
R1 = interp1(log10(f),R,log10(f1));
R2 = interp1(log10(f),R,log10(f2));
R3 = interp1(log10(f),R,log10(f3));

% Calculate masking 
M0 = 10*log10(10.^((B0+dkbo-bH-4)/10)+1); % steps 1b and 1c

OriginalFletcher = 1;
if OriginalFletcher
   %	Step 2a. "From the curve of R-M versus f found in step 1 read the
   %	values of R - M corresponding to each of the frequencies in
   %	column 1 of Chart 4 and record in column 3"
   %
   %	This implements the approach described by Fletcher and Galt.
   %	First, the difference R - M is calculated and then the curve is interpolated.
   %	Because the sample points in f0 are widely spaced, this approach
   %	can miss sharp resonant peaks. [see System I with damping 2.5
   %	(Fig 205 in Fltecher 1953)]
   R0_M0 = R0-M0;
   R1_M1 = interp1(log10(f0),R0-M0,log10(f1));
   R2_M2 = interp1(log10(f0),R0-M0,log10(f2));                                 
   R3_M3 = interp1(log10(f0),R0-M0,log10(f3));
else
   %	This approach interpolates the effective masking noise first and then calculates
   %	the difference R - M. Because M is the effective masking, it cannot have steep gradients.
   %	In general, M will be smoother than R. This approach does not interpolate R1, R2, or R3
   %	and minimizes the chance that sharp resonant peaks are "missed."
   
   MOatR1 = interp1(log10(f0),M0,log10(f1));
   MOatR2 = interp1(log10(f0),M0,log10(f2));                                     
   MOatR3 = interp1(log10(f0),M0,log10(f3));

   R1_M1  = R1 - MOatR1;
   R2_M2  = R2 - MOatR2;
   R3_M3  = R1 - MOatR3;

	clear MOatR1 MOatR2 MOatR3;
end;

% steps 6a - 6g 
R1bar = 10*log10(sum(10.^(R1/10))/20);
R4bar = 40*log10(sum(10.^(R2/40))/20);
Rbar = 0.5*(R1bar+R4bar);

% steps 2,3, and 4
RM1bar = 10*log10(sum(10.^(R1_M1/10))/20);
RM4bar = 40*log10(sum(10.^(R2_M2/40))/20);
RMbar = 0.5*(RM1bar + RM4bar);

% step 5
W = xW2W((RMbar-R3_M3).*r');
WFN0 = W;
FN0 = sum(W)/20;

% step 7
E0delta = sum(((W-0.99)>0).*delta');
nN0 = 0.5*sum((W-0.9).*((W-0.9)>0));

% SELF MASKING OF SPEECH - downward masking curve
% step 6
y_down = -1000*ones(size(f3)); % initialization

% masking curves start at f(INDX)
slope_critical = 40/(0.5*log10(2));
f_sc = (diff(R)./diff(log10(f)))>=slope_critical;
INDX = find(diff(f_sc)<0)+1;	

if length(INDX)>0
 for k=1:length(INDX)
  % need x support points to extend masking curve below 310 Hz
  x = ceil(-2/log10(2)*log10(310/f(INDX(k))));
  x = [0:x];
  % masking curve piecewise log-lin between fT[i]
  fT = f(INDX(k))*2.^(-x/2);
  yT(1) = R(INDX(k))-63+interp1(log10([100; f3; 1E4]),...
  [21.6 ZF 32],log10(f(INDX(k))));
  yT(2) = yT(1)-40;
  for l=3:length(x);
   yT(l) = yT(l-1) - 0.5*(yT(l-2)-yT(l-1));
  end;
  % interpolate masking curve at frequencies f3 
  y_down = max(y_down,interp1(log10([1E4+1 max(fT)+0.1 fT]),...
  [-1E3 -1E3 yT(1:length(fT))],log10(f3)));
  clear yT;
 end;	
end;	

% upward masking curve
gamma = xg2Ps(nN0/FN0) + nN0/FN0*E0delta; % tentative gamma
ZFf = interp1(log10([100; f3; 1E4]),[21.6 ZF 32],log10(f)');

% start iteration
CONVERGENCE = 0;
if ~CONVERGENCE,
 % critical slope (dB/octave) sigma as function of frequency 
 sigma = 75-0.5*((ZFf)'+R-RM1bar+gamma*(RM1bar-RM4bar));			    
 % masking curves start at f(INDX)
 INDX = find(diff([diff(R)./diff(log10(f)) 0]<(sigma/log10(0.5)))>0)+1;
 y_up = -1000*ones(size(f3)); % initialization

 if length(INDX)>0
  for k=1:length(INDX)
   yT(1) = R(INDX(k))-63+interp1(log10([100; f3; 1E4]),...
   [21.6 ZF 32],log10(f(INDX(k))));
   yT(2) = sigma(INDX(k))/log10(0.5)*log10(1E4/f(INDX(k)))+yT(1);
   % upward masking at frequencies f3 
   y_up = max(y_up,interp1(log10([99 f(INDX(k))-0.1, ...
   f(INDX(k)) 1E4+1]),[-1E3 -1E3 yT(1) yT(2)],log10(f3)));
   clear yT;
  end;
 end;	
 
 dR = max(y_down,y_up)-R3;
 r = 68./(ZF'+5-(((63-ZF'+dR)>0).*(63-ZF'+dR)));
 r = (r>0).*r + (r<0)*1000;
 W = xW2W((Rbar-R3).*r);
 WFNM = W;
 FNM = sum(W)/20;
 
 % step 7
 nNM = 0.5*sum((W-0.9).*((W-0.9)>0)); 
 EMdelta = sum(((W-0.99)>0).*delta');
 
 % step 8
 gamma_new = xg2Ps(0.5*(nN0/FN0+nNM/FNM))+0.5*(0.5*(nN0/FN0+nNM/FNM))...
 *(E0delta+EMdelta);
 if (abs(gamma-gamma_new)<(1E-5)*gamma_new) CONVERGENCE = 1; end;
 gamma = gamma_new;
end; % of iteration loop 

% step 9
a0 = bH-bt+12-RM1bar;

a = a0+xV+gamma*(RM1bar-RM4bar)*Phi;

% step 10
xE = a+R1bar+bt-bH-12;
E = xE2E(xE);

% step 11
F = 2/3*FNM + 1/3*FN0;

% step 12
if extNoise
 beta = dkbo+bo_dash'+B0;
 [am,beta_m,fm] = Step12a(f0,beta);
 if ((fm>4000)|(beta_m>100)|(beta_m<40)) 
  disp('Tables 5 and 6 on Chart 6 were extrapolated'); 
 end;
 Km = interp1(100*[1 2 3 4 5 7 10 14 20 30 40 100],...
 [9 11 13 14 15 14 12 10 7 -2 -11 -11],fm);
 H = 1-am*a2j(a-(beta_m-bt-R1bar-Km));
else
 H = ones(size([1:18]));
end;

% step 13
 AI = H.*F.*E.*V;

% delete values for which xE>129 dB
INDX = find(xE>129);
if length(INDX)==0 
 k=length(a); 
  else 
   k = INDX(1);
end;

% function output
a = a(1:k);
AI = AI(1:k); 
PRout = [p;bt;bH;(gamma*(RM1bar-RM4bar) > -21.739)];


set(Hnd(32),'enable','on');
set(Hnd(33),'enable','on');
set(Hnd(34),'enable','on');
set(Hnd(36),'enable','on');
set(Hnd(37),'enable','on');

set(Hnd(41),'enable','on');
set(Hnd(42),'enable','on');

% display internal variables:
disp('.');
disp('Internal variabes: ' )
disp(['gamma   : ' num2str(gamma)]); 
disp(['FN0     : ' num2str(FN0)]);
disp(['FNM     : ' num2str(FNM)]);
disp(['RM1bar  : ' num2str(RM1bar)]);
disp(['RM4bar  : ' num2str(RM4bar)]);
disp(['alpha_0 : ' num2str(a0)]);
disp('.');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            PRIVATE FUNCTIONS                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function W = xW2W(xw);
% polynomial approximation of Table (1) Chart 6
W = -2.090462E-16*xw.^10 + 7.753209E-14*xw.^9 - 1.199329E-11*xw.^8 ...
+ 1.003562E-9*xw.^7 - 4.932304E-8*xw.^6 + 1.451754E-6*xw.^5 ...
- 2.516821E-5*xw.^4 + 2.475973E-4*xw.^3 -1.605801E-3*xw.^2 + ...
1.659429E-4*xw + 0.9991724;

% set W(i)=1 if xw(i)<0
W = ((~(xw<=0)).*W)+(xw<=0);

% set W(i)=0 if xw(i)>68
W = W.*(~(xw>=68));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  Ps = xg2Ps(xg);
% approximation of Table (3) Chart 6

P0 = 1.2*(xg<0.35);

Ps_Table = [0.35 1.20
0.36	1.198
0.37	1.196
0.38	1.194
0.39	1.192
0.4 	1.185
0.41	1.176
0.42	1.167
0.43	1.158
0.44	1.149
0.45	1.14
0.46	1.128
0.47	1.114
0.48	1.097
0.49	1.083
0.5 	1.07
0.51	1.06
0.52	1.045
0.53	1.03
0.54	1.01
0.55	0.99
0.56	0.97
0.57	0.94
0.58	0.905
0.59	0.87
0.6 	0.84
0.61	0.805
0.62	0.77
0.63	0.74
0.64	0.71
0.65	0.689
0.66	0.655
0.67	0.625
0.68	0.595
0.69	0.56
0.7 	0.53
0.71	0.5
0.72	0.475
0.73	0.445
0.74	0.415
0.75	0.39
0.76	0.36
0.77	0.335
0.78	0.31
0.79	0.285
0.8 	0.265
0.81	0.24
0.82	0.215
0.83	0.185
0.84	0.155
0.85	0.13
0.86	0.11
0.87	0.09
0.88	0.07
0.89	0.055
0.9 	0.045
0.91	0.035
0.92	0.025
0.93	0.015
0.94	0.008
0.95	0.003
0.96	0.000];

P1 = spline(Ps_Table(:,1),Ps_Table(:,2),xg);
P1 = P1.*((xg<=0.96)&(xg>=0.35));

Ps = zeros(size(xg))+P0+P1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function E = xE2E(xE);
% polynomial approximation of Table (2) Chart 6
% resuts for xE>129dB invalid; eliminate corresponding data
% points at the end.

E = -1.800781E-18*xE.^10 + 1.002433E-15*xE.^9 -1.960203E-13*xE.^8 ...
+ 1.046832E-11*xE.^7 +1.472283E-9*xE.^6 -1.939179E-7*xE.^5 ...
-3.345258E-6*xE.^4 + 1.821276E-3*xE.^3 -0.1229845*xE.^2 + ...
3.283362*xE -26.45757;
E = ((xE>70).*E)+(xE<=70);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a,b,f] = Step12a(f0,beta);
% table look up of step 12a  Chart 6 Tables (5) and (6)
% a is a linear function of beta, but saturated at a=0;
% linear interpolation in frequency
b = [-500 300];
f = 100*[1 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 100];
slope = 0.001*[6 6 8 10 10 10 10.75 11 12 11 10.4 11 8 5 5 5 5]';
offset = -0.1*[4 4 4.9 5.8 5.3 4.8 4.975 5.6 6.8 5.6 4.8 5.2 ...
3.2 1.2 2.2 3.2 3.2]';

Tbl = (slope*b) + [offset offset];
a = diag(interp2(b,f,Tbl,beta,f));
INDX = find((a==max(a)));
b = beta(INDX(1));
f = f0(INDX(1)); 
a = max(a);
a = a*(a>0); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function J = a2j(a);
% polynomial approximation of Table (4) Chart 6
a = abs(a);
J = -2.400633E-13*a.^10 + 4.464949E-11*a.^9 -3.497583E-9*a.^8 + ...
1.502497E-7*a.^7 -3.869212E-6*a.^6 + 6.148607E-5*a.^5 ...
-5.97494E-4*a.^4 + 3.338717E-3*a.^3 -9.940277E-3*a.^2 + ...
9.203489E-3*a + 0.9996684;
J = (J.*(a>3))+(a<=3);
J = J.*(a<39);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
