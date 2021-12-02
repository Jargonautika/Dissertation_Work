function [ratemap,cf] = makeRateMap4IHC(x,fs,lowcf,highcf,numchans,frameshift,ti,compression, outermiddle, isIHC)
% [ratemap,cf] = makeRateMap4IHC(x,fs,lowcf,highcf,numchans,frameshift,ti,compression, outermiddle, isIHC, reflevel)
% generates a spectro-temporal excitation pattern (STEP) representation for the signal x, 
% modelling the smoothed and compressed representation of the envelope of the basilar 
% membrane response to sound. The waveform is initially processed by gammatone filterbanks 
% using a pole-mapping procedure described in Cooke (1993). The Hilbert envelope in each 
% channel of the filterbank is smoothed and downsampled. If specified, the frequency-dependent 
% IHC response is calculated using an approach introduced in Cooke (1993).
%
% input:
%        x             input signal
%        fs            sampling frequency in Hz (Default: 8000 Hz)
%        lowcf         centre frequency of lowest filter in Hz (Default: 100 Hz)
%        highcf        centre frequency of highest filter in Hz (Default: fs/2 Hz)
%        numchans      number of channels in filterbank (Default: 34 ERB)
%        frameshift    interval between successive frames in ms (Default: 10 ms)
%        ti            temporal integration in ms (Default: 8 ms)
%        compression   type of compression ['cuberoot','log','none'] (Default: 'none')
%        outermiddle   type of outer/middle ear model used to weight the frequencies ['iso', 'terhardt', 'none']
%                      (Default: 'none')
%        isIHC         A flag indicating the output type [0 1]. '0': STEP; '1': IHC response
%                      (Default: '0')
% output:
%        ratemap       STEP representation or IHC response of the input signal x
%        cf            centre frequences for each band
%
% Authors: Martin Cooke and Yan Tang
% Modified: July 28, 2015


if nargin < 2
   fs = 8000; %Hz
end

if nargin < 3
   lowcf = 100; %Hz
end

if nargin < 4
   highcf = round(fs / 2); %Hz
end

if nargin < 5
   numchans = 34; %GT filters
end

if nargin < 6
   frameshift = 10; %ms
end

if nargin < 7
   ti = 8; %ms
end

if nargin < 8
   compression = 'none';
end

if nargin < 9
   outermiddle = 'none';
end

if nargin < 10
   isIHC = false;
end

cf = MakeErbCFs(lowcf,highcf,numchans);
frameshift_samples = round(frameshift*fs/1000);
framecentres = 1:frameshift_samples:length(x);
numframes = length(framecentres);
x = x(:)';
xx=zeros(1,numframes*frameshift_samples);
xx(1:length(x))=x;

ratemap = zeros(numchans,numframes);
tp = 2*pi;
tpt = tp/fs;
wcf = 2*pi*cf;
kT = (0:length(xx)-1) / fs;
bw = erb(cf)*bwcorrection;
as = exp(-bw*tpt);

gain = ((bw*tpt).^4)/3;
% outer/middle ear transfer function integration
switch lower(outermiddle)
   case 'iso'
      gain = gain .* db2amp(-outerMiddleEar(cf));
   case 'terhardt'
      gain = gain .* db2amp(-terhardt_threshold(cf));
   case 'none'
end


intdecay=exp(-(1000/(fs*ti)));
intgain=1-intdecay;
haircellgain=10000;

for c=1:numchans
   a = as(c);
   q = exp(-1i*wcf(c)*kT).*xx;
   p = filter([1 0],[1 -4*a 6*a^2 -4*a^3 a^4],q);    % filter: part 1
   u = filter([1 4*a a^2 0],[1 0],p);                % filter: part 2
   
   %extract Hilbert envelope
   env = gain(c)*abs(u);
   
   % generated IHC response instead of simple STEP if specified, using approach in Cooke 1993.
   if isIHC
      env = haircellgain .* IHC_cooke(max(20*log10(env./2e-5), -30), fs);
   else
      env = max(env, 1e-10); % avoid zero amplitude which causes -inf value when doing log compression
   end  
      
   smoothed_env = filter(1,[1 -intdecay],env);         % temporal integration
   tmp = intgain.*mean(reshape(smoothed_env,frameshift_samples,numframes)); % downsampling to 100Hz
  
   ratemap(c, :) = tmp;
end

% do compression for STEP
if ~isIHC
   switch compression
      case 'log'
         ratemap = 20 * log10 (ratemap);
      case 'cuberoot'
         ratemap = ratemap .^ 0.3;
      case 'none'
   end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRIVATE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y=bwcorrection
y=1.019;


function y=MakeErbCFs(lfhz,hfhz,n)
% y=MakeErbCFs(lfhz,hfhz,n) generates n centre frequencies in Hz with the given 
% lower and upper boundaries, spaced according to the ERB scale.
%
% input:
%        lfhz:        centre frequency of low freq filter
%        hfhz:        centre frequency of high freq filter
%        n:           number of filters
% output:
%        y:           centre frequencies of the output of the filterbanks
%
% Author: Martin Cooke

y=ErbRateToHz(linspace(HzToErbRate(lfhz),HzToErbRate(hfhz),n));



function y=ErbRateToHz(x)
% y=ErbRateToHz(x) converts ERB rate to Hz
%
% input:
%        x:       ERB number
% output:
%        y:       corresponding Hz
%
% Author: Martin Cooke

y=(10.^(x/21.4)-1)/4.37e-3;



function y=HzToErbRate(x)
% y = HzToErbRate(x) converts Hz to ERB rate
%
% input:
%        x:       frequency in Hz
% output:
%        y:       corresponding ERB number
%
% Author: Martin Cooke

y=(21.4*log10(4.37e-3*x+1));



function y=erb(x)
% y = erb(x) equivalent rectangular bandwidth at a given frequency
%
% input:
%        x:       frequency in Hz
% output:
%        y:       corresponding ERB
%
% Author: Martin Cooke

y=24.7*(4.37e-3*x+1);



function thrsd = terhardt_threshold(cfs)
% y = terhardt_threshold(CFs) computes hearing threshold for given central frequencies 
% using equation supplied in "Calculating virtual pitch", Hearing Research, 
% vol. 1 pp. 155-182, 1979. by E. Terhardt
%
% input:
%        cfs       a vector of centre frequencies in Hz
% output:
%        thrsd     a vector of corresponding thresholds at the given frequencies
%
% Author: Yan Tang

cfs = cfs ./ 1000; % convert to kHz
thrsd = 3.64*(cfs.^(-0.8))-6.5*exp(-0.6*((cfs-3.3).^2))+ 10e-3*(cfs.^4);



function thrsd = outerMiddleEar(cfs)
% h=outerMiddleEar(cfs) returns the outer-middle ear transfer function at the given 
% centre frequencies using the data from ISO 387-9 (1996). Acoustics -- Reference zero 
% for the calibration of audiometric equipment. Part 7: Reference threshold of 
% hearing under free-filed and diffuse-field listening conditions
% Note, only defined for frequencies between 20 and 12,500 Hz.
%
% input:
%        cfs       a vector of centre frequencies in Hz
% output:
%        thrsd     a vector of corresponding thresholds at the specified frequencies
%

if ((min(cfs)<20) || (max(cfs)>12500))
   error('Centre frequency out of range');
end

f=[20 25 31.5 40 50 63 80 100 125 160 200 250 315 400 500 630 800 ...
   1000 1250 1600 2000 2500 3150 4000 5000 6300 8000 10000 12500];
tf=[74.3 65 56.3 48.4 41.7 35.5 29.8 25.1 20.7 16.8 13.8 11.2 8.9 ...
   7.2 6 5 4.4 4.2 3.7 2.6 1 -1.2 -3.6 -3.9 -1.1 6.6 15.3 16.4 11.6];

thrsd = interp1(f,tf,cfs,'PCHIP');



function amp=db2amp(level,ref)
% amp=db2amp(level,ref) converts decibels to amplitude with optional reference level
%
% input:
%        level       sound level in decibels
%        ref         reference level in decibels. It is useful to adopt a standard reference level
%                    of 80 dB to represent an amplitude of 1 for MATLAB sound output purposes.
% output:
%        amp         corresponding amplitude value
%
% Author: Martin Cooke
%

if nargin < 2
   amp=10.^(level./20);
else
   amp=10.^((level-ref)./20);
end

