function [gpoim, gpaud, gp] = GPoim(s, n, fs, s_o, prtLevel)
% gpoim = GPoim(s, n, fs, s_o, prtLevel) computes the normalised estimate of the number of
% above-threshold speech target regions represented at the level of the auditory nerve.
%
% input:
%        s           input speech signal
%        n           input noise signal
%        fs          sampling frequency, Hz
%        s_o         unmodified form of s. This argument only takes effect if s differs from s_o in
%                    duration
%        prtLevel    presentation level of s+n mixture, SPL [default: 74 dB SPL]
%
% output:
%        gpoim       index of glimpse metric
%        gp          original glimpse proportion in Cooke (2006)
%
% usage:
%        [gpoim, gp] = GPoim(speech, noise, fs)
%        [gpoim, gp] = GPoim(modified, noise, fs, unmodified)
%        [gpoim, gp] = GPoim(modified, noise, fs, unmodified, 63)
%
%
% Authors: Yan Tang and Martin Cooke
% Modified: July 28, 2015

if nargin < 5
   prtLevel = 74; %dB
end

if nargin < 4
   s_o = s;
end

fr_l        = 100;   %lower frequency bound (Hz)
fr_h        = 7500;  %upper frequency bound (Hz)
framerate   = 10;    %window size (ms)
ti          = 8;     %temporal integration (ms)
ERBnum      = 34;    %number of filters
a           = 3;     %local threshold for defining glimpse (dB)
HL          = 25;    %hearing threshold (dB)

% Adjust the signal level
rmslevel = power(10, prtLevel/20);
k   = rmslevel / rms(s + n);
s   = k .* s;
n   = k .* n;

% Make STEP representations for the signals
% STEP_s  =  makeRateMap4IHC(s, fs, fr_l, fr_h, ERBnum, framerate, ti, 'log', 'iso', 0);
% STEP_n  =  makeRateMap4IHC(n, fs, fr_l, fr_h, ERBnum, framerate, ti, 'log', 'iso', 0);
[STEP_s, ~, cfs]   = makeRateMap_IHC(s, fs,fr_l,fr_h,ERBnum,framerate,ti,'log','none','iso',1);
STEP_n             = makeRateMap_IHC(n, fs,fr_l,fr_h,ERBnum,framerate,ti,'log','none','iso',1);



% define putative glimpses using definition of Cooke (2006), as well as those above HL.
glimpses     =  STEP_s > (STEP_n + a);
glimpses_aud =  STEP_s > max(STEP_n + a, HL); %Eq.2

% compute IHC of the signal+noise mixture, and use it to further validate glimpses with a forward
% masking effect.
IHC_mix =  makeRateMap4IHC((s+n)./k, fs, fr_l, fr_h, ERBnum, framerate, ti, 'none', 'none', 1);
glimpses_FM = forwardmasking(IHC_mix, glimpses_aud);

% compute speed-up ratio
lambda = length(s_o) / length(s);

% compute normalised glimpse count and cap it to 1 after being weighted by speed-up factor
fnum = size(STEP_s, 2);
gpoim = min(1/lambda * sum(glimpses_FM(:))/(fnum * ERBnum), 1); % Eq. 4
gpaud = min(1/lambda * sum(glimpses_aud(:))/(fnum * ERBnum), 1); % Eq. 4

% compute original glimpse proportion
gp = sum(glimpses(:))/(fnum * ERBnum); % Eq. 1

% transform  to OIM index
% add a very small number to avoid log(0) issue.
delta = 0.01;
gpoim = log(1 + gpoim/delta) / log(1 + 1/delta); %Eq. 5
gpaud = log(1 + gpaud/delta) / log(1 + 1/delta); %Eq. 5



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRIVATE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function glimpses_FM = forwardmasking(IHC_mix, glimpses)
% glimpses_FM = forwardmasking(IHC_mix, glimpses) returns validated glimpses with non-simultaneous
% masking effect (forward masking).
%
% input:
%        IHC_mix        inner hair cell response of speech+noise mixture
%        glimpses       putative glimpses above hearing threshold
%
% output:
%        glimpses_FM    genuine glimpses
%
% Author: Yan Tang


[bands, frames] = size(IHC_mix);
glimpses_FM = zeros(bands,frames);

% across all frequencies
for idx = 1:bands  
   glimpses_f = glimpses(idx, :);
   IHC_f = IHC_mix(idx,:);
   glimpses_g = zeros(size(IHC_f));
   
   % detect peaks in this channel
   [~, peaks] = findpeaks(IHC_f);
   peaknum = length(peaks);
   
   % validate each peak
   for peakidx = 1:peaknum
      currentpeak = peaks(peakidx);
      if peakidx < peaknum
         % get the index of the next peak
         nextpeak = peaks(peakidx+1);
      else
         nextpeak = frames+1;
      end
      
      % current peak is speech-dominant
      if glimpses_f(currentpeak)
         % contains last valid index
         sequence = [];
         
         pointer_fore = currentpeak + 1;
         sequence = cat(2,sequence,currentpeak);
         while pointer_fore <= frames
            % check regions after the peak consecutively until next peak
            % the frame must be glimpsed according to STEP
            if pointer_fore < nextpeak && glimpses_f(pointer_fore)
               sequence = cat(2,sequence,pointer_fore);
               pointer_fore = pointer_fore +1;
            else
               break;
            end
         end         
         glimpses_g(sequence) = 1;
      end
   end 
   glimpses_FM(idx,:) = glimpses_g;
end