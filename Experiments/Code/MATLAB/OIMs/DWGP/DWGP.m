function objscore = DWGP(sig, noise, fs, ref, presentationLevel)
% objscore = DWGP(sig, ref, noise, fs) computes the distortion-weighted glimpse proportion
% indicating speech intelligibility in given noise signal.
%
% input:
%        sig           input modified speech signal
%        noise         input noise signal
%        ref           input original speech signal
%        fs            sampling frequency in Hz
%
% output:
%        objscore      objective scores in a structure: it returns both DWGP and original GP score.
%
% usage:
%        objscore = DWGP(sig, noise, fs, ref, presentationLevel);
%        If the input speech signal is a modified signal by any algorithm, the reference signal
%        should be the corresponding original/unmodified signal, otherwise the input speech and
%        reference speech signal should be identical. However if the duration of modified and
%        original signal is different, the modified speech itself will be always used as the
%        reference even if the original speech signal is supplied.
%
% Author: Yan Tang
% Date: 22.07.2012
% Modified: 12.03.2019
%
% External dependencies: makeRateMap, BIF

if nargin < 4
    ref = sig;
end

if nargin < 5
    presentationLevel = 63; % dB
end


fr_l    = 100; %lower frequency bound (Hz)
fr_h    = 7500;%upper frequency bound (Hz)
ERBnum  = 34;  %number of filters
framerate       = 10;	% window size (ms)
tmp_int			= 8;	% temporal integration (ms)
LC      = 3;   %local criteria forclc glimpses (dB)
lc_thrd = 25;  %local hearing threshold (dB)


% Adjust the signal level
rmslevel = power(10, presentationLevel/20);
k     = rmslevel / rms(sig); % if the  presentation level is meant for mixture, it then should be rms(sig+noise)
sig   =  k .* sig;
noise =  k .* noise;
ref   =  k .* ref;

% Make STEP representations for the signals
% [STEP_sig, ~, cfs]   = makeRateMap_IHC(sig, fs,fr_l,fr_h,ERBnum, framerate, tmp_int,'none','none','iso',1);
% STEP_orig   = makeRateMap_IHC(ref, fs,fr_l,fr_h,ERBnum, framerate, tmp_int,'none','none','iso',1);
% STEP_noise = makeRateMap_IHC(noise, fs,fr_l,fr_h,ERBnum, framerate, tmp_int,'none','none','iso',1);
% STEP_mix   = makeRateMap_IHC(sig + noise, fs,fr_l,fr_h,ERBnum, framerate, tmp_int,'none','none','iso',1);

[STEP_sig, ~, cfs]    =   makeRateMap(sig, fs,fr_l,fr_h,ERBnum,framerate,tmp_int,'none','iso');
STEP_orig             =   makeRateMap(ref, fs,fr_l,fr_h,ERBnum,framerate,tmp_int,'none','iso');
STEP_noise            =   makeRateMap(noise, fs,fr_l,fr_h,ERBnum,framerate,tmp_int,'none','iso');
STEP_mix              =   makeRateMap(sig + noise, fs,fr_l,fr_h,ERBnum,framerate,tmp_int,'none','iso');

% check if the duration of modified speech is changed after being modified
% if yes use modified signal itself as reference
if length(sig) == length(ref)
    STEP_ref = STEP_orig;
else
    STEP_ref = STEP_sig;
end

% compute sound pressure level in air
STEP_sigEN = 20 * log10(STEP_sig);
STEP_noiseEN = 20 * log10(STEP_noise);

% find those regions on speech with local SNR above given local criteira: 3 dB
mask_LC = STEP_sigEN > (STEP_noiseEN + LC);
% find those regions on speech above local hearing level: 20 dB
mask_audi = STEP_sigEN >= lc_thrd;
% the final valid glimpsing points must both outstand noise and be audible.
mask = mask_LC & mask_audi;


% get distortion factors
dstfctrs = getDistortionFactors(STEP_ref, STEP_mix);

weights = BIF(cfs,1);
xGPchan = zeros(ERBnum,1);
for chan = 1:ERBnum
    xGPchan(chan) = dstfctrs(chan) * weights(chan) * sum(mask(chan, :));
end

[~,framenum] = size(STEP_sig);
objscore.DWGP = sum(xGPchan)/framenum * (length(sig)/length(ref));
% log-compress DWGP score
delta = 0.01;
objscore.DWGP = log((objscore.DWGP+delta)/delta)/log((1+delta)/delta);

% original glimpse proportion
objscore.GP = 100 * sum(mask_LC(:)) / (ERBnum * framenum);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRIVATE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function factors = getDistortionFactors(STEP_sig, STEP_mix)
% factors = getDistortionFactors(STEP_sig, STEP_mix) computes the distortion degree of speech by
% given noise on STEP envelopes. The distortion degree for each channel is represented by the
% correlation coefficient between the clean speech and the correupted (modified speech + noise)
% signal. The lower the correlation is, the more the signal is distorted by the noise masker.
%
% input:
%        STEP_ref       STEP envelope of the reference signal
%        STEP_mix       STEP envelope of the noise-corrupted signal
% output:
%        factors        a vector of the distortion factors (between 0 and 1) for each channel on
%                       the STEP. The size of this vector should be as same as the number of rows
%                       on the STEP.
%
% Author: Yan Tang

[bands,~] = size(STEP_sig);
factors = zeros(1,bands);

for idx = 1: bands
    factors(idx) = corr(STEP_sig(idx,:)',STEP_mix(idx,:)');
end
factors = abs(factors);


