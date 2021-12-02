function objscore = FMGP(sig, ref, noise, fs)
% objscore = FMGP_LISTA(sig, noise, fs) extends the original glimpse proportion metric with taking
% into account an effect of forward masking by noise masker.
%
% input:
%        sig           input modified speech signal
%        ref           input original speech signal
%        fs            sampling frequency in Hz
%
% output:
%        objscore      objective scores in a structure: it returns both FMGP and original GP score.
%
% usage:
%        objscore = FMGP_LISTA(sig,  noise, 16000);
%
%
% Author: Yan Tang


fr_l    = 100; %lower frequency bound (Hz)
fr_h    = 7500;%upper frequency bound (Hz)
ERBnum  = 34;  %number of filters
P_air   = 2e-5;%reference pressure in air (uPa)
tarRMS  = 0.1; %assumed RMS level
winsize = 40;  %sample points
LC      = 3;   %local criteria for glimpses (dB)
lc_thrd = 20;  %local hearing threshold (dB)
mingnum = 5;   %minimal number of glimpsing points to form a valid glimpse region.

% signal level adjustment
k = tarRMS / rms(sig + noise);
sig   = k .* sig;
noise = k .* noise;

% make STEP representations for the signals
STEP_sig = makeRateMap_IHC(sig, fs,fr_l,fr_h,ERBnum,10,8,'none','none','iso',1);
STEP_ref = makeRateMap_IHC(ref, fs,fr_l,fr_h,ERBnum,10,8,'none','none','iso',1);
STEP_noise = makeRateMap_IHC(noise, fs,fr_l,fr_h,ERBnum,10,8,'none','none','iso',1);
% STEP_sig = makeRateMap_c(sig, fs,fr_l,fr_h,ERBnum,10,8,'none',0,1,0);
% STEP_ref = makeRateMap_c(ref, fs,fr_l,fr_h,ERBnum,10,8,'none',0,1,0);
% STEP_noise = makeRateMap_c(noise, fs,fr_l,fr_h,ERBnum,10,8,'none',0,1,0);
% STEP_sig    =   makeRateMap(sig, fs,fr_l,fr_h,ERBnum,10,8,'none','iso');
% STEP_ref    =   makeRateMap(ref, fs,fr_l,fr_h,ERBnum,10,8,'none','iso');
% STEP_noise  =   makeRateMap(noise, fs,fr_l,fr_h,ERBnum,10,8,'none','iso');

% compute sound pressure level in air
STEP_sigEN = 20*log10(STEP_sig/P_air);
STEP_noiseEN = 20*log10(STEP_noise/P_air);

% find those regions on speech with local SNR above given local criteira: 3 dB
mask_LC = STEP_sigEN > (STEP_noiseEN + LC);

% model a forward masking effect by modifying noise envelope
STEP_noiseEN = doForwardMasking(STEP_noiseEN, winsize, lc_thrd);
% find the glimpsing regions with over modified noise envelope
mask_LC2  = STEP_sigEN > (STEP_noiseEN + LC);
mask_audi = STEP_sigEN > lc_thrd;
mask      = mask_LC2 & mask_audi;
% do glmpse grouping
mask = GRegionValidator(mask, mingnum);

% compute average glimpse count across channel
[~, framenum_ori] = size(STEP_ref);
objscore.FMGP = 100 * sum(mask(:)) / (ERBnum * framenum_ori);
% log-compress FMGP score
objscore.FMGP = objscore.FMGP+1;%log(objscore.FMGP+1); % add 1 avoiding zero or minus value after log
% compute glimpse proportion
[~,framenum] = size(STEP_sig);
objscore.GP = (100 * sum(mask_LC(:)) / (ERBnum * framenum))+1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRIVATE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rectified = doForwardMasking(STEP_noise, samplenumber, threshold)
% rectified = doForwardMasking(STEP_noise, winsize, threshold) modifies the STEP noise envelope for
% each channel to model the forward masking effect.
%
% input:
%        STEP_noise      STEP envelope of the noise signal
%        samplenumber    number of sample point on the envelope
%        threshold       normalised hearing threshold.
%
% output:
%        rectified       modified noise envelope with a forwarding masking effect.
%
%
% Author: Yan Tang

[chans,frames] = size(STEP_noise);
rectified = zeros(chans,frames);
for idx = 1:chans
    tmp =  STEP_noise(idx,:);
    location = findpeaks(tmp,'q',10);
    lastpeak = 1;
    
    for peakidx = 1:length(location)
        peak = location(peakidx);
        left = floor(peak);
        right = ceil(peak);
        if tmp(left) > tmp(right)
            peak = left;
        else
            peak = right;
        end
        
        if STEP_noise(idx,peak) > threshold
            if (peak-lastpeak > samplenumber || STEP_noise(idx,peak) >= tmp(peak))
                newamp = tmp(peak) - (0:samplenumber) .* ((tmp(peak)-threshold)/samplenumber);
                if peak + samplenumber < frames
                    tmp(peak:peak+samplenumber) = max(newamp,  tmp(peak:peak+samplenumber));
                else
                    tmp(peak:end) = max(newamp(1:frames-peak+1),  tmp(peak:end));
                end
            end
        end
        lastpeak = peak;
    end
    rectified(idx,:) = tmp;
end