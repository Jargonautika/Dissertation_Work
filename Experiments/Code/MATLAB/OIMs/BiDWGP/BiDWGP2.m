function [osi, DW, MLD] = BiDWGP2(srcs, fs, SPL_)
% osi = BiDWGP2(srcs, fs, SPL_) estimates the intelligibility of a speech source in the presence of
% a given number of masking sources. The estimation is made using given binaural signals of each 
% source.
% 
% input:
%        srcs           A 3-D (N x 2 x R) array holding the binaural signals of all the sound
%                       sources, where N denotes the number of samples and R is the number
%                       of the sound sources. The target speech source should always be held
%                       at the first layer of the array, i.e., R = 1.
%        fs             Sampling frequency (Hz). All source signals are assumed to have the same
%                       sampling frequency. This version only supports signals with sampling
%                       frequencies of no less than 16000 Hz.
%        SPL_           The presentation level of the target speech source, in dB. If it is not 
%                       specified, 63 dB SPL will be assumed.
% 
% output:
%        osi            Objective speech intelligibility score. This is a numeric index falling
%                       between [0 1]. Larger values indicate higher intelligibility.
%
%
% Author:   Yan Tang
% Date:     Dec 12, 2014


% if 'SPL_' is not specified, a 63 dB SPL is assumed
if nargin < 3
   SPL_ = 63; %dB; a conversational level
end

% check the format of the input array
dims =  size(srcs);
if dims < 3
   error('srcs must be a 3-D (N x 2 x R) matrix, where N denotes the number of samples and R are the total number of the sound sources');
end

if dims(1) < dims(2)
   srcs = permute(srcs, [2 1 3]);
end

[~, num_chan, num_src] = size(srcs);
if num_chan ~= 2
   error(['Expect the binaural signals to have 2 channels. Given signal has ' num2str(num_chan) 'channels']);
end

% speech level adjustment
rms_tar = power(10, SPL_/20);
rms_L = rms(srcs(:, 1, 1));
rms_R = rms(srcs(:, 2, 1));
if rms_L > rms_R
   rms_c = rms_L;
else
   rms_c = rms_R;
end
k = rms_tar / rms_c;
srcs = srcs .* k;
   
%%%%%%%%%%%%%%%%%%% Intelligibility estimation %%%%%%%%%%%%%%%%%%%
% constants for STEP representation
fr_l        = 100;	% lower frequency bound (Hz)
fr_h        = 7500;	% upper frequency bound (Hz)
framerate   = 10;    % window size (ms)
tmp_int		= 8;     % temporal integration (ms)
nchans   	= 34;    % number of filters
LT          = 3;    % local SNR threshold for defining glimpses (dB) 3 dB default
HL 			= 20;    % absolute hearing level (dB)

% compute centre frequencies on ERB scale
cfs = MakeErbCFs(fr_l,fr_h, nchans)';

% compute STEP for the left and the right ears for all sources
[bi_STEP_s.L, bi_BM_s.L]   = makeRateMap_IHC(sum(srcs(:,1, 1), 3), fs,fr_l,fr_h,nchans, framerate, tmp_int, 'none','none','iso',1);
[bi_STEP_s.R, bi_BM_s.R]   = makeRateMap_IHC(sum(srcs(:,2, 1), 3), fs,fr_l,fr_h,nchans, framerate, tmp_int, 'none','none','iso',1);


[bi_STEP_n.L, bi_BM_n.L]   = makeRateMap_IHC(sum(srcs(:,1,2:end), 3), fs,fr_l,fr_h,nchans, framerate, tmp_int, 'log','none','iso',1);
[bi_STEP_n.R, bi_BM_n.R]   = makeRateMap_IHC(sum(srcs(:,2,2:end), 3), fs,fr_l,fr_h,nchans, framerate, tmp_int, 'log','none','iso',1);


% make STEP for the mixture of speech+noise
bi_STEP_mix.L   = makeRateMap_IHC(sum(srcs(:,1,:), 3), fs,fr_l,fr_h,nchans, framerate, tmp_int, 'none','none','iso',1);
bi_STEP_mix.R   = makeRateMap_IHC(sum(srcs(:,2,:), 3), fs,fr_l,fr_h,nchans, framerate, tmp_int, 'none','none','iso',1);

% compute band-dependent distortion factors on speech envelope
dstfctrs = getDSTFactors(bi_STEP_s, bi_STEP_mix);
DW = getDSTFactors(bi_STEP_s, bi_STEP_n);

bi_STEP_s.L = 20 * log10(bi_STEP_s.L);
bi_STEP_s.R = 20 * log10(bi_STEP_s.R);

% compute masking level difference between target and each maskers
% using approach in Culling 2005.
MLD = zeros(nchans, 1);

for i = 1:nchans
   % given center frequencies, compute interaural phase shift for both speech and noise signals
   % as well as the coherence of the noise masker
   [phase_s, ~] = phaseANDcoherence(bi_BM_s.L(i,:), bi_BM_s.R(i,:), fs, cfs(i));
   [phase_n, coher_n] = phaseANDcoherence(bi_BM_n.L(i,:), bi_BM_n.R(i,:), fs, cfs(i));
   
   MLD(i) = getMLD_Culling(cfs(i), coher_n, phase_s, phase_n);
end

MLD_s = repmat(MLD, 1, size(bi_STEP_s.L,2));

% get a-priori mask at each ear with absolute audibility checking
bi_mask.L = (bi_STEP_s.L + MLD_s > (bi_STEP_n.L + LT)) & (bi_STEP_s.L >= HL);
bi_mask.R = (bi_STEP_s.R + MLD_s > (bi_STEP_n.R + LT)) & (bi_STEP_s.R >= HL);
% find the final glimpsed regions with best ear
glimpses = bi_mask.L | bi_mask.R;

% Weighted by band importance function
weights = BIF(cfs,1);
gc_band = zeros(nchans,1);
for idx = 1:nchans
   gc_band(idx) = dstfctrs(idx) * weights(idx) * sum(glimpses(idx, :));
end

osi = sum(gc_band) / size(glimpses, 2);
% log compression accounting for glimpse redundancy
delta = 0.01;
osi = log((osi+delta)/delta)/log((1+delta)/delta);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRIVATE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function factors = getDSTFactors(STEP_s, STEP_mix)
% factors = getDSTFactors(STEP_sig, STEP_mix) computes the distortion degree of speech by
% given noise maskers on STEP envelopes for binaural signals.
%
% input:
%        STEP_target    STEP envelope of the target speech signal
%        STEP_mix       STEP envelope of the noise-corrupted signal
% output:
%        factors        A vector of the distortion factors (between 0 and 1) for each channel on
%                       the STEP. The size of this vector should be as same as the number of rows
%                       on the STEP.
%
% Author:   Yan Tang
% Date:     Sep 09, 2014

bands = size(STEP_s.L, 1);
factors = zeros(1,bands);

for idx = 1: bands 
   co.L = pearsoncoeff(STEP_s.L(idx,:)', STEP_mix.L(idx, :)');
   co.R = pearsoncoeff(STEP_s.R(idx,:)', STEP_mix.R(idx, :)');
   
   factors(idx) = mean([abs(co.L), abs(co.R)]);
end




