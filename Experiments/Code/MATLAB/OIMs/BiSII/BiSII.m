function bisii = BiSII(sig, noise, fs)
% implementation of the binaural SII based on Zurek (2993) and Culling et al.(2004, 2005).
% 
% input:
%        sig     anechoic binaural speech signal
%        noise   reverberant binauarl noise signal
%        fs    sampling frequency (Hz)
%
% Reference:
% Zurek, P.M., 1993. Acoustical Factors Affecting Hearing Aid Performance. Allyn and Bacon, Needham Heights, MA, pp. 255?276. chapter Binaural advantages and directional effects in speech intelligibility.
% Culling, J.F., Hawley, M.L., Litovsky, R.Y., 2004. The role of head-induced interaural time and level differences in the speech reception threshold for multiple interfering sound sources. J. Acoust. Soc. Am. 116 (2), 1057-1065.
%
% Author:   Yan Tang
% Date:     Oct 09, 2015


if fs < 20000
   sig	= resample(sig, 20000, fs);
   noise	= resample(noise, 20000, fs);
   fs = 20000;
end


[co_fir, frs] = getFirBank4SII(fs);
[level_s.L, bi_BM_s.L] = computeBandLevel(sig(:, 1), co_fir);
[level_s.R, bi_BM_s.R] = computeBandLevel(sig(:, 2), co_fir);

[level_n.L, bi_BM_n.L] = computeBandLevel(noise(:, 1), co_fir);
[level_n.R, bi_BM_n.R] = computeBandLevel(noise(:, 2), co_fir);

HTL = outerMiddleEar(frs.cfs');
nchans = length(HTL);

% compute masking level difference between target and each maskers
% using approach in Culling 2005.
MLD = zeros(nchans, 1);

for i = 1:nchans
   % given center frequencies, compute interaural phase shift for both speech and noise signals
   % as well as the coherence of the noise masker
   [phase_s, ~] = phaseANDcoherence(bi_BM_s.L(:, i), bi_BM_s.R(:, i), fs, frs.cfs(i));
   [phase_n, coher_n] = phaseANDcoherence(bi_BM_n.L(:, i), bi_BM_n.R(:, i), fs, frs.cfs(i));
   
   MLD(i) = getMLD_Culling(frs.cfs(i), coher_n, phase_s, phase_n);
end



bisii = BiSII_helper(level_s, level_n, MLD, HTL);





