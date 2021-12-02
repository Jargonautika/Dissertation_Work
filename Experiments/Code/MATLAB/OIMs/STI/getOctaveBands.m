function frs = getOctaveBands
% compute octave band for STI 
% 7 bands coving 125 to 8000

[~, b] = fract_oct_freq_band(1, 125, 8000, 1);

frs.cfs = b(:, 2);
frs.lfs = b(:, 1);
frs.ufs = b(:, 3);