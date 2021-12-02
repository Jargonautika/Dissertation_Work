function [co_fir, frs] = getFirBank4STI(fs)
%
% set critical bandwidths
%
frs = getOctaveBands;

Nc = length(frs.cfs);
Flength = 256;
co_fir = zeros(Flength+1, Nc);

for i=1:Nc
    co_fir(:,i) = fir1(Flength, [frs.lfs(i) frs.ufs(i)]/(fs/2));
end


 function frs = getOctaveBands
% compute octave band for STI 
% 7 bands coving 125 to 8000

[~, b] = fract_oct_freq_band(1, 125, 8000, 1);

frs.cfs = b(:, 2);
frs.lfs = b(:, 1);
frs.ufs = b(:, 3);