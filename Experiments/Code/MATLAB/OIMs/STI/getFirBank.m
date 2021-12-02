function [co_fir, frs] = getFirBank(fs)
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