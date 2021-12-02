function sti = MTF2STI(MTF)
% compute STI from MTF 
% based on standard: IEC (2011). BS EN 60268-16:2011
%
% Author:   Yan Tang
% Date:     Oct 15, 2015

% convert MTF to apparent SNR 
SNR = 10 .* log10(MTF./(1-MTF)); % A5.4
% limit the SNR range to [-15 15];
SNR = min(15, max(real(SNR), -15)); 
% convert SNR to TI
TI = (SNR + 15) ./ 30;  % A5.5
% take the mean across MFs for each frequency band
if size(MTF, 1) > 1
   TI = mean(TI);  % A5.6
end

% read from table A.3 for male talkers
alpha = [0.085 0.127 0.230 0.233 0.309 0.224 0.173];
beta =  [0.085 0.078 0.065 0.011 0.047 0.095];

stia = sum(TI .* alpha);
stib = zeros(1,6);
for idx = 1:6
   stib(idx) = beta(idx) * sqrt(TI(idx) * TI(idx+1));
end

sti = stia - sum(stib);
