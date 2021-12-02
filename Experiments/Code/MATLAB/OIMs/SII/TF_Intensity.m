function [IMat, Ta] = TF_Intensity(B,s, fs)
%
% compute the time-frequency intensity (dBspl) of signal(s)
% after applying filters in B
% at times specified by Ta, and the center frequencies specified in B.
% Sampling frequency in fs
%
%
% Yannis Stylianou 2012

Nc = size(B,2);
Ls = length(s);

windur = linspace(35, 10, Nc);  % window duration in ms: from the lowest freq. to the max. freq 
windur = round(windur*fs/1000); % window duration in samples

MinDur = windur(end);
ToStart = windur(1)-windur;  % starting point of the windows 
Nfr = floor((Ls-ToStart(end))/MinDur);   % no overlap for the smallest size window

IMat = zeros(Nc, Nfr);

Ta = (0:Nfr-1)*windur(end)+ToStart(Nc)+1;

for i=Nc:-1:1
    sout = filter(B(:,i),1,s);
    IMat(i,:) = Intensity(sout, ToStart(i), windur(i), MinDur, Nfr, fs);
end
