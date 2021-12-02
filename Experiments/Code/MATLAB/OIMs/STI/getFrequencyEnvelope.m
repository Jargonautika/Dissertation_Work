function env = getFrequencyEnvelope(signal, co_filter, co_lp)
if nargin < 3
   doEnvelope = 0;
else
   doEnvelope = 1;
end


numchan = size(co_filter,2);
env = zeros(length(signal), numchan);
for idx = 1:numchan
    bm = filter(co_filter(:,idx),1,signal);
    bm = bm .^ 2;
    if doEnvelope
      bm = filter(co_lp, 1, bm); %50 Hz low pass squared envelope
    end
    env(:, idx) = bm;
end