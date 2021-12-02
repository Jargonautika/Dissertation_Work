function sti = STI_Payton(s, sn, fs)

if fs < 23000
   s = resample(s, 23000, fs);
   sn = resample(sn, 23000, fs);
   fs = 23000;
end



[co_fir, frs] = getFirBank(fs);

% extracting envelope
spoint = 10 * fs/1000; %10ms
co_lp = fir1(spoint, 50/(fs/2),'low'); % 50 cutoff frequency 

%
band_s  = getFrequencyEnvelope(s, co_fir, co_lp); 
band_sn =  getFrequencyEnvelope(sn, co_fir, co_lp);


num_band = length(frs.cfs);
MTF = zeros(1, num_band);
for idx = 1:num_band
   en_s = downsample(band_s(:, idx), 50); % downsampling frequency fs/50 = 460 Hz
   en_sn = downsample(band_sn(:, idx), 50);

   MTF(idx) = calMTF_Payton(en_s, en_sn);
end

sti = MTF2STI(min(max(MTF, 0), 1));
