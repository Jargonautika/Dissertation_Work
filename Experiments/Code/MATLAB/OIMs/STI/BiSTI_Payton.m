function sti = BiSTI_Payton(s, sn, fs)
% calculate binaural STI using implementation of van Wijngaarden, S. J., and Drullman, R. (2008).
% The modulation transfer functions are compuated using the approach described in Payton, K.L. and 
% Shrestha, M. (2013).
%
% input:
%        s     anechoic binaural speech signal
%        sn    reverberant binauarl speech signal or reverberant noise-corrupted binuaral speech
%              signal
%        fs    sampling frequency (Hz)
%
% Reference:
% Drullman, R., Festen, J.M., Plomp, R., 1994. Effect of reducing slow temporal modulations on speech reception. J. Acoust. Soc. Am. 95 (5), 2670?2680.
% Payton, K.L., Shrestha, M., 2013. Comparison of a short-time speech-based intelligibility metric to the speech transmission index and intelligibility data. J. Acoust. Soc. Am. 134 (5), 3818?3827.
%
% Author:   Yan Tang
% Date:     Oct 26, 2015

if fs < 23000
   s = resample(s, 23000, fs);
   sn = resample(sn, 23000, fs);
   fs = 23000;
end

[co_fir, frs] = getFirBank(fs);
[~, MFs] = fract_oct_freq_band(3, 0.63, 12.7, 1); % get 1/3 ocatave modulation band

% extracting envelope
spoint = 10 * fs/1000; %10ms
co_lp = fir1(spoint, 50/(fs/2),'low'); % 50 cutoff frequency

band_s.L  = getFrequencyEnvelope(s(:, 1), co_fir);
band_s.R  = getFrequencyEnvelope(s(:, 2), co_fir);

band_sn.L =  getFrequencyEnvelope(sn(:, 1), co_fir);
band_sn.R =  getFrequencyEnvelope(sn(:, 2), co_fir);



num_band = length(frs.cfs);
MTF = zeros(1, num_band);
band_BI = [2 3 4]; % 500, 1000. 2000
for idx = 1:num_band
   en_s.L = band_s.L(:, idx);
   en_sn.L = band_sn.L(:, idx);
   
   en_s.R = band_s.R(:, idx);
   en_sn.R = band_sn.R(:, idx);
   
   if ismember(idx, band_BI)
      
      int_s = getMLD4STI(en_s.L, en_s.R, fs);
      int_sn = getMLD4STI(en_sn.L, en_sn.R, fs);
      
      num_frame = size(int_s, 2);
      holder = zeros(length(MFs), num_frame);
      for f = 1:num_frame
         holder(:, f) = getBinauralMTF(int_s(:, f), int_sn(:, f), 'PAYTON');
      end
      
      MTF(idx) = mean(holder(:));
   else
      en_s.L = filter(co_lp,1,en_s.L);
      en_s.R = filter(co_lp,1,en_s.R);
      en_sn.L = filter(co_lp,1,en_sn.L);
      en_sn.R = filter(co_lp,1,en_sn.R);
      
      mtf_L = calMTF_Payton(en_s.L, en_sn.L);
      mtf_R = calMTF_Payton(en_s.R, en_sn.R);
      
      MTF(idx) = max(mtf_L, mtf_R);
   end
   
end

sti = MTF2STI(min(max(MTF, 0), 1));

