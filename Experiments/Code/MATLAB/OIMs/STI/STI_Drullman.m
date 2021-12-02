function sti = STI_Drullman(s, sn, fs)
% STI implementation for running speech.
% The modulation transfer functions are compuated using the approach described in Drullman et al. (1994)
% 
% input:
%        s     anechoic binaural speech signal
%        sn    reverberant binauarl speech signal or reverberant noise-corrupted binuaral speech
%              signal
%        fs    sampling frequency (Hz)
%
% Reference:
% Drullman, R., Festen, J.M., Plomp, R., 1994. Effect of reducing slow temporal modulations on speech reception. J. Acoust. Soc. Am. 95 (5), 2670?2680.
%
% Author:   Yan Tang
% Date:     Oct 26, 2015



if fs < 23000
   s = resample(s, 23000, fs);
   sn = resample(sn, 23000, fs);
   fs = 23000;
end


[co_fir, frs] = getFirBank(fs);

% extracting envelope
spoint = 10 * fs/1000; %10ms
[~, MFs] = fract_oct_freq_band(3, 0.63, 12.7, 1); % get 1/3 ocatave modulation band
co_lp = fir1(spoint, 50/(fs/2),'low'); % 50 cutoff frequency 

%
band_s  = getFrequencyEnvelope(s, co_fir, co_lp); 
band_sn =  getFrequencyEnvelope(sn, co_fir, co_lp);


num_band = length(frs.cfs);
num_MF = size(MFs, 1);
MTF = zeros(num_MF, num_band);

fs_downsampled = 200; %downsampling frequency, Hz. decrease computational load
for idx = 1:num_band   
   MTF(:, idx) = calMTF_Drullman(band_s(:, idx), band_sn(:, idx), fs, fs_downsampled, MFs);
end

sti = MTF2STI(MTF);
