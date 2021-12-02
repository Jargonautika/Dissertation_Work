function MTF = calMTF_Drullman_orig(band_s, band_sn, fs, fs_downsampled, MFs)
fr_max =  round(fs_downsampled/2);

% downsample to 200 hz
en_s = resample(band_s, fs_downsampled, fs);
en_sn = resample(band_sn, fs_downsampled, fs);

len2_sig = round(length(en_s)/2);

%compute power level proportional to actual PSD
spc_s = fft(en_s);
spc_s = spc_s(1:len2_sig);
psd_s = (1/(fs_downsampled*len2_sig)) .*  (abs(spc_s) .* conj(spc_s)); % to get actuall PSD: normalised by 2 * (1/(fs*N))

spc_sn = fft(en_sn);
spc_sn = spc_sn(1:len2_sig);
% psd_sn = (1/(fs_downsampled*len2_sig)) * abs(spc_sn).^2;
psd_sn = (1/(fs_downsampled*len2_sig)) .* (abs(spc_s) .* conj(spc_sn));

% get corresponding sample index for the lower and upper bounds of each MF
MFs_sample_l = round(MFs(:, 1) .* len2_sig ./ fr_max); % get correponding sample points for each MF
%    MFs_sample = round(MFs(:, 2) .* len2_sig ./ fr_max);
MFs_sample_h = round(MFs(:, 3) .* len2_sig ./ fr_max);

% compute power for each MF
num_MF = size(MFs, 1);
EN_s = zeros(num_MF, 1);
EN_sn = zeros(num_MF, 1);
for k = 1:num_MF
   EN_s(k) = sum(psd_s(MFs_sample_l(k):MFs_sample_h(k)));
   EN_sn(k) = sum(psd_sn(MFs_sample_l(k):MFs_sample_h(k)));
end

% calcualte MTF
band_n = abs(band_sn - band_s);
mui_s  = mean(band_s);
% mui_sn = mean(band_sn);
mui_n  = mean(band_n);
mui = mui_s / (mui_s + mui_n); % normalising factor
% MTF = mui * max(real(EN_sn./ EN_s), 0);   % Eqn.B1 in Drullman et al. 1994
MTF = mui * sqrt(max(real(EN_sn./ EN_s), 0));   % Eqn.B1 in Drullman et al. 1994

MTF = min(max(MTF, 0), 1);