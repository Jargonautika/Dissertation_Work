function MTF = calMTF_Payton(band_s, band_sn)
% Eqn. 1 Payton and Shrestha, 2013
band_n = abs(band_sn - band_s);
mui_s  = mean(band_s);
mui_sn = mean(band_sn);
mui_n  = mean(band_n);
mui = mui_s / (mui_s + mui_n); % normalising factor

MTF = mui * ((mean(band_s .* band_sn) - mui_s*mui_sn)/(mean(band_s.^2) - mui_s^2));
MTF = min(max(MTF, 0), 1);
