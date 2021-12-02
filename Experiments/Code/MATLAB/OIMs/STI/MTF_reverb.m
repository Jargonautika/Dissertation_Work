function m = MTF_reverb(fm, T)

m = 1 ./ sqrt(1 + (2*pi*T * fm ./13.8).^2);