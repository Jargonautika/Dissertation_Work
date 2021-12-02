% Review Masking Noise
semilogx(N(1,:),N(2,:))
grid; xlabel('Frequency [Hz]'); ylabel('Masker Spectrum Level [dB/Hz]');

title(strcat('Masker:"',Mskfn(2),'"'))
set(Hnd(41),'enable','off');
