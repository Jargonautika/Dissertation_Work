% Review Audiogram
semilogx(TSN(1,:),TSN(2,:),TSN(1,:),TSN(2,:),'o');
xlabel('Frequency [Hz]'), ylabel('Audiogram [dB HL]'); grid on;
title(strcat('Audiogram: "',SNLoss(2),'"'));
set(Hnd(41),'enable','off');
