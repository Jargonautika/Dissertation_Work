% Review Response
semilogx(Resp(1,:),Resp(2,:));
grid; xlabel('Frequency [Hz]'), ylabel('Filter Response [dB]');
title(strcat('Response:"',Rspfn(2),'"'))
set(Hnd(41),'enable','off');
