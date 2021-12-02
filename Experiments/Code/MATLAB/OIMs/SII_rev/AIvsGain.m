function [Plt]=AIvsGain(a,AI,PRout,Hnd,Rspfn,Mskfn,SNLoss);

% plot AI vs. Gain
if PRout(4)	
 plot([min(a):1:max(a)],spline(a,AI,[min(a):1:max(a)]),a,AI,'o')	
else 
 title('gamma*(RM1bar-RM4bar) < -21.739 causes non-monotonic PI function, PRESS ANY KEY');
 pause;
 title('');
 plot(a,AI,'o');
end;	

ax = axis;
axis([ax(1) ax(2) 0 1]);
grid on;
title(strcat('Response: "',Rspfn(2),'"; Masker: "',Mskfn(2),'"; Audiogram: "',SNLoss(2),'"'));
xlabel('System Gain [dB]');
ylabel('Articulation Index');

text(ax(1)+0.1*(ax(2)-ax(1)),0.92,strcat('bt = ',num2str(PRout(2)),' dB'));
text(ax(1)+0.1*(ax(2)-ax(1)),0.87,strcat('bH = ',num2str(PRout(3)),' dB'));
set(Hnd(41),'enable','on');
Plt = [a;AI];
