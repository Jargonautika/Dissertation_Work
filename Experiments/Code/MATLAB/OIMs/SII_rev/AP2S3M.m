function [Plt]=Ap2s3M(a,AI,PRout,Hnd,Rspfn,Mskfn,SNLoss);

% transformation of AI to articulation (Fig. 179)
Ap= PRout(1).*AI;
S = 100*(1-10.^(-Ap/0.55));
plot([min(a):1:max(a)],spline(a,S,[min(a):1:max(a)]),a,S,'o'); 
ax = axis;
axis([ax(1) ax(2) 0 100]);
title(strcat('Response: "',Rspfn(2),'"; Masker: "',Mskfn(2),'"; Audiogram: "',SNLoss(2),'"'));
xlabel('System Gain [dB]'); 
ylabel('s3M Score [%]');
text(ax(1)+0.1*(ax(2)-ax(1)),97,strcat('p = ',num2str(PRout(1))));
text(ax(1)+0.1*(ax(2)-ax(1)),92,strcat('bt = ',num2str(PRout(2)),' dB'));
text(ax(1)+0.1*(ax(2)-ax(1)),87,strcat('bH = ',num2str(PRout(3)),' dB'));
grid on;
Plt = [a; S];
set(Hnd(41),'enable','on');
