function [TSN,SNLoss,bH]=LdThrSN(Hnd,SNLoss);

% Load AUDIOGRAM
Stemp = pwd; 
S = char(SNLoss(1));
cd(S(1,:));

[fname,pname]=uigetfile('*.*','AUDIOGRAM');
if pname ~= 0
 [fid,message] = fopen([pname fname],'rt');
 if fid==-1 error(message), end;
 Thr = fscanf(fid,'%f',[2,inf]);
 fclose(fid);
 
 SNLoss(1) = cellstr(pname);
 
 [fthr,i] = sort(Thr(1,:));
 for k=1:length(i) Thrx(k) = Thr(2,i(k)); end;
 if min(diff(fthr))<=eps 
  error('Threshold at some frequencies specified twice')
 end;
 if (min(fthr)>200)|(max(fthr)<8000) 
  error('Thresholds must span at least f=[200..8000] Hz')
 end;
 LoadFlag = 1;
else
 LoadFlag = 0;
end;

if LoadFlag
 set(Hnd(234),'enable','on');
 set(Hnd(42),'enable','on');
 set(Hnd(354),'enable','on');
 semilogx(fthr,Thrx,fthr,Thrx,'o');
 xlabel('Frequency [Hz]'), ylabel('Hearing Loss [dB HL]'); grid on;
 impModel = 1;
 SNLoss(2) = cellstr(fname);
 title(strcat('Audiogram: "',SNLoss(2),'"'));
 TSN = [fthr;Thrx];
 

else
 impModel = 0;
 SNLoss(2) = cellstr('0 dB HL');
 TSN = [199 8001;0 0];
 set(Hnd(234),'enable','off');
 set(Hnd(42),'enable','off');
 set(Hnd(354),'enable','off');
end;

 % Estimate the hearing loss for speech from the pure-tone
 % audiogram (p. 413 of Fletcher 1953):

 f0 = [200 300 400 500 600 700 800 900 1000 1250 1500 1750 2000,...
       2500 3000 4000 5000 6000 7000 8000]'; 
 bn0 = interp1(log10(TSN(1,:)),TSN(2,:),log10(f0));
 bH = interp1(log10(f0),bn0,log10([250 500 1000 2000 4000 8000]'));
 bH = -10*log10(sum([.003 .104 .388 .395 .106 .004]'.*10.^(-bH/10)));
 
 if impModel
  disp(['Beta_H (p.413 of Fletcher 1953) corresponding to this audiogram is ', num2str(bH),' dB']);
  disp('');
 end;

set(Hnd(32),'enable','off');
set(Hnd(33),'enable','off');
set(Hnd(34),'enable','off');
set(Hnd(36),'enable','off');
set(Hnd(37),'enable','off');

cd(Stemp); 

