function [N,Mskfn,extNoise] = LdNoise(Hnd,Mskfn);

% Load masking noise spectrum
Stemp = pwd; 
S = char(Mskfn(1));
cd(S(1,:));
[fname,pname]=uigetfile('*.*','MASKER');
if pname ~= 0
 [fid,message] = fopen([pname fname],'rt');
 if fid==-1 error(message), end;
 B = fscanf(fid,'%f',[2,inf]);
 fclose(fid);
 
 Mskfn(1) = cellstr(pname);
 
 [fB,i] = sort(B(1,:));
 for k=1:length(i) Bx(k) = B(2,i(k)); end;
 if min(diff(fB))<=eps 
  error('Noise spectrum specifies some frequencies twice')
 end;
 if (min(fB)>200)|(max(fB)<8000) 
  error('Noise spectrum must span at least f=[200..8000] Hz')
 end;
 LoadFlag = 1;
else
 LoadFlag = 0;
end;

if LoadFlag
 extNoise = 1;
 set(Hnd(222),'enable','on');
 set(Hnd(42),'enable','on');
 set(Hnd(352),'enable','on');
 semilogx(fB,Bx)
 grid; xlabel('Frequency [Hz]'); ylabel('Masker Spectrum Level [dB/Hz]');
 Mskfn(2) = cellstr(fname);
 title(strcat('Masker:"',Mskfn(2),'"'))
 clear N;
 N = [fB;Bx];
else
 N = [];
 extNoise = 0;
 Mskfn(2) = cellstr('none');
 set(Hnd(222),'enable','off');
 set(Hnd(41),'enable','off');
 set(Hnd(42),'enable','off');
 set(Hnd(352),'enable','off');
end;	


set(Hnd(32),'enable','off');
set(Hnd(33),'enable','off');
set(Hnd(34),'enable','off');
set(Hnd(36),'enable','off');
set(Hnd(37),'enable','off');

cd(Stemp); 
