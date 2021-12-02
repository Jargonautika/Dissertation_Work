function [TF,Rspfn]= LdResp(Hnd,Rspfn);

% load filter response
S = char(Rspfn(1));
Stemp = pwd;
cd(S(1,:));

[fname,pname]=uigetfile('*.*','RESPONSE');
if pname ~= 0
 [fid,message] = fopen([pname fname],'rt');
 if fid==-1 error(message), end;
 TF = fscanf(fid,'%f',[2,inf]);
 fclose(fid);

 Rspfn(1) = cellstr(pname);
 
 [f,i] = sort(TF(1,:));
 for k=1:length(i) R(k) = TF(2,i(k)); end;
 if min(diff(f))<=eps
  error('Transfer function specifies some frequencies twice')
 end;
 if (min(f)>200)|(max(f)<8000) 
  error('Transfer function must span at least f=[200..8000] Hz')
 end;
 if max(f)>1E4 
  error('Transfer function must not extend beyond 10kHz')
 end;
 if min(f)<100 
  error('Transfer function must not extend below 100Hz')
 end;

 LoadFlag = 1;
else 
 LoadFlag = 0;
end;

if LoadFlag	
 semilogx(f,R);
 grid; xlabel('Frequency [Hz]'), ylabel('Filter Response [dB]');
 Rspfn(2) = cellstr(fname);
 title(strcat('Response:"',Rspfn(2),'"'))
 TF = [f;R];

 set(Hnd(31),'enable','on'); 
 set(Hnd(351),'enable','on');
 set(Hnd(42),'enable','on');

else
   
 Rspfn(2) = cellstr('none');
 TF = [];
 set(Hnd(31),'enable','off');
 set(Hnd(351),'enable','off');
 set(Hnd(42),'enable','off');
 
end;

set(Hnd(32),'enable','off');
set(Hnd(33),'enable','off');
set(Hnd(34),'enable','off');
set(Hnd(36),'enable','off');
set(Hnd(37),'enable','off');

cd(Stemp); 
