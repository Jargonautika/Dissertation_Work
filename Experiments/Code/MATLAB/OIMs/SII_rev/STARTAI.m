% Initialize status variables
clear all

TSN = [];
N = [];
PRin = [];
bH = 0;

Mskfn = {pwd,'none'};
SNLoss = {pwd,'0 dB HL'};
Rspfn = {pwd,'none'};

v = version;
%FlagArr = [SetupFlag;CndcLoss;impModel;bHflag;extNoise];
FlagArr = [0;0;0;bH;0];
clear bH;

Hnd(10) = figure('Name','ARTICULATION INDEX (c)1999 Hannes Muesch','NumberTitle','off','Menubar','none');

if (((str2num(v(1)) == 5) & (str2num(v(3)) < 1)) | (str2num(v(1)) <= 4))
   
   close(Hnd(10));
   disp('Sorry, your MATLAB version appears to be older than V 5.1. and is not supported.');
   
else
   
% Setup Menu Bar
% PARAMETER LOADING

Hnd(20) = uimenu(Hnd(10),'Label','AI-Setup');
Hnd(21) = uimenu(Hnd(20),'Label','Load RESPONSE','callback',...
'[Resp,Rspfn]= LdResp(Hnd,Rspfn);');
Hnd(22) = uimenu(Hnd(20),'Label','MASKER');
Hnd(221) = uimenu(Hnd(22),'Label','LOAD Masker','callback','[N,Mskfn,FlagArr(5)] = LdNoise(Hnd,Mskfn);');
Hnd(222) = uimenu(Hnd(22),'Label','DELETE Masker','callback','DelMsk','enable','off');
Hnd(23) = uimenu(Hnd(20),'Label','AUDIOGRAM');
Hnd(233) = uimenu(Hnd(23),'Label','LOAD Audiogram',...
'callback','[TSN,SNLoss,FlagArr(4)]=LdThrSN(Hnd,SNLoss);');
Hnd(234) = uimenu(Hnd(23),'Label','ASSUME 0 dB HL','callback','DelSNl','enable','off');

Hnd(24) = uimenu(Hnd(20),'Label','Set "bt", "bH", "p" ...','callback',...
'[FlagArr(1),FlagArr(4),PRin,SNLoss(2)] = PrmSetup(Hnd,FlagArr(4),SNLoss(2));');

Hnd(35) = uimenu(Hnd(20),'Label','REVIEW Parameters');
Hnd(351) = uimenu(Hnd(35),'Label','Display Response',...
'enable','off','callback','rResp');
Hnd(352) = uimenu(Hnd(35),'Label','Display Masker Spectrum',...
'enable','off','callback','rMsk');
Hnd(354) = uimenu(Hnd(35),'Label','Display Audiogram',...
'enable','off','callback','rSNl');
Hnd(24) = uimenu(Hnd(20),'separator','on','Label','Quit','callback',...
'close(Hnd(10)); clear all');


% FLETCHER AI

Hnd(30) = uimenu(Hnd(10),'Label','Fletcher AI');
Hnd(31) = uimenu(Hnd(30),'enable','off','Label','CALCULATE NOW','callback',...
'[PRout,a,AI] = FAI(Resp,FlagArr,N,Hnd,PRin); [Plt]=AIvsGain(a,AI,PRout,Hnd,Rspfn,Mskfn,SNLoss);');

Hnd(32) = uimenu(Hnd(30),'Label','Display S3 Score vs Gain',...
'enable','off','callback','[Plt]=Ap2s3(a,AI,PRout,Hnd,Rspfn,Mskfn,SNLoss);');
Hnd(33) = uimenu(Hnd(30),'Label','Display s3M Score vs Gain',...
'enable','off','callback','[Plt]=Ap2s3M(a,AI,PRout,Hnd,Rspfn,Mskfn,SNLoss);');
Hnd(36) = uimenu(Hnd(30),'Label','Display S2 Score vs Gain',...
'enable','off','callback','[Plt]=Ap2s2(a,AI,PRout,Hnd,Rspfn,Mskfn,SNLoss);');
Hnd(37) = uimenu(Hnd(30),'Label','Display S23 Score vs Gain',...
'enable','off','callback','[Plt]=Ap2s23(a,AI,PRout,Hnd,Rspfn,Mskfn,SNLoss);');
Hnd(34) = uimenu(Hnd(30),'Label','Display AI vs Gain',...
'enable','off','callback','[Plt]=AIvsGain(a,AI,PRout,Hnd,Rspfn,Mskfn,SNLoss);');

% SAVE/PRINT

Hnd(40) = uimenu(Hnd(10),'Label','Save');
Hnd(41) = uimenu(Hnd(40),'enable','off','Label','SAVE Data',...
'callback','SaveData(Plt)');
Hnd(42) = uimenu(Hnd(40),'enable','off','Label','PRINT Graph',...
'callback','print -v');

% HELP

Hnd(50) = uimenu(Hnd(10),'Label','Help');
Hnd(51) = uimenu(Hnd(50),'Label','General',...
'callback','type HlpFile6'); 
Hnd(52) = uimenu(Hnd(50),'Label','Program Execution');
Hnd(521) = uimenu(Hnd(52),'Label','Response',...
'callback','type HlpFile1'); 
Hnd(522) = uimenu(Hnd(52),'Label','Masker',...
'callback','type HlpFile2'); 
Hnd(523) = uimenu(Hnd(52),'Label','Audiogram',...
'callback','type HlpFile3'); 
Hnd(524) = uimenu(Hnd(52),'Label','"bt", "bH", and "p"',...
'callback','type HlpFile4');
Hnd(525) = uimenu(Hnd(52),'Label','Output/Save',...
'callback','type HlpFile5'); 

end;

clear v;