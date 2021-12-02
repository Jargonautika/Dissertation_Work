function [SetupFlag,bH,prm,S] = PrmSetup(Hnd,initbH,oldS);

% Set Parameters p, bt, and bH
 prompt  = {'Proficiency factor','Talking level beta_T [dB]','Hearing Loss for Speech beta_H [dB]'};
 bH      = initbH;
 default = {'1','68',num2str(initbH)};	


 prmtitle   = 'Parameter EDIT';
 lineNo  = 1;
 badInpt = 0;
 prm  = inputdlg(prompt,prmtitle,lineNo,default);
 if length(prm) == 3
	if (isempty(prm{1}) | isempty(prm{2}) | isempty(prm{3}))
	   badInpt = 1;
	end;	
 end;	 
 if ((length(prm)==0) | badInpt)
	if badInpt  
       errordlg('The dialog box was not completed!  Default settings will be used.');
	end;
    SetupFlag = 0;
    S = oldS;
 else   
    SetupFlag = 1;
    bH        = str2num(char(prm(3)));
    if (round(100*bH) == round(100*initbH))
       % if bH match up to 1/10 of a dB, assume its the same # (for labeling only)
       S = oldS;
    else
       S = cellstr(['bH = 'char(prm(3))]);
       set(Hnd(354),'enable','off');
    end;
 end;   

set(Hnd(32),'enable','off');
set(Hnd(33),'enable','off');
set(Hnd(34),'enable','off');
set(Hnd(36),'enable','off');
set(Hnd(37),'enable','off');
