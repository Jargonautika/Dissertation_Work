% function  [siivalue,eng_sig] = mysii(sig,noise,fs)
% 
% ctrl = true;
% 
% if nargin < 2
%     ctrl = false;
%     fs = 16000;
% end
% if nargin < 3
%     fs = 16000;
% end
% 
% 

%  
%  
% eng_sig = zeros(1,length(f));
% eng_noise = zeros(1,length(f));
% 
% for idx = 1: length(f)
%     eng = calEnergy(gammatoneZeroPhase(sig,fs,f(idx)));
%     eng_sig(idx) = eng;
%     
%     if ctrl        
%         eng = calEnergy(gammatoneZeroPhase(noise,fs,f(idx)));
%         eng_noise(idx) = eng;
%     end
% end
% 
% if ctrl
%     siivalue = sii('E',eng_sig,'N',eng_noise);
% else
%     siivalue = sii('E',eng_sig);
% end


function  siivalue = mysii(sig, noise, fs)

if nargin < 3
    error('Not enough arguments.');
end
% f = [160 200 250 315 400 500 630 800 1000 1250 1600 2000, ...
%      2500 3150 4000 5000 6300 8000];
% 
% Es = mySPL(sig,fs,f, fs/1000 * 50);
% Ns = mySPL(noise,fs,f, fs/1000 * 50);

[Es,cfs] = spectrumlevel(sig,fs);
Ns = spectrumlevel(noise,fs);

% thresholds = thrshd_terhardt(cfs);
% thresholds = thrshd_fay(cfs);
 
siivalue = sii('E',10*log10(Es),'N',10*log10(Ns), 'I',7);
% siivalue = SIIn(Es',Ns', fs);


% siivalue=sii('E',10*log10(Es),'I',7);
