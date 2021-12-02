function rate = IHC_cooke(env, fs)
% rate = IHC_cooke(env, fs) returns frequency-dependent inner hair cell response
% using the model of Cooke (1993)
%
% input:
%        env         envelope of BM vibration
%        fs          sampling frequency, Hz
%
% output:
%        rate        IHC response
%
%
% Author:   Martin Cooke
% Created:  Dec 16, 2008
% Modified: July 28, 2015


candfConst = 100;  % Crawford and Fettiplace 'c' value
release =  24.0;   % release fraction 
refill =  6.0;     % replenishment fraction 
spont  = 50;       % desired spontaneous firing rate
normSpike = 1500;  % maximum possible firing rate

vmin = spont/normSpike;
k = release/fs;    % normalised release fraction
l = refill/fs;     % normalised replenishment fraction

% initial conditions
vimm = vmin;
vrel = 0.0;
crel = 0.0;
vres = 1.0-vmin;
cimm = l/(vmin*k+l); % eq 23

% instantaneous compression
rp = env./(env+candfConst); % eq 5

% hair cell model (not yet fully vectorised)
rate=zeros(1,length(env));
hc_in=max(rp,vmin);
for i=1:length(rp)
   SPMin=hc_in(i);
   if SPMin > vimm   % stimulus increment
      if SPMin > (vimm + vrel)  % use both relax and reserve sites
         delta = SPMin -(vrel+vimm); % [x(t)-vrel(t)-vimm(t)]
         cimm = (cimm*vimm + crel*vrel + delta)/SPMin; %eq 10
         vrel = 0;
      else
         delta = SPMin-vimm; % i.e. [x(t)-vimm(t)]
         cimm = (cimm*vimm + delta * crel)/SPMin;  % eq 11
         vrel = vrel-delta; % take out of relax
      end
   elseif vimm > SPMin % stimulus decrement
      delta = vimm-SPMin;     % i.e. [vimm(t)-x(t)]
      crel = (delta*cimm + vrel*crel)/(delta+vrel); % eq 12
      vrel = vrel + delta;
   end
   
   vimm = SPMin;
   rate(i) = k*cimm*vimm;    % eq 7
   cimm = cimm-rate(i);
   cimm = cimm + l*(1-cimm); % eq 8
   crel = crel + l*(1-crel); % eq 9
end
