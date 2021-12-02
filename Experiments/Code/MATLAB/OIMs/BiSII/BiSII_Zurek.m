function bisii = BiSII_Zurek(sig, noise, AZ_s_, AZ_n_, fs, ear, DST_, HTL)
if fs < 20000
   sig	= resample(sig, 20000, fs); 
   noise	= resample(noise, 20000, fs);
   fs = 20000;
end

if nargin < 7
    DST_ = 2; % metre;
end
if length(DST_) > 2
   error('Distance should have a size of 2'); 
else
    if length(DST_) < 2
        DST_s_ = DST_;
        DST_n_ = DST_;
    else
        DST_s_ = DST_(1);
        DST_n_ = DST_(2);
    end
end

DST_ref = 2;
sig = sig ./ (DST_s_ / DST_ref); 
noise = noise ./ (DST_n_ / DST_ref);


if nargin < 6
    ear = 'B';
else
    ear = upper(ear);
end

[co_fir, frs] = getFirBank(fs);
level_s = computeBandLevel(sig, co_fir);
level_n = computeBandLevel(noise, co_fir);
if nargin  < 8
   HTL = outerMiddleEar(frs.cfs'); 
end

bisii = BiSII_Zurek_helper(level_s, level_n, AZ_s_, AZ_n_, HTL, ear);




