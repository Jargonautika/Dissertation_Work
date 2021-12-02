function bxgp = BiHEGP(sig, noise, fs, AZ_, DST_, ear, SPL)
% Yan Tang


% if presenation level for speech is not specified, a 63 dB SPL is assumed
if nargin < 7
    SPL = 63; %dB; a conversational level
end

% if listening mode is not specified, binaural listening with two ears is assumed
if nargin < 6
    ear = 'B';
else
    ear = upper(ear);
    if ~ismember(ear, {'B', 'L', 'R', 'M'})
        error('Listening mode must be ''B'', ''L'', ''R'' or ''M''.');
    end
end

% if distance is not given, 2-m distance is used as default.
if nargin < 5
    DST_ = 2; % metre; default distance is 2 metres.
end
if length(DST_) > 2
    error('Distance should have a size of 1 or 2');
else
    if length(DST_) < 2
        DST_s_ = DST_;
        DST_n_ = DST_;
    else
        DST_s_ = DST_(1);
        DST_n_ = DST_(2);
        
        %if speech and noise sources are not located on the same radius
        if DST_s_ ~= DST_n_
            c = 344;    % m/sec, speec of sound in air
            
            dist_diff = DST_s_ - DST_n_;
            time_diff = abs(dist_diff) / c;
            padding = zeros(round(time_diff * fs), 1);
            if dist_diff > 0    % speech is further than noise
                sig = [padding; sig];
                noise = [noise; padding];
            else                % speech is closer than noise
                sig = [sig; padding];
                noise = [padding; noise];
            end
        end
    end
end

% if azimuth is not given, 0 degree is used as default
% The azimuth is relative to the north
if nargin < 4
    AZ_ = 0; % degree;
end
if length(AZ_) > 2
    error('Azimuth should have a size of 2');
else
    if length(AZ_) < 2
        AZ_s_ = AZ_;
        AZ_n_ = AZ_;
    else
        AZ_s_ = AZ_(1);
        AZ_n_ = AZ_(2);
    end
end
% convert source azimuth to the value between 0 - 360 degrees
AZ_s_ = mod(AZ_s_, 360);
AZ_n_ = mod(AZ_n_, 360);

% speech level adjustment
rmslevel = power(10, SPL/20);
k     = rmslevel / rms(sig);
sig   = k .* sig;
noise = k .* noise;

% compute signal applitude attenuation due to distance using 1/r rule
% A distance of 2-metre is used as reference
DST_ref = 2; % m
sig = sig ./ (DST_s_ / DST_ref);
noise = noise ./ (DST_n_ / DST_ref);


% constants for STEP representation
fr_l        = 100;	% lower frequency bound (Hz)
fr_h    	= 7500;	% upper frequency bound (Hz)
framerate   = 10;	% window size (ms)
tmp_int		= 8;	% temporal integration (ms)
nchans  	= 34;	% number of filters
LT          = 3;    % local SNR threshold for defining glimpses (dB)
HL 			= 20;	% absolute hearing level (dB)
% P_air     = 2e-5; %reference pressure in air (uPa)

% compute centre frequencies
cfs = MakeErbCFs(fr_l,fr_h,nchans)';


if ~strcmp(ear, 'M')
    % estimate SPL transfer functions of two ears.
    transFn_s = SPLTrans(cfs, AZ_s_, ear, 1);
    transFn_n = SPLTrans(cfs, AZ_n_, ear, 1);
    
    if isstruct(transFn_s) && isstruct(transFn_n)
        fn_s = [transFn_s.L, transFn_s.R];
        fn_n = [transFn_n.L, transFn_n.R];
    else
        dum = zeros(size(transFn_s));
        if strcmp(ear, 'L')
            fn_s = [transFn_s, dum];
            fn_n = [transFn_n, dum];
        else
            fn_s = [dum, transFn_s];
            fn_n = [dum, transFn_n];
        end
    end
    
    % generate STEP represenations of speech and masker signals for both ears using SPL transfer
    % functions. Estiamte basilar membrane response for given centre frequencies
    [bi_STEP_s.L, bi_STEP_s.R, bi_BM_s.L, bi_BM_s.R] = makeBiSTEP(sig, fs, fn_s, fr_l,fr_h,nchans, framerate, tmp_int,'log','iso');
    [bi_STEP_n.L, bi_STEP_n.R, bi_BM_n.L, bi_BM_n.R] = makeBiSTEP(noise, fs, fn_n, fr_l,fr_h,nchans, framerate, tmp_int,'log','iso');
    
    % compute band-dependent mean energy level of speech+noise mixture
    mixTH = getMixHE(bi_BM_s, bi_BM_n, fs, ear);   
    
    frames = size(bi_STEP_s.L,2);
    if strcmp('B', ear) % if binaural listening
        % compute masking level difference
        MLD = getMLD(cfs, AZ_s_, AZ_n_, transFn_s, transFn_n);
        MLD_s = repmat(MLD, 1, frames);
        mixTH.L = repmat(mixTH.L, 1, frames);
        mixTH.R = repmat(mixTH.R, 1, frames);
        
        % get a-priori mask at each ear with absolute audibility checking
        bi_mask.L = ((bi_STEP_s.L + MLD_s) > (bi_STEP_n.L + LT)) & (bi_STEP_s.L >= max(HL,mixTH.L));
        bi_mask.R = ((bi_STEP_s.R + MLD_s) > (bi_STEP_n.R + LT)) & (bi_STEP_s.R >= max(HL,mixTH.R));
           
        % find the final glimpsed regions with best ear
        glimpses = bi_mask.L | bi_mask.R;
    else % if unilateral listening
        mixTH = repmat(mixTH.(ear), 1, frames);
        glimpses = (bi_STEP_s.(ear) > (bi_STEP_n.(ear) + LT)) & (bi_STEP_s.(ear) >= max(HL,mixTH));
    end
    
else % monaural listening
    % compute STEP for speech and noise in monoaural listening
    STEP_sig   = makeRateMap_IHC(sig, fs,fr_l,fr_h,nchans, framerate, tmp_int,'log','none','iso',1);
    STEP_noise = makeRateMap_IHC(noise, fs,fr_l,fr_h,nchans, framerate, tmp_int,'log','none','iso',1);
    STEP_mix   = makeRateMap_IHC(sig+noise, fs,fr_l,fr_h,nchans, framerate, tmp_int,'log','none','iso',1);
   
    % original glimpse proportion and glimpses
    glimpses  = STEP_sig > (STEP_noise + LT);  	% local SNR criterion
    
    % extended glimpse proportion
    % (1) local SNR criterion AND must be above hearing level
    % (2) use only high energy glimpses
    meanEN = mean(STEP_mix, 2); % mean energy of each channel from the mixture signal
    for idx = 1:nchans
        glimpses(idx, :) = glimpses(idx, :) & (STEP_mix(idx,:) >= max(meanEN(idx), HL));
    end
end


% apply BIF
bif = importanceFnc(cfs,1,0);
gc_band = zeros(nchans,1);
for idx = 1:nchans
    gc_band(idx) =  bif(idx) * sum(glimpses(idx, :));
end

bxgp = sum(gc_band) / size(glimpses, 2);
% log compressiong accounting for glimpse redundency
delta = 0.01;
bxgp = log((bxgp+delta)/delta)/log((1+delta)/delta);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRIVATE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mixTH = getMixHE(bm_s, bm_n, fs, ear)

[bands, samples] = size(bm_s.L);

% prepare parameters for smoothing envelope
intdecay = exp(-(1000/(fs*8)));
intgain = 1-intdecay;

frameshift_samples = round(10*fs/1000);
framecentres = 1:frameshift_samples:samples;
numframes = length(framecentres);

mixTH.L = zeros(bands, 1);
mixTH.R = zeros(bands, 1);

for idx = 1: bands
    if strcmp(ear, 'B') % case of binaural listening
        bi_env_mix.L = smoothEnv(bm_s.L(idx,:) + bm_n.L(idx,:), intgain, intdecay, frameshift_samples, numframes);
        bi_env_mix.R = smoothEnv(bm_s.R(idx,:) + bm_n.R(idx,:), intgain, intdecay, frameshift_samples, numframes);
        
        mixTH.L(idx) = mean(20*log10(bi_env_mix.L));
        mixTH.R(idx) = mean(20*log10(bi_env_mix.R));       
    else % case of unilateral listening
        bi_env_mix.(ear) = smoothEnv(bm_s.(ear)(idx,:) + bm_n.(ear)(idx,:), intgain, intdecay, frameshift_samples, numframes);
        mixTH.(ear)(idx) =  mean(20*log10(bi_env_mix.(ear)));
    end
end





