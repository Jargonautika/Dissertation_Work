function osi = BiDWGP1_rvb(srcs, fs, varargin)
% MY FASTER VERSION COMPARED TO THE ORIGINAL BIDWGP1


% osi = BiDWGP1(src, fs, varargin) estimates intelligibility of a speech source in the presence of
% a given number of masking sources. The estimation is done using the monaural signal, the location
% of each source, as well as the relative distance between the target speech and masking source.
%
% input:
%        srcs           A matrix holding all source signals. Conventionally individual sources
%                       should be held in columns. The target speech source should always be held
%                       in the first column.
%        fs             Sampling frequency (Hz). All source signals are assumed to have the same
%                       sampling frequency. This version only supports signals with sampling
%                       frequencies of no less than 16000 Hz.
%        varargin       Optional parameters in the form of name-value pair.
%                       'location': Azimuths of all the sources relative to the front centre of
%                                   listener (0?) in a horizontal plane, in degrees. All values
%                                   should be in a single array. The first element is taken as the
%                                   location of the target speech signal. The remaining elements
%                                   indicate the location of the masking sources, matching the
%                                   order in 'srcs'. If this argument is not specified, the
%                                   location of all the sources will be assumed to be at 0?.
%                                   e.g., BiDWGP1_reverb(srcs, fs, 'location',[0, 30, -60]).
%                       'distance': Distance of all the sources from the listener on a horizontal
%                                   plane, in metres. The data format is the same as for 'location'.
%                                   The length of 'distance' should be the same as the number of the
%                                   sources. If the length of the array is not equal to the number
%                                   of sources, all the masking sources will be assumed to be
%                                   on the same radius of the target speech source (i.e., the first
%                                   value in 'distance' will be used for all). If this argument is
%                                   not specified, a value of 2 will be assumed as the distance
%                                   between all the sources and listener.
%                                   e.g., BiDWGP1_reverb(srcs, fs, 'distance',[2, 3, 1.5]).
%				 				'rt': RT60 in seconds.
%										e.g., BiDWGP1_reverb(srcs, fs, 'rt', 0.6).
%                       'mode': A flag ('L', 'R', 'B' and 'M') indicating the listening mode:
%                               'L' - unilateral listening with the left ear.
%                               'R' - unilateral listening with the right ear.
%                               'B' - binaural listening.
%                               'M' - monaural listening.
%                               If it is not specified, binaural listening is assumed.
%                               e.g., BiDWGP1_reverb(srcs, fs, 'mode','B').
%                       'spl': The presentation level of the target speech source, in dB SPL. If it
%                              is not specified, 63 dB SPL will be assumed.
%
% output:
%        osi            Objective speech intelligibility score. This is a numeric index falling
%                       between [0 1]. Larger values indicate higher intelligibility.
%
% usage: osi = BiDWGP1_reverb([speech, masker1, masker2], fs, ...
%                    'location',[30, 90, -45],...
%                    'distance',[2],...
%							'rt', 0.6,...	
%                    'mode','B')
%        This example predicts the binaural intelligibility of a speech source which is 2 metres
%        away at 30? relative to the front centre of the listener, in the presence of two masking
%        sources also situated 2 metres away at 90? and -45?, respectively.
%
%
% Author:   Yan Tang
% Date:     Nov 26, 2014



%%%%%%%%%%%%%%%%%%% Parameter checking and preparation %%%%%%%%%%%%%%%%%%%
[a, b] = size(srcs);
if a < b
   srcs = srcs';
end

[len_sample, num_src] = size(srcs);

if num_src < 2
   error('No masking sources found!');
end

if fs < 16000
   error(['Sampling frequency: ' num2str(fs) 'Hz.',...
      'This version only supports signals with sampling frequency no less than 16000 Hz.']);
end


if nargin > 2
   for k = 1:length(varargin)/2
      var_name = varargin{2*k-1};
      var_val  = varargin{2*k};
      
      switch lower(var_name)
         case {'location', 'distance'}
            if ~isnumeric(var_val) || isempty(var_val)
               error(['Invalid value for ''' var_name '''! A vector or scalar is expected.']);
            end
            if length(var_val) < num_src
               var_val = ones(1, num_src) .* var_val(1);
            elseif length(var_val) > num_src
               var_val = var_val(1:num_src);
            end
         case 'mode'
            var_val = upper(var_val);
            if ~strcmp('L', var_val) && ~strcmp('R', var_val) && ~strcmp('B', var_val) && ~strcmp('M', var_val)
               error(['Invalid value for ''' var_name '''! Flag ''L'', ''R'', ''B'' or ''M'' is expected.']);
            end
         case 'spl'
            if ~isnumeric(var_val) || ~isscalar(var_val)
               error(['Invalid value for ''' var_name '''! A scalar is expected.']);
            end
         case 'rt'
            if ~isnumeric(var_val) || ~isscalar(var_val)
               error(['Invalid value for ''' var_name '''! A scalar is expected.']);
            end          
            if var_val < 0
               error(['Invalid value for ''' var_name '''! A postive value is expected.']);
            end
         otherwise
            error(['Unknown parameter: ''' var_name '''!']);
      end
      cmd = sprintf('%s=var_val;',var_name);
      eval(cmd);
   end
end

if ~exist('rt', 'var')
   rt = 0;
end

% if 'spl' is not specified, a 63 dB SPL is assumed
if ~exist('spl', 'var')
   spl = 63; %dB; a conversational level
end
% speech level adjustment
rmslevel = power(10, spl/20);
k        = rmslevel / rms(srcs(:,1));
srcs     = k .* srcs;

if ~exist('mode', 'var')
   mode = 'B';
end

% if 'location' is not given, 0? is used as default
% The azimuth is relative to the listener's front centre (0?)
if ~exist('location', 'var')
   location = zeros(1, num_src);
else
   % convert source azimuth to the value between 0 - 360 degrees
   location = mod(location, 360);
end

% if 'distance is not given',  metre is used as default.
if ~exist('distance', 'var')
   distance = ones(1, num_src) .* 2;
else
   distance(distance < 0.5) = 0.5; % limit the shortest distance to 0.5m
   
   dist_max = max(distance);
   dist_min = min(distance);
   
   %if speech and noise sources are on different radius, the phase shift is taken into account by
   %padding zeros before or after the signals.
   if dist_max ~= dist_min;
      c = 344;    % m/sec, speed of sound in air
      
      max_diff = abs(dist_max - dist_min);
      time_diff = abs(max_diff) / c;
      padding = zeros(round(time_diff * fs), 1);
      
      srcs_new = zeros(len_sample + length(padding), num_src);
      
      for i = 1 : num_src
         % for the furthest case
         dist_diff = abs(dist_max - distance(i));
         time_diff = abs(dist_diff) / c;
         len_padding_tail = round(time_diff * fs);
         padding_tail = zeros(len_padding_tail, 1);
         padding_head = zeros(length(padding) - len_padding_tail, 1);
         
         % for the closest case
         %          dist_diff = abs(distance(i) - dist_min);
         %          time_diff = abs(dist_diff) / c;
         %          padding_head = zeros(round(time_diff * fs), 1);
         srcs_new(:,i) = [padding_head; srcs(:, i); padding_tail];
      end
      srcs = srcs_new;
   end
   
   % compute signal amplitude attenuation due to distance using 1/r rule
   % A distance of 0.5 metre is used as reference
   DST_ref = 2; % m
   ratio = distance ./ DST_ref;
   ratio = repmat(ratio, size(srcs, 1),  1);
   srcs =  srcs ./ ratio;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Intelligibility estimation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% constants for STEP representation
fr_l        = 100;	% lower frequency bound (Hz)
fr_h        = 7500;	% upper frequency bound (Hz)
framerate   = 10;    % window size (ms)
tmp_int		= 8;     % temporal integration (ms)
nchans   	= 34;    % number of filters
LT          = 3;    % local SNR threshold for defining glimpses (dB)
HL 			= 20;    % absolute hearing level (dB)

gain_algorithm = 2.82;
% compute centre frequencies on ERB scale
cfs = MakeErbCFs(fr_l,fr_h, nchans)';

if ~strcmp(mode, 'M')
   % estimate SPL transfer functions of two ears.
   transFn = cell(1, num_src);
   
   % generate STEP representations of speech and masker signals for both ears using SPL transfer
   % functions. Estimate basilar membrane response for given centre frequencies
   for i = 1:num_src
      transFn{i} = SPLTrans(cfs, location(i), mode, 1);
      if isstruct(transFn{i})
         fn = [transFn{i}.L, transFn{i}.R];
      else
         dum = zeros(size(transFn{i}));
         if strcmp(mode, 'L')
            fn = [transFn{i}, dum];
         else
            fn = [dum, transFn{i}];
         end
      end
      
      if i == 1
         compression = 'none';
      else
         compression = 'log';
      end
      
      [bi_STEP(i).L, bi_STEP(i).R, bi_BM(i).L, bi_BM(i).R] = makeBiSTEP(srcs(:,i), fs, fn, fr_l, fr_h, nchans, framerate, tmp_int, compression, 'iso');
   end
   
   
   for i = 1:num_src
      for f = 1:nchans
         sig_rec = reconstructSig(location(i), cfs(f), [bi_BM(i).L(f, :)' bi_BM(i).R(f, :)'], fs);
         
         bi_BM(i).L(f, :) = sig_rec(:, 1);
         bi_BM(i).R(f, :) = sig_rec(:, 2);
      end
   end 
   
   
   bi_BM_n.L = bi_BM(2).L;
   bi_BM_n.R = bi_BM(2).R;
   
   if  num_src > 2
      for j = 3:num_src
         bi_BM_n.L = bi_BM_n.L + bi_BM(j).L;
         bi_BM_n.R = bi_BM_n.R + bi_BM(j).R;
      end
      bi_STEP_n = makeNoiseSTEP(bi_BM_n, fs);
   else
      bi_STEP_n = bi_STEP(2);
   end
   % compute band-dependent distortion factors on speech envelope
   dstfctrs = getDSTFactors_B(bi_BM(1), bi_BM_n, bi_STEP(1), fs, mode);
   
   bi_STEP_s.L = 20 * log10(bi_STEP(1).L);
   bi_STEP_s.R = 20 * log10(bi_STEP(1).R);
   
   if strcmp(mode, 'B') % if binaural listening
      % compute masking level difference

      MLD = zeros(nchans, 1);
      for i = 1:nchans
         % given center frequencies, compute interaural phase shift for both speech and noise signals
         % as well as the coherence of the noise masker
         [phase_s, ~] = phaseANDcoherence(bi_BM(1).L(i,:), bi_BM(1).R(i,:), fs, cfs(i));
         [phase_n, coher_n] = phaseANDcoherence(bi_BM_n.L(i,:), bi_BM_n.R(i,:), fs, cfs(i));
         
         MLD(i) = getMLD_Culling(cfs(i), coher_n, phase_s, phase_n);
      end

      MLD_s = repmat(MLD, 1, size(bi_STEP_s.L,2));
      % get a-priori mask at each ear with absolute audibility checking
      bi_mask.L = (bi_STEP_s.L + MLD_s > (bi_STEP_n.L + LT)) & (bi_STEP_s.L >= HL);
      bi_mask.R = (bi_STEP_s.R + MLD_s > (bi_STEP_n.R + LT)) & (bi_STEP_s.R >= HL);
      
      % find the final glimpsed regions with best ear
      glimpses = bi_mask.L | bi_mask.R;
   else % if unilateral listening
      glimpses = (bi_STEP_s.(mode) > (bi_STEP_n.(mode) + LT)) & (bi_STEP_s.(mode) >= HL);
   end

else % monaural listening
   % compute STEP for speech and noise in monaural listening
   s_target = srcs(:, 1);
   s_masker = sum(srcs(:, 2:end),2);
   
   STEP_target = makeRateMap_IHC(s_target, fs,fr_l,fr_h,nchans, framerate, tmp_int,'none','none','iso',1);
   STEP_masker = makeRateMap_IHC(s_masker, fs,fr_l,fr_h,nchans, framerate, tmp_int,'log','none','iso',1);
   STEP_mix    = makeRateMap_IHC(sum(srcs, 2), fs,fr_l,fr_h,nchans, framerate, tmp_int,'none','none','iso',1);
  
   
   % compute band-dependent distortion factors on speech envelope
   dstfctrs = getDSTFactors_M(STEP_target, STEP_mix);
   
   STEP_target = 20 * log10(STEP_target);
   glimpses = (STEP_target > (STEP_masker + LT)) & (STEP_target >= HL);
end

if rt > 0
   MFs = [0.63; 0.8; 1; 1.25; 1.60; 2; 2.50; 3.15; 4; 5; 6.30; 8; 10; 12.50];
   m_rvb = mean(MTF_reverb(MFs, rt));
else
   m_rvb = 1;
end



% Weighted by band importance function
weights = BIF(cfs,1);
gc_band = zeros(nchans,1);
for idx = 1:nchans
   gc_band(idx) = dstfctrs(idx) * weights(idx) * sum(glimpses(idx, :)) * m_rvb^0.5;
end

osi = sum(gc_band) / size(glimpses, 2);
% log compression accounting for glimpse redundancy
delta = 0.01;
osi = log((osi+delta)/delta)/log((1+delta)/delta);


