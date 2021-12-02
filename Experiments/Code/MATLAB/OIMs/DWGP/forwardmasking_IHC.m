function glimpses_FM = forwardmasking_IHC(IHC_mix, glimpses)
% glimpses_FM = forwardmasking(IHC_mix, glimpses) returns validated glimpses with non-simultaneous
% masking effect (forward masking).
%
% input:
%        IHC_mix        inner hair cell response of speech+noise mixture
%        glimpses       putative glimpses above hearing threshold
%
% output:
%        glimpses_FM    genuine glimpses
%
% Author: Yan Tang


[bands, frames] = size(IHC_mix);
glimpses_FM = zeros(bands,frames);

% across all frequencies
for idx = 1:bands  
   glimpses_f = glimpses(idx, :);
   IHC_f = IHC_mix(idx,:);
   glimpses_g = zeros(size(IHC_f));
   
   % detect peaks in this channel
   [~, peaks] = findpeaks(IHC_f);
   peaknum = length(peaks);
   
   % validate each peak
   for peakidx = 1:peaknum
      currentpeak = peaks(peakidx);
      if peakidx < peaknum
         % get the index of the next peak
         nextpeak = peaks(peakidx+1);
      else
         nextpeak = frames+1;
      end
      
      % current peak is speech-dominant
      if glimpses_f(currentpeak)
         % contains last valid index
         sequence = [];
         
         pointer_fore = currentpeak + 1;
         sequence = cat(2,sequence,currentpeak);
         while pointer_fore <= frames
            % check regions after the peak consecutively until next peak
            % the frame must be glimpsed according to STEP
            if pointer_fore < nextpeak && glimpses_f(pointer_fore)
               sequence = cat(2,sequence,pointer_fore);
               pointer_fore = pointer_fore +1;
            else
               break;
            end
         end         
         glimpses_g(sequence) = 1;
      end
   end 
   glimpses_FM(idx,:) = glimpses_g;
end