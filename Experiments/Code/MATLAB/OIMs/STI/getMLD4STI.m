function [iaccs, itd] = getMLD4STI(sig_L, sig_R, fs)

len_sig = length(sig_L);

time_shift = 30; % 30ms
sample_shift = round(fs / 1000 * time_shift);
sample_ms = round(sample_shift / time_shift) * 2; %2ms
s_start = 1:sample_shift:len_sig;
num_fram = length(s_start)-1;

itd =  zeros(1, num_fram);


t = (1:2 * sample_shift - 1) ./ fs;
t = t*1000 - time_shift;
loc_0 = find(t==0);
sample_kept = loc_0 - sample_ms : loc_0 + sample_ms;


iaccs = zeros(length(sample_kept), num_fram); % in the range of 2ms
for f = 1:num_fram
   s_tail = s_start(f)+sample_shift-1;
   if s_tail > len_sig
      s_tail = len_sig;
   end

   seg_L = sig_L(s_start(f):s_tail);
   seg_R = sig_R(s_start(f):s_tail);
   
   iacc = xcorr(formatchecker(seg_L), formatchecker(seg_R), 'coeff');
   iacc = iacc - min(iacc);
   
   [~, loc] = max(iacc);   
   itd(f) = t(loc);
   
   iaccs(:, f) = iacc(sample_kept); 
end


function s = formatchecker(s)
y = size(s);
if y(1) < y(2)
   s = s';
end