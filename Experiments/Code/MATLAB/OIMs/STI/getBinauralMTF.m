function MTF = getBinauralMTF(en_s, en_sn, metric, MFs)
if nargin < 4
   MFs = 0;
end

num_MF = length(MFs);
num_sp = round(length(en_s)/4);

points = [fliplr(0:-2:-num_sp) 0:2:num_sp];
num_trail = length(points);

holder = zeros(num_MF, num_trail);
for i = 1:num_trail
   pad = zeros(abs(points(i)), 1);
   if points(i) < 0
      p_s = [pad; en_s];
      p_sn = [en_sn; pad];
   else
      p_s = [en_s; pad];
      p_sn = [pad; en_sn];     
   end
   
   switch upper(metric)
      case 'PAYTON'
         holder(:, i) = calMTF_Payton(p_s, p_sn);
      case 'DRULLMAN'
         holder(:, i) = getMTF4Binaural(p_s, p_sn, MFs);
      case 'NCM'
         holder(:, i) = calMTF_NCM(p_s, p_sn);
   end  
end

avg_mtf = mean(holder, 1);
[~, loc] = max(avg_mtf);

MTF = holder(:, loc);

