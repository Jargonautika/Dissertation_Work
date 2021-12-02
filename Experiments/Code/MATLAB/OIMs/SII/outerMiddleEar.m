function h=outerMiddleEar(cf)
% OUTERMIDDLEEAR   Outer-middle ear transfer function
%
%    h=outerMiddleEar(f)
%    Returns the outer-middle ear transfer function at the given
%    frequency, f. The data is from the ISO standard for equal
%    loudness contours. Note, only defined for frequencies between 20
%    and 12,500 Hz.
%

if ((min(cf)<20) || (max(cf)>12500))
  error('centre frequency out of range');
end

f=[20 25 31.5 40 50 63 80 100 125 160 200 250 315 400 500 630 800 ...
      1000 1250 1600 2000 2500 3150 4000 5000 6300 8000 10000 12500];
tf=[74.3 65 56.3 48.4 41.7 35.5 29.8 25.1 20.7 16.8 13.8 11.2 8.9 ...
      7.2 6 5 4.4 4.2 3.7 2.6 1 -1.2 -3.6 -3.9 -1.1 6.6 15.3 16.4 11.6];

h=interp1(f,tf,cf,'PCHIP');
% h=db2amp(-h,30);
% h=interp1([min(h) max(h)], [0 1], h);

% the figures are gains in db
% - so convert to get attenuation in amplitude units

% h=db2amp(-h); 