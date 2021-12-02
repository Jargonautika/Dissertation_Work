function y=newOuterMiddleEar(x,fs)
% from Guy and Stuart, 2008

numpoints=1024;%512;

% frequencies and db gain measured from figure 2 of Meddis and Hewitt
% (1991) JASA 89 (6) p.2868
freq=[0 100 200 300 400 500 600 1000 2000 3000 4000 5000 6000 7000  8000 9000 10000 fs/2];
db=[0 5 15 20 25 26 27 32 29 32 26 26 25 16 15 4 3 0];

% fir2 requires frequencies as a fraction of the nyquist rate
f=freq/(fs/2);

% note that our breakpoints are in db, and we need to turn these into
% magnitudes for fir2 to work
b = fir2(numpoints,f,10.^(db/20));

y = filter(b,[1],x); 