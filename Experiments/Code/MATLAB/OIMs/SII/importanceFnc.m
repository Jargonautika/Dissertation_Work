function h=importanceFnc(cfs, tst, reverse)
% tst = 1:	Average speech as specified in Table 3
%		2:	various nonsense syllable tests where most English
%			phonemes occur equally often
%		3:	CID-22
%		4:	NU6
%		5:	Diagnostic Rhyme test
%		6:	short passages of easy reading material
%		7:	SPIN

if ~((tst==1)||(tst==2)||(tst==3)||(tst==4)||(tst==5)||(tst==6)||(tst==7)),
	error('Band Importance function must be integer between 1 and 7');
end;
if length(cfs) < 15
  error('not enough bands');
end

if ((min(cfs)<20) || (max(cfs)>12500))
  error('centre frequency out of range');
end

f = [160 200 250 315 400 500 630 800 1000 1250 1600 2000, ...
     2500 3150 4000 5000 6300 8000];

BIArr= [0.0083	0		0.0365	0.0168	0		0.0114	0
		0.0095	0		0.0279	0.013	0.024	0.0153	0.0255
		0.015	0.0153	0.0405	0.0211	0.033	0.0179	0.0256
		0.0289	0.0284	0.05	0.0344	0.039	0.0558	0.036
		0.044	0.0363	0.053	0.0517	0.0571	0.0898	0.0362
		0.0578	0.0422	0.0518	0.0737	0.0691	0.0944	0.0514
		0.0653	0.0509	0.0514	0.0658	0.0781	0.0709	0.0616
		0.0711	0.0584	0.0575	0.0644	0.0751	0.066	0.077
		0.0818	0.0667	0.0717	0.0664	0.0781	0.0628	0.0718
		0.0844	0.0774	0.0873	0.0802	0.0811	0.0672	0.0718
		0.0882	0.0893	0.0902	0.0987	0.0961	0.0747	0.1075
		0.0898	0.1104	0.0938	0.1171	0.0901	0.0755	0.0921
		0.0868	0.112	0.0928	0.0932	0.0781	0.082	0.1026
		0.0844	0.0981	0.0678	0.0783	0.0691	0.0808	0.0922
		0.0771	0.0867	0.0498	0.0562	0.048	0.0483	0.0719
		0.0527	0.0728	0.0312	0.0337	0.033	0.0453	0.0461
		0.0364	0.0551	0.0215	0.0177	0.027	0.0274	0.0306
		0.0185	0		0.0253	0.0176	0.024	0.0145	0];

h=interp1(f,BIArr(:,tst),cfs,'PCHIP');
if reverse
    h = -h-min(-h);
end
h(h<0) = 0;
h = h ./ sum(h);
