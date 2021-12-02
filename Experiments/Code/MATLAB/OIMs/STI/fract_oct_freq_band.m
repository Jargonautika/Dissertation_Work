function [fexact_l_c_u, fnominal_l_c_u, fnominal_str_l_c_u] = fract_oct_freq_band(N, min_f, max_f, SI_prefixes, base10)
% % fract_oct_freq_band: Calculates the 1/nth octave band center, lower, and upper bandedge frequency limits
% % 
% % Syntax;    
% % 
% % [fexact_l_c_u, fnominal_l_c_u, fnominal_str_l_c_u] = fract_oct_freq_band(N, min_f, max_f, SI_prefixes, base10);
% % 
% % **********************************************************************
% % 
% % Description
% % 
% % This program calculates the fractional, 1/N octave band center 
% % frequencies and the lower and upper bandedge limits of every 
% % frequency band from min_f to max_f (Hz).
% % 
% % The exact and nominal values of the frequencies are output. 
% % The exact values are useful for calculations and making filters.  
% % The nominal values are useful for tables, plots, figures, and graphs.
% % 
% % The frequency band nominal values are rounded to three or more 
% % significant digits based on the number of bands per octave.  
% % Full octave and third octave bands are rounded to three significant digits. 
% % 
% % The exact and nominal frequency values and character strings are output.  
% % 
% % 1000 Hz is a center frequency when N is odd, odd fractional octave bands.  
% % 1000 Hz is an edge frequency when N is even, even fractional octave bands.  
% % 
% % fract_oct_freq_band produces exact and nominal frequency bands which 
% % satisfy ANSI S1.6-R2006 and ANSI S1.11-R2014.    
% % 
% % 
% % fract_oct_freq_band is a modification of centr_freq
% % centr_freq can be found on Matlab Central File Exchange
% % The Matlab ID is 17590.
% % 
% % **********************************************************************
% % 
% % Input Variables
% % 
% % N=3;            % 3 three bands per octave.
% %                 % 1 one band per octave.  
% %                 % N is the number of frequency bands per octave.  
% %                 % N can be any integrer > 0.  
% %                 % Default is 3 for third octave bands.  
% % 
% % min_f=20;       % min_f is the minimum frequency band to calculate (Hz).
% %                 % min_f > 0. Must be graeater than 0.  
% %                 % default is 20;
% % 
% % max_f=20000;    % max_f is the maximum frequency band to calculate (Hz).
% %                 % max_f > 0. Must be graeater than 0.  
% %                 % default is 20000;
% % 
% % SI_prefixes=0;  % This parameter controls the output for the character 
% %                 % strings of the nominal frequencies.
% %                 % 1 for using SI prefixes (i.e. k for 1000)
% %                 % 0 for not using prefixes.
% %                 % default is SI_prefixes=0;
% % 
% % base10=1;       % 1 for using base 10 filter frequencies
% %                 % 0 for using base 2 filter frequencies
% %                 % otherwise  base 10 filter frequencies
% %                 % default base 10 filter frequencies
% %
% % **********************************************************************
% % 
% % Output Variables
% % 
% % fexact_l_c_u is a 2D array of exact frequency bands in Hz.
% % fexact_l_c_u( frequecy bands, [lower=1, center=2, upper =3;] );
% % 
% % fnominal_l_c_u is a 2D array of nominal frequency bands in Hz.
% % fnominal_l_c_u{ frequecy bands, [lower=1, center=2, upper =3;] };
% % 
% % fnominal_str_l_c_u is a 2D cell array of nominal frequency band strings in Hz.
% % fnominal_str_l_c_ufnominal_l_c_u{ frequecy bands, [lower=1, center=2, upper =3;] };
% % 
% % **********************************************************************
% 
% 
% Example='1';
% 
% % Full octave band center frequencies from 20 Hz to 20000 Hz
% close('all');
% [fexact_l_c_u, fnominal_l_c_u, fnominal_str_l_c_u] = fract_oct_freq_band(1, 20, 20000);
% figure(1);
% semilogy(fexact_l_c_u(:, 1), ':ro', 'linewidth', 4, 'markersize', 10);
% hold on;
% semilogy(fexact_l_c_u(:, 2), '-gp', 'linewidth', 4, 'markersize', 10);
% semilogy(fexact_l_c_u(:, 3), '-.cs', 'linewidth', 4, 'markersize', 10);
% semilogy(fnominal_l_c_u(:, 1), '-.kx', 'linewidth', 3, 'markersize', 14);
% semilogy(fexact_l_c_u(:, 2), ':m*', 'linewidth', 3, 'markersize', 14);
% semilogy(fexact_l_c_u(:, 3), '--b+', 'linewidth', 3, 'markersize', 10);
% legend({'Exact lower band', 'Exact center', 'Exact upper band', ...
%         'Nominal lower band','Nominalcenter', 'Nominal upper band'}, ...
%         'location', 'northwest');
% set(gca, 'fontsize', 24);
% maximize(gcf);
% 
% 
% % third octave band center frequencies from 20 Hz to 20000 Hz
% [fexact_l_c_u, fnominal_l_c_u, fnominal_str_l_c_u] = fract_oct_freq_band(3, 20, 20000);
% 
% % twelveth octave band center frequencies from 100 Hz to 10000 Hz
% [fexact_l_c_u, fnominal_l_c_u, fnominal_str_l_c_u] = fract_oct_freq_band(12, 100, 10000);
% 
% % twenty-fourth octave band center frequencies from 0.001 Hz to 10000000 Hz
% [fexact_l_c_u, fnominal_l_c_u, fnominal_str_l_c_u] = fract_oct_freq_band(24, 0.001, 10000000);
% 
% 
% % **********************************************************************
% % 
% % References
% % 
% % 1)  ANSI S1.6-R2006 Preferred Frequencies, Frequency Levels, and 
% %     Band Numbers for Acoustical Measurements, 1984.
% % 
% % 2)  ANSI S1.11-2014, (2014). American National Standard Electroacoustics
% %     - Octave-band and Fractional-octave-band Filters - Part 1: 
% %     Specifications (a nationally adopted international standard). 
% %     American National Standards Institute, Acoustical Society of 
% %     America, New York 
% % 
% % 
% % **********************************************************************
% % 
% % fract_oct_freq_band is a modification of centr_freq
% % centr_freq can be found on Matlab Central File Exchange
% % The Matlab ID is 17590.
% % 
% % Written by   Author  Eleftheria  
% %              E-mail  elegeor@gmail.com 
% %              Company/University: University of Patras 
% % 
% % 
% % **********************************************************************
% % 
% % List of Dependent Subprograms for 
% % fract_oct_freq_band
% % 
% % FEX ID# is the File ID on the Matlab Central File Exchange
% % 
% % 
% % Program Name   Author   FEX ID#
% % 1) sd_round		Edward L. Zechmann			
% % 
% % 
% % **********************************************************************
% % 
% % 
% % Program Modified by Edward L. Zechmann
% % 
% % modified  3 March       2008    Original modification of program
% %                                 updated comments 
% % 
% % modified 13 August      2008    Updated comments.  
% %   
% % modified 18 August      2008    Added rounding last digit to nearest 
% %                                 multiple of 5.  Added Examples.  
% %                                 Updated comments.  
% %    
% % modified 21 August      2008    Fixed a bug in usign min_f and max_f 
% %                                 which does not include 1000 Hz.  
% %                                 Zipped the depended program sd_round. 
% %                                 Updated comments.  
% % 
% % modified 18 November    2008   	Added additional rounding 
% % 
% % modified  8 December    2008    Updated comments.
% % 
% % modified 18 December    2008    Updated comments.
% % 
% % modified  4 January     2008    Changed rounding of the lower and upper 
% %                                 bandedge frequency limits.
% % 
% % modified  6 October     2009    Updated comments
% % 
% % modified 22 January     2010    Modified the number of significant 
% %                                 digits for rounding. The number of  
% %                                 significnat digits increases as the
% %                                 number of bands per octave increases.  
% %                                 This supports high resolution octave
% %                                 band analysis.
% % 
% % modified  4 October     2014    Modified the even octave bands to have
% %                                 1000 Hz as an edge frequency.  
% %                                 Changed the number of significant
% %                                 digits calculation for rounding. 
% %                                 Limited the number of bands per octave 
% %                                 from 1 to 43.
% % 
% % 
% % **********************************************************************
% % 
% % Please Feel Free to Modify This Program
% %   
% % See Also: centr_freq, sd_round
% %   

if (nargin < 1 || isempty(N)) || ~isnumeric(N)
    N=3;
end

% N must be an integrer;
N=round(N);

% N must be between 1 and 43.  Filters with more than 43 bands per octave 
% have machine precision rounding errors in the band edge frequencies.    
N(N < 1)=1;
N(N > 43)=43;

if (nargin < 2 || isempty(min_f)) || (logical(min_f < 0) || ~isnumeric(min_f))
    min_f=20;
end

if (nargin < 3 || isempty(max_f)) || (logical(max_f < 0) || ~isnumeric(max_f))
    max_f=20000;
end

if (nargin < 4 || isempty(SI_prefixes)) ||  ~isnumeric(SI_prefixes)
    SI_prefixes=0;
end

if (nargin < 5 || isempty(base10)) || ~isnumeric(base10)
    base10=1;
end

if base10 ~= 0;
    base10=1;
end

% Determine if N is odd or even.  
b=mod(N-1, 2);
% b == 0 if N is even
% b == 1 if N is odd


% Now the nominal frequencies can be calculated

% For base-2 filter frequencies
% 
% 2^N=floor(3+n_sd_digits*log10(2));

% base-10 filter frequencies
% 
% G^N=floor(3+n_sd_digits*log10(G));

% Calculate the base number G  
if base10 == 1
    G=10^(3/10);
else
    G=2;
end

% Calculate the maximum number of octave bands above 1000 Hz.
Nmax=round(N*ceil(log(max_f/1000)/log(G))+1);

if Nmax < 1
    Nmax=1;
end

f_a=zeros(Nmax, 1);
f_a(1)=1000*(G.^(b/((1+b)*N))); 

for e1=1:Nmax;          
    % center frequencies over 1000 Hz
    % Center frequencies are based on a 1/3 octave band
    % falling exactly at each factor of 10
    f_a(e1+1, 1)=f_a(e1)*(G.^(1/N)); 
end

% Remove bands above the extended max_f limit
f_a=f_a(f_a < max_f*(G^(0.5/N)));


% Calculate number of bands below 1000 Hz
Nmin=round(N*ceil(log(1000/min_f)/log(G))+1);

if Nmin < 1
    Nmin=1;
end

f_b=zeros(Nmin, 1);
f_b(1)=1000/(G.^(b/((1+b)*N))); 

for e1=1:Nmin;           
    % center frequencies below 1000 Hz
    % In the base 10 system, the center frequencies fall exactly at each 
    % factor of 10 based on third octave intervals.  
    f_b(e1+1, 1)=f_b(e1)/(G.^(1/N));
end

% Remove bands below the extended min_f limit
f_b=f_b(f_b > min_f*(G^(-0.5/N)));

% Concatenate center frequency bands
% Make center frequency bands unique
fc = unique([f_b;f_a]);           

% Remove bands above the extended max_f limit
fc=fc(fc < max_f*(G^(0.5/N)));

% Remove bands below the extended min_f limit
fc=fc(fc > min_f*(G^(-0.5/N)));

% Calculate the number of center frequency bands
num_bands=length(fc);

% fc is the array of exact center frequencies;
fc_exact=fc;

% ***********************************************************************
% 
% Calculate the lower and upper bounds for the lower and upper
% band edge frequencies
flu = [fc(1)./G.^(1./(2.*N)); fc.*( G.^(1./(2.*N)) )];     

% Form the numeric arrays for the exact lower frequency band edge limits.
fl_exact=flu(1:num_bands);

% Form the numeric arrays for the exact upper frequency band edge limits.
fu_exact=flu(1+(1:num_bands));

% ************************************************************************
% 
% Calculate the number of significant digits for rounding the exact
% frequencies into the nominal frequencies.
% 
% Full octave band and third octave band must have 3 significant digits.  
% the number of digits per ocrtave increases N*log10(G).  
% This is derived by solving the equation 2^N=floor(3+n_sd_digits*log10(2));
% for n_sd_digits.
% 
num_sd_digits=floor(3+N*log10(G));
m=10.^(num_sd_digits-1);

% ************************************************************************
% 
% Calculations for nominal frequencies
% 
% Apply appropriate rounding to the center frequencies
[fc,   fc_str]   = sd_round(fc, num_sd_digits, 1, 5, SI_prefixes);
[fc2,  fc_str2]  = sd_round(fc, num_sd_digits, 1, 100, SI_prefixes);

% If the center frequency rounded to 3 significant digits and the last 
% digit rounded to the nearest multiple of 5 is within 1% of the center 
% frequency rounded to 1 significant digit, then round to 1 significant 
% digit. 
%
ix=find(abs(m*(1-fc./fc2)) < 1);
fc(ix)=fc2(ix);
fc_str(ix)=fc_str2(ix);

% Nominal frequencies
fc_nom=fc;
fc_nom_str=fc_str;


% Apply the same rounding technique to the lower and upper frequency 
% bandedge limits as was applied to the center frequencies.  
[flu,   flu_str]   = sd_round(flu, num_sd_digits, 1, 5, SI_prefixes);
[flu2,  flu_str2]  = sd_round(flu, num_sd_digits, 1, 100, SI_prefixes);


% If the center frequency rounded to 3 significant digits and the last 
% digit rounded to the nearest multiple of 5 is within 1% of the center 
% frequency rounded to 1 significant digit, then round to 1 significant 
% digit. 
%
ix=find(abs(m*(1-flu./flu2)) < 1);
flu(ix)=flu2(ix);
flu_str(ix)=flu_str2(ix);

% Form the numeric and string arrays for the nominal lower frequency 
% band edge limits.
fl_nom=flu(1:num_bands);
fl_nom_str=flu_str(1:num_bands);

% Form the numeric and string arrays for the nominal upper frequency 
% band edge limits.
fu_nom=flu(1+(1:num_bands));
fu_nom_str=flu_str(1+(1:num_bands));

% Concatenate the outputs
fexact_l_c_u=[fl_exact, fc_exact, fu_exact];
fnominal_l_c_u=[fl_nom, fc_nom, fu_nom];

fnominal_str_l_c_u=cell(num_bands, 3);
for e1=1:num_bands;
    fnominal_str_l_c_u{e1, 1}=fl_nom_str{e1}; 
    fnominal_str_l_c_u{e1, 2}=fc_nom_str{e1}; 
    fnominal_str_l_c_u{e1, 3}=fu_nom_str{e1}; 
end

