function y = thrshd_terhardt(CFs)

% To compute threshold of hearing for given frequency bands according to the
% equation in E. Terhardt, "Calculating virtual pitch", Hearing Res., vol. 1
% pp. 155-182, 1979.

% Argument CFs: a vector of central frequencies.
% Output   thresholds : a vector of corresponding thresolds of give frequencies



tcf = CFs ./ 1000;
y = 3.64*(tcf.^(-0.8))-6.5*exp(-0.6*((tcf-3.3).^2))+ 10e-3*(tcf.^4);
