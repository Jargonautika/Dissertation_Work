function y=thrshd_fay(f)

% fay - Fay's 1988 approximation to the hearing threshold as a function
% of cf (input f)

f = f./1000;
y=4.08./f + 17.47 - 45.23*f + 45.76*(f.^2) - 19.59*(f.^3) + 4.11*(f.^4) - 0.41*(f.^5) + 0.02*(f.^6);