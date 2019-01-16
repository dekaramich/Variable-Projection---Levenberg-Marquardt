function [ynorm, maxvarytr, minvarytr] = normalizationy(y, a, b);
ytrT = y';
%max, min values
maxvarytr = max(ytrT);
minvarytr = min(ytrT);

Ly = (ytrT(1,:)- minvarytr)/(maxvarytr-minvarytr);
ytrn(1,:) = a+Ly*(b-a);

ynorm = ytrn';
end