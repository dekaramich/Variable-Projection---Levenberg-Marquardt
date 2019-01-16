function [ynorm, maxvarytr, minvarytr] = normalizationy_val(y, a, b,minvarytr,maxvarytr);
ytrT = y';

Ly = (ytrT(1,:)- minvarytr)/(maxvarytr-minvarytr);
ytrn(1,:) = a+Ly*(b-a);

ynorm = ytrn';
end