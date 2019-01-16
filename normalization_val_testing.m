function [xnorm] = normalization_val_testing(x, no_var, a, b, allmaxx, allminx);
xT = x';
for i = 1:no_var
        L = (xT(i,:)-allminx(1,i))/( allmaxx(1,i)- allminx(1,i));
        xn(i,:) = a+L*(b-a); 
end

xnorm = xn';
end


