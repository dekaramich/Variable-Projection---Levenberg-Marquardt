function [xnorm, allmaxx, allminx] = normalizationx(x, no_var, a, b);
    xT = x';
    for i = 1:no_var
        %max, min values of each variable
        maxvar = max(xT(i,:));
        minvar = min(xT(i,:));

        Lx = (xT(i,:)- minvar)/(maxvar-minvar);
        xn(i,:) = a+Lx*(b-a);

        allmaxx(1,i) = maxvar;
        allminx(1,i) = minvar;
    end

    xnorm = xn';    
end