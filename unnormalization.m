function [ytrue] = unnormalization(a, b, minvary, maxvary, ytruenorm, data);
    ytrue =  minvary+((ytruenorm-a)*(maxvary-minvary))/(b-a);
end
        