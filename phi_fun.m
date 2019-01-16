function [phi] = phi_fun(cnorm, xnorm, N,data, no_var, SIGMA)
    phi(1:data,1:N+1) = 0;
    xx=repmat(xnorm,1,N);
    cnormnew(1,1:no_var)=cnorm(1,:);

    k = no_var + 1;
    for u = 2:N
        cnormnew(1, k:k+no_var-1)=cnorm(u,:);
        k = k+ no_var;
    end

    c=repmat(cnormnew,data,1);

    dist=xx-c;

    d=sqrt(sum(dist(:,1:no_var).^2, 2));
    %Gaussian kernel
    phi(:,1)=exp(-(d.^2)/(2*SIGMA(1,1)^2));

    k = no_var +1;
    for u=2:N
        d=sqrt(sum(dist(:, k:k+no_var-1).^2, 2));
        %Gaussian kernel
        phi(:,u)=exp(-(d.^2)/(2*SIGMA(1,u)^2));   
        k = k+ no_var;
    end
    phi(:,N+1) = 1;
end

%%2 loops
% function [phi] = phi_fun(c,xtrnorm,N, data, no_var,ytrnorm,SIGMA)
% tic
%     phi2(1:data,1:N+1) = 0;
%     for i=1:data
%         for j=1:N
%             dist = xtrnorm(i,:)-c(j,:);
%             d=sqrt(sum(dist.^2, 2));
%             phi2(i,j) = exp(-(d.^2)/(2*SIGMA(1,j)^2));
%         end
%     end
%     phi2(:,N+1) = 1;
%     old = toc
% end