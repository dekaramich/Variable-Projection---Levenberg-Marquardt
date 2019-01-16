function [DPHIwSIGMA] = grad_fun_SIGMA(c,w,xtrnorm,N, data,SIGMA)
tic
DPHIwSIGMA(1:data,1:N) = 0;
for cent = 1:N
    %norm squared                     
    %d = abs(xtrnorm-repmat(c(cent,:),data,1));                     
    diff = repmat(c(cent,:),data,1) - xtrnorm;                    
    %Gaussian                                          
    dphidSIGMA_ =  -(1/(SIGMA(1,cent)^3))*exp((-sum(diff.^2,2))/(2*SIGMA(1,cent)^2)).*sum(diff.^2,2);                                     
    DPHIwSIGMA(:,cent) = w(cent,1)*dphidSIGMA_;
end          
gradnew = toc
end

%%%%%%% 2 loops
% function [DPHIwSIGMA1] = grad_fun_SIGMA(x,w,phi,xtrnorm,N, data, no_var,ytrnorm,SIGMA)
 
%  DPHIwSIGMA1(1:data,1:N) = 0;
%  tic
%  for cent = 1:N
%      for i=1:data
%          diff = c(cent,:)-xtrnorm(i,:);
%          dphidSIGMA = ((diff*diff')/(SIGMA(1,cent)^3))*exp((-diff*diff')/(2*SIGMA(1,cent)^2));         
%          DPHIwSIGMA1(i,cent) = w(cent,1)*dphidSIGMA;
%      end             
%  end
%  gradold = toc
% end         