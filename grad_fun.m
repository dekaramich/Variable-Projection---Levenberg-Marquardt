function [DPHIw] = grad_fun(c,w,xtrnorm,N, data, no_var,SIGMA)
%  tic
DPHIw(1:N,1:N*no_var) = 0;
n_arx=1;
n_tel=no_var;
  
for cent = 1:N
    %norm squared                  
    %d = abs(xtrnorm-repmat(c(cent,:),data,1));                      
    diff = repmat(c(cent,:),data,1) - xtrnorm;                         
    %Gaussian                      
    dphidc_ =  -(2*(-1/(2*SIGMA(1,cent)^2))*exp((-sum(diff.^2,2))/(2*SIGMA(1,cent)^2)).*diff);                      
    DPHIw(1:data,n_arx:n_tel) = w(cent,1)*dphidc_;                   
    n_arx=n_tel+1;                  
    n_tel=n_tel+no_var;          
end           
% gradnew = toc
end

%% 2 loops
% function [DPHIw] = grad_fun(c,w,phi,xtrnorm,N, data, no_var,ytrnorm,SIGMA)
% tic
%  DPHIw2(1:N,1:N*no_var) = 0;
%  n_arx=1;
%  n_tel=no_var;
%   
%  for cent = 1:N
%      for i=1:data
%          diff = c(cent,:)-xtrnorm(i,:);
%          dphidc = 2*(-1/(2*SIGMA(1,cent)^2))*exp((-diff*diff')/(2*SIGMA(1,cent)^2))*diff;
%          DPHIw2(i,n_arx:n_tel) = -w(cent,1)*dphidc;
%      end
%      n_arx=n_tel+1;
%      n_tel=n_tel+no_var;              
%  end
%  gradold = toc
% end