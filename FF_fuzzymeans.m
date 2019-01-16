%% Algorithm 1 Levenberg-Marquardt method
%  Full Functional, Fuzzy means, constant sigma
clear
%% datasets
%training, validation and testing
load('training_data_set1.mat');
load('validation_data_set2.mat');
load('testing_data_set3.mat');

%number of examples & number of inputs
data = size(xtr,1); no_var = size(xtr,2);

%% normalization
%normalization limits [a,b]
a = -1; b = 1;

%training dataset
[xtrnorm, allmaxxtr, allminxtr] = normalizationx(xtr, no_var, a, b);
[ytrnorm, maxvarytr, minvarytr] = normalizationy(ytr, a, b);
%validation dataset
[xvanorm] = normalization_val_testing(xva, size(xva,2), a, b, allmaxxtr, allminxtr);
%testing dataset
[xtenorm] = normalization_val_testing(xte, size(xte,2), a, b, allmaxxtr, allminxtr);


%% RBF center selection 
%% fuzzy means
% [center locations, number of centers] = ...
%... SFMfunction(number of inputs, number of examples, x, number of fuzzy sets)
[c, N] = SFMfunction(no_var, data, xtrnorm, 7);
cinitial = c;

%% sigma ( P-nearest neighbors )
[SIGMA] = Pnn(N, c);
SIGMAinitial = SIGMA;  

%% count training time
count = tic;

%% PHI matrix (m x N)
[phi] = phi_fun(c, xtrnorm, N, data, no_var, SIGMA);
%% initial weights ( Linear Least Square Solution )
w = phi\ytrnorm;      
winitial = w;
%% initial cost
hh(1,1) = 0.5*norm(ytrnorm - phi*w)^2;
%% RMSE
%% training dataset
itytruetrnorm = phi*w;
[itytruetr] = unnormalization(a, b, minvarytr, maxvarytr, itytruetrnorm, size(ytr,1));
itRMSEtr(1,1) = sqrt(mean((ytr-itytruetr).^2));
%% validation dataset
[itphiva] = phi_fun(c, xvanorm, N, size(xva,1), size(xva,2), SIGMA); 
itytruevanorm = itphiva*w;
[itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1)); 
itRMSEva(1,1)= sqrt(mean((yva-itytrueva).^2));
%% save this value as the min value of RMSE in the validation dataset
minva(1,1) = itRMSEva(1,1);
%and the current parameters
cmin = c;
wmin = w;
SIGMAmin = SIGMA;
%% 1/2 || A d - b ||^2 + lambda/2 ||d||^2
%Jacobian of PHI matrix
%DPHIwc (m x Nno_var)
DPHIwc = grad_fun(c, w, xtrnorm, N, data, no_var, SIGMA);
%A m x (Nno_var + N)
A = [DPHIwc -phi];
%b m x 1
B = -(ytrnorm - phi*w);
%I (Nno_var x Nno_var)
I = eye(size(A'*A,1),size(A'*A,2));

%% damping parameter
lambda = 10;

%% successful LM iterations (r>0) 
%which reduce RMSE in the validation dataset
kva = 1; %counter
%% k total iterations counter
k = 1;
%% exit flag
flagg = false;
%% 1st stop criterion
%if the RMSEval doesn't improve within MAX ( = 5) iterations
flagit = 0;
%% 2nd stop criterion
%if the objective h doesn't improve within MAX ( = 5) iterations
flagit2 = 0;

count = toc(count);

%tictoc counter for succesfull iterations
tictoc = 0;
succesfulliterations = tic;
while flagg == false
    if k > 1
        succesfulliterations = tic;
        %Jacobian, A, b...      
        DPHIwc = grad_fun(c, w, xtrnorm, N, data, no_var, SIGMA);      
        A = [DPHIwc -phi];
        B = -(ytrnorm - phi*w);      
    end   
    %% compute LM direction   
    %Cholesky factorization  
    %lower triangular matrix
    %L (Nno_var) x (Nno_var)    
    L = chol(A'*A + lambda*I,'lower');    
    %d vector (Nno_var + N) x 1
    dvec = L'\(L\(A'*B));
    %d centers
    dcvec = dvec(1:N*no_var);
    dc =vec2mat(dcvec,no_var);
    %d weights
    dw = dvec(N*no_var+1:N*no_var+N+1);
    %trial center locations  
    trialc = c + dc;
    %trial weights  
    trialw = w + dw; 
    %% QR factorization of Phi at trialc = c+d    
    [phicd] = phi_fun(trialc, xtrnorm, N, data, no_var, SIGMA);   
    %% gain ratio, r    
    num = 0.5*norm(ytrnorm - phi*w)^2 - 0.5*norm(ytrnorm - phicd*trialw)^2;
    denom = 0.5*norm(ytrnorm - phi*w)^2 - 0.5*norm(ytrnorm - phi*w + A*dvec)^2;
    %cost at current c   
    hh(k+1,1) = 0.5*norm(ytrnorm - phi*w)^2; 
    %cost at c + d   
    hh(k+1,2) = 0.5*norm(ytrnorm - phicd*trialw)^2;
    %linear model cost
    hh(k+1,3) = 0.5*norm(ytrnorm - phi*w + A*dvec)^2;
    r = num/denom;
    %% update lambda   
    if r < 0.25    
        %small value of r, increase lambda        
        %go as the steepest descent       
        %reduce step length
        lambda = 4*lambda;       
    elseif r > 0.75   
        %large value of rho, decrease lambda        
        %go as the gauss-newton
        %increase step length      
        lambda = lambda/2;       
    end    
    %%
    if r > 0  
        %update centers
        c = c + dc; 
        %update weights      
        w = w + dw;
        %PHI matrix
        [phi] = phi_fun(c, xtrnorm, N, data, no_var, SIGMA);        
        %calculate RMSE in the validation dataset    
        [itphiva] = phi_fun(c, xvanorm, N, size(xva,1), size(xva,2), SIGMA);
        itytruevanorm = itphiva*w;
        %RMSE validation path
        [itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1));            
        itRMSEva(kva+1,1)= sqrt(mean((yva-itytrueva).^2));   
        %RMSE training path
        itytruetrnorm = phi*w;
        [itytruetr] = unnormalization(a, b, minvarytr, maxvarytr, itytruetrnorm, size(ytr,1));
        itRMSEtr(kva+1,1) = sqrt(mean((ytr-itytruetr).^2));
        %is this new value of RMSEva lower than the minimum one?       
        minva(1,1) = round(minva(1,1),4);       
        itRMSEva(kva+1) = round(itRMSEva(kva+1),4);         
        if (minva(1,1) > itRMSEva(kva+1)) 
            %update the min value  
            minva(1,1) = itRMSEva(kva+1);       
            %and the parameters        
            cmin = c;         
            wmin = w;         
            SIGMAmin = SIGMA;       
            %RMSEval is improved         
            flagit = 0;             
            tictoc = tictoc + toc(succesfulliterations);
        else  
            %RMSEval is not improved           
            flagit = flagit + 1;            
        end      
        %next iteration LM
        kva = kva+1;
        %h is improved
        flagit2 = 0;
    else
        %h is not improved
        flagit2 = flagit2 + 1;
    end
    %criterions check
    if  flagit >= 5 || flagit2 >= 5
        flagg = true; %exit while
    end 
    k = k+1;%succesful + unsuccesful iterations
end
TrainingTime =  count + tictoc;
%% final best values of the parameters in the current run
c = cmin;
w = wmin;
SIGMA = SIGMAmin; 
%% final results RMSE
%% training dataset
[itphitr] = phi_fun(c,xtrnorm,N, data, no_var,SIGMA);   
itytruetrnorm = itphitr*w;
[itytruetr] = unnormalization(a, b, minvarytr, maxvarytr, itytruetrnorm, size(ytr,1));
RMSEtr = sqrt(mean((ytr-itytruetr).^2));
%% validation dataset
[itphiva] = phi_fun(c,xvanorm, N, size(xva,1), size(xva,2),SIGMA);
itytruevanorm = itphiva*w;
[itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1));
RMSEva = sqrt(mean((yva-itytrueva).^2));
%% testing dataset
[itphite] = phi_fun(c,xtenorm, N, size(xte,1), size(xte,2),SIGMA);
itytruetenorm = itphite*w;
[itytruete] = unnormalization(a, b, minvarytr, maxvarytr, itytruetenorm, size(yte,1));  
RMSEte = sqrt(mean((yte-itytruete).^2));

%iterations needed & min RMSE in the validation dataset
[minRMSEva,indexminn]=min(round(itRMSEva,4));
it = indexminn-1;

%% RMSE
%validation
bva = minRMSEva
%testing
bte_va = RMSEte
%training
btr_va = RMSEtr
%LM iterations needed
bit = it
%Training Time
TrainingTime
%RMSE path in the validation dataset
RMSEvaPath = itRMSEva;
%RMSE path in the training dataset
RMSEtrPath = itRMSEtr;

%% Performance Plot
linx = 0:1:size(RMSEvaPath,1)-1;
plot(linx,RMSEvaPath,'Color',[0.8500, 0.3250, 0.0980])
hold on
plot(linx,RMSEtrPath,'Color',[0, 0.4470, 0.7410])
hold on
scatter(bit,bva,'k')
title(['Best Validation Performance is ', num2str(bva), ' at iteration ',num2str(it)])
xlabel('successful iterations')
ylabel('Root Mean Square Error (RMSE)')
legend('Validation','Train','Best')
legend('Location','northwest')


%% INITIAL VALUES OF THE PARAMETERS
winitial; cinitial; SIGMAinitial;
%% FINAL VALUES OF THE PARAMETERS
w; c; SIGMA;