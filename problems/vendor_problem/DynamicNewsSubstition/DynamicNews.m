function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = DynamicNews(x, runlength, seed, ~)
%function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = DynamicNews(x, runlength, seed, other)
% x is the row vector, quantity to buy of each product
% runlength is the number of days of demand to simulate
% seed is the index of the substreams to use (integer >= 1)
% other is not used
% Returns Mean and Variance of Profit

%   *************************************************************
%   ***             Written by Danielle Lertola               ***
%   ***         dcl96@cornell.edu    June 27th, 2012          ***
%   ***              Edited by Bryan Chong                    ***
%   ***        bhc34@cornell.edu    October 15th, 2014        ***
%   *************************************************************

constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;
FnGrad=NaN;
FnGradCov=NaN;

if (max(x < 0)>0) || (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed),
    fprintf('All values in x should be >= 0, runlength should be positive integer, seed must be a positive integer\n');
    fn = NaN;
    FnVar = NaN;
    FnGrad = NaN;
    FnGradCov = NaN;
else
    %%% Setting 1 %%%
    
    n=2; % number of products
    T=5; % number of customers
    u=ones(n,1); % product constant
    mu=1;
    
    %%%%%%%%%%%%%%%%%
    
    
    % %%% Setting 2 %%%
    %
    % n=10; % number of products
    % T=30; % number of customers
    % u=ones(n,1)*5; % product constant
    % for i=1:n
    %    u(i)=u(i)+i;
    % end
    % mu=1;
    %
    % %%%%%%%%%%%%%%%%%
    
    cost = ones(1,n)*5;
    sellPrice = ones(1,n)*9;
    
    % Generate a new stream for random numbers
    OurStream = RandStream.create('mrg32k3a');
    
    % Set the substream to the "seed"
    OurStream.Substream = seed;
    
    % Compute Gumbel RV's for Utility
    OldStream = RandStream.setGlobalStream(OurStream);
    Gumbel= evrnd(mu*-psi(1),mu,[n,runlength,T]);
    RandStream.setGlobalStream(OldStream);
    
    % Determine Utility Function
    Utility=zeros(n,runlength,T);
    for i=1:n
        Utility(i,:,:)=u(i)+ Gumbel(:,:,i);
    end
    
    % Run Simulation
    initial=x'*ones(1,runlength);
    inventory=initial;
    
    for j=1:T
        available=(inventory>0);
        decision=available.*Utility(:,:,j);
        [maxVal, index]=max(decision);
        itembought=maxVal>0;
        for k=1:runlength
            inventory(index(k),k)=inventory(index(k),k)-itembought(k);
        end
    end
    
    % Compute daily profit
    numSold =initial - inventory;
    unitProfit=sellPrice-cost;
    singleRepProfit=unitProfit*numSold;
    fn = mean(singleRepProfit);
    FnVar = var(singleRepProfit)/runlength;
end