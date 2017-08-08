function [error] = pmf(num_p, num_m, train_vec, probe_vec, epsilon, lambda, maxepoch, num_feat);

rand('state',0);
randn('state',0);

momentum = 0.8;
num_batches = 9;

w1_M1 = 0.1*randn(num_m, num_feat);
w1_P1 = 0.1*randn(num_p, num_feat);
w1_M1_inc = zeros(num_m, num_feat);
w1_P1_inc = zeros(num_p, num_feat);

mean_rating = mean(train_vec(:,3));

pairs_tr = length(train_vec); % training data
pairs_pr = length(probe_vec); % validation data

error = mean_rating

for epoch = epoch:maxepoch
  rr = randperm(pairs_tr);
  train_vec = train_vec(rr,:);
  clear rr

  for batch = 1:numbatches
    fprintf(1,'epoch %d batch %d \r',epoch,batch);
    N=100000; % number training triplets per batch

    aa_p   = double(train_vec((batch-1)*N+1:batch*N,1));
    aa_m   = double(train_vec((batch-1)*N+1:batch*N,2));
    rating = double(train_vec((batch-1)*N+1:batch*N,3));

    rating = rating-mean_rating; % Default prediction is the mean rating.

    %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
    pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
    f = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));

    %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
    IO = repmat(2*(pred_out - rating),1,num_feat);
    Ix_m=IO.*w1_P1(aa_p,:) + lambda*w1_M1(aa_m,:);
    Ix_p=IO.*w1_M1(aa_m,:) + lambda*w1_P1(aa_p,:);

    dw1_M1 = zeros(num_m,num_feat);
    dw1_P1 = zeros(num_p,num_feat);

    for ii=1:N
      dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
      dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
    end

    %%%% Update movie and user features %%%%%%%%%%%

    w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N;
    w1_M1 =  w1_M1 - w1_M1_inc;

    w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N;
    w1_P1 =  w1_P1 - w1_P1_inc;
  end
end

%%%%%%%%%%%%%% Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
f_s = sum( (pred_out - rating).^2 + ...
    0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
err_train(epoch) = sqrt(f_s/N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%%
NN=pairs_pr;

aa_p = double(probe_vec(:,1));
aa_m = double(probe_vec(:,2));
rating = double(probe_vec(:,3));

pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;
ff = find(pred_out>5); pred_out(ff)=5; % Clip predictions
ff = find(pred_out<1); pred_out(ff)=1;

err_valid(epoch) = sqrt(sum((pred_out- rating).^2)/NN);
fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', ...
          epoch, batch, err_train(epoch), err_valid(epoch));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error = err_valid(epoch);
