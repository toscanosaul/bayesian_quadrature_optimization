function [error] = pmf(num_p, num_m, train_vec, probe_vec, epsilon, lambda, maxepoch, num_feat, l_rating, u_rating);

rand('state',0);
randn('state',0);

momentum = 0.8;
numbatches = 9;
epoch=1;

w1_M1 = 0.1*randn(num_m, num_feat);
w1_P1 = 0.1*randn(num_p, num_feat);
w1_M1_inc = zeros(num_m, num_feat);
w1_P1_inc = zeros(num_p, num_feat);

mean_rating = mean(train_vec(:,3));

pairs_tr = length(train_vec); % training data
pairs_pr = length(probe_vec); % validation data

N = pairs_tr / numbatches;
N_int = int64(N);

[n1,n2]= size(train_vec);

for epoch = epoch:maxepoch
  rr = randperm(pairs_tr);
  train_vec = train_vec(rr,:);
  clear rr

  for batch = 1:numbatches

    next = min(batch*N_int, n1);
    aa_p   = double(train_vec((batch-1)*N_int+1:next,1));
    aa_m   = double(train_vec((batch-1)*N_int+1:next,2));
    rating = double(train_vec((batch-1)*N_int+1:next,3));

    if batch*N_int > n1
        [size1, size2]= size(aa_p);
        size_2 = N_int - size1;
        aa_p_2   = double(train_vec(1:size_2,1));
        aa_m_2   = double(train_vec(1:size_2,2));
        rating_2 = double(train_vec(1:size_2,3));
        aa_p = cat(1, aa_p, aa_p_2);
        aa_m = cat(1, aa_m, aa_m_2);
        rating = cat(1, rating, rating_2);
    end

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

    for ii=1:N_int
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


%%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%%
NN=pairs_pr;

aa_p = double(probe_vec(:,1));
aa_m = double(probe_vec(:,2));
rating = double(probe_vec(:,3));

pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;
ff = find(pred_out>u_rating); pred_out(ff)=u_rating; % Clip predictions
ff = find(pred_out<l_rating); pred_out(ff)=l_rating;

err_valid(epoch) = sqrt(sum((pred_out- rating).^2)/NN);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error = err_valid(epoch);
