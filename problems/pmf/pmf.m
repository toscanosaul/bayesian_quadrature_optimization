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
error = mean_rating