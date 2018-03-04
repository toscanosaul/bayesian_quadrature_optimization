========================================
Bayesian Quadrature Optimization
========================================
Bayesian Quadrature Optimization is a software package to perform Bayesian optimization. In particular, it includes the algorithm Bayesian Quadrature Optimization from the paper 'Bayesian Optimization with Expensive Integrands' by Saul Toscano-Palmerin and Peter Frazier. 

Installation
------------
In the project root, run:
```
make env
```

Running Tests
-------------
```
make test
```

Documentation
-------------

* Steps to use BQO:

1) Import library: from stratified_bayesian_optimization.services.bgo import bgo

2) Define the objective function g(x) where x are the arguments of the function as a list, and the refunction returns [float]. 

3) For BQO define the integrand function f(x) where x are the arguments of the function as a list, and the function returns [float]. If the function is noisy, include the parameter n_samples, f(x, n_samples), and the function returns [(float) value of function, (float) variance].

4) Define the bounds of the domain of x as a list: [[(float) lower_bound, (float) upper_bound]].

5) For BQO define the bounds_domain_w as a list: [[(float) lower_bound, (float) upper_bound] or [(float) range]]. In the second case, the  list represents the points of the domain of that entry (e.g. when W is finite).

6) Define type_bounds as a list of size equal to the dimension of the domain of the integrand of 0's and 1's. If the entry is 0, it means that the bounds are an interval: [lower_bound, upper_bound]. If the entry is 1, it means that the bounds contains all the possible points of that entry of the domain.

7) Define (str) name_method to choose the BO method to optimize the function. For example, name_method = 'bqo' or 'ei'.

8) n_iterations is the number of points chosen by the BO method.

* Optional Arguments:

9) (Optional) (int) random_seed to run the BO method.

10) (Optional) (int) n_training is the number of training points for the BO method.

11) (Optional) Define problem_name as a string. If None, problem_name='user_problem'

12) (Optional) (int) n_restarts, Number of starting points to optimize the acquisition function. Default is 10.

13) (Optional) (int) n_restarts_mean, Number of starting points to optimize the posterior mean. Default is 100.

14) (Optional) (int) n_best_restarts_mean,  Number of best starting points chosen from the n_restart points. Default is 10.

15) (Optional) (int) maxepoch: Maximum number of iterations of the SGD when optimizing the acquisition function. Default is 50.

16) (Optional) (int) maxepoch_mean: Maxepoch for the optimization of the posterior mean. Default is 50.

17) (Optional) (int) n_burning: Number of burnings samples for slice sampling. Default is 500.

18) (Optional) (int) thinning: Thinning parameter for slice sampling to obtain a sample of hyperparameters. Default is 50.

19) (Optional) (int) default_n_samples_parameters: Number of samples of Z for the discretization-free estimation of the BQO. Default is 100.

20) (Optional) (int) default_n_samples: Number of samples of hyperparameters to estimate BQO. Default is 20. 

* Run BQO:

21) sol = bgo(
    g, bounds_domain_x, integrand_function=f, bounds_domain_w=bounds_domain_w, type_bounds=type_bounds,
    name_method=name_method, n_iterations=n_iterations, random_seed=random_seed, n_training=n_training, 
    problem_name=problem_name, n_restarts=n_restarts, n_restarts_mean=n_restarts_mean, n_best_restarts_mean=n_best_restarts_mean,
    maxepoch=maxepoch, maxepoch_mean=maxepoch_mean, n_burning=n_burning, thinning=thinning, default_n_samples_parameters=default_n_samples_parameters, default_n_samples=default_n_samples)

22) The output sol is a dictionary. The entry 'optimal_solution' contains the solution given by the BO algorithm, and 'optimal_value' is the objective function evaluated at the 'optimal_solution'.
   
* Files generated:
 
 1) The training data is written in problems/problem_name/data
 2) The Guassian process model is written as a json file in data/gp_models/problem_name. The entry 'data' contains all the training data plus the points that have been chosen by the Bayesian optimization algorithm.
 3) The results of the algorithm are written in problems/problem_name/partial_results
