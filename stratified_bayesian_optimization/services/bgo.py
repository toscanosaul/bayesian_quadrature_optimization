import numpy as np

from stratified_bayesian_optimization.services.gp_fitting import GPFittingService
from stratified_bayesian_optimization.services.domain import (
    DomainService
)
from stratified_bayesian_optimization.entities.domain import (
    BoundsEntity,
)
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature
from stratified_bayesian_optimization.acquisition_functions.sbo import SBO
from stratified_bayesian_optimization.acquisition_functions.ei import EI
from stratified_bayesian_optimization.lib.constant import (
    SBO_METHOD,
    MULTI_TASK_METHOD,
    TASKS,
    DOGLEG,
    LBFGS_NAME,
    SGD_NAME,
    EI_METHOD,
    SDE_METHOD,
    SCALED_KERNEL,
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    UNIFORM_FINITE,
    GAMMA,
)
from stratified_bayesian_optimization.services.bayesian_global_optimization import BGO
from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)

_possible_optimization_methods = [SBO_METHOD, EI_METHOD]


def bgo(objective_function, bounds_domain_x, integrand_function=None, simplex_domain=None,
        noise=False, n_samples_noise=0, bounds_domain_w=None, type_bounds=None, distribution=None,
        parameters_distribution=None, name_method='bqo', n_iterations=50, type_kernel=None,
        dimensions_kernel=None, n_restarts=10, n_best_restarts=0, problem_name=None,
        n_training=None, random_seed=1, mle=False, n_samples_parameters=5, maxepoch=50, thinning=50,
        n_burning=500, max_steps_out=1000, parallel=True, same_correlation=True,
        monte_carlo_sbo=True, n_samples_mc=5, n_restarts_mc=5, n_best_restarts_mc=0, factr_mc=1e12,
        maxiter_mc=10, method_opt_mc=LBFGS_NAME, n_restarts_mean=100, n_best_restarts_mean=10,
        n_samples_parameters_mean=5, maxepoch_mean=50, parallel_training=False,
        default_n_samples_parameters=None, default_n_samples=None):
    """
    Maximizes the objective function.

    :param objective_function: function G to be maximized:
        If the function is noisy-free, G(([float])point) and returns [float].
        If the function is noisy, G(([float])point, (int) n_samples) and
            returns [(float) value, (float) variance]
    :param bounds_domain_x: [(float, float)]
    :param integrand_function: function F:
        If the function is noisy-free, F(([float])point) and returns [float].
        If the function is noisy, F(([float])point, (int) n_samples) and
            returns [(float) value, (float) variance]
    :param simplex_domain: (float) {sum[i, from 1 to domain]=simplex_domain}
    :param noise: (boolean) True if the evaluations of the objective function are noisy
    :param n_samples_noise: (int)  If noise is true, we take n_samples of the function to estimate
            its value.
    :param bounds_domain_w: [([float, float] or [float])], the first case is when the bounds
            are lower or upper bound of the respective entry; in the second case, it's list of
            finite points representing the domain of that entry (e.g. when W is finite).
    :param type_bounds: [0 or 1], 0 if the bounds are lower or upper bound of the respective
            entry, 1 if the bounds are all the finite options for that entry.
    :param distribution: str, probability distributions for the Bayesian quadrature (i.e. the
        distribution of W)
    :param parameters_distribution: {str: float}
    :param name_method: str, Options: 'SBO', 'EI'
    :param n_iterations: int
    :param type_kernel: [str] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL].
            If we want to use a scaled NAME_1_KERNEL, the parameter must be
            [SCALED_KERNEL, NAME_1_KERNEL].
    :param dimensions_kernel: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product, and
            the total dimension of the product_kernels_separable too in the first entry.
    :param n_restarts: (int) Number of starting points to optimize the acquisition function
    :param n_best_restarts:  (int) Number of best starting points chosen from the n_restart
            points.
    :param problem_name: str
    :param n_training: (int) number of training points
    :param random_seed: int
    :param mle: (boolean) If true, fits the GP by MLE. Otherwise, we use a fully Bayesian approach.
    :param n_samples_parameters: (int) Number of samples of the parameter to estimate the stochastic
        gradient when optimizing the acquisition function.
    :param maxepoch: (int) Maximum number of iterations of the SGD when optimizing the acquisition
        function.
    :param thinning: int
    :param n_burning: (int) Number of burnings samples for slice sampling.
    :param max_steps_out: (int)  Maximum number of steps out for the stepping out  or
        doubling procedure in slice sampling.
    :param parallel: (boolean)
    :param same_correlation: (boolean) If true, it uses the same correlations for the task kernel.
    :param monte_carlo_sbo: (boolean) If True, the code estimates the objective function and
        gradient with the discretization-free approach.
    :param n_samples_mc: (int) Number of samples for the MC method.
    :param n_restarts_mc: (int) Number of restarts to optimize the posterior mean given a sample of
        the normal random variable.
    :param n_best_restarts_mc:  (int) Number of best restarting points used to optimize the
        posterior mean given a sample of the normal random variable.
    :param factr_mc: (float) Parameter of LBFGS to optimize a sample of BQO when using the
        discretization-free approach.
    :param maxiter_mc: (int) Max number of iterations to optimize a sample of BQO when using the
        discretization-free approach.
    :param method_opt_mc: (str) Optimization method used when using the discretization-free approach
        of BQO.
    :param n_restarts_mean: (int) Number of starting points to optimize the posterior mean.
    :param n_best_restarts_mean: int
    :param n_samples_parameters_mean: (int) Number of sample of hyperparameters to estimate the
        stochastic gradient inside of the SGD when optimizing the posterior mean.
    :param maxepoch_mean: (int) Maxepoch for the optimization of the posterior mean.
    :param parallel_training: (boolean) Train in parallel if it's True.
    :param default_n_samples_parameters: (int) Number of samples of Z for the discretization-free
        estimation of the VOI.
    :param default_n_samples: (int) Number of samples of the hyperparameters to estimate the VOI.
    :return: {'optimal_solution': np.array(n),
            'optimal_value': float}
    """

    np.random.seed(random_seed)
    # default_parameters

    dim_x = len(bounds_domain_x)
    x_domain = range(dim_x)

    if name_method == 'bqo':
        name_method = SBO_METHOD

    dim_w = 0
    if name_method == SBO_METHOD:
        if type_bounds is None:
            dim_w = len(bounds_domain_w)
        elif type_bounds is not None and type_bounds[-1] == 1:
            dim_w = 1
        elif type_bounds is not None:
            dim_w = len(type_bounds[dim_x:])

    total_dim = dim_w + dim_x

    if type_bounds is None:
        type_bounds = total_dim * [0]

    if bounds_domain_w is None:
        bounds_domain_w = []
    bounds_domain = [list(bound) for bound in bounds_domain_x + bounds_domain_w]

    training_name = None

    if problem_name is None:
        problem_name = 'user_problem'

    if type_kernel is None:
        if name_method == SBO_METHOD:
            if type_bounds[-1] == 1:
                type_kernel = [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]
                dimensions_kernel = [total_dim, dim_x, len(bounds_domain[-1])]
            else:
                type_kernel = [SCALED_KERNEL, MATERN52_NAME]
                dimensions_kernel = [total_dim]
        elif name_method == EI_METHOD:
            type_kernel = [SCALED_KERNEL, MATERN52_NAME]
            dimensions_kernel = [total_dim]

    if dimensions_kernel is None:
        raise Exception("Not enough inputs to run the BGO algorithm")

    if n_training is None:
        if type_bounds[-1] == 1:
            n_training = len(bounds_domain[-1])
        else:
            n_training = 5

    if distribution is None:
        if type_bounds[-1] == 1:
            distribution = UNIFORM_FINITE
        else:
            distribution = GAMMA

    method_optimization = name_method

    name_model = 'gp_fitting_gaussian'

    if name_method == SBO_METHOD:
        training_function = integrand_function
    elif name_method == EI_METHOD:
        training_function = objective_function

    bounds_domain_x = BoundsEntity.to_bounds_entity(bounds_domain_x)

    spec = {
        'name_model': name_model,
        'problem_name': problem_name,
        'type_kernel': type_kernel,
        'dimensions': dimensions_kernel,
        'bounds_domain': bounds_domain,
        'type_bounds': type_bounds,
        'n_training': n_training,
        'noise': noise,
        'training_data': None,
        'points': None,
        'training_name': training_name,
        'mle': mle,
        'thinning': thinning,
        'n_burning': n_burning,
        'max_steps_out': max_steps_out,
        'n_samples': n_samples_noise,
        'random_seed': random_seed,
        'kernel_values': None,
        'mean_value': None,
        'var_noise_value': None,
        'cache': True,
        'same_correlation': same_correlation,
        'use_only_training_points': True,
        'optimization_method': method_optimization,
        'n_samples_parameters': n_samples_parameters,
        'parallel_training': parallel_training,
        'simplex_domain': simplex_domain,
        'objective_function': training_function,
        'dim_x': dim_x,
        'choose_noise': True,
        'bounds_domain_x': bounds_domain_x,
    }

    gp_model = GPFittingService.from_dict(spec)

    quadrature = None
    acquisition_function = None

    domain = DomainService.from_dict(spec)

    if method_optimization not in _possible_optimization_methods:
        raise Exception("Incorrect BGO method")

    if method_optimization == SBO_METHOD:
        quadrature = BayesianQuadrature(gp_model, x_domain, distribution,
                                        parameters_distribution=parameters_distribution)

        acquisition_function = SBO(quadrature, np.array(domain.discretization_domain_x))
    elif method_optimization == EI_METHOD:
        acquisition_function = EI(gp_model, noisy_evaluations=noise)

    bgo_obj = BGO(acquisition_function, gp_model, n_iterations, problem_name, training_name,
              random_seed, n_training, name_model, method_optimization, minimize=False,
              n_samples=n_samples_noise, noise=noise, quadrature=quadrature, parallel=parallel,
              number_points_each_dimension_debug=None, n_samples_parameters=n_samples_parameters,
              use_only_training_points=True, objective_function=objective_function,
              training_function=training_function)

    opt_params_mc = {}

    if factr_mc is not None:
        opt_params_mc['factr'] = factr_mc
    if maxiter_mc is not None:
        opt_params_mc['maxiter'] = maxiter_mc

    result = bgo_obj.optimize(debug=False, n_samples_mc=n_samples_mc, n_restarts_mc=n_restarts_mc,
                          n_best_restarts_mc=n_best_restarts_mc,
                          monte_carlo_sbo=monte_carlo_sbo, n_restarts=n_restarts,
                          n_best_restarts=n_best_restarts,
                          n_samples_parameters=n_samples_parameters,
                          n_restarts_mean=n_restarts_mean,
                          n_best_restarts_mean=n_best_restarts_mean,
                          random_seed=bgo_obj.random_seed, method_opt_mc=method_opt_mc,
                          n_samples_parameters_mean=n_samples_parameters_mean,
                          maxepoch_mean=maxepoch_mean,
                          maxepoch=maxepoch, threshold_sbo=0.001,
                          optimize_only_posterior_mean=False,
                          start_optimize_posterior_mean=0, optimize_mean_each_iteration=False,
                          default_n_samples_parameters=default_n_samples_parameters,
                          default_n_samples=default_n_samples, **opt_params_mc)

    return result

