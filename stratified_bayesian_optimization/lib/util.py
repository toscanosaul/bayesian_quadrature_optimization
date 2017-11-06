from __future__ import absolute_import

from collections import defaultdict

import numpy as np

from stratified_bayesian_optimization.lib.constant import(
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    LENGTH_SCALE_NAME,
    SIGMA2_NAME,
    LOWER_TRIANG_NAME,
    SCALED_KERNEL,
    SAME_CORRELATION,
    BAYESIAN_QUADRATURE,
)
from stratified_bayesian_optimization.lib.affine_break_points import (
    AffineBreakPointsPrep,
    AffineBreakPoints,
)
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.bayesian.bayesian_evaluations import BayesianEvaluations

logger = SBOLog(__name__)


def convert_dictionary_gradient_to_simple_dictionary(dictionary, order_keys):
    """

    :param dictionary: {
            (str) name: {(int): nxn} or nxn,
        }
    :param order_keys: ([(name_key, keys)]) (keys can be other list if we have a dictionary of
        dictionaries, otherwise keys=None)
    :return: {
        (int) : nxn
    } the indexes are set based on order_keys.
    """

    result = []

    for key in order_keys:
        if key[1] is None:
            result += [np.array(dictionary[key[0]])]
        else:
            for sub_key in key[1]:
                result += [np.array(dictionary[key[0]][sub_key[0]])]

    simple_dictionary = defaultdict()

    for index, element in enumerate(result):
        simple_dictionary[index] = element

    return simple_dictionary


def convert_dictionary_from_names_kernels_to_only_parameters(dictionary, order_kernels):
    """

    :param dictionary: {
            (str) kernel: {keys_kernel: value_kernel},
        }
    :param order_kernels: [str]
    :return: {
        {keys_kernel + 'kernel_name': value_kernel}
    } the indexes are set based on order_keys.
    """
    result = {}
    for kernel in order_kernels:
        for parameter in dictionary[kernel]:
            result[parameter] = dictionary[kernel][parameter]
    return result


def separate_numpy_arrays_in_lists(array, division):
    """
    Separate the m-axis of the array in a list such that [array(nxdivision), array(nx(division:))]
    :param array: np.array(nxm) or np.array(n)
    :param division: (int)
    :return: [np.array(nxdivision), np.array(nx(division:))]
    """
    if len(array.shape) == 2:
        return [array[:, 0: division], array[:, division: array.shape[1]]]
    else:
        return [array[0: division], array[division: len(array)]]


def wrapper_fit_gp_regression(self, **kwargs):
    """
    Wrapper of fit_gp_regression
    :param self: instance of class GPFittingGaussian
    :param kwargs:
        - 'start': (np.array(n)) starting point of the optimization of the llh.
        - 'random_seed': int
    :return: updated self
    """

    return self.fit_gp_regression(**kwargs)


def wrapper_evaluate_objective_function(point, cls_, name_module, n_samples):
    """
    Wrapper of evaluate_function in training_data
    :param cls: TrainingDataService
    :param name_module: (str) Name of the module of the problem
    :param point: [float]
    :param n_samples: int. If noise is true, we take n_samples of the function to estimate its
        value.
    :return: float
    """

    module = __import__(name_module, globals(), locals(), -1)

    return cls_.evaluate_function(module, point, n_samples)


def get_number_parameters_kernel(kernel_name, dim, **kernel_parameters):
    """
    Returns the number of parameters associated to the kernel.

    :param kernel_name: [str]
    :param dim: [int], for standard kernels the list consists of only one element. For the product
        of kernels the lists consists of the dimension of the product of kernels, and each of the
        kernels of the product.
    :param kernel_parameters: additional kernel parameters,
        - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.
    :return: int
    """

    if kernel_name[0] == MATERN52_NAME:
        return dim[0]

    if kernel_name[0] == TASKS_KERNEL_NAME:
        same_correlation = kernel_parameters.get(SAME_CORRELATION, False)

        if not same_correlation:
            return np.cumsum(xrange(dim[0] + 1))[dim[0]]
        else:
            return min(dim[0], 2)

    if kernel_name[0] == PRODUCT_KERNELS_SEPARABLE:
        n_params = 0
        for name, dimension in zip(kernel_name[1:], dim[1:]):
            n_params += get_number_parameters_kernel([name], [dimension], **kernel_parameters)
        return n_params

    raise NameError(kernel_name[0] + " doesn't exist")


def get_default_values_kernel(kernel_name, dim, same_correlation=False, **parameters_priors):
    """
    Returns default values for the kernel_name.

    :param kernel_name: [str]
    :param dim: [int]
    :param same_correlation: (boolean) Parameter for the task kernel
    :param parameters_priors:
            -SIGMA2_NAME: float,
            -LENGTH_SCALE_NAME: [float],
            -LOWER_TRIANG_NAME: [float],
    :return: [float]
    """

    if kernel_name[0] == SCALED_KERNEL:
        sigma2 = [parameters_priors.get(SIGMA2_NAME, 1.0)]
        if kernel_name[1] == MATERN52_NAME:
            ls = parameters_priors.get(LENGTH_SCALE_NAME, dim[0] * [1.0])
            return ls + sigma2

    if kernel_name[0] == MATERN52_NAME:
        ls = parameters_priors.get(LENGTH_SCALE_NAME, dim[0] * [1.0])
        return ls

    if kernel_name[0] == TASKS_KERNEL_NAME:
        n_params = get_number_parameters_kernel(kernel_name, dim,
                                                **{SAME_CORRELATION: same_correlation})
        tasks_kernel_chol = parameters_priors.get(LOWER_TRIANG_NAME, n_params * [0.0])
        return tasks_kernel_chol

    if kernel_name[0] == PRODUCT_KERNELS_SEPARABLE:
        values = []

        for name, dimension in zip(kernel_name[1:], dim[1:]):
            values += get_default_values_kernel([name], [dimension], **parameters_priors)

        return values


def convert_list_to_dictionary(ls):
    """

    :param ls: [object]
    :return: {i: ls[i]}
    """

    d = {}
    for index, value in enumerate(ls):
        d[index] = value

    return d


def convert_dictionary_to_list(dictionary):
    """
    Return list where list[i] = b iff dictionary[i] = b.

    :param dictionary: {(int) i: object}
    :return: [object]
    """
    ls = [None] * len(dictionary)

    for key, value in dictionary.iteritems():
        ls[key] = value

    return ls


def expand_dimension_vector(x, change_indexes, default_x):
    """
    Expand the dimension of x, where the new_x[i] = x[index] iff i is in
    change_indexes, otherwise new_x[i] = default_x[index].

    :param x: np.array(n)
    :param change_indexes: [int], where its length is less than n.
    :param default_x: np.array(m)

    :return: np.array(m)
    """

    index = 0
    new_x = default_x.copy()
    for j in change_indexes:
        new_x[j] = x[index]
        index += 1
    return new_x


def reduce_dimension_vector(x, change_indexes):
    """
    Reduce the dimension of the vector, where the entries in change_indexes are conserved.

    :param x: np.array(n)
    :param change_indexes: [int]

    :return: np.array(m)
    """

    new_x = np.zeros(len(change_indexes))
    index = 0
    for j in change_indexes:
        new_x[index] = x[j]
        index += 1
    return new_x


def combine_vectors(vector1, vector2, indexes1):
    """
    Combine vector1 and vector2 into one vector v where
    v[indexes1[i]] = vector1[i], and
    v[indexes2[i]] = vector2[i] where indexes2 = np.arange(dimension) - indexes1

    :param vector1: np.array(n1)
    :param vector2: np.array(n2)
    :param indexes1: [int]

    :return: np.array(n1+n2)
    """

    dimension = len(vector1) + len(vector2)
    vector = np.zeros(dimension)

    for index, index1 in enumerate(indexes1):
        vector[index1] = vector1[index]

    indexes2 = [i for i in list(np.arange(dimension)) if i not in indexes1]
    for index, index2 in enumerate(indexes2):
        vector[index2] = vector2[index]

    return vector


def separate_vector(vector, indexes1):
    """
    Separate vector into two vectors vector1, vector2 where
    vector1 = vector[indexes1],
    vector2 = vector[indexes1^c]

    :param vector: np.array(n)
    :param indexes1: [int]
    :return: [np.array(n1), np.array(n2)] where n = n1 + n2
    """
    dimension = len(vector)

    vector1 = vector[indexes1]

    indexes2 = [i for i in list(np.arange(dimension)) if i not in indexes1]
    vector2 = vector[indexes2]

    return [vector1, vector2]


def wrapper_optimization(start, *args):
    """
    Wrapper of optimization.optimize


    :param start: np.array(n), starting point of the optimization
    :param args: args[0] is an optimization instance, and args[1:] are arrguments to pass to
        the objective function and its gradient.

    :return: {
        'solution': np.array(n),
        'optimal_value': float,
        'gradient': np.array(n),
        'warnflag': int,
        'task': str
    }
    """

    return args[0].optimize(start, *args[1:])


def wrapper_objective_voi(point, self, monte_carlo=False, n_samples=1, n_restarts=1,
                          n_best_restarts=0, opt_params_mc=None, n_threads=0,
                          n_samples_parameters=0, method_opt_mc=None):
    """
    Wrapper of objective_voi
    :param self: instance of the acquisition function
    :param point: np.array(n)
    :param monte_carlo: (boolean) If True, estimates the function by MC.
    :param n_samples: (int) Number of samples for the MC method.
    :param n_restarts: (int) Number of restarts to optimize a_{n+1} given a sample.
    :param n_best_restarts: (int)
    :param opt_params_mc: {
        -'factr': int
        -'maxiter': int
    }
    :param n_threads: (int)
    :param n_samples_parameters: (int)

    :return: float
    """

    if opt_params_mc is None:
        opt_params_mc = {}

    if n_samples_parameters == 0:
        value = self.objective_voi(point, monte_carlo=monte_carlo, n_samples=n_samples,
                                   n_restarts=n_restarts, n_best_restarts=n_best_restarts,
                                   n_threads=n_threads, method_opt=method_opt_mc, **opt_params_mc)
    else:
        value = self.objective_voi_bayesian(
            point, monte_carlo=monte_carlo, n_samples_parameters=n_samples_parameters,
            n_samples=n_samples, n_restarts=n_restarts, n_best_restarts=n_best_restarts,
            n_threads=n_threads, compute_max_mean=False, method_opt=method_opt_mc, **opt_params_mc)

    return value

def wrapper_gradient_voi(point, self, monte_carlo=False, n_samples=1, n_restarts=1,
                         n_best_restarts=0, opt_params_mc=None, n_threads=0,
                         n_samples_parameters=0, method_opt_mc=None):
    """
    Wrapper of objective_voi (an acquisition function)
    :param self: instance of the acquisition function
    :param point: np.array(n)
    :param monte_carlo: (boolean) If True, estimates the function by MC.
    :param n_samples: (int) Number of samples for the MC method.
    :param n_restarts: (int) Number of restarts to optimize a_{n+1} given a sample.
    :param n_best_restarts: (int)
    :param opt_params_mc:{
        -'factr': int
        -'maxiter': int
    }
    :param n_threads: (int)
    :param n_samples_parameters: (int)
    :return: np.array(n)
    """

    if opt_params_mc is None:
        opt_params_mc = {}

    if n_samples_parameters == 0:
        value = self.grad_obj_voi(point, monte_carlo=monte_carlo, n_samples=n_samples,
                                  n_restarts=n_restarts, n_best_restarts=n_best_restarts,
                                  n_threads=n_threads, method_opt=method_opt_mc, **opt_params_mc)
    else:
        value = self.grad_obj_voi_bayesian(
            point, monte_carlo, n_samples_parameters, n_samples, n_restarts,
            n_best_restarts, n_threads, compute_max_mean=False, method_opt=method_opt_mc,
            **opt_params_mc)

    return value

def wrapper_evaluate_quadrature_cross_cov(point, historical_points, parameters_kernel, self):
    """
    Wrapper of evaluate quadrature cross cov

    :param point: np.array(1xn)
    :param historical_points: np.array(mxk)
    :param parameters_kernel: np.array(l)
    :param self: instance of bayesian quadrature
    :return: np.array(1xm)
    """

    return self.evaluate_quadrature_cross_cov(
                point, historical_points, parameters_kernel)


def wrapper_compute_vector_b(point, compute_vec_covs, compute_b_new, historical_points,
                             parameters_kernel, candidate_point, self):
    """
    Wrapper of the function that computes B(x, i) and B(new, i)

    :param point: np.array(1xn)
    :param compute_vec_covs: boolean
    :param compute_b_new: boolean
    :param historical_points: np.array(kxm)
    :param parameters_kernel: np.array(l)
    :param candidate_point: np.array(1xm)
    :param self: bq instance
    :return: {
        'b_new': float,
        'vec_covs': np.array(k),
    }
    """
    b_new = None
    vec_covs = None

    if compute_vec_covs:
        vec_covs = self.evaluate_quadrature_cross_cov(
            point, historical_points, parameters_kernel)
    if compute_b_new:
        b_new = self.evaluate_quadrature_cross_cov(
            point, candidate_point, parameters_kernel)

    return {
        'b_new': b_new,
        'vec_covs': vec_covs,
    }

def wrapper_evaluate_sbo_mc(candidate_points, task, self, n_samples, n_restarts):
    """

    :param candidate_points: np.array(rxn)
    :param task: (int)
    :param self: sbo instance
    :param n_samples: (int) Number of samples for the MC method.
    :param n_restarts: (int) Number of restarts to optimize a_{n+1} given a sample.

    :return: np.array(r)
    """
    tasks =  candidate_points.shape[0] * [task]
    tasks = np.array(tasks).reshape((len(tasks), 1))

    candidate_points = np.concatenate((candidate_points, tasks), axis=1)


    r = candidate_points.shape[0]

    values = np.zeros(r)

    points = {}
    for i in xrange(r):
        points[i] = candidate_points[i, :]

    args = (False, None, False, 0, self, True, n_samples, n_restarts)
    val = Parallel.run_function_different_arguments_parallel(
        wrapper_objective_voi, points, *args)

    for i in xrange(r):
        if val.get(i) is None:
            logger.info("Computation of VOI failed for new_point %d" % i)
            continue
        values[i] = val[i]

    return values


def wrapper_evaluate_sbo(candidate_points, task, self):
    """

    :param candidate_points: np.array(rxn)
    :param task: (int)
    :param self: sbo instance
    :return: np.array(r)
    """
    tasks =  candidate_points.shape[0] * [task]
    tasks = np.array(tasks).reshape((len(tasks), 1))

    candidate_points = np.concatenate((candidate_points, tasks), axis=1)

    vectors = self.bq.compute_posterior_parameters_kg_many_cp(
        self.discretization, candidate_points
    )

    a = vectors['a']
    b = vectors['b']

    r = candidate_points.shape[0]

    values = np.zeros(r)


    b_vectors = {}
    for i in xrange(r):
        b_vectors[i] = b[:, i]

    args = (False, None, True, 0, a, self,)
    val = Parallel.run_function_different_arguments_parallel(
        wrapper_hvoi, b_vectors, *args)

    for i in xrange(r):
        if val.get(i) is None:
            logger.info("Computation of VOI failed for new_point %d" % i)
            continue
        values[i] = val[i]

    return values


def wrapper_hvoi(b, a, self):
    """

    :param b:
    :param a:
    :param self:
    :return:
    """

    if not np.all(np.isfinite(b)):
        return 0.0

    a, b, keep = AffineBreakPointsPrep(a, b)

    keep1, c = AffineBreakPoints(a, b)
    keep1 = keep1.astype(np.int64)

    return self.hvoi(b, c, keep1)

def wrapper_GPFittingGaussian(training_data_sets, model, type_kernel, dimensions, bounds_domain,
                              thinning, n_burning, max_steps_out, random_seed, problem_name,
                              training_name, **kernel_parameters):
    """

    :param training_data_sets:
    :param type_kernel:
    :param dimensions:
    :param bounds_domain:
    :param thinning:
    :param n_burning:
    :param max_steps_out:
    :param random_seed:
    :param problem_name:
    :param training_name:
    :param kernel_parameters:
    :return: GP-model instance.
    """

    gp = model(type_kernel, training_data_sets, dimensions=dimensions, bounds_domain=bounds_domain,
               thinning=thinning, n_burning=n_burning, max_steps_out=max_steps_out,
               random_seed=random_seed, problem_name=problem_name, training_name=training_name,
               **kernel_parameters)

    return gp

def wrapper_evaluate_sbo_by_sample(start_sample, self, candidate_point, var_noise, mean,
                                   parameters_kernel, n_threads, method_opt, **opt_params_mc):
    """

    :param start_sample: [np.array(n), float], the first element is the starting point, and the
        second element is the sampled element from the Gaussian r.v.
    :param self: sbo-instance
    :param candidate_point: np.array(1xm)
    :param var_noise: float
    :param mean: float
    :param parameters_kernel: np.array(l)
    :param opt_params_mc:
        -'factr': int
        -'maxiter': int
    :param n_threads: int
    :return: {'max': float, 'optimum': np.array(n)}
    """

    return self.evaluate_sbo_by_sample(
        candidate_point, start_sample[1], start=start_sample[0], var_noise=var_noise, mean=mean,
        parameters_kernel=parameters_kernel, n_restarts=0, parallel=False, n_threads=n_threads,
        method_opt=method_opt, tol=None, **opt_params_mc)

def wrapper_evaluate_sbo_by_sample_2(start, self, sample, candidate_point, var_noise, mean,
                                   parameters_kernel, n_threads, method_opt, **opt_params_mc):
    """

    :param start: np.array(n)
    :param self: sbo-instance
    :param sample: float
    :param candidate_point: np.array(1xm)
    :param var_noise: float
    :param mean: float
    :param parameters_kernel: np.array(l)
    :param opt_params_mc:
        -'factr': int
        -'maxiter': int
    :param n_threads: int
    :return: {'max': float, 'optimum': np.array(n)}
    """

    return self.evaluate_sbo_by_sample(
        candidate_point, sample, start=start, var_noise=var_noise, mean=mean,
        parameters_kernel=parameters_kernel, n_restarts=0, parallel=False, n_threads=n_threads,
        method_opt=method_opt, tol=None, **opt_params_mc)

def wrapper_evaluate_sbo_by_sample_no_sp(
        sample_candid_parameters, self, n_threads, method_opt, n_restarts,
        **opt_params_mc):

    sample = sample_candid_parameters[1]
    candidate_point = sample_candid_parameters[0]
    params = sample_candid_parameters[2]

    return self.evaluate_sbo_by_sample(
        candidate_point, sample, start=None, var_noise=params[0], mean=params[1],
        parameters_kernel=params[2:], n_restarts=n_restarts, parallel=False, n_threads=n_threads,
        method_opt=method_opt, tol=None, **opt_params_mc)


def wrapper_evaluate_sbo_by_sample_bayesian(start_sample_parameters, self, candidate_point,
                                            n_threads, method_opt, **opt_params_mc):
    """

    :param start_sample_parameters: [np.array(n), float, np.array(l)], the first element is the
        starting point, and the second element is the sampled element from the Gaussian r.v.
    :param self: sbo-instance
    :param candidate_point: np.array(1xm)
    :param var_noise: float
    :param mean: float
    :param parameters_kernel: np.array(l)
    :param opt_params_mc:
        -'factr': int
        -'maxiter': int
    :param n_threads: int
    :param method_opt: str
    :return: {'max': float, 'optimum': np.array(n)}
    """
    sample = start_sample_parameters[1]
    start = start_sample_parameters[0]
    params = start_sample_parameters[2]

    return self.evaluate_sbo_by_sample(
        candidate_point, sample, start=start, var_noise=params[0], mean=params[1],
        parameters_kernel=params[2:], n_restarts=0, parallel=False, n_threads=n_threads,
        method_opt=method_opt, tol=None, **opt_params_mc)

def wrapper_evaluate_sbo_by_sample_bayesian_2(start_sample_parameters_candidate, self, n_threads,
                                            method_opt, **opt_params_mc):
    """

    :param start_sample_parameters: [np.array(n), float, np.array(l)], the first element is the
        starting point, and the second element is the sampled element from the Gaussian r.v.
    :param self: sbo-instance
    :param candidate_point: np.array(1xm)
    :param var_noise: float
    :param mean: float
    :param parameters_kernel: np.array(l)
    :param opt_params_mc:
        -'factr': int
        -'maxiter': int
    :param n_threads: int
    :return: {'max': float, 'optimum': np.array(n)}
    """
    sample = start_sample_parameters_candidate[2]
    candidate_point = start_sample_parameters_candidate[1]
    start = start_sample_parameters_candidate[0]
    params = start_sample_parameters_candidate[3]

    return self.evaluate_sbo_by_sample(
        candidate_point, sample, start=start, var_noise=params[0], mean=params[1],
        parameters_kernel=params[2:], n_restarts=0, parallel=False, n_threads=n_threads,
        method_opt=method_opt, tol=None, **opt_params_mc)

def wrapper_evaluate_sample(point, self, *args):
    """

    :param point:
    :param self: sbo instance
    :param args:
    :return:
    """
    if type(point) == list:
        val = self.evaluate_sample(point[0], point[1], point[2], *args)
    else:
        val = self.evaluate_sample(point, *args)

    return val

def wrapper_evaluate_sample_bayesian(point, self):
    point_ = point[0]
    candidate_point = point[1]
    sample = point[2]
    params = point[3]

    val = self.evaluate_sample(point_, candidate_point, sample, params[0], params[1], params[2:],
                               True, 0, False)
    return val

def wrapper_evaluate_gradient_sample(point, self, *args):
    """

    :param point:
    :param self: sbo instance
    :param args:
    :return:
    """
    return self.evaluate_gradient_sample(point, *args)

def wrapper_evaluate_hessian_sample(point, self, *args):
    """

    :param point:
    :param self: sbo instance
    :param args:
    :return:
    """
    return self.evaluate_hessian_sample(point, *args)


def wrapper_optimize(point, self, *args):
    """
    Wrapper of optimization.optimize
    :param point: starting point
    :param self: optimization instance
    :param args: additional arguments for the objective and gradient functions
    :return: optimize results
    """

    return self.optimize(point, *args)

def wrapper_sgd(point_rs, self, *args, **kwargs):
    point = point_rs[0]
    rs = point_rs[1]

    np.random.seed(rs)

    return self.SGD(point, *args, **kwargs)

def wrapper_objective_posterior_mean_bq(point, self, var_noise=None, mean=None,
                                        parameters_kernel=None, n_samples_parameters=0):
    """
    Wrapper of the objective posterior mean of a bq model
    :param point: np.array(k)
    :param self: bayesian-quadrature instance
    :param n_samples_parameters: int
    :return: float
    """
    if n_samples_parameters == 0:
        val = self.objective_posterior_mean(point, var_noise=var_noise, mean=mean,
                                            parameters_kernel=parameters_kernel)
    else:
        val = BayesianEvaluations.evaluate(self.objective_posterior_mean, point, self.gp,
                                           n_samples_parameters, None)[0]

    return val

def wrapper_grad_posterior_mean_bq(point, self, var_noise=None, mean=None, parameters_kernel=None,
                                   n_samples_parameters=0):
    """
    Wrapper of the gradient of the posterior mean of a bq model
    :param point: np.array(k)
    :param self: bayesian-quadrature instance
    :param n_samples_parameters: int
    :return: np.array(k)
    """

    if n_samples_parameters == 0:
        val = self.grad_posterior_mean(point, var_noise=var_noise, mean=mean,
                                       parameters_kernel=parameters_kernel)
    else:
        val = BayesianEvaluations.evaluate(self.grad_posterior_mean, point, self.gp,
                                           n_samples_parameters, None)[0]

    return val

def wrapper_hessian_posterior_mean_bq(
        point, self, var_noise=None, mean=None, parameters_kernel=None, n_samples_parameters=0):
    """
    ONLY FOR n_samples_parameters = 0
    :param point: np.array(k)
    :param self:
    :param var_noise:
    :param mean:
    :param parameters_kernel:
    :param n_samples_parameters:
    :return:
    """
    point = point.reshape((1, len(point)))
    val = self.hessian_posterior_mean(point, var_noise, mean, parameters_kernel)

    return val

def wrapper_objective_acquisition_function(point, self, n_samples_parameters=0, *params):
    """
    Wrapper of an acquisition function that's not SBO or KG.

    :param point: np.array(n)
    :param self: acquisition function instance
    :param n_samples_parameters: int
    :param params: additional parameters of the function
    :return: float
    """

    point = point.reshape((1, len(point)))

    if n_samples_parameters == 0:
        value = self.evaluate(point, *params)
    else:
        if self.gp.name_model == BAYESIAN_QUADRATURE:
            gp_model = self.gp.gp
        else:
            gp_model = self.gp

        value = BayesianEvaluations.evaluate(self.evaluate, point, gp_model, n_samples_parameters,
                                             None, *params)[0]

    return value

def wrapper_posterior_mean_gp_model(point, self, n_samples_parameters=0, *params):
    point = point.reshape((1, len(point)))

    if n_samples_parameters == 0:
        value = self.compute_posterior_parameters(point, *params, only_mean=True)['mean']
    else:
        def evaluate(point, var_noise=None, mean=None, parameters_kernel=None):
            return self.compute_posterior_parameters(
                point, var_noise, mean, parameters_kernel, only_mean=True)['mean']
        value = BayesianEvaluations.evaluate(evaluate, point, self, n_samples_parameters,
                                             None, *params)[0]
    return value

def wrapper_gradient_posterior_mean_gp_model(point, self, n_samples_parameters=0, *params):
    point = point.reshape((1, len(point)))
    if n_samples_parameters == 0:
        value = self.gradient_posterior_parameters(point, *params, only_mean=True)['mean']
    else:
        def evaluate(point, var_noise=None, mean=None, parameters_kernel=None):
            return self.gradient_posterior_parameters(
                point, var_noise, mean, parameters_kernel, only_mean=True)['mean']
        value = BayesianEvaluations.evaluate(evaluate, point, self, n_samples_parameters,
                                             None, *params)[0]
    return value

def wrapper_gradient_acquisition_function(point, self, n_samples_parameters=0, *params):
    """
    Wrapper of the gradient of an acquisition function that's not SBO or KG.

    :param point: np.array(n)
    :param self: acquisition function instance
    :param n_samples_parameters: int
    :param params: additional parameters of the function
    :return: float
    """
    point = point.reshape((1, len(point)))

    if n_samples_parameters == 0:
        value = self.evaluate_gradient(point, *params)
    else:
        if self.gp.name_model == BAYESIAN_QUADRATURE:
            gp_model = self.gp.gp
        else:
            gp_model = self.gp

        value = BayesianEvaluations.evaluate(self.evaluate_gradient, point, gp_model,
                                             n_samples_parameters, None, *params)[0]

    return value

def wrapper_get_parameters_for_samples(parameters, point, self, *args):
    return self.bq.get_parameters_for_samples(True, point, parameters[0], parameters[1],
                                              parameters[2], clear_cache=False)

def wrapper_get_parameters_for_samples_2(parameters, self, *args):
    return self.bq.get_parameters_for_samples(True, parameters[3], parameters[0], parameters[1],
                                              parameters[2], clear_cache=False)

def wrapper_grad_voi_sgd(point, self, *args, **opt_params_mc):

    return self.grad_voi_sgd(point, *args,  **opt_params_mc)

def wrapper_optimize_posterior_mean(parameter, self, random_seed, method_opt, n_restarts):
    opt_result = self.optimize_posterior_mean(
        random_seed=random_seed, n_treads=0, var_noise=parameter[0],
        parameters_kernel=parameter[2:], mean=parameter[1], n_restarts=n_restarts,
        method_opt=method_opt, parallel=False, n_best_restarts=0)
    return opt_result

def wrapper_evaluate_gradient_ei_sample_params(point, self):

    return self.evaluate_gradient_sample_params(point)

def wrapper_evaluate_gradient_sample_params_bq(point, self):
    return self.evaluate_gradient_sample_params(point)

def wrapper_evaluate_gradient_sample_params_gp(point, self):
    return self.evaluate_gradient_sample_params(point)
