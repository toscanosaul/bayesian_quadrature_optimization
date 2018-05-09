from __future__ import absolute_import

import numpy as np
import os
import argparse

from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.initializers.log import SBOLog
from multi_start.parametric_functions import ParametricFunctions

logger = SBOLog(__name__)

def SGD(start, gradient, n, function, exact_gradient=None, args=(), kwargs={}, bounds=None, learning_rate=0.1,
        momentum=0.0, maxepoch=250, adam=True, betas=None, eps=1e-8, simplex_domain=None,
        name_model='1', method='real_gradient', n_epochs=1, n_samples=100, gradient_samples=None,
        problem=None, exact_objective=None):
    """
    SGD to minimize sum(i=0 -> n) (1/n) * f(x). Batch sizes are of size 1.
    ADAM: https://arxiv.org/pdf/1412.6980.pdf
    :param start: np.array(n)
    :param gradient:
    :param n:
    :param learning_rate:
    :param momentum:
    :param maxepoch:
    :param args: () arguments for the gradient
    :param kwargs:
    :param bounds: [(min, max)] for each point
    :return: np.array(n)
    """
    values = []
    points = []

    gradients = []
    exact_values = []
    stochastic_gradients = []

    if method == 'grad_epoch':
        gradients = {}

    gradient_batch = []
    # points.append(np.array(start))
    # values.append(function(start))

    logger.info('start_value')
    logger.info(function(start))

    # if exact_gradient is not None and method == 'real_gradient':
    #     gradients.append(exact_gradient(start))

    project = False
    if bounds is not None or simplex_domain is not None:
        project = True

    if betas is None:
        betas = (0.9, 0.999)

    m0 = np.zeros(len(start))
    v0 = np.zeros(len(start))

    point = start
    v = np.zeros(len(start))
    times_out_boundary = 0
    t_ = 0

    lr = learning_rate

    for iteration in xrange(maxepoch):
        learning_rate = lr / float(iteration + 1)
        previous = point.copy()
        t_ += 1
        grad = []

        for j in xrange(n):
            gradient_ = gradient(point, *args, **kwargs)

            while gradient_ is np.nan:
                norm_point = np.sqrt(np.sum(point ** 2))
                perturbation = norm_point * 1e-6

                if project:
                    parameters_uniform = []
                    for i in range(len(bounds)):
                        bound = bounds[i]
                        dist = point[i] - bound[0]
                        lb = min(perturbation, dist)
                        dist = bound[1] - point[i]
                        ub = min(perturbation, dist)
                        parameters_uniform.append([-lb, ub])
                else:
                    parameters_uniform = len(point) * [[-perturbation, perturbation]]

                perturbation = []
                for i in range(len(point)):
                    lb = parameters_uniform[i][0]
                    ub = parameters_uniform[i][1]
                    perturbation.append(np.random.uniform(lb, ub))
                perturbation = np.array(perturbation)
                point = point + perturbation
                gradient_ = gradient(point, *args, **kwargs)
            grad.append(gradient_)
        gradient_ = np.mean(np.array(grad), axis=0)
        stochastic_gradients.append(gradient_)

        if exact_gradient is not None and method == 'real_gradient':
            gradients.append(exact_gradient(point))
        points.append(np.array(point))
        values.append(function(point))

        if exact_objective is not None:
            exact_values.append(exact_objective(point))

        if not adam:
            v = momentum * v + gradient_
            old_p = point.copy()
            point -= learning_rate * v
        else:
            m0 = betas[0] * m0 + (1 - betas[0]) * gradient_
            v0 = betas[1] * v0 + (1 - betas[1]) * (gradient_ ** 2)
            m_1 = m0 / (1 - (betas[0]) ** (t_))
            v_1 = v0 / (1 - (betas[1]) ** (t_))
            point = point - learning_rate * m_1 / (np.sqrt(v_1) + eps)



        in_domain = True
        if project:
            for dim, bound in enumerate(bounds):
                if bound[0] is not None and point[dim] < bound[0]:
                    in_domain = False
                    break
                if bound[1] is not None and point[dim] > bound[1]:
                    in_domain = False
                    break
                if simplex_domain is not None:
                    if np.sum(point) > simplex_domain:
                        in_domain = False
                        break
                    #TODO:Only for citibike, generalize later
                    if simplex_domain - np.sum(point) > 3717.0:
                        in_domain = False
                        break

        if project and not in_domain:
            for dim, bound in enumerate(bounds):
                if bound[0] is not None:
                    point[dim] = max(bound[0], point[dim])
                if bound[1] is not None:
                    point[dim] = min(bound[1], point[dim])
            if simplex_domain is not None:
                if np.sum(point) > simplex_domain:
                    point = simplex_domain * (point / np.sum(point))

                if simplex_domain - np.sum(point) > 3717.0:
                    point = (simplex_domain - 3717.0) * (point / np.sum(point))
            if not adam:
                for dim, bound in enumerate(bounds):
                    v[dim] = (point[dim] - old_p[dim]) / learning_rate

        # points.append(np.array(point))
        # values.append(function(point))

        #    gradients.append(np.array(gradient_))


    if exact_gradient is not None and method == 'real_gradient':
        gradients.append(exact_gradient(point))
    points.append(np.array(point))
    values.append(function(point))

    if exact_objective is not None:
        exact_values.append(exact_objective(point))

    gradient_ = np.array(gradient(point, *args, **kwargs))
    stochastic_gradients.append(gradient_)

    if method == 'grad_epoch':
        for iteration in range(maxepoch):
            if iteration % n_epochs == (n_epochs - 1):
                gradients[iteration] = gradient_samples(points[iteration], n_samples)

    results = {'points': points,
               'values': values,
               'gradients': gradients,
               'n_epochs': n_epochs,
               'stochastic_gradients': stochastic_gradients,
               'exact_values': exact_values}

    f_name = 'data/multi_start/analytic_example/training_results/'

    if not os.path.exists('data/multi_start'):
        os.mkdir('data/multi_start')
    if not os.path.exists('data/multi_start/' + problem):
        os.mkdir('data/multi_start/' + problem)
    f_name = 'data/multi_start/' + problem + '/'
    if not os.path.exists(f_name + 'training_results'):
        os.mkdir(f_name + 'training_results')
    f_name += 'training_results' + '/' + name_model


    JSONFile.write(results, f_name)

    return results

def objective_parabola(x):
    return 0.5 * (np.array(x) ** 2)

def exact_gradient_parabola(x):
    return np.array(x)

def gadient_parabola(x, std, m):
    epsilon = np.random.normal(0, std, m)
    return exact_gradient_parabola(x) + np.mean(epsilon)


def rastrigin(x):
    n = len(x)
    return 10. * n + np.sum((x**2) - 10.0 * np.cos(2.0 * np.pi * x))

def rastrigin_noisy(x, std):
    return rastrigin(x) + np.random.normal(0, std, 1)

def exact_gradient_rastrigin(x):
    n = len(x)
    gradient = np.zeros(n)
    for i in range(n):
        value = 2.0 * x[i] + 10.0 * np.sin(2.0 * np.pi * x[i]) * (2.0 * np.pi)
        gradient[i] = value
    return gradient

def gradient_rastrigin(x, std, m):
    n = len(x)
    epsilon = np.random.normal(0, std, (n, m))
    return exact_gradient_rastrigin(x) + np.mean(epsilon, axis=1)


def problem_6(x):
    return -(x + np.sin(x)) * np.exp(-x**2)

def exact_gradient_problem_6(x):
    return np.exp(-x**2) * (-2.0 * x) * (-1.0) * (x + np.sin(x)) + np.exp(-x**2) * (-1.0 - np.cos(x))

def gradient_problem_6(x, std, m):
    epsilon = np.random.normal(0, std, m)
    return exact_gradient_problem_6(x) + np.mean(epsilon)

def problem_5(x):
    return -(1.4 - 3.0*x) * np.sin(18.0 * x)

def exact_gradient_problem_5(x):
    return -1.4 * 18.0 * np.cos(18.0 * x) + (3.0 * np.sin(18.0 * x) + 3.0 * x * np.cos(18.0*x) * 18.0)

def gradient_problem_5(x, std, m):
    epsilon = np.random.normal(0, std, m)
    return exact_gradient_problem_5(x) + np.mean(epsilon)



if __name__ == '__main__':
    # python -m multi_start.problems.sgd_specific_function 123 1 100 1 10 1.0 10.0 real_gradient rastrigin 0 2
    parser = argparse.ArgumentParser()
    parser.add_argument('rs', help='5')
    parser.add_argument('batch_size', help='2')
    parser.add_argument('n_epochs', help=20)
    parser.add_argument('lb', help=-1)
    parser.add_argument('ub', help=1)
    parser.add_argument('std', help=1.0)
    parser.add_argument('learning_rate', default=1.0)
    parser.add_argument('method', help='real_gradient, grad_epoch, no_gradient')
    parser.add_argument('problem', help='rastrigin, quadratic, problem6')
    parser.add_argument('choose_sign_st')
    parser.add_argument('dimension', help='dimension of domain')


    args = parser.parse_args()

    random_seed = int(args.rs)
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    lb = [float(args.lb)]
    ub = [float(args.ub)]
    std = float(args.std)
    lr = float(args.learning_rate)
    method = args.method
    problem = args.problem
    choose_sign_st = bool(int(args.choose_sign_st))
    dimension = int(args.dimension)

    bounds = None


    if problem == 'quadratic':
        objective = objective_parabola
        exact_gradient = exact_gradient_parabola

        def gradient(x, n_samples=1):
            epsilon = np.random.normal(0, std, n_samples)
            return np.array(x) + np.mean(epsilon)

        def gradient_samples(z, m):
            epsilon = np.random.normal(0, std, m)
            return np.array(z) + np.mean(epsilon)
    elif problem == 'rastrigin':
        exact_gradient = exact_gradient_rastrigin

        exact_objective = rastrigin
        exact_gradient = exact_gradient_rastrigin

        def objective(x):
            return rastrigin_noisy(x, std)

        def gradient(x, n_samples=1):
            return gradient_rastrigin(x, std, n_samples)

        def gradient_samples(z, m):
            return gradient_rastrigin(z, std, m)

        bounds = dimension * [[-5.12, 5.12]]
        lb = dimension * [-5.12]
        ub = dimension * [5.12]

    elif problem == 'problem6':
        objective = problem_6
        exact_gradient = exact_gradient_problem_6

        def gradient(x, n_samples=1):
            return gradient_problem_6(x, std, n_samples)

        def gradient_samples(z, m):
            return gradient_problem_6(z, std, m)
    elif problem == 'problem5':
        objective = problem_5
        exact_gradient = exact_gradient_problem_5

        def gradient(x, n_samples=1):
            return gradient_problem_5(x, std, n_samples)

        def gradient_samples(z, m):
            return gradient_problem_5(z, std, m)


        bounds = [[0.0, 1.2]]


    np.random.seed(random_seed)
    start = np.zeros(dimension)
    for i in range(dimension):
        start[i] = np.random.uniform(lb[i], ub[i], 1)
    sign = np.random.binomial(1, 0.5)

    if choose_sign_st:
        if sign == 0:
            start = -1.0 * start

    logger.info('start')
    logger.info(start)



    results = SGD(start, gradient, batch_size, objective, maxepoch=n_epochs, adam=False,
                  name_model='std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s' % (std, random_seed, lb[0], ub[0], lr, method),
                  exact_gradient=exact_gradient, learning_rate=lr, method=method, n_epochs=5,
                  n_samples=100, gradient_samples=gradient_samples, problem=problem, bounds=bounds,
                  exact_objective=exact_objective)
    logger.info('sol')
    logger.info(results['points'][-1])
    logger.info(results['values'][-1])