from __future__ import absolute_import

import numpy as np

from multi_start.problems.sgd_specific_function import *
from multi_start.stat_model_domain_several_dimensions import StatModelDomainMultiDimensional

from multi_start.greedy_policy_multivariate import GreedyPolicy
from multi_start.uniform_policy_multivariate import UniformPolicy
from multi_start.random_policy_multivariate import RandomPolicy


def create_model_multivariate(args, dimensions, n_training=3, n_epochs=100, burning=True, point=None):

    rs = int(args['rs'])
    lb = [float(t) for t in args['lb'] ]
    ub = [float(t) for t in args['ub']]
    std = float(args['std'])
    lr = float(args['lr'])
    method = args['method']
    problem_name = args['problem_name']

    #TODO: ADD THIS AS A PARAMETER
    lipschitz_cte = 2.0

    method_ = method
    if method == 'lipschitz' or method == 'approx_lipschitz':
        method_ = 'real_gradient'

    if point is None:
        name_model = 'std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s' % (std, rs, lb[0], ub[0], lr, method_)
    else:
        name_model = 'std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s_point_%d' % (std, rs, lb[0], ub[0], lr, method_, point)
    dir_data = 'data/multi_start/' + problem_name + '/' + 'training_results/'

    data = JSONFile.read(dir_data + name_model)

    if point is None:
        name_model = 'std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s' % (std, rs, lb[0], ub[0], lr, method)
    else:
        name_model = 'std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s_point_%d' % (std, rs, lb[0], ub[0], lr, method, point)

    if method == 'real_gradient':
        data['gradients'] = [-1.0 * np.array(t) for t in data['gradients']]
    elif method == 'grad_epoch':
        new_grads = {}
        for t in data['gradients']:
            new_grads[int(t)] = -1.0 * np.array(data['gradients'][t])
        data['gradients'] = new_grads

    data['stochastic_gradients'] = [-1.0 * np.array(t) for t in data['stochastic_gradients']]

    data['values'] = [-1.0 * np.array(t) for t in data['values']]
    data['exact_values'] = [-1.0 * np.array(t) for t in data['exact_values']]
    data['points'] = [np.array(t) for t in data['points']]



    training_data = {'points': data['points'][0:n_training],
                     'values': data['exact_values'][0:n_training], 'gradients': [],
                     'stochastic_gradients':data['stochastic_gradients'][0:n_training],
                     'exact_values': data['exact_values'][0:n_training]}
    if method == 'real_gradient':
        training_data['gradients'] = data['gradients'][0:n_training]
    elif method == 'grad_epoch':
        training_data['gradients'] = {}
        for j in range(n_training):
            if j in data['gradients']:
                training_data['gradients'][j] = data['gradients'][j]

    points_domain = data['points'][0: n_training]
    best_results = np.max(training_data['values'])
    functions_get_value = get_values
    kwargs = {'data': data, 'method': method}

    n_burning = 50


    n_batches = 1
    total_iterations = n_epochs * n_batches

    model = StatModelDomainMultiDimensional(
        training_data, best_results, n_training, functions_get_value,
        points_domain[-1], 0,
        n_training, specifications=name_model, problem_name=problem_name,
        max_iterations=total_iterations, parametric_mean=False, lower=None, upper=None,
        n_burning=n_burning, total_batches=n_batches, type_model=method, lipschitz=lipschitz_cte,
        n_thinning=10, kwargs_get_value_next_iteration=kwargs, burning=burning, dimensions=dimensions)

    return model


def get_values(i, data, method):
    data_ = data
    return {'point': data_['points'][i - 1], 'value': data_['values'][i - 1],
            'gradient': None, 'stochastic_gradient': data_['stochastic_gradients'][i-1],
            'exact_value': data_['exact_values'][i-1]}


if __name__ == '__main__':
    # python -m multi_start.script_run_policies_multivariate 1 20 0.1 0.01 100 rastrigin 2
    parser = argparse.ArgumentParser()
    parser.add_argument('rs', help=5)
    parser.add_argument('n_restarts', help=10)
    parser.add_argument('std', help=1.0)
    parser.add_argument('lr', default=1.0)
    parser.add_argument('n_epochs', default=20)
    parser.add_argument('problem_name', help='quadratic, problem6, problem5, rosenbrock')
    parser.add_argument('dimension', help='dimension of the domain')

    args = parser.parse_args()

    random_seed = int(args.rs)
    n_restarts = int(args.n_restarts)
    std = float(args.std)
    lr = float(args.lr)
    method = 'real_gradient'
    n_epochs = int(args.n_epochs)
    problem_name = args.problem_name
    problem = problem_name
    dimension = int(args.dimension)

    lb = [-1.0]
    ub = [1.0]

    bounds = None
    exact_objective = None

    if problem == 'quadratic':
        objective = objective_parabola
        exact_gradient = exact_gradient_parabola

        def gradient(x, n_samples=1):
            epsilon = np.random.normal(0, std, n_samples)
            return np.array(x) + np.mean(epsilon)

        def gradient_samples(z, m):
            epsilon = np.random.normal(0, std, m)
            return np.array(z) + np.mean(epsilon)

        lb = [-10.0]
        ub = [10.0]
    elif problem == 'rastrigin':
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

        lb = [-10.0]
        ub = [10.0]

        bounds = [[-10.0, 10.0]]
    elif problem == 'problem5':
        objective = problem_5
        exact_gradient = exact_gradient_problem_5

        def gradient(x, n_samples=1):
            return gradient_problem_5(x, std, n_samples)

        def gradient_samples(z, m):
            return gradient_problem_5(z, std, m)

        lb = [0.0]
        ub = [1.2]

        bounds = [[0.0, 1.2]]
    elif problem == 'rosenbrock':
        exact_objective = rosenbrock
        exact_gradient = exact_gradient_rosenbrock

        def objective(x):
            return rosenbrock_noisy(x, std)

        def gradient(x, n_samples=1):
            return gradient_rosenbrock(x, std, n_samples)

        def gradient_samples(z, m):
            return gradient_rosenbrock(z, std, m)

        bounds = dimension * [[-3.0, 3.0]]
        lb = dimension * [-3.0]
        ub = dimension * [3.0]

    np.random.seed(random_seed)

    start_points = np.random.uniform(lb, ub, (n_restarts, dimension))

    batch_size = 1

    method_ = method
    if method == 'approx_lipschitz':
        method_ = 'real_gradient'

    for i in range(n_restarts):
        start = start_points[i,:]
        logger.info('start')
        logger.info(start)

        results = SGD(start, gradient, batch_size, objective, maxepoch=n_epochs, adam=False,
                      name_model='std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s_point_%d' % (
                      std, random_seed, lb[0], ub[0], lr, method_, i),
                      exact_gradient=exact_gradient, learning_rate=lr, method=method, n_epochs=5,
                      n_samples=100, gradient_samples=gradient_samples, problem=problem,
                      bounds=bounds, exact_objective=exact_objective)
        logger.info('sol')
        logger.info(results['points'][-1])
        logger.info(results['values'][-1])

    ### run policies

    parameters = {}
    n_training = 3
    for i in range(n_restarts):
        j = i - n_training
        tmp_d = {}
        tmp_d['rs'] = random_seed
        tmp_d['std'] = std
        tmp_d['method'] = method
        tmp_d['lr'] = lr
        tmp_d['problem_name'] = problem_name

        tmp_d['lb'] = lb
        tmp_d['ub'] = ub

        parameters[i] = tmp_d

    np.random.seed(random_seed)
    stat_models = {}
    stat_models_2 = {}
    stat_models_3 = {}
    points_index = range(n_restarts)

    print (n_restarts)
    for i in points_index:
        stat_models[i] = create_model_multivariate(parameters[i],dimension, n_training=n_training, n_epochs=n_epochs, burning=False, point=i)
        stat_models_2[i] = create_model_multivariate(parameters[i],dimension, n_training=n_training, n_epochs=n_epochs, burning=False, point=i)
        stat_models_3[i] = create_model_multivariate(parameters[i],dimension, n_training=n_training, n_epochs=n_epochs, burning=False, point=i)

    policy_greedy = GreedyPolicy(stat_models, method, problem_name, type_model=method, random_seed=random_seed, n_restarts=n_restarts)
    policy_uniform = UniformPolicy(stat_models_2, method, problem_name, type_model=method, random_seed=random_seed, n_restarts=n_restarts)
    policy_random = RandomPolicy(stat_models_3, method, problem_name, type_model=method, random_seed=random_seed, n_restarts=n_restarts)

    policy_greedy.run_policy(n_epochs - n_training, sufix="train")
    policy_uniform.run_policy(n_epochs - n_training, sufix="train")
    policy_random.run_policy(n_epochs - n_training, sufix="train")