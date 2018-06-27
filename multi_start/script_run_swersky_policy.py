from __future__ import absolute_import

import numpy as np

from multi_start.problems.sgd_specific_function import *
from multi_start.parametric_stat_model_hutter import ParametricModel
from multi_start.stat_model_swersky import StatModelSwersky
from multi_start.greedy_policy_one_dimension_hutter import HutterGreedyPolicy
from multi_start.greedy_swersky import SwerskyGreedy

def get_values(i, data, method):
    data_ = data

    return {'point': data_['points'][i - 1], 'value': data_['values'][i - 1]}

def create_model(args, n_training=3, n_epochs=100, burning=True, point=None):

    rs = int(args['rs'])
    lb = [args['lb']]
    ub = [args['ub']]
    std = float(args['std'])
    lr = float(args['lr'])
    method = args['method']
    problem_name = args['problem_name']

    method_ = method

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

    # if method == 'real_gradient':
    #     data['gradients'] = [-1.0 * np.array(t) for t in data['gradients']]
    # elif method == 'grad_epoch':
    #     new_grads = {}
    #     for t in data['gradients']:
    #         new_grads[int(t)] = -1.0 * np.array(data['gradients'][t])
    #     data['gradients'] = new_grads

    data['stochastic_gradients'] = []

    data['values'] = [-1.0 * np.array(t) for t in data['values']]
    data['points'] = [np.array(t) for t in data['points']]



    training_data = {'points': data['points'][0:n_training],
                     'values': data['values'][0:n_training], 'gradients': [],
                     'stochastic_gradients': [] }
    # if method == 'real_gradient':
    #     training_data['gradients'] = data['gradients'][0:n_training]
    # elif method == 'grad_epoch':
    #     training_data['gradients'] = {}
    #     for j in range(n_training):
    #         if j in data['gradients']:
    #             training_data['gradients'][j] = data['gradients'][j]

    points_domain = data['points'][0: n_training]
    best_results = np.max(training_data['values'])
    functions_get_value = get_values
    kwargs = {'data': data, 'method': method}

    n_burning = 50


    n_batches = 1
    total_iterations = n_epochs * n_batches


    model = StatModelSwersky(
        training_data, best_results, n_training, functions_get_value,
        points_domain[-1], 0,
        n_training, problem_name=problem_name,
        max_iterations=total_iterations, lower=None, upper=None,
        n_burning=n_burning, total_batches=n_batches,
        thinning=10, kwargs_get_value_next_iteration=kwargs)



    return model

if __name__ == '__main__':
    # python -m multi_start.script_run_swersky_policy 1 20 0.1 0.1 100 problem5
    parser = argparse.ArgumentParser()
    parser.add_argument('rs', help=5)
    parser.add_argument('n_restarts', help=10)
    parser.add_argument('std', help=1.0)
    parser.add_argument('lr', default=1.0)
    parser.add_argument('n_epochs', default=20)
    parser.add_argument('problem_name', help='quadratic, problem6, problem5')

    args = parser.parse_args()

    random_seed = int(args.rs)
    n_restarts = int(args.n_restarts)
    std = float(args.std)
    lr = float(args.lr)
    n_epochs = int(args.n_epochs)
    problem_name = args.problem_name
    problem = problem_name

    lb = -1.0
    ub = 1.0

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

        lb = -10.0
        ub = 10.0
    elif problem == 'rastrigin':
        objective = rastrigin
        exact_gradient = exact_gradient_rastrigin

        def gradient(x, n_samples=1):
            return gradient_rastrigin(x, std, n_samples)

        def gradient_samples(z, m):
            return gradient_rastrigin(z, std, m)
    elif problem == 'problem6':
        objective = problem_6
        exact_gradient = exact_gradient_problem_6

        def gradient(x, n_samples=1):
            return gradient_problem_6(x, std, n_samples)

        def gradient_samples(z, m):
            return gradient_problem_6(z, std, m)

        lb = -10.0
        ub = 10.0

        bounds = [[-10.0, 10.0]]
    elif problem == 'problem5':
        objective = problem_5
        exact_gradient = exact_gradient_problem_5

        def gradient(x, n_samples=1):
            return gradient_problem_5(x, std, n_samples)

        def gradient_samples(z, m):
            return gradient_problem_5(z, std, m)

        lb = 0.0
        ub = 1.2

        bounds = [[0.0, 1.2]]

    np.random.seed(random_seed)

    start_points = list(np.random.uniform(lb, ub, (n_restarts, 1)))

    batch_size = 1

    for i in range(n_restarts):
        start = start_points[i]
        logger.info('start')
        logger.info(start)

        results = SGD(start, gradient, batch_size, objective, maxepoch=n_epochs, adam=False,
                      name_model='std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s_point_%d' % (
                      std, random_seed, lb, ub, lr, 'swersky', i),
                      exact_gradient=None, learning_rate=lr, n_epochs=5,
                      n_samples=100, gradient_samples=0, problem=problem,
                      bounds=bounds, exact_objective=objective)

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
        tmp_d['method'] = 'swersky'
        tmp_d['lr'] = lr
        tmp_d['problem_name'] = problem_name

        tmp_d['lb'] = lb
        tmp_d['ub'] = ub

        parameters[i] = tmp_d

    np.random.seed(random_seed)
    stat_models = {}
    points_index = range(n_restarts)

    print (n_restarts)
    for i in points_index:
        stat_models[i] = create_model(parameters[i], n_training=n_training, n_epochs=n_epochs, burning=False, point=i)

    policy_greedy = SwerskyGreedy(stat_models, 'swersky', problem_name, random_seed=random_seed, n_restarts=n_restarts)

    policy_greedy.run_policy(n_epochs - n_training)




