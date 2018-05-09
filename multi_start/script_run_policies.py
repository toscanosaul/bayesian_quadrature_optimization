from __future__ import absolute_import


from stratified_bayesian_optimization.util.json_file import JSONFile
import numpy as np
from multi_start.gredy_policy import GreedyPolicy
from multi_start.uniform_policy import UniformPolicy

import argparse

from multi_start.stat_model_domain import StatModel
from multi_start.stat_model_domain_lipschitz import StatModelLipschitz


def create_model(args, n_training=3, n_epochs=100, burning=True, point=None):

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
    data['points'] = [np.array(t) for t in data['points']]



    training_data = {'points': data['points'][0:n_training],
                     'values': data['values'][0:n_training], 'gradients': [],
                     'stochastic_gradients':data['stochastic_gradients'][0:n_training] }
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

    if method == 'approx_lipschitz' or method == 'lipschitz':
        if method == 'approx_lipschitz':
            lipschitz_cte = None
        model = StatModelLipschitz(
            training_data, best_results, n_training, functions_get_value,
            points_domain[-1], 0,
            n_training, specifications=name_model,problem_name=problem_name,
            max_iterations=total_iterations, parametric_mean=False, lower=None, upper=None,
            n_burning=n_burning, total_batches=n_batches, type_model=method, lipschitz=lipschitz_cte,
            n_thinning=10, kwargs_get_value_next_iteration=kwargs, burning=burning)
    else:
        model = StatModel(
            training_data, best_results, n_training, functions_get_value,
            points_domain[-1], 0,
            n_training, specifications=name_model, problem_name=problem_name,
            max_iterations=total_iterations, parametric_mean=False, lower=None, upper=None,
            n_burning=n_burning, total_batches=n_batches,model_gradient=method,
            n_thinning=10, kwargs_get_value_next_iteration=kwargs, burning=burning)
    return model




def get_values(i, data, method):
    data_ = data
    if method == 'real_gradient':
        return {'point': data_['points'][i - 1], 'value': data_['values'][i - 1],
                'gradient': data_['gradients'][i - 1]}
    elif method == 'grad_epoch':
        if i - 1 in data_['gradients']:
            grad = data_['gradients'][i-1]
        else:
            grad = None
        return {'point': data_['points'][i - 1], 'value': data_['values'][i - 1],
                'gradient': grad}
    else:
        return {'point': data_['points'][i - 1], 'value': data_['values'][i - 1],
                'gradient': None}


bounds = {}
bounds['problem6'] = {}
bounds['problem6']['lb'] =[0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0, -2.5]
bounds['problem6']['ub'] =[0.5, 1.0, 1.5, 2.0, 2.5, 0.0, -0.5, -1.0, -1.5,-2.0]

bounds['problem5'] = {}
bounds['problem5']['lb'] =[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0]
bounds['problem5']['ub'] =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.1]




if __name__ == '__main__':
    # python -m multi_start.script_run_policies real_gradient uniform problem6 10 100 10.0 0.1 10
    parser = argparse.ArgumentParser()
    parser.add_argument('method', help='approx_lipschitz')
    parser.add_argument('policy', help='uniform, greedy', default='greedy')
    parser.add_argument('problem_name', help='analytic_example')
    parser.add_argument('n_starting_points', help=10)
    parser.add_argument('rs', help=100)
    parser.add_argument('std', help=0.5)
    parser.add_argument('lr', help=0.1)
    parser.add_argument('n_iterations', help=200)

    n_training = 3

    args_ = parser.parse_args()

    problem_name = args_.problem_name
    method = args_.method
    type_policy = args_.policy
    n_points = int(args_.n_starting_points)
    n_iterations = int(args_.n_iterations)

    points_index = range(n_points)
    random_seed = 5

    rs = int(args_.rs)
  #  method = 'approx_lipschitz'
    std = float(args_.std)
    lr = float(args_.lr)
    #lr = 10.0

    parameters = {}
    for i in range(n_points):
        j = i - n_training
        tmp_d = {}
        tmp_d['rs'] = rs
        tmp_d['std'] = std
        tmp_d['method'] = method
        tmp_d['lr'] = lr
        tmp_d['problem_name'] = problem_name

        if problem_name == 'quadratic':
            lb = 10 ** j
            ub = 10 ** (j + 1)

            tmp_d['lb'] = lb
            tmp_d['ub'] = ub
        else:
            tmp_d['lb'] = bounds[problem_name]['lb'][i]
            tmp_d['ub'] = bounds[problem_name]['ub'][i]

        parameters[i] = tmp_d

    np.random.seed(random_seed)
    stat_models = {}
    for i in points_index:
        stat_models[i] = create_model(parameters[i], n_training=n_training, n_epochs=n_iterations)

    if type_policy == 'greedy':
        policy = GreedyPolicy(stat_models, method, problem_name, type_model=method)
    elif type_policy == 'uniform':
        policy = UniformPolicy(stat_models, method, problem_name, type_model=method)

    print(policy.type_model)

    policy.run_policy(n_iterations)