from __future__ import absolute_import


from stratified_bayesian_optimization.util.json_file import JSONFile
import numpy as np
from multi_start.gredy_policy import GreedyPolicy

import argparse

from multi_start.stat_model_domain import StatModel
from multi_start.stat_model_domain_lipschitz import StatModelLipschitz


def create_model(args):

    rs = int(args['rs'])
    lb = float(args['lb'])
    ub = float(args['ub'])
    std = float(args['std'])
    lr = float(args['lr'])
    method = args['method']

    method_ = method
    if method == 'lipschitz' or method == 'approx_lipschitz':
        method_ = 'real_gradient'

    name_model = 'std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s' % (std, rs, lb, ub, lr, method_)
    dir_data = 'data/multi_start/analytic_example/training_results/'

    data = JSONFile.read(dir_data + name_model)

    name_model = 'std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s' % (std, random_seed, lb, ub, lr, method)

    if method == 'real_gradient':
        data['gradients'] = [-1.0 * np.array(t) for t in data['gradients']]
    elif method == 'grad_epoch':
        new_grads = {}
        for t in data['gradients']:
            new_grads[int(t)] = -1.0 * np.array(data['gradients'][t])
        data['gradients'] = new_grads

    data['values'] = [-1.0 * np.array(t) for t in data['values']]


    n_training = 3

    training_data = {'points': data['points'][0:n_training],
                     'values': data['values'][0:n_training], 'gradients': []}
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

    n_epochs = 100
    n_batches = 1
    total_iterations = n_epochs * n_batches

    if method == 'approx_lipschitz' or method == 'lipschitz':
        model = StatModelLipschitz(
            training_data, best_results, n_training, functions_get_value,
            points_domain[-1], 0,
            n_training, problem_name=name_model,
            max_iterations=total_iterations, parametric_mean=False, lower=None, upper=None,
            n_burning=n_burning, total_batches=n_batches, type_model=method, lipschitz=None,
            n_thinning=10, kwargs_get_value_next_iteration=kwargs)
    else:
        model = StatModel(
            training_data, best_results, n_training, functions_get_value,
            points_domain[-1], 0,
            n_training, problem_name=name_model,
            max_iterations=total_iterations, parametric_mean=False, lower=None, upper=None,
            n_burning=n_burning, total_batches=n_batches,model_gradient=method,
            n_thinning=10, kwargs_get_value_next_iteration=kwargs)
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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('method', help='approx_lipschitz')


    args_ = parser.parse_args()

    method = args_.method

    n_points = 9
    points_index = range(n_points)
    random_seed = 5

    rs = 1540
  #  method = 'approx_lipschitz'
    std = 1.0
    lr = 10.0

    parameters = {}
    for i in range(n_points):
        j = i - 3
        tmp_d = {}
        tmp_d['rs'] = rs
        tmp_d['std'] = std
        tmp_d['method'] = method
        tmp_d['lr'] = lr

        lb = 10 ** j
        ub = 10 ** (j + 1)

        tmp_d['lb'] = lb
        tmp_d['ub'] = ub

        parameters[i] = tmp_d

    np.random.seed(random_seed)
    stat_models = {}
    for i in points_index:
        stat_models[i] = create_model(parameters[i])

    policy = GreedyPolicy(stat_models, method, type_model=method)

    n_iterations = 1000
    policy.run_policy(n_iterations)