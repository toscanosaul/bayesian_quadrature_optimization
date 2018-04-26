from __future__ import absolute_import

import numpy as np
import argparse

from multi_start.parametric_model import ParametricModel
from multi_start.stat_model import StatModel


from stratified_bayesian_optimization.util.json_file import JSONFile
import numpy as np
from multi_start.stat_model_domain import StatModel
from multi_start.stat_model_domain_lipschitz import StatModelLipschitz


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rs', help='5')
    parser.add_argument('lb', help=-1)
    parser.add_argument('ub', help=1)
    parser.add_argument('std', help=1.0)
    parser.add_argument('learning_rate', default=1.0)
    parser.add_argument('method', help='real_gradient, grad_epoch, lipschitz, approx_lipschitz')
    parser.add_argument('lipschitz')

    args = parser.parse_args()

    random_seed = int(args.rs)
    lb = float(args.lb)
    ub = float(args.ub)
    std = float(args.std)
    lr = float(args.learning_rate)
    method = args.method

    np.random.seed(random_seed)

    method_ = method
    if method == 'lipschitz' or method == 'approx_lipschitz':
        method_ = 'real_gradient'

    name_model = 'std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s' % (std, random_seed, lb, ub, lr, method_)
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


    def get_values(i):
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


    training_data = {}
    best_results = {}
    functions_get_value = {}
    points_domain = {}
    arguments = {}

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

    points_domain= data['points'][0: n_training]
    best_results = np.max(training_data['values'])
    functions_get_value = get_values

    n_burning = 50

    n_epochs = 20
    n_batches = 1
    total_iterations = n_epochs * n_batches

    if method == 'lipschitz':
        lipschitz = float(args.lipschitz)
        model = StatModelLipschitz(
            training_data, best_results, n_training, functions_get_value,
            points_domain[-1], 0,
            n_training, problem_name=name_model,
            max_iterations=total_iterations, parametric_mean=False, lower=None, upper=None,
            n_burning=n_burning, total_batches=n_batches, type_model=method, lipschitz=lipschitz)
    elif method == 'approx_lipschitz':
        model = StatModelLipschitz(
            training_data, best_results, n_training, functions_get_value,
            points_domain[-1], 0,
            n_training, problem_name=name_model,
            max_iterations=total_iterations, parametric_mean=False, lower=None, upper=None,
            n_burning=n_burning, total_batches=n_batches, type_model=method, lipschitz=None)
    else:
        model = StatModel(
            training_data, best_results, n_training, functions_get_value,
            points_domain[-1], 0,
            n_training, problem_name=name_model,
            max_iterations=total_iterations, parametric_mean=False, lower=None, upper=None,
            n_burning=n_burning, total_batches=n_batches, model_gradient=method)


    results = model.accuracy(model.gp_model, start=n_training, iterations=total_iterations, sufix=name_model,
                             model=method)

    model.plot_accuracy_results(
        results[0], results[1], 0.0,
        start=n_training, sufix=name_model, final_iteration=total_iterations)