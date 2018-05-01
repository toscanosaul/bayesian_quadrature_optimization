from __future__ import absolute_import

import numpy as np

from multi_start.problems.sgd_specific_function import *
from multi_start.script_run_policies import *

if __name__ == '__main__':
    # python -m multi_start.script_run_policies_from_beginning 1 20 0.1 0.1 approx_lipschitz 0 100 problem5
    parser = argparse.ArgumentParser()
    parser.add_argument('rs', help=5)
    parser.add_argument('n_restarts', help=10)
    parser.add_argument('std', help=1.0)
    parser.add_argument('lr', default=1.0)
    parser.add_argument('method', help='real_gradient, grad_epoch, lipschitz, approx_lipschitz')
    parser.add_argument('lipschitz')
    parser.add_argument('n_epochs', default=20)
    parser.add_argument('problem_name', help='quadratic, problem6, problem5')

    args = parser.parse_args()

    random_seed = int(args.rs)
    n_restarts = int(args.n_restarts)
    std = float(args.std)
    lr = float(args.lr)
    method = args.method
    lipschitz = float(args.lipschitz)
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

    method_ = method
    if method == 'approx_lipschitz':
        method_ = 'real_gradient'

    for i in range(n_restarts):
        start = start_points[i]
        logger.info('start')
        logger.info(start)

        results = SGD(start, gradient, batch_size, objective, maxepoch=n_epochs, adam=False,
                      name_model='std_%f_rs_%d_lb_%f_ub_%f_lr_%f_%s' % (
                      std, random_seed, lb, ub, lr, method_),
                      exact_gradient=exact_gradient, learning_rate=lr, method=method, n_epochs=5,
                      n_samples=100, gradient_samples=gradient_samples, problem=problem,
                      bounds=bounds)
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
    points_index = range(n_restarts)
    for i in points_index:
        stat_models[i] = create_model(parameters[i], n_training=n_training, n_epochs=n_epochs, burning=False)
        stat_models_2[i] = create_model(parameters[i], n_training=n_training, n_epochs=n_epochs, burning=False)

    policy_greedy = GreedyPolicy(stat_models, method, problem_name, type_model=method, random_seed=random_seed, n_restarts=n_restarts)
    policy_uniform = UniformPolicy(stat_models_2, method, problem_name, type_model=method, random_seed=random_seed, n_restarts=n_restarts)

    policy_greedy.run_policy(n_epochs - n_training)
    policy_uniform.run_policy(n_epochs - n_training)



