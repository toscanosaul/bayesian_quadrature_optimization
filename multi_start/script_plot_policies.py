from __future__ import absolute_import

import numpy as np
import os


import matplotlib.pyplot as plt
plt.switch_backend('agg')

import argparse
from stratified_bayesian_optimization.util.json_file import JSONFile


def get_best_values(data, n_restarts=9, n_training=3):
    chosen = data['chosen_index']
    current_value = []
    for i in range(n_restarts):
        current_value.append(data['evaluations'][str(i)][n_training - 1])

    index_points = {}

    for i in range(n_restarts):
        index_points[i] = n_training
    best_values = []
    best_values.append(np.max(current_value))
    n_iterations = len(chosen)

    for i in range(n_iterations):
        j = int(data['chosen_index'][i])
        current_value[j] = data['evaluations'][str(j)][index_points[j]]
        best_values.append(np.max(current_value))
        index_points[j] += 1
    return best_values

def plot_animation_policy(data, type, problem_name, n_restarts=9, n_training=3, n_iterations=200):
    current_index = {}
    for i in range(n_restarts):
        current_index[i] = n_training
    chosen = data['chosen_index']
    paths = {}
    for i in range(n_restarts):
        if problem_name == 'parabola':
            paths[i] = [-np.log(-t[0]) for t in data['evaluations'][str(i)][0:n_training]]
        else:
            paths[i] = [t[0] for t in data['evaluations'][str(i)][0:n_training]]

    points = range(n_training)
    plt.figure()
    for i in range(n_restarts):
        plt.plot(points, paths[i], label=str(i))

    plt.legend()
    plt.ylabel('Objective function')
    plt.xlabel('Iteration')

    file_name = 'data/multi_start/plot_policies/animation/'

    if not os.path.exists(file_name):
        os.mkdir(file_name)

    file_name += problem_name + '/'

    if not os.path.exists(file_name):
        os.mkdir(file_name)

    file_name += type + '/'


    if not os.path.exists(file_name):
        os.mkdir(file_name)

    file_name += str(0) + '_paths'

    plt.savefig(file_name + '.pdf')


    for j in range(n_iterations):
        i = chosen[j]
        index = current_index[i]

        if problem_name == 'parabola':
            paths[i].append(-np.log(-data['evaluations'][str(i)][index][0]))
        else:
            paths[i].append(data['evaluations'][str(i)][index][0])

        current_index[i] += 1
        plt.figure()
        for i in range(n_restarts):
            n_points = len(paths[i])
            points = range(n_points)
            plt.plot(points, paths[i], label=str(i))

        plt.legend()
        plt.ylabel('Objective function')
        plt.xlabel('Iteration')

        file_name = 'data/multi_start/plot_policies/animation/'
        file_name += problem_name + '/'
        file_name += type + '/'

        file_name += str(j) + '_paths'

        plt.savefig(file_name + '.pdf')


if __name__ == '__main__':
    # python -m multi_start.script_plot_policies real_gradient problem6 10 10

    parser = argparse.ArgumentParser()
    parser.add_argument('method', help='approx_lipschitz')
    parser.add_argument('problem_name', help='analytic_example')
    parser.add_argument('n_starting_points', help=10)
    parser.add_argument('n_iterations', help=200)
   # parser.add_argument('method_2', help='real_gradient')



    args_ = parser.parse_args()
    n_iterations = int(args_.n_iterations)
    method = args_.method
    problem = args_.problem_name


    file_1 = 'data/multi_start/' + problem + '/' + 'greedy_policy/' + method + '.json'
    file_2 = 'data/multi_start/' + problem + '/' +'uniform_policy/' + method + '.json'

    n_restarts = int(args_.n_starting_points)
    n_training = 3
    n = n_iterations


    data = JSONFile.read(file_1)
    data_2 = JSONFile.read(file_2)

    data_list = [data, data_2]

    best_values = {}
    data_dict = {}
    type_1 = 'greedy_' + method
    types = [type_1, 'uniform']

    data_dict[type_1] = data
    data_dict['uniform'] = data_2

    for t in types:
        best_values[t] = get_best_values(data_dict[t], n_restarts, n_training)

    points = range(n_training, n_training + n)

    plt.figure()

    for t in types:
        z = best_values[t][0:n]
        plt.plot(points[0:n], z, label=t)

    plt.legend()
    plt.ylabel('Best Value')
    plt.xlabel('Iteration')

    file_name = 'data/multi_start/plot_policies/'

    if not os.path.exists(file_name):
        os.mkdir(file_name)

    file_name += problem + '/'

    if not os.path.exists(file_name):
        os.mkdir(file_name)


    file_name += type_1 + '_uniform' + '_' + 'best_solution'

    plt.savefig(file_name + '.pdf')

    for t in data_dict:
        plot_animation_policy(
            data_dict[t], t, problem, n_restarts=n_restarts, n_training=n_training, n_iterations=n)
