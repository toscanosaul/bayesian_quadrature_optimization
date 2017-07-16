import numpy as np

from os import path

from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.lib.sample_functions import SampleFunctions


decimals = 10
random_seed = 5
np.random.seed(random_seed)
n_points = 1000
points = np.linspace(0, 100, n_points)
points = np.round(points, decimals=decimals)
points = points.reshape([n_points, 1])

tasks = np.array([[0, 1]])

add = [10, -10]
kernel = Matern52.define_kernel_from_array(1, np.array([100.0, 1.0]))
function = SampleFunctions.sample_from_gp(points, kernel)
function = function[0, :]

final_function = {}

for task in range(2):
    final_function[task] = []
    for i in xrange(n_points):
        point = np.concatenate((points[i, :], np.array([task])))
        final_function[task].append(function[i] + add[task])

filename = path.join('problems', 'test_simulated_gp', 'simulated_function_with_%d_%d' %
                     (n_points, random_seed))

JSONFile.write({'function': final_function, 'points': points}, filename)
