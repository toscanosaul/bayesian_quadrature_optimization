import warnings

import numpy as np
import numpy.testing as npt

from copy import deepcopy

from stratified_bayesian_optimization.acquisition_functions.sbo import SBO
from stratified_bayesian_optimization.entities.domain import DomainEntity
from stratified_bayesian_optimization.lib.constant import (
    PRODUCT_KERNELS_SEPARABLE,
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    UNIFORM_FINITE,
    TASKS,
)
from stratified_bayesian_optimization.services.gp_fitting import GPFittingService
from stratified_bayesian_optimization.models.gp_fitting_gaussian import GPFittingGaussian
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.lib.sample_functions import SampleFunctions
from stratified_bayesian_optimization.entities.domain import(
    BoundsEntity,
    DomainEntity,
)
from stratified_bayesian_optimization.services.domain import DomainService
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences
from stratified_bayesian_optimization.lib.affine_break_points import (
    AffineBreakPoints,
)
from stratified_bayesian_optimization.lib.parallel import Parallel


dim_x = 4
bounds_domain_x = [(0.01, 1.01), (0.1, 2.1), (1, 21), (1, 201)]
problem_name = 'movies_collaborative'
name_model = 'gp_fitting_gaussian'
training_name = None
type_kernel = [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]
dimensions = [5, 4, 5]
bounds_domain = [[0.01, 1.01], [0.1, 2.1], [1, 21], [1, 201], [0, 1, 2, 3, 4]]
n_training = 100
random_seed = 5
type_bounds = [0, 0, 0, 0, 1]
x_domain = [0, 1, 2, 3]
number_points_each_dimension = [10, 10, 11, 10]
mle = True
distribution = UNIFORM_FINITE
parallel = True
thinning = 5
n_burning = 100
max_steps_out = 1000
n_iterations = 100
same_correlation = True
debug = False
number_points_each_dimension_debug = [10, 10, 10, 10]
noise = False
training_data = None
points = None
n_samples = 0
kernel_values = None
mean_value = None
var_noise_value = None
cache = True
parameters_distribution = None

gp = GPFittingService.get_gp(name_model, problem_name, type_kernel, dimensions, bounds_domain, type_bounds,
                     n_training, noise, training_data, points, training_name, mle, thinning,
                     n_burning, max_steps_out, n_samples, random_seed, kernel_values, mean_value,
                     var_noise_value, cache, same_correlation)
quadrature = BayesianQuadrature(gp, x_domain, distribution,
                                parameters_distribution=parameters_distribution)
gp.data = gp.convert_from_list_to_numpy(gp.training_data)

bq=quadrature
bounds_x = [bq.gp.bounds[i] for i in xrange(len(bq.gp.bounds)) if i in
         bq.x_domain]

points =  DomainService.get_points_domain(11000, bounds_x,
                                            type_bounds=len(bounds_x) * [0])

sbo = SBO(quadrature, np.array(points))
point = np.array([[   0.78777778,    0.32222222,   21.        ,  178.77777778,    1.        ]])
z = sbo.evaluate(point)

print z