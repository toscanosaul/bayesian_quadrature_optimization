from __future__ import absolute_import

import sys


SPECS_DIR = 'data/specs'

# Directory for multiple specifications
MULTIPLESPECS_DIR = 'data/multiple_specs'

# Directory for domain
DOMAIN_DIR = 'domain'

# Directory of diagnostics kernel
DIAGNOSTIC_KERNEL_DIR = 'results/diagnostic_kernel'
RESULTS_DIR = 'results/'

# Directory of GP models
GP_DIR = 'data/gp_models'

# Directory of log messages
LOG_DIR = 'data/log'

# Directory of problems
PROBLEM_DIR = 'problems'

# Name of the file to call problems
FILE_PROBLEM = 'main'

MATERN52_NAME = 'Matern52'
TASKS_KERNEL_NAME = 'Tasks_Kernel'
SAME_CORRELATION = 'same_correlation'
PRODUCT_KERNELS_SEPARABLE = 'Product_of_kernels_with_separable_domain'
SCALED_KERNEL = 'Scaled_kernel'
ORNSTEIN_KERNEL = 'Ornstein_kernel'
DIFFERENCES_KERNEL = 'Differences_kernel'

# Default names for the parameters
LENGTH_SCALE_NAME = 'length_scale'
SIGMA2_NAME = 'sigma2'
LENGTH_SCALE_ORNSTEIN_NAME = 'length_scale_ornstein'


LOWER_TRIANG_NAME = 'lower_triangular'

MEAN_NAME = 'mean'
VAR_NOISE_NAME = 'var_noise'

# Optimization methods
LBFGS_NAME = 'lbfgs'
SGD_NAME = 'sgd'
NEWTON_CG_NAME = 'newton_cg'
TRUST_N_CG = 'trust-ncg'
DOGLEG = 'dogleg'
NELDER = 'Nelder-Mead'

# Constants
SMALLEST_POSITIVE_NUMBER = 1e-10
SMALLEST_NUMBER = -sys.float_info.max
LARGEST_NUMBER = sys.float_info.max

# Cached
CHOL_COV = 'chol_cov'
SOL_CHOL_Y_UNBIASED = 'sol_chol_y_unbiased'

# Random
DEFAULT_RANDOM_SEED = 1

#Distributions
UNIFORM_FINITE = 'uniform_finite'
TASKS = 'n_tasks'
EXPONENTIAL = 'exponential'
GAMMA = 'gamma'
WEIGHTED_UNIFORM_FINITE = 'weighted_uniform_finite'
WEIGHTS = 'weights'
MULTINOMIAL_DISTRIBUTION = 'multinomial_distribution'

#Cache Quadrature
QUADRATURES = 'quadrature'
POSTERIOR_MEAN = 'posterior_mean'
B_NEW = 'quadratures_with_candidate'

#BGO methods
SBO_METHOD = 'sbo'
MULTI_TASK_METHOD = 'multi_task'
EI_METHOD = 'ei'
SDE_METHOD = 'sde'

#Directory of solutions of BGO in the different iterations
PARTIAL_RESULTS = 'partial_results'
AGGREGATED_RESULTS = 'aggregated_results'

#Directory of debugging
DEBUGGING_DIR = 'data/debugging'

BAYESIAN_QUADRATURE = 'bayesian_quadrature'

# Default number of sampled parameters
DEFAULT_N_PARAMETERS = 20

DEFAULT_N_SAMPLES = 100