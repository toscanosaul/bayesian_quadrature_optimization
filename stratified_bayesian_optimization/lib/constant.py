import sys


SPECS_DIR = 'data/specs'

# Directory for multiple specifications
MULTIPLESPECS_DIR = 'data/multiple_specs'

# Directory for domain
DOMAIN_DIR = 'domain'

# Directory of diagnostics kernel
DIAGNOSTIC_KERNEL_DIR = 'results/diagnostic_kernel'

# Directory of GP models
GP_DIR = 'data/gp_models'

# Directory of problems
PROBLEM_DIR = 'problems'

# Name of the file to call problems
FILE_PROBLEM = 'main'

MATERN52_NAME = 'Matern52'
TASKS_KERNEL_NAME = 'Tasks_Kernel'
PRODUCT_KERNELS_SEPARABLE = 'Product_of_kernels_with_separable_domain'
SCALED_KERNEL = 'Scaled_kernel'

# Default names for the parameters
LENGTH_SCALE_NAME = 'length_scale'
SIGMA2_NAME = 'sigma2'

LOWER_TRIANG_NAME = 'lower_triangular'

MEAN_NAME = 'mean'
VAR_NOISE_NAME = 'var_noise'

# Optimization methods
LBFGS_NAME = 'lbfgs'

# Constants
SMALLEST_POSITIVE_NUMBER = 1e-10
SMALLEST_NUMBER = -sys.float_info.max
LARGEST_NUMBER = sys.float_info.max

# Cached
CHOL_COV = 'chol_cov'
SOL_CHOL_Y_UNBIASED = 'sol_chol_y_unbiased'

# Random
DEFAULT_RANDOM_SEED = 1
