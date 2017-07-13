import ujson

from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
)


if __name__ == '__main__':
    # usage: python -m scripts.generate_spec > data/specs/test_spec.json

    # script used to generate spec file to run BGO

    dim_x = 1
    bounds_domain_x = [(0, 100)]
    problem_name = 'test_problem_with_tasks'
    training_name = 'test_global'
    type_kernel = [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]
    dimensions = [2, 1, 2]
    bounds_domain = [[0, 100], [0, 1]]
    n_training = 100
    random_seed = 5
    type_bounds = [0, 1]
    x_domain = [0]


    spec = SpecService.generate_dict_spec(problem_name, dim_x, bounds_domain_x, training_name,
                                          type_kernel, dimensions, bounds_domain=bounds_domain,
                                          n_training=n_training, random_seed=random_seed,
                                          type_bounds=type_bounds, x_domain=x_domain)

    print ujson.dumps(spec, indent=4)

