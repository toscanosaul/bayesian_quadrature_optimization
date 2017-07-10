import ujson

from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
)


if __name__ == '__main__':
    # usage: python -m scripts.generate_multiple_spec > data/multiple_specs/multiple_test_spec.json

    # script used to generate spec file to run BGO

    dim_xs = [1]
    bounds_domain_xs = [[(1, 100)]]
    problem_names = ['test_problem']
    training_names = ['test_global']
    type_kernels = [[SCALED_KERNEL, MATERN52_NAME]]
    dimensionss = [[1]]

    specs = SpecService.generate_dict_multiple_spec(1, problem_names, dim_xs, bounds_domain_xs,
                                                    training_names, type_kernels, dimensionss)

    print ujson.dumps(specs, indent=4)
