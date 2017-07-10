import ujson

from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
)


if __name__ == '__main__':
    # usage: python -m scripts.generate_spec > data/specs/test_spec.json

    # script used to generate spec file to run BGO

    dim_x = 1
    bounds_domain_x = [(1, 100)]
    problem_name = 'test_problem'
    training_name = 'test_global'
    type_kernel = [SCALED_KERNEL, MATERN52_NAME]
    dimensions = [1]

    spec = SpecService.generate_dict_spec(problem_name, dim_x, bounds_domain_x, training_name,
                                          type_kernel, dimensions)

    print ujson.dumps(spec, indent=4)
