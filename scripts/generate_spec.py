import ujson

from stratified_bayesian_optimization.services.spec import SpecService


if __name__ == '__main__':
    # usage: python -m scripts.generate_spec > data/specs/test_spec.json

    # script used to generate spec file to run BGO

    dim_x = 2
    choose_noise = True
    bounds_domain_x = [(0, 1), (3, 6)]
    number_points_each_dimension = [5, 5]
    problem_name = 'toy_example'
    method_optimization = 'SBO'

    spec = SpecService.generate_dict_spec(problem_name, dim_x, choose_noise, bounds_domain_x,
                                          number_points_each_dimension, method_optimization)

    print ujson.dumps(spec, indent=4)