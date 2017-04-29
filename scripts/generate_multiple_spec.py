import ujson

from stratified_bayesian_optimization.services.spec import SpecService


if __name__ == '__main__':
    # usage: python -m scripts.generate_multiple_spec > data/multiple_specs/multiple_test_spec.json

    # script used to generate spec file to run BGO

    dim_xs = [2]
    choose_noises = [True]
    bounds_domain_xs = [[(0, 1), (3, 6)]]
    number_points_each_dimensions = [[5, 5]]
    problem_names = ['toy_example']
    method_optimizations = ['SBO']

    specs = SpecService.generate_dict_multiple_spec(problem_names, dim_xs, choose_noises,
                                                    bounds_domain_xs, number_points_each_dimensions,
                                                    method_optimizations)

    print ujson.dumps(specs, indent=4)