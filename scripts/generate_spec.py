import ujson

from stratified_bayesian_optimization.services.spec import SpecService


if __name__ == '__main__':
    # usage: python -m scripts.generate_spec > data/specs/test_spec.json

    # script used to generate spec file to run BGO

    dim_x = 1
    bounds_domain_x = [(1, 100)]
    problem_name = 'test_problem'
    training_name = 'test_global'

    spec = SpecService.generate_dict_spec(problem_name, dim_x, bounds_domain_x, training_name)

    print ujson.dumps(spec, indent=4)
