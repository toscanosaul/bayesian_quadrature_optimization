from __future__ import absolute_import

import argparse
import ujson

from stratified_bayesian_optimization.entities.run_spec import MultipleSpecEntity
from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.services.bayesian_global_optimization import BGO


if __name__ == '__main__':
    # Example usage:
    # python -m scripts.run_multiple_spec arxiv_10_training_random_seeds.json 2

    parser = argparse.ArgumentParser()
    parser.add_argument('multiple_spec', help='e.g. test_multiple_spec.json')
    parser.add_argument('spec', help="e.g. 1, number of specification")
    parser.add_argument('--niter', type=int, help='number of iterations', default=5)
    parser.add_argument('--output_file', type=str, help='output file', default='output.json')

    args = parser.parse_args()


    output_file = args.output_file
    n_spec = int(args.spec)

    output_file = 'spec_%d' % n_spec + '_' + output_file
    multiple_spec = MultipleSpecEntity.from_json(args.mspec)


    spec = SpecService.generate_specs(n_spec, multiple_spec)

    result = BGO.run_spec(spec)

    with open(args.output_file, 'w') as f:
        ujson.dump(result, f)

    # results = []
    # for spec in specs:
    #     result = {
    #         'problem_name': spec.problem_name,
    #         'method_optimization': spec.method_optimization,
    #         'result': BGO.run_spec(spec)
    #     }
    #     results.append(result)


