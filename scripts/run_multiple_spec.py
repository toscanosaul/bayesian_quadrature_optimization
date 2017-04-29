from __future__ import absoluste_import

import argparse
import ujson

from stratified_bayesian_optimization.entities.run_spec import MultipleSpecEntity
from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.services.bayesian_global_optimization import BGO


if __name__ == '__main__':
    # Example usage:
    # python -m scripts.run_multiple_spec sample_multiple_spec.json

    parser = argparse.ArgumentParser()
    parser.add_argument('multiple_spec', help='e.g. test_multiple_spec.json')
    parser.add_argument('--niter', type=int, help='number of iterations', default=5)
    parser.add_argument('--output_file', type=str, help='output file', default='output.json')

    args = parser.parse_args()
    multiple_spec = MultipleSpecEntity.from_json(args.mspec)

    specs = SpecService.generate_specs(multiple_spec)

    results = []
    for spec in specs:
        result = {
            'problem_name': spec.problem_name,
            'method_optimization': spec.method_optimization,
            'result': BGO.run_spec(spec)
        }
        results.append(result)

    with open(args.output_file, 'w') as f:
        ujson.dump(results, f)
