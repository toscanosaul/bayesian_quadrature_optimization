from __future__ import absolute_import

import ujson
import argparse

from stratified_bayesian_optimization.entities.run_spec import RunSpecEntity
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.services.bayesian_global_optimization import BGO

logger = SBOLog(__name__)


if __name__ == '__main__':
    # Example:
    # python -m scripts.run test_spec.json --output_file "dump.json"
    parser = argparse.ArgumentParser()
    parser.add_argument('specfile', help='e.g. sample_spec.json')
    parser.add_argument('--niter', type=int, help='number of iterations', default=5)
    parser.add_argument('--output_file', type=str, help='output file', default='dump.json')
    args = parser.parse_args()

    spec = RunSpecEntity.from_json(args.specfile)
    result = BGO.run_spec(spec)

    with open(args.output_file, 'w') as f:
        ujson.dump(result, f)
