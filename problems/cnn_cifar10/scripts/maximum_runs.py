from __future__ import absolute_import

import argparse
import os
import numpy as np

from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


if __name__ == '__main__':
    # Example usage:
    # python -m problems.cnn_cifar10.maximum_runs 500 600

    parser = argparse.ArgumentParser()
    parser.add_argument('min_rs', help='e.g. 500')
    parser.add_argument('max_rs', help='e.g. 600')

    args = parser.parse_args()
    min_rs = int(args.min_rs)
    max_rs = int(args.max_rs)

    max_values = []
    for i in xrange(min_rs, max_rs):
        file_name = 'problems/cnn_cifar10/runs_random_seeds/' + 'rs_%d' % i + '.json'
        if not os.path.exists(dir):
            continue
        data = JSONFile.read(file_name)
        max_values.append(data['test_error_images'])

    max = np.max(max_values)
    min = np.min(max_values)

    logger.info('max is: %f' % max)
    logger.info('min is: %f' % min)

