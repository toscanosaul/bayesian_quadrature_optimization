import ujson

import argparse

from stratified_bayesian_optimization.entities.run_spec import MultipleSpecEntity
from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    UNIFORM_FINITE,
    SBO_METHOD,
    DOGLEG,
    MULTI_TASK_METHOD,
)


if __name__ == '__main__':
    # usage: python -m scripts.aggregate_results_multiple_spec arxiv_10_training_random_seeds.json

    # script used to aggregate the results of the multiple runs defined by a multiple_spec file

    parser = argparse.ArgumentParser()
    parser.add_argument('specfile', help='e.g. multiple_spec.json')
    args = parser.parse_args()

    multiple_spec_file = args.specfile
    multiple_spec = MultipleSpecEntity.from_json(multiple_spec_file)

    SpecService.collect_multi_spec_results(multiple_spec)
