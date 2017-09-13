import ujson

import argparse

from plots.plot_aggregate_results import plot_aggregate_results
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
    # usage: python -m scripts.aggregate_results_multiple_spec combine_arxiv_runs.json --niter 31

    # script used to aggregate the results of the multiple runs defined by a multiple_spec file

    parser = argparse.ArgumentParser()
    parser.add_argument('specfile', help='e.g. multiple_spec.json')
    parser.add_argument('--niter', help='number of iterations', default=-1)
    args = parser.parse_args()

    n_iterations = int(args.niter)

    if n_iterations == -1:
        n_iterations = None

    multiple_spec_file = args.specfile
    multiple_spec = MultipleSpecEntity.from_json(multiple_spec_file)

    SpecService.collect_multi_spec_results(multiple_spec, total_iterations=n_iterations)

    plot_aggregate_results(multiple_spec)
