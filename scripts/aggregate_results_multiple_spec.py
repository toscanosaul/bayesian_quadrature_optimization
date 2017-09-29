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
    # usage: python -m scripts.aggregate_results_multiple_spec sbo_mt_arxiv.json --niter 30 --rs_lw 0 --rs_up 10

    # script used to aggregate the results of the multiple runs defined by a multiple_spec file


    parser = argparse.ArgumentParser()
    parser.add_argument('specfile', help='e.g. combine_arxiv_runs.json')
    parser.add_argument('--niter', help='number of iterations', default=-1)
    parser.add_argument('--rs_lw', help='number of iterations', default=0)
    parser.add_argument('--rs_up', help='number of iterations', default=-1)
    args = parser.parse_args()

    n_iterations = int(args.niter)
    rs_lw = int(args.rs_lw)
    rs_up = int(args.rs_up)

    if rs_up == 1:
        rs_up = None

    if n_iterations == -1:
        n_iterations = None

    multiple_spec_file = args.specfile
    multiple_spec = MultipleSpecEntity.from_json(multiple_spec_file)

    SpecService.collect_multi_spec_results(
        multiple_spec, total_iterations=n_iterations, rs_lw=rs_lw, rs_up=rs_up)


    plot_aggregate_results(multiple_spec)
