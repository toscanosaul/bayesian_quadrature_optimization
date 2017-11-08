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
    # usage: python -m scripts.aggregate_results_multiple_spec sbo_mt_arxiv.json --niter 30 --sign 1 --rs_lw 0 --rs_up 10

    # script used to aggregate the results of the multiple runs defined by a multiple_spec file


    parser = argparse.ArgumentParser()
    parser.add_argument('specfile', help='e.g. combine_arxiv_runs.json')
    parser.add_argument('--niter', help='number of iterations', default=-1)
    parser.add_argument('--sign', help='the results are multiplied by sign', default=-1)
    parser.add_argument('--rs_lw', help='number of iterations', default=0)
    parser.add_argument('--rs_up', help='number of iterations', default=-1)
    parser.add_argument('--only_plot', help='only plot', default=0)
    parser.add_argument('--same_rs', help='0', default=0)
    parser.add_argument('--y_label', help='e.g. y_label')

    args = parser.parse_args()

    n_iterations = int(args.niter)
    rs_lw = int(args.rs_lw)
    rs_up = int(args.rs_up)

    same_rs = int(args.same_rs)

    if int(args.only_plot) == 0:
        only_plot = False
    else:
        only_plot = True

    if int(args.sign) == -1:
        sign = True
    else:
        sign = False

    if same_rs == 0:
        same_rs = False
    else:
        same_rs = True

    if rs_up == -1:
        rs_up = None

    if n_iterations == -1:
        n_iterations = None

    multiple_spec_file = args.specfile
    multiple_spec = MultipleSpecEntity.from_json(multiple_spec_file)

    if not only_plot:
        SpecService.collect_multi_spec_results(
            multiple_spec, total_iterations=n_iterations, rs_lw=rs_lw, rs_up=rs_up,
            same_random_seeds=same_rs, sign=sign)

    plot_aggregate_results(multiple_spec, y_label=args.y_label, n_iterations=n_iterations)
