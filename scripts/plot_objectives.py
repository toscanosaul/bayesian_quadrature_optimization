from __future__ import absolute_import

from plots.plot_objective import make_plots


if __name__ == '__main__':
    # usage: python -m scripts.plot_objectives

    # script used to generate plots of the objective function, and functions considered by
    # the acquisition function (i.e. F(x, w))

    problem_name = 'test_simulated_gp'
    n_tasks = 2
    bounds = [[0, 100]]
    n_samples = 0

    make_plots(problem_name, n_tasks, bounds, n_samples=n_samples)

