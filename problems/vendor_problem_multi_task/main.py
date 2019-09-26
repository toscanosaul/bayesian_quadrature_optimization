from __future__ import absolute_import

import numpy as np
from copy import deepcopy

from scipy.stats import gamma

from problems.vendor_problem.vendor import simulation


runlength = 5
n_customers = 1000
n_products = 2
cost = [5, 10]
sell_price = [8, 18]

set_constraints = {}
median = gamma.median(n_customers, scale=1)
intervals = [[0.0, median], [median, np.inf]]
for i in xrange(2):
    for j in xrange(2):
        set_constraints[i * 2 + j] = [intervals[i], intervals[j]]

# n=5
# x=[2, 3, 4, 1, 1]
# replications=5
# customers=10
# simulation(x, replications, customers, n, [5,6, 3, 1, 1], [8, 12, 7, 2, 3])

def toy_example(n_samples, x):
    """

    :param n_samples: int
    :param x: (n_products * [int] + [int]) inventory levels, and total sum over the
        number of custombers of the Gumbel random vector associated to the n_products.
    :return: [float, float]

    """
    inv_levels = x[0:-1]
    inv_levels = [int(a) for a in inv_levels]

    task = x[-1]
    val = simulation(inv_levels, n_samples, n_customers, n_products, cost, sell_price,
                     set_sum_exp=set_constraints[task], seed=1)

    return val


def integrate_toy_example(x):
    """

    :param x: n_products * [int]
    :return: [float]
    """
    val = simulation(x, 100000, n_customers, n_products, cost, sell_price, seed=1)
    return val

def main(n_samples, *params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(n_samples, *params)

def main_objective(n_samples, *params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)