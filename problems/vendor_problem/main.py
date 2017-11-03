from __future__ import absolute_import

import numpy as np
from copy import deepcopy

from problems.vendor_problem.vendor import simulation


runlength = 5
n_customers = 5
n_products = 2
cost = [5, 6]
sell_price = [8, 10]

set_constraints = {}
intervals = [[-np.inf, 0.0], [0.0, np.inf]]
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
    :param x: (n_products * [int] + [n_products * [float]]) inventory levels, and total sum over the
        number of custombers of the Gumbel random vector associated to the n_products.
    :return: [float, float]

    """
    inv_levels = x[0:-n_products]
    inv_levels = [int(a) for a in inv_levels]

    sum = x[-n_products:]
    val = simulation(inv_levels, n_samples, n_customers, n_products, cost, sell_price,
                     sum_exp=sum, seed=1)

    return val


def integrate_toy_example(x):
    """

    :param x: n_products * [int]
    :return: [float]
    """
    val = simulation(x, 1000, n_customers, n_products, cost, sell_price, seed=1)
    return [val[0]]

def main(n_samples, *params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(n_samples, *params)

def main_objective(n_samples, *params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)