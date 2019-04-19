from __future__ import absolute_import

import numpy as np
from copy import deepcopy

from problems.aircraft_ei.fuel_burn import *


n_scenarios = 8
points_, weights_ = weights_points(n_scenarios)

heights = [40000, 50000, 30000, 10000, 20000, 7000]
mach_numbers = [0.8322926591514808, 0.8338867889125835, 0.8366730174675625, 0.8389325581307193,
                0.8410674418692806, 0.8433269825324374, 0.8461132110874164, 0.8477073408485192]
domain_random = [[m, h] for m in mach_numbers for h in heights]

def toy_example(n_samples, x):
    """

    :param x: [float, float, int, int, int]
    :return: [float]
    """
    thickness_cp = np.array(x[0: 3])
    twist_cp = np.array(x[3:])

    index = np.random.choice(len(domain_random), n_samples, replace=False)

    sol = 0.0
    values = []
    for t in index:
        match_number = domain_random[t][0]
        height = domain_random[t][1]
        ans = get_burn(thickness_cp, twist_cp, match_number, height)
        values.append(ans)
        sol += ans

    return [-1.0 * sol / float(n_samples), float(np.var(values)) / float(n_samples)]


def integrate_toy_example(x):
    """

    :param x: [float, float]
    :return: [float]
    """
    thickness_cp = np.array(x[0: 3])
    twist_cp = np.array(x[3:])

    value = get_burn_flight_conditions(thickness_cp, twist_cp, points=points_, weight=weights_)
    return [-1.0 * value, 0.0]


def main(n_samples, *params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(n_samples, *params)

def main_objective(n_samples, *params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)