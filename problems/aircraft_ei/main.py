from __future__ import absolute_import

import numpy as np
from copy import deepcopy

from problems.aircraft.fuel_burn import *


n_scenarios = 8
points_, weights_ = weights_points(n_scenarios)

def toy_example(x):
    """

    :param x: [float, float, int, int, int]
    :return: [float]
    """
    thickness_cp = np.array(x[0: 3])
    twist_cp = np.array(x[3:])

    value = get_burn_flight_conditions(thickness_cp, twist_cp, points=points_, weight=weights_)

    return [value]

def integrate_toy_example(x):
    """

    :param x: [float, float]
    :return: [float]
    """
    thickness_cp = np.array(x[0: 3])
    twist_cp = np.array(x[3:])

    value = get_burn_flight_conditions(thickness_cp, twist_cp, points=points_, weight=weights_)
    return [value]


def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)