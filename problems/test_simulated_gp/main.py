from __future__ import absolute_import

from stratified_bayesian_optimization.util.json_file import JSONFile
import numpy as np

read = JSONFile.read("problems/test_simulated_gp/simulated_function_with_1000_5")
points = read['points']
function = read['function']


def find_point_in_domain(x, array=np.array(points)):
    """
    Find the index of the closest point in array to x
    :param x: float
    :param array: np.array(float)
    :return: int
    """
    idx =  np.abs(array - x).argmin()
    return idx

def toy_example(x):
    """

    :param x: [float, int]
    :return: [float]
    """
    id = find_point_in_domain(x[0])

    return [function[str(int(x[1]))][id]]

def integrate_toy_example(x):
    """

    :param x: [float]
    :return: [float]
    """
    point = [x[0], 0]
    a = toy_example(point)

    point = [x[0], 1]
    b = toy_example(point)

    return [(a[0] + b[0]) / 2.0]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)

