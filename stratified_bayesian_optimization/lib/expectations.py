from __future__ import absolute_import

import numpy as np


def uniform_finite(f, point, index_points, domain_random, index_random, double=False):
    """
    Computes the expectation of f(z), where z=(point, x) which is equal to:
        mean(f((point, x)): x in domain_random), where
    z[index_points[i]] = point[i].

    If double is True, it computes the mean over all the points.

    :param f: function
    :param point: np.array(1xk)
    :param index_points: [int]
    :param domain_random: np.array(mxl)
    :param index_random: [int]
    :param double: boolean
    :return: np.array
    """

    dim_random = domain_random.shape[1]

    new_points = np.zeros((domain_random.shape[0], dim_random + point.shape[1]))
    new_points[:, index_points] = np.repeat(point, domain_random.shape[0], axis=0)

    new_points[:, index_random] = domain_random

    values = f(new_points)

    if double:
        return np.mean(values)

    return np.mean(values, axis=0)

def gradient_uniform_finite(f, point, index_points, domain_random, index_random):
    """
    Computes the gradient of the expectation of f(z), where z=(point, x)

    :param f: function
    :param point: np.array(1xk)
    :param index_points: [int]
    :param domain_random: np.array(mxl)
    :param index_random: [int]

    :return: np.array
    """

    dim_random = domain_random.shape[1]

    new_points = np.zeros((domain_random.shape[0], dim_random + point.shape[1]))
    new_points[:, index_points] = np.repeat(point, domain_random.shape[0], axis=0)

    new_points[:, index_random] = domain_random

    values = f(new_points)

    if double:
        return np.mean(values)

    return np.mean(values, axis=0)


