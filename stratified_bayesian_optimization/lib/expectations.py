from __future__ import absolute_import

import numpy as np

from scipy.stats import gamma


def uniform_finite(f, point, index_points, domain_random, index_random, double=False):
    """
    Computes the expectation of f(z), where z=(point, x) which is equal to:
        mean(f((point, x)): x in domain_random), where
    z[index_points[i]] = point[i].

    If double is True, it computes the mean over all the points. Used for the variance.

    :param f: function
    :param point: np.array(1xk)
    :param index_points: [int]
    :param domain_random: np.array(n_tasksx1)
    :param index_random: [int]
    :param double: boolean
    :return: np.array
    """

    dim_random = domain_random.shape[1]

    new_points = np.zeros((domain_random.shape[0], dim_random + point.shape[1]))

    new_points[:, index_points] = np.repeat(point, domain_random.shape[0], axis=0)

    new_points[:, index_random] = domain_random



    if double:
        new_points = np.concatenate([new_points, new_points], axis=1)
        values = f(new_points)
        return np.mean(values)

    values = f(new_points)
    return np.mean(values, axis=0)

def gamma(f, point, index_points, index_random, parameters_dist, n_samples=50, double=False):
    """
    Computes the expectation of f(z), where z=(point, x) which is equal to:
        mean(f((point, x)): x in domain_random), where
    z[index_points[i]] = point[i].

    If double is True, it computes the mean over all the points. Used for the variance.

    :param f: function
    :param point: np.array(1xk)
    :param index_points: [int]
    :param index_random: [int]
    :param parameters_dist: {'scale':float, 'a': int}
    :param n_samples: int
    :param double: boolean
    :return: np.array
    """
    a = parameters_dist['a']
    scale = parameters_dist['scale']

    if double:
        n_samples *= n_samples

    new_points = np.zeros((n_samples, len(index_random) + point.shape[1]))

    new_points[:, index_points] = np.repeat(point, n_samples, axis=0)

    z = gamma.rvs(a, scale=scale, size=n_samples)
    if double:
        w = gamma.rvs(a, scale=scale, size=n_samples)

        random = []
        for s in z:
            for t in w:
                random.append([s, t])
        random = np.array(random)
    else:
        random = np.array([z])

    new_points[:, index_random] = random

    z = f(new_points)

    if double:
        return np.mean(z)

    return np.mean(z, axis=0)


def gradient_uniform_finite(f, point, index_points, domain_random, index_random, points_2,
                            parameters_kernel):
    """
    Computes the gradient of the expectation of f(z, point_), where z=(point, x), for each
    point_ in points_2.

    :param f: function
    :param point: np.array(1xk)
    :param index_points: [int]
    :param domain_random: np.array(n_tasks x 1)
    :param index_random: [int]
    :param points_2: np.array(mxk')
    :param parameters_kernel: np.array(n)
    :return: np.array(kxm)
    """

    dim_random = domain_random.shape[1]

    new_points = np.zeros((domain_random.shape[0], dim_random + point.shape[1]))
    new_points[:, index_points] = np.repeat(point, domain_random.shape[0], axis=0)

    new_points[:, index_random] = domain_random

    gradients = np.zeros((new_points.shape[0], point.shape[1], points_2.shape[0]))

    for i in xrange(new_points.shape[0]):
        value = f(new_points[i:i+1, :], points_2, parameters_kernel)[:, index_points]
        gradients[i, :, :] = value.transpose()

    gradient = np.mean(gradients, axis=0)

    return gradient

def gradient_gamma():
    pass

def hessian_uniform_finite(f, point, index_points, domain_random, index_random, points_2,
                            parameters_kernel):
    """
    Computes the Hessian of the expectation of f(z, point_), where z=(point, x), for each
    point_ in points_2.

    :param f: function
    :param point: np.array(1xk)
    :param index_points:
    :param domain_random:
    :param index_random:
    :param points_2: np.array(mxk')
    :param parameters_kernel:
    :return: np.array(mxkxk)
    """

    dim_random = domain_random.shape[1]

    new_points = np.zeros((domain_random.shape[0], dim_random + point.shape[1]))

    new_points[:, index_points] = np.repeat(point, domain_random.shape[0], axis=0)

    new_points[:, index_random] = domain_random

    hessian = np.zeros((new_points.shape[0], points_2.shape[0], point.shape[1], point.shape[1]))

    for i in xrange(new_points.shape[0]):
        value = f(new_points[i:i+1, :], points_2, parameters_kernel)[:, index_points]
        value = value[:, :, index_points]
        hessian[i, :, :, :] = value

    hessian = np.mean(hessian, axis=0)

    return hessian

def gradient_gamma_resp_candidate():
    pass

def hessian_gamma():
    pass


def gradient_uniform_finite_resp_candidate(f, candidate_point, index_points, domain_random,
                                           index_random, points, parameters_kernel):
    """
    Computes the gradient of the expectation of f(z, candidate_point) respect to candidate_point,
    where z=(point, x) for each point in points.

    :param f: function
    :param candidate_point: np.array(1xk)
    :param index_points: [int]
    :param domain_random: np.array(n_tasks x 1)
    :param index_random: [int]
    :param points: np.array(mxk')
    :param parameters_kernel: np.array(n)
    :return: np.array(kxm)
    """

    dim_random = domain_random.shape[1]

    new_points = np.zeros((domain_random.shape[0], dim_random + points.shape[1]))

    gradients = np.zeros((candidate_point.shape[1], points.shape[0]))

    for i in xrange(points.shape[0]):
        point = points[i : i+1, :]
        new_points[:, index_points] = np.repeat(point, domain_random.shape[0], axis=0)
        new_points[:, index_random] = domain_random
        values = f(candidate_point, new_points, parameters_kernel)
        gradients[:, i] = np.mean(values, axis=0)

    return gradients
