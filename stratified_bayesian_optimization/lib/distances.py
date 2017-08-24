from __future__ import absolute_import

from scipy.spatial.distance import cdist
import numpy as np


class Distances(object):

    @staticmethod
    def dist_square_length_scale(ls, x1, x2=None):
        """
        Compute the squared distance between each point of x1 and x2, given the length scales ls.
        :param ls: np.array(d)
        :param x1: np.array(nxd)
        :param x2: np.array(mxd)

        :return: np.array(nxm)
        """

        if x2 is None:
            x1 = x1 / ls
            x2 = x1
        else:
            x1 = x1 / ls
            x2 = x2 / ls

        distance_2 = cdist(x1, x2, 'sqeuclidean')

        return distance_2

    @classmethod
    def gradient_distance_length_scale_respect_ls(cls, ls, x1, x2=None):
        """
        Compute gradient of r respect to ls

        :param ls: np.array(d)
        :param x1: np.array(nxd)
        :param x2: np.array(mxd)
        :return: {
            (int) ls[i]: np.array(nxm)
        }
        """

        N = x1.shape[0]

        r2 = np.abs(cls.dist_square_length_scale(ls, x1, x2))
        r = np.sqrt(r2)

        if x2 is None:
            x2 = x1

        gradient = {}

        for i in range(len(ls)):
            x_i = x1[:, i:i + 1]
            x_2_i = x2[:, i:i + 1]
            x_dist = cdist(x_i, x_2_i, 'sqeuclidean')
            product_1 = (1.0 / r) * x_dist
            product_2 = - 1.0 / (ls[i] ** 3)
            derivative = product_1 * product_2

            for j in range(N):
                derivative[j, j] = 0

            gradient[i] = derivative

        return gradient

    @classmethod
    def gradient_distance_length_scale_respect_point(cls, ls, point, x, second=False):
        """
        Compute gradient of r = dist(point, x) respect to x.

        :param ls: np.array(d)
        :param point: np.array(1xd)
        :param x: np.array(nxd)
        :param second: (boolean) Hessian if it's True
        :return: np.array(nxd) or {'first': np.array(nxd), 'second': np.array(nxdxd)}
        """

        r2 = np.abs(cls.dist_square_length_scale(ls, point, x))
        r = np.sqrt(r2)

        gradient = np.zeros((x.shape[0], len(ls)))

        differences = {}

        for i in range(len(ls)):
            differences[i] = (point[0:1, i] - x[:, i]) / (ls[i] ** 2)
            factor = differences[i] / r
            gradient[:, i] = factor

        if not second:
            return gradient

        cross_partial = {}

        r3 = r ** 3
        gradient_2 = np.zeros((x.shape[0], len(ls)))
        for i in xrange(len(ls)):
            factor_1 = (r * (ls[i] ** 2)) ** -1
            factor_2 = differences[i] ** 2 / r3

            factor = factor_1 - factor_2
            gradient_2[:, i] = factor
            for j in xrange(i + 1, len(ls)):
                cross_partial[(i, j)] = -1.0 * differences[i] * differences[j] /r3

        hessian = np.zeros((x.shape[0], len(ls), len(ls)))
        for j in xrange(len(ls)):
            hessian[:, j, j] = gradient_2[:, j]
            for h in xrange(j + 1, len(ls)):
                hessian[:, j, h] = cross_partial[(j, h)][0, :]
                hessian[:, h, j] = hessian[:, j, h]

        return {'first': gradient, 'second': hessian}
