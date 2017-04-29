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
    def gradient_distance_length_scale_respect_point(cls, ls, point, x):
        """
        Compute gradient of r = dist(point, x) respect to x.

        :param ls: np.array(d)
        :param point: np.array(1xd)
        :param x: np.array(nxd)
        :return: np.array(nxd)
        """

        r2 = np.abs(cls.dist_square_length_scale(ls, point, x))
        r = np.sqrt(r2)

        gradient = np.zeros((x.shape[0], len(ls)))

        for i in range(len(ls)):
            factor = (1.0 / (ls[i] ** 2)) * (point[0:1, i] - x[:, i]) / r
            gradient[:, i] = factor

        return gradient