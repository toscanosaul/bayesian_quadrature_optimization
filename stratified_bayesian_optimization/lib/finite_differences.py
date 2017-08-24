from __future__ import absolute_import

from collections import defaultdict

from copy import deepcopy

import numpy as np


class FiniteDifferences(object):

    @staticmethod
    def forward_difference(f, x, dh):
        """
        Computes the gradient using forward differences.

        :param f: scalar or vectorial function (output as np.array(mxk))
        :param x: np.array(n)
        :param dh: np.array(n) or np.array(1)
        :return: {(int) i: np.array(mxk)}, each entry of the dictionary are the finite differences
            of f respect to the ith entry of x.
        """

        n = len(x)
        if n != len(dh):
            dh = np.zeros(n) + dh[0]

        finite_differences = defaultdict()

        base_value = f(x)

        for i in xrange(n):
            new_x = deepcopy(x)
            new_x[i] = x[i] + dh[i]
            new_evaluation = f(new_x)
            finite_differences[i] = (new_evaluation - base_value) / dh[i]

        return finite_differences

    @staticmethod
    def second_order_central(f, x, dh):
        """
        Computes the Hessian using second order central differences.

        :param f: scalar or vectorial function (output as np.array(mxk))
        :param x: np.array(n)
        :param dh: np.array(n) or np.array(1)
        :return: {(i, j): np.array(mxk)}, each entry of the dictionary are the finite differences
            of f respect to the ith and jth entry of x.
        """

        n = len(x)
        if n != len(dh):
            dh = np.zeros(n) + dh[0]

        finite_differences = defaultdict()

        base_value = f(x)

        for i in xrange(n):
            forward_x = deepcopy(x)
            forward_x[i] = x[i] + dh[i]
            value_f = f(forward_x)

            backward_x = deepcopy(x)
            backward_x[i] = x[i] - dh[i]
            value_b = f(backward_x)

            val = (value_f + value_b - 2.0 * base_value) / (dh[i] ** 2)

            finite_differences[(i, i)] = val

            for j in xrange(i + 1, n):
                forward = deepcopy(forward_x)
                forward[j] = forward_x[j] + dh[j]

                val_0 = f(forward)

                forward_2 = deepcopy(forward_x)
                forward_2[j] = forward_x[j] - dh[j]

                val_1 = f(forward_2)

                backward = deepcopy(backward_x)
                backward[j] = backward_x[j] - dh[j]

                val_2 = f(backward)

                backward_2 = deepcopy(backward_x)
                backward_2[j] = backward_x[j] + dh[j]

                val_3 = f(backward_2)

                val = val_0 + val_2 - val_1 - val_3


                val /= 4.0 * dh[j] * dh[i]

                finite_differences[(i, j)] = val
                finite_differences[(j, i)] = val

        return finite_differences


