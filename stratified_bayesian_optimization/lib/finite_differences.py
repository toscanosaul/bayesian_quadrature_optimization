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
