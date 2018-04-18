from __future__ import absolute_import

import numpy as np
import numpy.random as npr
from stratified_bayesian_optimization.lib.util import (
    combine_vectors,
)


class SliceSampling(object):

    def __init__(self, log_prob, indexes, ignore_index=None, **slice_sampling_params):
        """
        For details of the procedure see  Slice Sampling by Radford Neal (2003).

        :param log_prob: function that computes the logarithm of the probability density,
            log_prob(point, *args_log_prob) (the point is the full vector. It doesn't matter
            if we're sampling only a subset of the full vector).
        :param indexes: ([int]) indexes of the parameters to be sampled.
        :param ignore_index: ([int]) we do not move the index of indexes if component_wise is
            selected.
        :param slice_sampling_params:
            - sigma: (float) Parameter to randomly choose the x interval:
                upper ~ U(0, sigma), lower = upper - sigma.

                It's also the step size of the stepping out procedure.

            - step_out: (boolean) If true, the stepping out or doubling procedure is used,
                which finds a "good" x interval.
            - max_steps_out: (int) Maximum number of steps out for the stepping out  or
                doubling procedure.
            - component_wise (boolean) If true, slice sampling is applied to each component of the
                vector point.
            - doubling_step: (boolean) If true, the doubling procedure is used. Otherwise, the
                stepping out procedure is used if ste_out is true.
        """
        if ignore_index is None:
            ignore_index = []
        self.ignore_index = ignore_index

        self.log_prob = log_prob
        self.indexes = indexes
        self.sigma = slice_sampling_params.get('sigma', 1.0)
        self.step_out = slice_sampling_params.get('step_out', True)
        self.max_steps_out = slice_sampling_params.get('max_steps_out', 1000)
        self.component_wise = slice_sampling_params.get('component_wise', True)
        self.doubling_step = slice_sampling_params.get('doubling_step', True)

    def slice_sample(self, point, fixed_parameters, *args_log_prob):
        """
        Same a point from self.log_prob using slice sampling.

        :param point: (np.array(n)) starting point
        :param fixed_parameters: (np.array(l)) values of the parameters that are fixed in the order
            of the model (i.e. variance of noise, mean, parameters of the kernel)
        :param args_log_prob: additional arguments of the log_prob function.
        :return: np.array(n)
        """
        dimensions = len(point)

        if self.component_wise:
            dims = range(dimensions)
            npr.shuffle(dims)
            new_point = point.copy()
            for d in dims:
                if d not in self.ignore_index:
                    direction = np.zeros(dimensions)
                    direction[d] = 1.0
                    new_point = self.direction_slice(direction, new_point, fixed_parameters,
                                                     *args_log_prob)
        else:
            direction = npr.randn(dimensions)
            for d in self.ignore_index:
                direction[d] = 0.0
            if np.all(direction == 0):
                return point

            direction = direction / np.sqrt(np.sum(direction ** 2))
            new_point = self.direction_slice(direction, point, fixed_parameters, *args_log_prob)

        return new_point

    def directional_log_prob(self, x, direction, point, fixed_parameters=None, *args_log_prob):
        """
        Computes log_prob(direction * x + point)
        :param x: (float) magnitude of the movement towards the direction
        :param direction: np.array(n), unitary vector
        :param point: np.array(n)
        :param fixed_parameters: (np.array(l)) values of the parameters that are fixed in the order
            of the model (i.e. variance of noise, mean, parameters of the kernel)
        :return: float
        """
        new_point = point + x * direction


        if fixed_parameters is not None:
            new_point = combine_vectors(new_point, fixed_parameters, self.indexes)

        return self.log_prob(new_point, *args_log_prob)

    def acceptable(self, z, llh, L, U, direction, point, fixed_parameters, *args_log_prob):
        """
        Accepts whether z * direction + point is an acceptable next point, where
        z is in [L, U] and {z: llh < log_prob(direction * z + point)}.

        :param z: float
        :param llh: float
        :param L: (float)
        :param U: (float)
        :param direction: (np.array(n)), unitary vector
        :param point: (np.array(n))
        :param fixed_parameters: (np.array(l)) values of the parameters that are fixed in the order
            of the model (i.e. variance of noise, mean, parameters of the kernel)
        :param args_log_prob: additional arguments of the log_prob function.

        :return: boolean
        """

        if not self.doubling_step:
            return True

        while (U - L) > 1.1 * self.sigma:
            old_U = U
            old_L = L

            middle = 0.5 * (L + U)

            if z < middle:
                U = middle
            else:
                L = middle

            if U == old_U and L == old_L:
                return False

            D = (middle > 0 and z >= middle) or (middle <= 0 and z < middle)

            lp0 = self.directional_log_prob(U, direction, point, fixed_parameters, *args_log_prob)
            lp1 = self.directional_log_prob(L, direction, point, fixed_parameters, *args_log_prob)
            if D and llh >= lp0 and llh >= lp1:
                return False
        return True

    def find_x_interval(self, llh, lower, upper, direction, point, fixed_parameters,
                        *args_log_prob):
        """
        Finds the magnitude of the lower bound and upper bound of a x-interval in slice sampling
        such that:
            0) the interval is defined by
                (new_lower * direction + point, new_upper * direction + point)
            i) containts point
            ii) contains much of the slice {x: y < log_prob(x)}, where
                y ~ uniform(0, log_prob(point))
        We can do it by using the doubling procedure.

        :param llh: (float) llh ~ uniform(0, log_prob(point))
        :param lower: (float) starting magnitude of the lower bound of the x-interval:
            lower * direction + point
        :param upper: (float) starting magnitude of the upper bound of the x-interval:
            upper * direction + point
        :param direction: (np.array(n))
        :param point: (np.array(n))
        :param fixed_parameters: (np.array(l)) values of the parameters that are fixed in the order
            of the model (i.e. variance of noise, mean, parameters of the kernel)
        :param args_log_prob: additional arguments of the log_prob function.

        :return: (float, float) upper, lower
        """
        l_steps_out = 0
        u_steps_out = 0

        if self.doubling_step:
            lp0 = self.directional_log_prob(
                lower, direction, point, fixed_parameters, *args_log_prob)
            lp1 = self.directional_log_prob(
                upper, direction, point, fixed_parameters, *args_log_prob)
            while (lp0 > llh or lp1 > llh) and (l_steps_out + u_steps_out < self.max_steps_out):
                if npr.rand() < 0.5:
                    l_steps_out += 1
                    lower -= (upper - lower)
                    lp0 = self.directional_log_prob(
                        lower, direction, point, fixed_parameters, *args_log_prob)
                else:
                    u_steps_out += 1
                    upper += (upper - lower)
                    lp1 = self.directional_log_prob(
                        upper, direction, point, fixed_parameters, *args_log_prob)
        else:
            lp1 = self.directional_log_prob(
                lower, direction, point, fixed_parameters, *args_log_prob)
            while lp1 > llh and l_steps_out < self.max_steps_out:
                l_steps_out += 1
                lower -= self.sigma
                lp1 = self.directional_log_prob(
                    lower, direction, point, fixed_parameters, *args_log_prob)

            lp2 = self.directional_log_prob(
                upper, direction, point, fixed_parameters, *args_log_prob)
            while lp2 > llh and u_steps_out < self.max_steps_out:
                u_steps_out += 1
                upper += self.sigma
                lp2 = self.directional_log_prob(
                    upper, direction, point, fixed_parameters, *args_log_prob)

        return upper, lower

    def find_sample(self, lower, upper, llh, direction, point, fixed_parameters, *args_log_prob):
        """
        Sample magnitude z to define new sample towards the direction: z * direction + point.
        z is sampled from {x: llh < log_prob(x * direction + point)} intersected with [lower, upper]

        :param lower: (float) magnitude of the lower bound of the x-interval:
            lower * direction + point
        :param upper: (float) magnitude of the upper bound of the x-interval:
            upper * direction + point
        :param llh: (float) llh ~ uniform(0, log_prob(point))
        :param direction: np.array(n)
        :param point: np.array(n)
        :param fixed_parameters: (np.array(l)) values of the parameters that are fixed in the order
            of the model (i.e. variance of noise, mean, parameters of the kernel)
        :param args_log_prob: additional arguments of the log_prob function.

        :return: float
        """
        start_upper = upper
        start_lower = lower

        steps_in = 0
        while True:
            steps_in += 1
            new_z = (upper - lower) * npr.rand() + lower
            new_llh = self.directional_log_prob(new_z, direction, point, fixed_parameters,
                                                *args_log_prob)
            if np.isnan(new_llh):
                new_llh = -np.inf #manual fix
               # raise Exception("Slice sampler got a NaN")

            if new_llh > llh and self.acceptable(new_z, llh, start_lower, start_upper, direction,
                                                 point, fixed_parameters, *args_log_prob):
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        return new_z

    def direction_slice(self, direction, point, fixed_parameters, *args_log_prob):
        """

        Sample a new point by doing slice sampling, and only moving the point towards the
        direction vector.

        :param direction: (np.array(n)) Unitary vector
        :param point: (np.array(n)) starting point
        :param fixed_parameters: (np.array(l)) values of the parameters that are fixed in the order
            of the model (i.e. variance of noise, mean, parameters of the kernel)
        :param args_log_prob: additional arguments of the log_prob function.

        :return: (np.array(n)) Sample a new point
        """

        upper = self.sigma * npr.rand()
        lower = upper - self.sigma
        llh = np.log(npr.rand()) + self.directional_log_prob(0.0, direction, point,
                                                             fixed_parameters, *args_log_prob)
        if self.step_out:
            upper, lower = self.find_x_interval(llh, lower, upper, direction, point,
                                                fixed_parameters, *args_log_prob)

        new_z = self.find_sample(lower, upper, llh, direction, point,
                                 fixed_parameters, *args_log_prob)

        return new_z * direction + point
