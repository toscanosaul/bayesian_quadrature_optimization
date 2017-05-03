from __future__ import absolute_import

import numpy as np
import numpy.random as npr


class SliceSampling(object):

    def __init__(self, log_prob, **slice_sampling_params):
        """

        :param log_prob: function that computes the logarithm of the probability density,
            log_prob(point, *args_log_prob)
        :param thinning
        :param slice_sampling_params:
            - sigma
            - step_out
            - max_steps_out
            - component_wise (boolean) If true, slice sampling is applied to each component of the
                vector point.
            - doubling_step
        """
        self.log_prob = log_prob
        self.sigma = slice_sampling_params.get('sigma', 1.0)
        self.step_out = slice_sampling_params.get('step_out', True)
        self.max_steps_out = slice_sampling_params.get('max_steps_out', 1000)
        self.component_wise = slice_sampling_params.get('component_wise', True)
        self.doubling_step = slice_sampling_params.get('doubling_step', True)

    def sample(self, model):
        """
        Sample parameters of the model from their posterior.

        :param model: gp_fitting_gaussian

        :return: np.array(n)
        """
        parameters = model.get_value_parameters_model

        for i in xrange(self.thinning + 1):
            parameters =  self.slice_sample(
                parameters
            )

        return parameters

    def slice_sample(self, point):
        """
        Same a point from a probability density using slice sampling.

        :param point: (np.array(n)) starting point
        :return: np.array(n)
        """

        dimensions = len(point)

        if self.component_wise:
            dims = range(dimensions)
            npr.shuffle(dims)
            new_point = point.copy()
            for d in dims:
                direction = np.zeros(dimensions)
                direction[d] = 1.0
                new_point = self.direction_slice(direction, new_point)
        else:
            direction = npr.randn(dimensions)
            direction = direction / np.sqrt(np.sum(direction ** 2))
            new_point = self.direction_slice(direction, point)

        return new_point

    def directional_log_prob(self, x, direction, point):
        """
        Computes log_prob(direction * x + point)
        :param x: (float) magnitude of the movement towards the direction
        :param direction: np.array(n), unitary vector
        :param point: np.array(n)
        :return: float
        """
        return self.log_prob(point + x * direction)

    def acceptable(self, z, llh, L, U, direction, point):
        """
        :param z:
        :param llh: (float) value of log_likelihood
        :param L: (float)
        :param U: (float)
        :param direction: (np.array(n)), unitary vector
        :param point: (np.array(n))
        :return: boolean
        """
        while (U - L) > 1.1 * self.sigma:
            middle = 0.5 * (L + U)

            if z < middle:
                U = middle
            else:
                L = middle

            is_split = (middle > 0 and z >= middle) or (middle <= 0 and z < middle)

            if is_split and llh >= self.directional_log_prob(U, direction, point) and \
                            llh >= self.directional_log_prob(L, direction, point):
                return False
        return True

    def find_x_interval(self, llh, lower, upper, direction, point):
        """

        :return:
        """
        l_steps_out = 0
        u_steps_out = 0

        if self.doubling_step:
            while (self.directional_log_prob(lower, direction, point) > llh or
                           self.directional_log_prob(upper, direction, point) > llh) and \
                    (l_steps_out + u_steps_out < self.max_steps_out):
                if npr.rand() < 0.5:
                    l_steps_out += 1
                    lower -= (upper - lower)
                else:
                    u_steps_out += 1
                    upper += (upper - lower)
        else:
            while self.directional_log_prob(lower, direction, point) > llh and \
                            l_steps_out < self.max_steps_out:
                l_steps_out += 1
                lower -= self.sigma
            while self.directional_log_prob(upper, direction, point) > llh and \
                            u_steps_out < self.max_steps_out:
                u_steps_out += 1
                upper += self.sigma

        return upper, lower

    def find_sample(self, lower, upper, llh, direction, point):
        """

        :param lower:
        :param upper:
        :param llh:
        :param direction:
        :param point:
        :return:
        """
        start_upper = upper
        start_lower = lower

        steps_in = 0
        while True:
            steps_in += 1
            new_z = (upper - lower) * npr.rand() + lower
            new_llh = self.directional_log_prob(new_z, direction, point)

            if np.isnan(new_llh):
                raise Exception("Slice sampler got a NaN")

            if new_llh > llh and self.acceptable(new_z, llh, start_lower, start_upper, direction,
                                                 point):
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        return new_z


    def direction_slice(self, direction, point):
        """

        :param direction:
        :param point:

        :return:
        """
        upper = self.sigma * npr.rand()
        lower = upper - self.sigma
        llh = np.log(npr.rand()) + self.directional_log_prob(0.0, direction, point)

        if self.step_out:
            upper, lower = self.find_x_interval(llh, lower, upper, direction, point)

        new_z = self.find_sample(lower, upper, llh, direction, point)

        return new_z * direction + point

