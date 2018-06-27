from __future__ import absolute_import

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit, leastsq, fmin_bfgs, fmin_l_bfgs_b, nnls
import sys
import os
import matplotlib.pyplot as plt

from multi_start.parametric_functions import ParametricFunctions
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.samplers.slice_sampling import SliceSampling
from multi_start.parametric_functions import ParametricFunctions


class ParametricModel(object):
    # Implementation of 'Speeding up Automatic Hyperparameter Optimization of Deep Neural Networks
    # by Extrapolation of Learning Curves'.

    def __init__(self,raw_results, best_result, current_iteration, get_value_next_iteration,
                 starting_point, current_batch_index, current_epoch,
                 kwargs_get_value_next_iteration=None,
                 problem_name=None, specifications=None, max_iterations=1000,
                 lower=None, upper=None, total_batches=10, n_burning=500, n_thinning=10,
                 model_gradient='real_gradient', burning=True):
        self.specifications = specifications
        self.starting_point = starting_point
        self.current_point = starting_point
        self.current_batch_index = current_batch_index
        self.total_batches = total_batches
        self.current_epoch = current_epoch
        self.n_burning = n_burning
        self.n_thinning = n_thinning
        self.burning = burning

        self.model_gradient = model_gradient

        self.raw_results = raw_results #dictionary with points and values, and gradients
        self.best_result = best_result
        self.current_iteration = current_iteration

        self.parametrics = ParametricFunctions(max_iterations, lower, upper)

        # self.differences = None
        # self.points_differences = None
        self.training_data = None
        self.process_data()

        self.problem_name = problem_name
        self.max_iterations = max_iterations

        self.sampler = None
        self.samples_parameters = None
        self.set_samplers()

        self.get_value_next_iteration = get_value_next_iteration
        self.kwargs = {}
        if kwargs_get_value_next_iteration is not None:
            self.kwargs = kwargs_get_value_next_iteration

    def process_data(self):
        self.training_data = self.raw_results['values']

    def log_prob_slice(self, params, training_data):
        return self.parametrics.log_prob(params[0:-1], params[-1], training_data)

    def set_samplers(self):
        slice_parameters = {
            'max_steps_out': 1000,
            'component_wise': False,
        }
        n_params = self.parametrics.n_weights + self.parametrics.n_parameters + 1

        sampler = SliceSampling(self.log_prob_slice, range(0, n_params), **slice_parameters)

        self.sampler = sampler

        start = self.parametrics.get_starting_values_mle(self.training_data)

        samples_parameters = []
        samples_parameters.append(start)
        if self.n_burning > 0:
            parameters = self.sample_parameters(float(self.n_burning) / (self.n_thinning + 1), start, sampler,
                                                self.training_data)
            samples_parameters = []
            samples_parameters.append(parameters[-1])

        self.samples_parameters = samples_parameters


    def sample_parameters(self, n_samples, start_point, sampler, historical_data, random_seed=None):
        """
        Sample parameters of the model from the posterior without considering burning.

        :param n_samples: (int)
        :param start_point: np.array(n_parameters)
        :param random_seed: intf

        :return: n_samples * [np.array(float)]
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        samples = []

        n_samples *= (self.n_thinning + 1)
        n_samples = int(n_samples)

        for sample in range(n_samples):
            start_point_ = None
            n_try = 0
            while start_point_ is None and n_try < 10:
                #  start_point_ = sampler.slice_sample(start_point, None)
                start_point_ = sampler.slice_sample(start_point, None, *(historical_data,))
                try:
                    start_point_ = sampler.slice_sample(start_point, None, *(historical_data, ))
                except Exception as e:
                    n_try += 1
                    start_point_ = None
            if start_point_ is None:
                print('program failed to compute a sample of the parameters')
                sys.exit(1)
            start_point = start_point_
            samples.append(start_point)
        return samples[::self.n_thinning + 1]


    def compute_posterior_params_marginalize(self, n_samples=10, burning_parameters=True, get_vectors=True):
        if burning_parameters:
            parameters = self.sample_parameters(
                float(self.n_burning) / (self.n_thinning + 1), self.samples_parameters[-1],self.sampler, self.training_data)
            self.samples_parameters = []
            self.samples_parameters.append(parameters[-1])
            self.start_point_sampler = parameters[-1]


        parameters = self.sample_parameters(n_samples,self.samples_parameters[-1],self.sampler, self.training_data)

        means = []
        stds = []
        for s in parameters:
            mean = self.weighted_combination(self.max_iterations, s[0:self.n_functions], s[self.n_functions:-1])
            std = s[-1]
            means.append(mean)
            stds.append(std)
            # val = 1.0 - norm.cdf(y, loc=mean, scale=std)
            # values.append(val)

        if get_vectors:
            return {'means': means, 'stds': stds}


    def add_observations(self, point, y, new_point_in_domain=None):

        self.current_iteration = point
        self.best_result = max(self.best_result, y)

        self.training_data.append(y)
        self.raw_results['values'].append(y)

        if new_point_in_domain is not None:
            self.raw_results['points'].append(new_point_in_domain)

        if self.current_batch_index + 1 > self.total_batches:
            self.current_epoch += 1

        self.current_batch_index = (self.current_batch_index + 1) % self.total_batches

        if new_point_in_domain is not None:
            self.current_point = new_point_in_domain







