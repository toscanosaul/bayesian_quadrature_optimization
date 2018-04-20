from __future__ import absolute_import

import numpy as np
from scipy.stats import norm

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.util.json_file import JSONFile

logger = SBOLog(__name__)


class GreedyPolicy(object):

    def __init__(self, dict_stat_models, forward_point, epsilon=0.01, total_iterations=100,
                 a_learning_rate=0.5, total_batches=10):
        self.dict_stat_models = dict_stat_models
        self.epsilon = epsilon
        self.total_iterations = total_iterations
        self.forward_point = forward_point
    #    self.a_learning_rate = a_learning_rate
        self.total_batches = total_batches
        self.chosen_points = []


    def get_current_best_value(self):
        best_values = []
        for model in self.dict_stat_models:
            best_values.append(model.gp_model.best_result)
        return np.max(best_values)

    def probability_being_better(self, n_samples=10):
        y = self.get_current_best_value()

        probabilities = {}
        for index in self.dict_stat_models:
            model = self.dict_stat_models[index]
            params = model.compute_posterior_params_marginalize(model.gp_model, n_samples=n_samples)
            mean = params[0]
            std = params[1]

            val = 1.0 - norm.cdf(y + self.epsilon, loc=mean, scale=std)
            probabilities[index] = val

        return probabilities

    def choose_move_point(self, n_samples=10):
        probabilites = self.probability_being_better(n_samples=n_samples)

        point_ind = max(probabilites, key=probabilites.get)

        move_model = self.dict_stat_models[point_ind]
        current_point = move_model.current_point

        self.chosen_points.append(point_ind)

        current_iteration = move_model.current_iteration

        # lr = float(current_iteration) / self.a_learning_rate
        # lr = 1.0 / (1.0 + lr)

        batch_index = move_model.current_batch_index

        new_point, new_value = self.forward_point(
            epoch=move_model.epoch, batch_index=batch_index, current_point=current_point)
        move_model.current_point = new_point
        move_model.current_iteration += 1

        if move_model.current_batch_index + 1 > self.total_batches:
            move_model.current_epoch += 1

        move_model.current_batch_index = (move_model.current_batch_index + 1) % self.total_batches



        move_model.add_observations(move_model.gp_model, current_iteration, new_value)

        logger.info('Point chosen is: ')
        logger.info(move_model.starting_point)
        logger.info('value is: ')
        logger.info(new_value)

        self.save_data()


    def save_data(self, sufix=None):
        data = {}
        data['chosen_points'] = self.chosen_points

        file_name = 'data/multi_start/chosen_points'

        if sufix is not None:
            file_name += '_' + sufix

        JSONFile.write(data, file_name + '.json')

        for i in self.dict_stat_models:
            model = self.dict_stat_models[i]
            model.save_model(str(i))
