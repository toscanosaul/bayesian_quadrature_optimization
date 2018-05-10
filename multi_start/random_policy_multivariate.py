from __future__ import absolute_import

import numpy as np
import os
from scipy.stats import norm
from scipy.stats import foldnorm

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.util.json_file import JSONFile

logger = SBOLog(__name__)


class RandomPolicy(object):

    def __init__(self, dict_stat_models, name_model, problem_name, type_model='grad_epoch', n_epochs=1,
                 stop_iteration_per_point=100, random_seed=None, n_restarts=None):
        self.dict_stat_models = dict_stat_models
        self.points_index = range(len(self.dict_stat_models))
        self.current_index = 0

        self.type_model = type_model
        self.problem_name = problem_name

        self.name_model = name_model
        self.n_epochs = n_epochs

        self.chosen_points = {}
        self.evaluations_obj = {}

        self.stop_iteration_per_point = stop_iteration_per_point

        self.chosen_index = []

        for i in dict_stat_models:
            self.chosen_points[i] = list(dict_stat_models[i].gp_model[0].raw_results['points'])
            self.evaluations_obj[i] = list(dict_stat_models[i].gp_model[0].raw_results['values'])

        self.random_seed = random_seed
        self.n_restarts = n_restarts
        np.random.seed(random_seed)


    def get_current_best_value(self):
        best_values = []
        for index in self.dict_stat_models:
            best_values.append(self.dict_stat_models[index].raw_results['values'][-1])
        return np.max(best_values)

    def choose_move_point(self):
        point_ind = np.random.randint(0, self.n_restarts)
        move_model = self.dict_stat_models[point_ind]

        for t in range(self.n_epochs):
            current_point = move_model.current_point
            i = move_model.gp_model[0].current_iteration
            data_new = move_model.get_value_next_iteration(i + 1, **move_model.kwargs)

            type_model = self.type_model

            self.chosen_index.append(point_ind)
            self.chosen_points[point_ind].append(data_new['point'])
            self.evaluations_obj[point_ind].append(data_new['exact_value'])

            for j in move_model.gp_model:
                move_model.gp_model[j].current_iteration += 1

            # move_model.add_observations(move_model.gp_model, i + 1, data_new['value'],
            #                             data_new['point'], data_new['gradient'], type_model)

            logger.info('Point chosen is: ')
            logger.info(point_ind)
            logger.info(move_model.starting_point)
            logger.info('value is: ')
            logger.info(data_new['value'])

       # self.current_index = (self.current_index + 1) % len(self.dict_stat_models)
        self.save_data()

    def run_policy(self, number_iterations=100):
        steps = number_iterations / self.n_epochs
        for i in range(steps):
            self.choose_move_point()
            self.save_data()
        logger.info('best_solution is: ')
        logger.info(self.get_current_best_value())

    def save_data(self, sufix=None):
        data = {}
        data['chosen_points'] = self.chosen_points
        data['evaluations'] = self.evaluations_obj
        data['chosen_index'] = self.chosen_index

        file_name = 'data/multi_start/'

        file_name += self.problem_name + '/'

        if sufix is None:
            sufix = self.name_model

        if not os.path.exists(file_name):
            os.mkdir(file_name)

        file_name += 'random_policy' + '/'

        if not os.path.exists(file_name):
            os.mkdir(file_name)

        file_name += sufix

        if self.random_seed is not None:
            file_name += '_random_seed_' + str(self.random_seed)

        if self.n_restarts is not None:
            file_name += '_n_restarts_' + str(self.n_restarts)

        JSONFile.write(data, file_name + '.json')
        #
        # for i in self.dict_stat_models:
        #     model = self.dict_stat_models[i]
        #     model.save_model(str(i))
