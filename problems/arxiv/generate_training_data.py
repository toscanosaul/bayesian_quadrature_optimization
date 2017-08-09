import gzip
import json
import os
from os import path
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.initializers.log import SBOLog
import ujson
from bisect import bisect_left
import numpy as np
import random

logger = SBOLog(__name__)


class TrainingData(object):
    _name_file_final = 'problems/arxiv/data/{year}_{month}_top_users.json'.format
    _name_training_data = 'problems/arxiv/data/{year}_{month}_training_data.json'.format
    _name_fold_data_training = 'problems/arxiv/data/{year}_{month}_fold_{fold}_training_data' \
                               '.json'.format
    _name_fold_data_validation = 'problems/arxiv/data/{year}_{month}_fold_{fold}_validation_data' \
                               '.json'.format
    _name_fold_indexes = 'problems/arxiv/data/{year}_{month}_fold_indexes' \
                               '.json'.format



    @classmethod
    def get_training_data(cls, year, month):
        """
        Creates a file with the training data:
            [[user_id, paper_id, rating]], where rating is 1 if the paper wasn't seen by the user,
            or 2 otherwise.

        :param year: str
        :param month: str (e.g. '1', '12')

        """
        file_name = cls._name_file_final(year=year, month=month)
        data = JSONFile.read(file_name)

        papers = data[0].keys()

        users_data = data[1]
        users = users_data.keys()

        training_data = []

        key_paper = {}
        for i, paper in enumerate(papers):
            key_paper[paper] = i

        for i, user in enumerate(users):
            for paper in users_data[user]['diff_papers']:
                training_data.append([i, key_paper[paper], 2])
            for paper in (set(papers) - set(users_data[user]['diff_papers'])):
                training_data.append([i, key_paper[paper], 1])

        file_name = cls._name_training_data(year=year, month=month)
        JSONFile.write(training_data, file_name)

    @classmethod
    def cv_data_sets(cls, year, month, n_folds=5, random_seed=1):
        """
        Creates n_folds files with pairs of datasets: (training_data, validation_data).

        :param year: str
        :param month: str (e.g. '1', '12')

        """
        random.seed(random_seed)

        file_name = cls._name_training_data(year=year, month=month)
        data = JSONFile.read(file_name)

        indexes_data = range(len(data))
        random.shuffle(indexes_data)

        n_batch = len(indexes_data) / n_folds
        random_indexes = [indexes_data[i * n_batch: n_batch + i * n_batch] for i in xrange(n_folds)]

        extra = 0
        for j in xrange(len(indexes_data) % n_folds):
            random_indexes[j].append(indexes_data[n_batch + extra + (n_folds - 1) * n_batch])
            extra += 1

        file_name = cls._name_fold_indexes(year=year, month=month)
        JSONFile.write(random_indexes, file_name)

        # for i in xrange(n_folds):
        #     validation = [data[index] for index in random_indexes[i]]
        #
        #     training_indexes = []
        #     for j in xrange(n_folds):
        #         if j != i:
        #             training_indexes += random_indexes[j]
        #
        #     training = [data[index] for index in training_indexes]
        #
        #     file_name = cls._name_fold_data_training(year=year, month=month, fold=i)
        #     JSONFile.write(training, file_name)
        #
        #     file_name = cls._name_fold_data_validation(year=year, month=month, fold=i)
        #     JSONFile.write(validation, file_name)

