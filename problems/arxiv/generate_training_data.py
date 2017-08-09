import gzip
import json
import os
from os import path
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.initializers.log import SBOLog
import ujson
from bisect import bisect_left
import numpy as np

logger = SBOLog(__name__)


class TrainingData(object):
    _name_file_final = 'problems/arxiv/data/{year}_{month}_top_users.json'.format
    _name_training_data = 'problems/arxiv/data/{year}_{month}_training_data.json'.format

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
