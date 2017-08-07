import gzip
import json
import os
from os import path
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.initializers.log import SBOLog
import ujson
from bisect import bisect_left

logger = SBOLog(__name__)


class StatisticstProcessedData(object):

    _name_file_ = 'problems/arxiv/data/{year}_{month}_processed_data.json'.format
    _name_file_final = 'problems/arxiv/data/{year}_{month}_top_users.json'.format

    @classmethod
    def top_users_papers(cls, year, month, n_entries=100, different_papers=20, top_n=5000):
        """
        Returns the users that accessed to at least n_entries papers, and at least different_papers
        were different and were in the top_n papers in the month of the year.

        Returns the top_n papers based on how many times they were seen.

        :param year: (str)
        :param month: (str) e.g. '1', '12'
        :param n_entries: (int) n_
        :return: [ {'paper': (int) number of times seen},
            {'user': {'stats': ((int) # entries, (int) # different papers in the top_n papers),
                      'diff_papers': [str]
                }
            }
        ]
        """

        file_name = cls._name_file_(year=year, month=month)
        data = JSONFile.read(file_name)

        users = data[0]
        papers = data[1]

        n_papers = []
        paper_ls = []
        for paper in papers:
            paper_ls.append(paper)
            n_papers.append(papers[paper]['views'])
        index_top_papers = sorted(range(len(n_papers)), key=lambda k: n_papers[k])
        index_top_papers = index_top_papers[-top_n:]

        rank_papers = {}
        for index in index_top_papers:
            rank_papers[paper_ls[index]] = n_papers[index]

        paper_ls = rank_papers.keys()
        rank_user = {}

        users_ls = []
        n_entries_ls = []

        for user in users:
            users_ls.append(user)
            n_entries_ls.append(sum(users[user].values()))

        index_top_users = sorted(range(len(n_entries_ls)), key=lambda k: n_entries_ls[k])
        users_ls = [users_ls[i] for i in index_top_users]
        n_entries_ls = [n_entries_ls[i] for i in index_top_users]
        ind_bis = bisect_left(n_entries_ls, n_entries)

        users_ls = users_ls[ind_bis:]
        n_entries_ls = n_entries_ls[ind_bis:]

        for user, n in zip(users_ls, n_entries_ls):
            diff_papers = set(users[user].keys()).intersection(set(paper_ls))
            n_diff = len(diff_papers)
            if n_diff < different_papers:
                continue
            rank_user[user] = {'stats': (n, n_diff), 'diff_papers': diff_papers}

        file_name = cls._name_file_final(year=year, month=month)
        JSONFile.write(file_name)

        return [rank_papers, rank_user]


    def top_papers_year(self, n, year):
        """
        Returns the top n papers seen in the year
        :param n: int
        :param year: str
        :return: n * [str]
        """
        pass
