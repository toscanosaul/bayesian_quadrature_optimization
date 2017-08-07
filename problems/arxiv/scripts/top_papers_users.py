from problems.arxiv.statistics_processed_data import StatisticstProcessedData

import argparse

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


if __name__ == '__main__':
    # python -m problems.arxiv.scripts.top_papers_users '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('month', help='e.g. 23, 1')
    args = parser.parse_args()
    month = args.month
    year = '2016'


    StatisticstProcessedData.top_users_papers(year, month)
