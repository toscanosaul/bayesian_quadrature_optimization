from problems.arxiv.statistics_processed_data import StatisticsProcessedData

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

    # StatisticsProcessedData.top_users_papers(year, month)
    StatisticsProcessedData.top_users_papers_selecting_categories(year, month)
