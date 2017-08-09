from problems.arxiv.generate_training_data import TrainingData

import argparse

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


if __name__ == '__main__':
    # python -m problems.arxiv.scripts.run_training_data '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('month', help='e.g. 23, 1')
    args = parser.parse_args()
    month = args.month
    year = '2016'

    TrainingData.get_training_data(year, month)