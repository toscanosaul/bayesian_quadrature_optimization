from problems.arxiv.generate_training_data import TrainingData

import argparse

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


if __name__ == '__main__':
    # python -m problems.arxiv.scripts.run_create_folds '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('month', help='e.g. 23, 1')
    args = parser.parse_args()
    month = args.month
    year = '2016'

    TrainingData.cv_data_sets(year, month)