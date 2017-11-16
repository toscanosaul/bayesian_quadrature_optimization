from problems.arxiv.process_raw_data import ProcessRawData

import argparse


if __name__ == '__main__':
    # python -m problems.arxiv.scripts.run_process_data
    parser = argparse.ArgumentParser()
    parser.add_argument('month', help='e.g. 23')
    args = parser.parse_args()

    month = int(args.month)

    ProcessRawData.get_click_data(
        ['/data/json/usage/2017/170203_usage.json.gz'],"test.json")