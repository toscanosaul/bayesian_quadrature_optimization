from problems.arxiv.process_raw_data import ProcessRawData


if __name__ == '__main__':
    # python -m problems.arxiv.scripts.run_year_data

    files = ProcessRawData.generate_filenames_year(2016)
    ProcessRawData.get_click_data(
        files,"problems/arxiv/data/2016_processed_data.json")