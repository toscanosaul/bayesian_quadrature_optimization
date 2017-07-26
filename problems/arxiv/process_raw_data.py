import gzip
import json
import os
from os import path
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.initializers.log import SBOLog
import ujson

logger = SBOLog(__name__)


class ProcessRawData(object):

     _years = [2017, 2016, 2015, 2014, 2013, 2012]
     _data_path = '/data/json/usage/'
     _papers_path = '/data/json/idcat/'
     _store_path = 'problems/arxiv/data/click_data_{year}.json'.format

     @classmethod
     def get_click_data(cls, filenames, store_filename):
          """
          Get click data from filenames. Writes a JSON file with the format:

          {
               'cookie_hash': {'arxiv_id'}
          }

          :param filenames: [str]
          :param store_filename: str

          """
          paper = {}

          process_data = {}

          for filename in filenames:
               logger.info("Processing filename: %s" % filename)

               f = gzip.open(filename, 'rb')

               data = json.load(f)
               entries = data['entries']

               for entry in entries:
                    if 'arxiv_id' in entry and 'cookie_hash' in entry:
                         arxiv_id = entry['arxiv_id']

                         if 'v' in arxiv_id:
                             index = arxiv_id.index('v')
                             arxiv_id = arxiv_id[0: index]

                         user = entry['cookie_hash']

                         if arxiv_id not in paper:
                              paper[arxiv_id] = {'views': 0, 'cat': cls.get_cats(
                                   arxiv_id, arxiv_id[0: 2], arxiv_id[2: 4])}

                         paper[arxiv_id]['views'] += 1

                         if user not in process_data:
                              process_data[user] = {}
                              process_data[user][arxiv_id] = 0
                         elif arxiv_id not in process_data[user]:
                              process_data[user][arxiv_id] = 0
                         process_data[user][arxiv_id] += 1

          JSONFile.write(process_data, store_filename)

     @classmethod
     def generate_filenames_year(cls, year):
          """
          Generate all file names of one year
          :param year: (int)
          :return: [str]
          """
          data_path = path.join(cls._data_path, str(year))

          files = []

          for (dirpath, dirnames, filenames) in os.walk(data_path):
               files = filenames

          files = [data_path + f for f in files]

          return files


     @classmethod
     def get_cats(cls, arxiv_id, year, month):
          """
          Get category of a file

          :param arxiv_id: (str)
          :param year: (str) e.g. '07', '10'
          :param month: (str) e.g. '12', '02'
          :return: str

          """

          filename = os.join(cls._papers_path, '20' + year)
          date = year + month

          for day in xrange(1, 32):
               if day < 10:
                    date = date + '0' + day
               else:
                    date = date + day
          filename_ = os.join(filename, date)

          data = None
          if path.exists(filename_):
               with open(filename) as f:
                    data = ujson.load(f)

          if data is None:
               logger.info("Couldn't find category of paper %s" % arxiv_id)

          for dicts in data['new']:
               if dicts['id'] == arxiv_id:
                    cats = [a.lower() for a in dicts["cat"].split(":")]
                    break

          return cats[0]
