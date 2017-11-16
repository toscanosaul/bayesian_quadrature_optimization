from stratified_bayesian_optimization.util.json_file import JSONFile
from os import path
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.initializers.log import SBOLog
import ujson

logger = SBOLog(__name__)

class Categories(object):

    _name_file_ = 'problems/arxiv/data/{year}_{month}_processed_data.json'.format
    _name_file_categories = 'problems/arxiv/data/{year}_{month}_papers.json'.format
    _name_file_final = 'problems/arxiv/data/{year}_{month}_top_users.json'.format
    _name_file_final_categ = 'problems/arxiv/data/{year}_{month}_top_users_top_' \
                             'categories.json'.format
    _papers_path = '/data/json/idcat/'
    _histogram_papers = 'problems/arxiv/data/{year}_{month}_histogram_papers.pdf'.format
    _histogram_users = 'problems/arxiv/data/{year}_{month}_histogram_users.pdf'.format
    _name_file_categories_users = 'problems/arxiv/data/{year}_{month}_users_papers.json'.format
    _name_file_categories_users_hist = \
        'problems/arxiv/data/{year}_{month}_users_papers_hist.json'.format

    @classmethod
    def assign_categories_date_year(cls, year, month):
        """
        :param year: (str)
        :param month: (str) e.g. '1', '12'
        :return:
        """

        file_name = cls._name_file_final(year=year, month=month)
        data = JSONFile.read(file_name)
        papers = data[0].keys()
        papers = cls.assign_categories(papers, year, month)
        return papers


    @classmethod
    def assign_categories(cls, list_papers, year, month):
        """

        :param list_papers: [str]
        :return: {paper_name (str):  category (str)}
        """
        papers = {}
        for paper in list_papers:

            before_2007 = False
            arxiv_id = paper

            if '/' in arxiv_id:
                before_2007 = True
                index = arxiv_id.index('/')
                cat = arxiv_id[0: index]
                arxiv_id = arxiv_id[index + 1:]

            if 'v' in arxiv_id:
                index = arxiv_id.rfind('v')
                arxiv_id = arxiv_id[0: index]

            if not before_2007:
                cat = cls.get_cats(arxiv_id, arxiv_id[0: 2], arxiv_id[2: 4])

            papers[paper] = cat

        JSONFile.write(papers, cls._name_file_categories(year=year, month=month))
        return papers


    @classmethod
    def get_cats(cls, arxiv_id, year, month):
        """
        Get category of a file

        :param arxiv_id: (str)
        :param year: (str) e.g. '07', '10'
        :param month: (str) e.g. '12', '02'
        :return: str

        """

        filename = path.join(cls._papers_path, '20' + year)
        date = year + month

        cats = None

        for day in xrange(1, 32):
            if day < 10:
                date_ = date + '0' + str(day)
            else:
                date_ = date + str(day)

            date_ += '_idcat.json'
            filename_ = path.join(filename, date_)

            data = None
            if path.exists(filename_):
                with open(filename_) as f:
                    data = ujson.load(f)

            if data is not None:
                for dicts in data['new']:
                    if dicts['id'] == arxiv_id:
                        cats = [a.lower() for a in dicts["cat"].split(":")]
                        break

            if cats is not None:
                return cats[0]

        new_month = int(month) + 1

        if new_month == 13:
            new_month = 1
            year = int(year) + 1
            if year < 10:
                year = '0' + str(year)
            else:
                year = str(year)

        if new_month < 10:
            new_month = '0' + str(new_month)
        else:
            new_month = str(new_month)

        filename = path.join(cls._papers_path, '20' + year)

        for day in xrange(1, 10):
            date = year + new_month + '0' + str(day) + '_idcat.json'
            filename_ = path.join(filename, date)

            data = None
            if path.exists(filename_):
                with open(filename_) as f:
                    data = ujson.load(f)

            if data is not None:
                for dicts in data['new']:
                    if dicts['id'] == arxiv_id:
                        cats = [a.lower() for a in dicts["cat"].split(":")]
                        break

            if cats is not None:
                return cats[0]

        if cats is None:
            logger.info("Couldn't find category of paper %s" % arxiv_id)

        return cats