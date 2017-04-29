from __future__ import absolute_import

from os import path
import ujson

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


class JSONFile(object):

    @staticmethod
    def read(filename):
        """
        Read json file or return None.
        :param filename: (str)

        :return: content or None
        """

        if path.exists(filename):
            logger.info('Loading %s' % filename)
            with open(filename) as f:
                return ujson.load(f)

        return None

    @staticmethod
    def write(data, filename):
        """
        Write data into file

        :param data: content
        :param filename: str
        """
        with open(filename, 'w') as f:
            ujson.dump(data, f)
