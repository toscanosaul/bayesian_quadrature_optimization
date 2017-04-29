from __future__ import absolute_import

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class SBOLog(object):

    def __init__(self, name):
        self._log = logging.getLogger(name)

    def info(self, msg, *args, **kwargs):
        """
        :param msg: dict
        """

        self._log.info(msg, *args, **kwargs)
