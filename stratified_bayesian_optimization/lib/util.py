from __future__ import absolute_import

from collections import defaultdict

import numpy as np


def convert_dictionary_gradient_to_simple_dictionary(dictionary, order_keys):
    """

    :param dictionary: {
            (str) name: {(int): nxn} or nxn,
        }
    :param order_keys: ([(name_key, keys)]) (keys can be other list if we have a dictionary of
        dictionaries, otherwise keys=None)
    :return: {
        (int) : nxn
    } the indexes are set based on order_keys.
    """

    result = []

    for key in order_keys:
        if key[1] is None:
            result += [np.array(dictionary[key[0]])]
        else:
            for sub_key in key[1]:
                result += [np.array(dictionary[key[0]][sub_key[0]])]

    simple_dictionary = defaultdict()

    for index, element in enumerate(result):
        simple_dictionary[index] = element

    return simple_dictionary
