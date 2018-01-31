from __future__ import absolute_import

from os import path

import argparse
import ujson

from stratified_bayesian_optimization.lib.constant import (
    SPECS_DIR,
    MULTIPLESPECS_DIR,
    DEFAULT_RANDOM_SEED,
    LBFGS_NAME,
    DOGLEG,
)
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.entities.run_spec import MultipleSpecEntity

if __name__ == '__main__':
    # Example usage:
    # python -m scripts.combine_multiple_specs arxiv_10_training_random_seeds.json  multiple__spec_arxiv_rs_500.json

    parser = argparse.ArgumentParser()
    parser.add_argument('multiple_spec_2', help='e.g. test_multiple_spec.json')
    parser.add_argument('multiple_spec_1', help='e.g. test_multiple_spec.json')
    parser.add_argument('output_file', help='e.g. combine_multiple_spec.arxiv.json')

    args = parser.parse_args()

    spec_1 = args.multiple_spec_1
    spec_2 = args.multiple_spec_2
    output = args.output_file

    with open(path.join(MULTIPLESPECS_DIR, spec_1)) as specfile:
        spec_1 = ujson.load(specfile)

    with open(path.join(MULTIPLESPECS_DIR, spec_2)) as specfile:
        spec_2 = ujson.load(specfile)

    new_spec = {}

    n_specs = len(spec_1.keys())

    keys = set(spec_1.keys() + spec_2.keys())

    if spec_1.get('random_seeds') is None:
        spec_1['random_seeds'] = 3 * [1]

    rs_1 = len(spec_1['random_seeds'])
    rs_2 = len(spec_2['random_seeds'])


    for key in keys:
        values_1 = None
        values_2 = None
        if key in spec_1:
            values_1 = spec_1[key]
        if key in spec_2:
            values_2 = spec_2[key]
        if values_1 is None:
            values_1 = rs_1 * [None]

        if values_2 is None:
            values_2 = rs_2 * [None]

        new_spec[key] = []
        for i in xrange(max(len(values_1), len(values_2))):
            if i < len(values_1):
                value_1 = values_1[i]
                new_spec[key] += [value_1]
            if i < len(values_2):
                value_2 = values_2[i]
                new_spec[key] += [value_2]

    # for key in spec_1:
    #     value_1 = spec_1[key]
    #     value_2 = spec_2[key]
    #     new_spec[key] = value_1 + value_2

    JSONFile.write(new_spec, path.join(MULTIPLESPECS_DIR, output))