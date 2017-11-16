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
    # Repeat each entry of spec n times
    # Example usage:
    # python -m scripts.repeat_spec name_file.json n_times output

    parser = argparse.ArgumentParser()
    parser.add_argument('multiple_spec', help='e.g. test_multiple_spec.json')
    parser.add_argument('n', help='e.g. 5')
    parser.add_argument('output', help='output.json')

    args = parser.parse_args()

    spec_1 = args.multiple_spec
    n = int(args.n)

    with open(path.join(MULTIPLESPECS_DIR, spec_1)) as specfile:
        spec = ujson.load(specfile)

    new_spec = {}

    n_specs = len(spec.keys())
    keys = set(spec.keys())

    for key in keys:
        values = spec[key]
        new_entry = []
        for i in xrange(len(values)):
            new_entry += n * [values[i]]
        new_spec[key] = new_entry

    output = args.output
    JSONFile.write(new_spec, path.join(MULTIPLESPECS_DIR, output))