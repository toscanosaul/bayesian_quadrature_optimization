from __future__ import absolute_import

import numpy as np
from copy import deepcopy

def toy_example(x, cuda=False):
    """

    :param x: [int, float, int, int, int]
    :return: [float]

    """

    n_epochs = max(int(x[0]), 1)
    batch_size = max(int(x[1]), 4)
    lr = x[2]
    number_chanels_first = max(int(x[3]), 3)
    number_hidden_units = max(int(x[4]), 100)
    size_kernel = max(int(x[5]), 2)
    task = int(x[6])

    training, validation = get_training_test(task)

    val = train_nn(
        random_seed=1, n_epochs=n_epochs, batch_size=batch_size, lr=lr, weight_decay=0.0,
        number_chanels_first=number_chanels_first, number_hidden_units=number_hidden_units,
        size_kernel=size_kernel, cuda=cuda, trainset=training, testset=validation)

    val = -1.0 * val
    return [val]


def integrate_toy_example(x):
    """

    :param x: [float, float, int, int]
    :return: [float]
    """

    points = {}
    for task in xrange(n_folds):
        point = deepcopy(x)
        point.append(task)
        points[task] = point
        # val = toy_example(point)
        # values.append(val[0])

    errors = Parallel.run_function_different_arguments_parallel(
        toy_example, points, parallel=False)

    values = convert_dictionary_to_list(errors)

    return [np.mean(np.array(values))]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)