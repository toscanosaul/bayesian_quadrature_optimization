from __future__ import absolute_import


def toy_example(x):
    """

    :param x: [float, int]
    :return: [float]
    """
    tasks = x[1]
    value = x[0]
    if tasks == 0:
        value += 10
    else:
        value -= 10

    return [value]

def integrate_toy_example(x):
    """

    :param x: [float]
    :return: [float]
    """
    point = [x[0], 0]
    a = toy_example(point)

    point = [x[0], 1]
    b = toy_example(point)

    return [(a[0] + b[0]) / 2.0]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)

