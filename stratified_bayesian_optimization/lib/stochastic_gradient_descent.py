from __future__ import absolute_import



import numpy as np

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


def SGD(start, gradient, n, args=(), kwargs={}, bounds=None, learning_rate=0.1, momentum=0.5,
        maxepoch=250):
    """
    SGD to minimize sum(i=0 -> n) (1/n) * f(x). Batch sizes are of size 1.
    :param start: np.array(n)
    :param gradient:
    :param n:
    :param learning_rate:
    :param momentum:
    :param maxepoch:
    :param args: () arguments for the gradient
    :param kwargs:
    :param bounds: [(min, max)] for each point
    :return: np.array(n)
    """

    project = False
    if bounds is not None:
        project = True

    point = start
    v = np.zeros(len(start))
    for i in xrange(maxepoch):
        logger.info('echoch %d'%i)
        for j in xrange(n):
            logger.info('iteration %d'%j)
            v = momentum * v + learning_rate * gradient(point, *args, **kwargs)
            point -= v
            if project:
                for dim, bound in enumerate(bounds):
                    if bound[0] is not None:
                        point[dim] = max(bound[0], point[dim])
                    if bound[1] is not None:
                        point[dim] = min(bound[1], point[dim])

    return point
