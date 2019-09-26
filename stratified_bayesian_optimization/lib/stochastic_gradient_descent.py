from __future__ import absolute_import



import numpy as np

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


def SGD(start, gradient, n, args=(), kwargs={}, bounds=None, learning_rate=0.1, momentum=0.9,
        maxepoch=250, adam=True, betas=None, eps=1e-8, simplex_domain=None):
    """
    SGD to minimize sum(i=0 -> n) (1/n) * f(x). Batch sizes are of size 1.
    ADAM: https://arxiv.org/pdf/1412.6980.pdf
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
    if bounds is not None or simplex_domain is not None:
        project = True

    if betas is None:
        betas = (0.9, 0.999)

    m0 = np.zeros(len(start))
    v0 = np.zeros(len(start))

    point = start
    v = np.zeros(len(start))
    times_out_boundary = 0
    t_ = 0

    for iteration in xrange(maxepoch):
        previous = point.copy()
        t_ += 1
        grad = []
        for j in xrange(n):
            gradient_ = gradient(point, *args, **kwargs)

            while gradient_ is np.nan:
                norm_point = np.sqrt(np.sum(point ** 2))
                perturbation = norm_point * 1e-6

                if project:
                    parameters_uniform = []
                    for i in range(len(bounds)):
                        bound = bounds[i]
                        dist = point[i] - bound[0]
                        lb = min(perturbation, dist)
                        dist = bound[1] - point[i]
                        ub = min(perturbation, dist)
                        parameters_uniform.append([-lb, ub])
                else:
                    parameters_uniform = len(point) * [[-perturbation, perturbation]]

                perturbation = []
                for i in range(len(point)):
                    lb = parameters_uniform[i][0]
                    ub = parameters_uniform[i][1]
                    perturbation.append(np.random.uniform(lb, ub))
                perturbation = np.array(perturbation)
                point = point + perturbation
                gradient_ = gradient(point, *args, **kwargs)
            grad.append(gradient_)
        gradient_ = np.mean(np.array(grad), axis=0)

        if not adam:
            v = momentum * v + gradient_
            old_p = point.copy()
            point -= learning_rate * v
        else:
            m0 = betas[0] * m0 + (1 - betas[0]) * gradient_
            v0 = betas[1] * v0 + (1 - betas[1]) * (gradient_ ** 2)
            m_1 = m0 / (1 - (betas[0]) ** (t_))
            v_1 = v0 / (1 - (betas[1]) ** (t_))
            point = point - learning_rate * m_1 / (np.sqrt(v_1) + eps)

        in_domain = True
        if project:
            for dim, bound in enumerate(bounds):
                if bound[0] is not None and point[dim] < bound[0]:
                    in_domain = False
                    break
                if bound[1] is not None and point[dim] > bound[1]:
                    in_domain = False
                    break
                if simplex_domain is not None:
                    if np.sum(point[2:]) > simplex_domain:
                        in_domain = False
                        break
                    #TODO:Only for citibike, generalize later
                 #   if simplex_domain - np.sum(point) > 3717.0:
                 #       in_domain = False
                 #       break

        if project and not in_domain:
            for dim, bound in enumerate(bounds):
                if bound[0] is not None:
                    point[dim] = max(bound[0], point[dim])
                if bound[1] is not None:
                    point[dim] = min(bound[1], point[dim])
            if simplex_domain is not None:
                if np.sum(point[2:]) > simplex_domain:
                    point[2:] = simplex_domain * (point[2:] / np.sum(point[2:]))

             #   if simplex_domain - np.sum(point) > 3717.0:
             #       point = (simplex_domain - 3717.0) * (point / np.sum(point))
            if not adam:
                for dim, bound in enumerate(bounds):
                    v[dim] = (point[dim] - old_p[dim]) / learning_rate

        den_norm = (np.sqrt(np.sum(previous ** 2)))

        if den_norm == 0:
            norm = np.sqrt(np.sum((previous - point) ** 2)) / 1e-2
        else:
            norm = np.sqrt(np.sum((previous - point) ** 2)) / den_norm
        if norm < 0.01:
            break

    return point
