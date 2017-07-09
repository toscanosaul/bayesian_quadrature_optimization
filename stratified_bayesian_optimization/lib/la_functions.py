from __future__ import absolute_import

import numpy as np
from scipy.linalg import lapack
from scipy import linalg


def cholesky(cov, max_tries=5):
    """
    Computest the Cholesky decomposition L of the matrix cov: L*L^T = cov
    :param cov: np.array(nxn)
    :param max_tries: int
    :return: L
    """

    cov = np.ascontiguousarray(cov)
    L, info = lapack.dpotrf(cov, lower=1)

    if info == 0:
        return L

    diag_cov = np.diag(cov)

    if np.any(diag_cov <= 0.):
        raise linalg.LinAlgError("not positive definite matrix")

    jitter = diag_cov.mean() * 1e-6

    n_tries = 1
    while n_tries <= max_tries and np.isfinite(jitter):
        try:
            L = linalg.cholesky(cov + np.eye(cov.shape[0]) * jitter, lower=True)
            return L
        except:
            jitter *= 10
        finally:
            n_tries += 1
    raise linalg.LinAlgError("not positive definite, even with jitter.")


def cho_solve(chol, y):
    """
    Solves the systems chol * chol^T * x = y
    :param cov: np.array(nxn)
    :param y: np.array(n)
    :return: np.array(n)
    """

    chol = np.asfortranarray(chol)
    return lapack.dpotrs(chol, y, lower=1)[0]
