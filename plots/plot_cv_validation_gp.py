from __future__ import absolute_import

import matplotlib.pyplot as plt

import numpy as np

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


def plot_histogram(y_eval, means, std_vec, filename_histogram):
    """
    Plot the histogram of the vector (y_eval-means)/std_vec. We'd expect to have the histogram of a
    standard Gaussian random variable.

    :param y_eval: np.array(n)
    :param means: np.array(n)
    :param std_vec: np.array(n)
    :param filename_histogram: str
    """
    indices = np.where(std_vec != 0)

    y_eval = y_eval[indices]
    means = means[indices]
    std_vec = std_vec[indices]

    plt.figure()
    plt.hist((y_eval - means) / std_vec, bins=15)
    plt.savefig(filename_histogram)

def plot_diagnostic_plot(y_eval, means, std_vec, n_data, filename_plot):
    """
    Plot a diagnostic plot of the GP fitting. For each point of the test fold, we plot the value of
    the function in that point, and its C.I. based on the GP model. We would expect that around
    95% of the test points stay within their C.I.

    :param y_eval: np.array(n)
    :param means: np.array(n)
    :param std_vec: np.array(n)
    :param n_data: int
    :param filename_plot: str
    """

    indices = np.where(std_vec != 0)

    y_eval = y_eval[indices]
    means = means[indices]
    std_vec = std_vec[indices]

    plt.figure()
    plt.ylabel('Prediction of the function')
    plt.errorbar(np.arange(n_data), means, yerr=2.0 * std_vec, fmt='o')
    plt.scatter(np.arange(n_data), y_eval, color='r')
    plt.savefig(filename_plot)
