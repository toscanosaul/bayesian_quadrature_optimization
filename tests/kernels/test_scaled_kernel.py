from __future__ import absolute_import

import unittest

from doubles import expect

import copy
import numpy as np
import numpy.testing as npt

from stratified_bayesian_optimization.kernels.matern52 import Matern52, GradientLSMatern52
from stratified_bayesian_optimization.kernels.scaled_kernel import ScaledKernel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.distances import Distances
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences
from stratified_bayesian_optimization.priors.uniform import UniformPrior
from stratified_bayesian_optimization.lib.constant import (
    SMALLEST_NUMBER,
    LARGEST_NUMBER,
    MATERN52_NAME,
    SMALLEST_POSITIVE_NUMBER,
    SIGMA2_NAME,
    LENGTH_SCALE_NAME,
)


class TestScaledKernel(unittest.TestCase):

    def setUp(self):
        self.x =1