import unittest

from mock import create_autospec
from doubles import expect

import numpy.testing as npt

from copy import deepcopy

import numpy as np

from stratified_bayesian_optimization.services.bayesian_global_optimization import BGO
from stratified_bayesian_optimization.services.domain import DomainService
from stratified_bayesian_optimization.entities.run_spec import RunSpecEntity
from stratified_bayesian_optimization.entities.domain import BoundsEntity, DomainEntity
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    TASKS_KERNEL_NAME,
    UNIFORM_FINITE,
    TASKS,
)
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.lib.sample_functions import SampleFunctions
from stratified_bayesian_optimization.models.gp_fitting_gaussian import GPFittingGaussian
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature
from stratified_bayesian_optimization.acquisition_functions.sbo import SBO


class TestBGOService(unittest.TestCase):

    def setUp(self):

        self.bounds_domain_x = BoundsEntity({
            'lower_bound': 0,
            'upper_bound': 100,
        })

        spec_domain = {
            'dim_x': 1,
            'choose_noise': True,
            'bounds_domain_x': [self.bounds_domain_x],
            'number_points_each_dimension': [100],
            'problem_name': 'a',
        }

        self.domain = DomainService.from_dict(spec_domain)

        dict = {
            'problem_name': 'test_problem_with_tasks',
            'dim_x': 1,
            'choose_noise': True,
            'bounds_domain_x': [BoundsEntity({'lower_bound': 0, 'upper_bound': 100})],
            'number_points_each_dimension': [100],
            'method_optimization': 'sbo',
            'training_name': 'test_bgo',
            'bounds_domain': [[0, 100], [0, 1]],
            'n_training': 4,
            'type_kernel': [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            'noise': False,
            'random_seed': 5,
            'parallel': False,
            'type_bounds': [0, 1],
            'dimensions': [2, 1, 2],
            'name_model': 'gp_fitting_gaussian',
            'mle': True,
            'thinning': 0,
            'n_burning': 0,
            'max_steps_out': 1,
            'training_data': None,
            'x_domain': [0],
            'distribution': UNIFORM_FINITE,
            'parameters_distribution': None,
            'minimize': False,
            'n_iterations': 5,
        }

        self.spec = RunSpecEntity(dict)


        self.acquisition_function = None
        self.gp_model = None

        ###Define other BGO object
        np.random.seed(5)
        n_points = 100
        points = np.linspace(0, 100, n_points)
        points = points.reshape([n_points, 1])
        tasks = np.random.randint(2, size=(n_points, 1))

        add = [10, -10]
        kernel = Matern52.define_kernel_from_array(1, np.array([100.0, 1.0]))
        function = SampleFunctions.sample_from_gp(points, kernel)

        for i in xrange(n_points):
            function[0, i] += add[tasks[i, 0]]
        points = np.concatenate((points, tasks), axis=1)
        self.points = points

        function = function[0, :]

        points_ls = [list(points[i, :]) for i in xrange(n_points)]

        training_data_med = {
            'evaluations': list(function[0:5]),
            'points': points_ls[0:5],
            "var_noise": [],
        }

        self.training_data = training_data_med

        training_data = {
            'evaluations': list(function),
            'points': points,
            "var_noise": [],
        }

        gaussian_p = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data, [2, 1, 2], bounds_domain=[[0, 100], [0, 1]], type_bounds=[0, 1])
        gaussian_p = gaussian_p.fit_gp_regression(random_seed=1314938)

        params = gaussian_p.get_value_parameters_model
        self.params = params

        dict = {
            'problem_name': 'test_simulated_gp',
            'dim_x': 1,
            'choose_noise': True,
            'bounds_domain_x': [BoundsEntity({'lower_bound': 0, 'upper_bound': 100})],
            'number_points_each_dimension': [100],
            'method_optimization': 'sbo',
            'training_name': 'test_sbo',
            'bounds_domain': [[0, 100], [0, 1]],
            'n_training': 5,
            'type_kernel': [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            'noise': False,
            'random_seed': 5,
            'parallel': False,
            'type_bounds': [0, 1],
            'dimensions': [2, 1, 2],
            'name_model': 'gp_fitting_gaussian',
            'mle': False,
            'thinning': 0,
            'n_burning': 0,
            'max_steps_out': 1,
            'training_data': training_data_med,
            'x_domain': [0],
            'distribution': UNIFORM_FINITE,
            'parameters_distribution': None,
            'minimize': False,
            'n_iterations': 50,
            'var_noise_value': [params[0]],
            'mean_value': [params[1]],
            'kernel_values': list(params[2:]),
            'cache': False,
            'debug': False,
        }

        self.spec_2 = RunSpecEntity(dict)

        #self.bgo = BGO(self.acquisition_function, self.gp_model)

    def test_from_spec(self):
       # bgo = BGO.from_spec(self.spec)
        assert True
        # TODO: FINISH THIS TEST

    def test_run_spec(self):
        # bgo = create_autospec(BGO)
        # expect(BGO).from_spec.and_return(bgo)
        # domain = create_autospec(DomainEntity)
        # expect(DomainService).from_dict.and_return(domain)
        # expect(bgo).optimize.and_return({})
        #
        # assert BGO.run_spec(self.spec) == {}
        assert 1 == 1

    def test_optimize(self):
        expect(JSONFile).read.and_return(None)
        bgo = BGO.from_spec(self.spec)
        z = bgo.optimize(random_seed=1)

        print z
     #   assert z['optimal_solution'] == np.array([100.0])

    def test_optimize_2(self):
        bgo_2 = BGO.from_spec(self.spec_2)
        sol = bgo_2.optimize(random_seed=1)

        # test that after the first iterations the new points are added correctly
        training_data = deepcopy(self.training_data)

        training_data['points'] = np.concatenate((training_data['points'],
                                                  np.array([[99.9863644153, 0]])), axis=0)
        training_data['evaluations'] = np.concatenate((training_data['evaluations'],
                                                       np.array([9.0335599603])))
        gaussian_p_med = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data, [2, 1, 2], bounds_domain=[[0, 100], [0, 1]], type_bounds=[0, 1])
        gaussian_p_med.update_value_parameters(self.params)
        gp_med = BayesianQuadrature(gaussian_p_med, [0], UNIFORM_FINITE, {TASKS: 2})
        sbo = SBO(gp_med, np.array(self.domain.discretization_domain_x))

        point = sbo.optimize(start=np.array([[10 ,0]]))

        npt.assert_almost_equal(point['optimal_value'], 542.4598435381, decimal=4)
        npt.assert_almost_equal(point['solution'], np.array([61.58743036, 0]))
        print "with other way"
        print point
        print "new"
        print sol
        assert 1 == 0

