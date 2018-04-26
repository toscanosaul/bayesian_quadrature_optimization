from __future__ import absolute_import

import numpy as np

class ParametricFunctions(object):

    def __init__(self, total_iterations=100, lower=0.0, upper=1.0):
        self.functions = {
            'linear': ParametricFunctions.linear,
            'ilog2': ParametricFunctions.ilog2,
            'weibull': ParametricFunctions.weibull,
            'janoschek': ParametricFunctions.janoschek,
            'exp_4': ParametricFunctions.exp_4,
            'mmf': ParametricFunctions.mmf,
            'pow_4': ParametricFunctions.pow_4,
            'log_power': ParametricFunctions.log_power,
            'hill_3': ParametricFunctions.hill_3,
            'log_log_linear': ParametricFunctions.log_log_linear,
            'pow_3': ParametricFunctions.pow_3,
            'vapor_pressure': ParametricFunctions.vapor_pressure,
        }

        self.gradients_functions = {
            'linear': ParametricFunctions.grad_linear,
            'ilog2': ParametricFunctions.grad_ilog2,
            'weibull': ParametricFunctions.grad_weibull,
            'janoschek': ParametricFunctions.grad_janoschek,
            'exp_4': ParametricFunctions.grad_exp_4,
            'mmf': ParametricFunctions.grad_mmf,
            'pow_4': ParametricFunctions.grad_pow_4,
            'log_power': ParametricFunctions.grad_log_power,
            'hill_3': ParametricFunctions.grad_hill_3,
            'log_log_linear': ParametricFunctions.grad_log_log_linear,
            'pow_3': ParametricFunctions.grad_pow_3,
            'vapor_pressure': ParametricFunctions.grad_vapor_pressure,
        }

        self.list_functions = [
            'linear',
             'hill_3',
             'vapor_pressure',
             'mmf',
             'janoschek',
             'log_power',
             'log_log_linear',
             'weibull',
             'exp_4',
             'pow_4',
             'pow_3',
             'ilog2'
        ]

        self.parameters_functions = {
            'linear': ['a', 'b'],
            'ilog2': ['a', 'c'],
            'weibull': ['alpha', 'beta', 'k', 'delta'],
            'janoschek': ['alpha', 'beta', 'k', 'delta'],
            'exp_4': ['a', 'b', 'c', 'alpha'],
            'mmf': ['alpha', 'beta', 'k', 'delta'],
            'pow_4': ['a', 'b', 'c', 'alpha'],
            'log_power': ['a', 'b', 'c'],
            'hill_3': ['eta', 'k', 'theta', 'alpha'],
            'log_log_linear': ['a', 'b'],
            'pow_3': ['a', 'c', 'alpha'],
            'vapor_pressure': ['a', 'b', 'c'],
        }

        self.default_values = {
            'linear': [0.5, 0.5],
            'ilog2': [0.40999, 0.78],
            'weibull': [.7, 0.1, 0.02, 1],
            'janoschek': [0.73, 0.07,  0.355,  0.46],
            'exp_4': [0.8, -0.8,0.7, 0.3],
            'mmf': [.7,  0.1, 0.01,  5],
            'pow_4': [200, 0., 1.001,  0.1],
            'log_power': [0.77,  2.98, -0.51],
            'hill_3': [0.586449, 2.460843, 0.772320, 0.1],
            'log_log_linear': [1.0, 1.00001],
            'pow_3':[0.52, 0.84, 0.01],
            'vapor_pressure': [-0.622028, -0.470050,  0.042322],
        }

        self.min_values = {
            'linear': [0.00001, 0.0001],
            'ilog2': [None, 0.6],
            'weibull': [None, 0.0, 1e-100, None],
            'janoschek': [0.000001, 0.000001, None, None],
            'exp_4': [0.5, None, 1.0, None],
            'mmf': [None, None, 1e-10, None],
            'pow_4': [None, 0., 1.000001, 1e-10],
            'log_power': [0.0000001, None, None],
            'hill_3': [-10., 1e-10, 1e-10, 1e-10],
            'log_log_linear': [1e-10, 1.000001],
            'pow_3': [None, 0.60, None],
            'vapor_pressure': None,
        }


        self.max_values = {
            'linear': None,
            'ilog2': [0.41, None],
            'weibull': None,
            'janoschek': None,
            'exp_4': [None, 0.3, None, 1.5],
            'mmf': None,
            'pow_4': None,
            'log_power': None,
            'hill_3': [10., None, 100., None],
            'log_log_linear': None,
            'pow_3': [0.59999, None, None],
            'vapor_pressure': None,
        }

        n_parameters = 0
        for f in self.list_functions:
            n_parameters += len(self.parameters_functions[f])
        self.n_parameters = n_parameters
        self.n_weights = len(self.list_functions)

        self.total_iterations = total_iterations
        self.lower = lower
        self.upper = upper

    def weighted_combination(self, x, weights, params):
        """
        weights: [float]
        params: [float], they are in the order given by list_functions
        """

        val = 0.0

        for index_f, funct in enumerate(self.list_functions):
            function = self.functions[funct]
            parameters = self.parameters_functions[funct]

            n_params = len(parameters)
            par = params[0: n_params]
            params = params[n_params:]

            param_tmp = {}

            for index, p in enumerate(parameters):
                param_tmp[p] = par[index]

            val += weights[index_f] * function(x, **param_tmp)
            if val != val:
                print function
                print x, param_tmp
                df

        return val

    def gradient_weighted_combination(self, x, weights, params):
        grad = np.zeros(len(weights) + len(params))

        start_index = 0
        n_functions = self.n_weights
        for index_f, funct in enumerate(self.list_functions):
            function = self.functions[funct]
            gradient_function = self.gradients_functions[funct]
            parameters = self.parameters_functions[funct]

            n_params = len(parameters)
            par = params[0: n_params]
            params = params[n_params:]

            param_tmp = {}

            for index, p in enumerate(parameters):
                param_tmp[p] = par[index]

            val = weights[index_f] * gradient_function(x, **param_tmp)
            val_2 = function(x, **param_tmp)

            grad[n_functions + start_index: n_functions + start_index + n_params] = val
            start_index += n_params

            grad[index_f] = val_2

        return grad

    def log_prior(self, params, weights):
        """
        log_prob for only one starting point
        Historical data of only one starting point.

        x[0:kernel_params] are the kenrel params
        then we have the weights, and then the parameters of te functions
        historical_data: [float]
        """


        bounds = []

        index = 0
        val = 0.0
        for f in self.list_functions:
            if self.min_values[f] is not None or self.max_values[f] is not None:
                bd = []
                if self.min_values[f] is not None:
                    n_bounds = len(self.min_values[f])
                else:
                    n_bounds = len(self.max_values[f])

                for i in range(n_bounds):
                    bd_ = [None, None]
                    if self.min_values[f] is not None:
                        bd_[0] = self.min_values[f][i]
                    if self.max_values[f] is not None:
                        bd_[1] = self.max_values[f][i]
                    bd += [bd_]
                bounds += bd

            else:
                bd = len(self.parameters_functions[f]) * [[None, None]]
                bounds += bd

        for i in range(len(params)):
            if bounds[i][0] is not None and params[i] < bounds[i][0]:
          #      print "wrong_bo"
                val = -np.inf
                return val
            if bounds[i][1] is not None and params[i] > bounds[i][1]:
           #     print "wrong_bo"
                val = -np.inf
                return val

        if self.weighted_combination(1, weights, params) >= \
                self.weighted_combination(self.total_iterations, weights, params):
           # print "no incease"
            val = -np.inf

        if self.lower is not None and self.weighted_combination(1, weights, params) < self.lower:
           # print "no lower"

            val = - np.inf
        if self.upper is not None \
                and self.weighted_combination(self.total_iterations, weights, params) > self.upper:
          #  print "no upper"
            val = -np.inf
        return val

    def get_starting_values(self):
        params_st = np.ones(self.n_weights + self.n_parameters)
        start_index = 0
        n_functions = self.n_weights
        for f in self.list_functions:
            n_par = len(self.parameters_functions[f])
            if self.default_values[f] is not None:
                parameters = self.default_values[f]
                params_st[n_functions + start_index:n_functions + start_index + n_par] = parameters
            start_index += n_par
        return params_st

    def get_bounds(self):
        bounds = []

        bounds_weights = self.n_weights * [[None, None]]
        bounds += bounds_weights

        for f in self.list_functions:
            bounds_f = []
            if self.min_values[f] is not None:
                bd = []
                for t in self.min_values[f]:
                    bd += [[t, None]]
                bounds_f += bd

            if self.max_values[f] is not None:
                bd = []
                for it, t in enumerate(self.max_values[f]):
                    if t is not None and len(bounds_f) > 0:
                        bounds_f[it][1] = t
                    elif len(bounds_f) == 0:
                        bd += [[None, t]]

                if len(bounds_f) == 0:
                    bounds_f += bd

            if self.max_values[f] is None and self.min_values[f] is None:
                bounds_f = len(self.parameters_functions[f]) * [[None, None]]
            bounds += bounds_f

        return bounds

    @staticmethod
    def linear(x, a, b):
        x = float(x)
        a = float(a)
        b = float(b)

        return a * x + b

    @staticmethod
    def grad_linear(x, a, b):
        x = float(x)
        a = float(a)
        b = float(b)

        grad = np.zeros(2)

        grad[0] = x
        grad[1] = 1.0

        return grad

    @staticmethod
    def vapor_pressure(x, a, b, c):
        x = float(x)
        a = float(a)
        b = float(b)
        c = float(c)

        return np.exp(a + (b / x) + c * np.log(x))

    @staticmethod
    def grad_vapor_pressure(x, a, b, c):
        x = float(x)
        a = float(a)
        b = float(b)
        c = float(c)

        grad = np.zeros(3)
        value = ParametricFunctions.vapor_pressure(x, a, b, c)

        grad[0] = value
        grad[1] = value * (1.0 / x)
        grad[2] = value * np.log(x)

        return grad

    @staticmethod
    def pow_3(x, a, c, alpha):
        x = float(x)
        a = float(a)
        alpha = float(alpha)
        c = float(c)

        val = c - a * (np.power(x, -1.0 * alpha))

        return val

    @staticmethod
    def grad_pow_3(x, a, c, alpha):
        x = float(x)
        a = float(a)
        alpha = float(alpha)
        c = float(c)

        grad = np.zeros(3)
        grad[0] = -np.power(x, -1.0 * alpha)
        grad[1] = 1.0
        grad[2] = a * (np.power(x, -1.0 * alpha)) * np.log(x)

        return grad

    @staticmethod
    def log_log_linear(x, a, b):
        x = float(x)
        a = float(a)
        b = float(b)

        return np.log(a * np.log(x) + b)

    @staticmethod
    def grad_log_log_linear(x, a, b):
        x = float(x)
        a = float(a)
        b = float(b)

        grad = np.zeros(2)
        grad[0] = np.log(x) / (a * np.log(x) + b)
        grad[1] = 1.0 / (a * np.log(x) + b)

        return grad

    @staticmethod
    def hill_3(x, eta, k, theta, alpha):
        x = float(x)
        eta = float(eta)
        k = float(k)
        theta = float(theta)
        alpha = float(alpha)

        return alpha + (theta * np.power(x, eta)) / (np.power(k, eta) + np.power(x, eta))

    @staticmethod
    def grad_hill_3(x, eta, k, theta, alpha):
        x = float(x)
        eta = float(eta)
        k = float(k)
        theta = float(theta)
        alpha = float(alpha)

        grad = np.zeros(4)
        grad[0] = theta * (
        np.power(k, eta) * (np.power(k, eta) + np.power(x, eta)) * np.log(x) - np.power(x, eta) * (
        np.power(k, eta) * np.log(k) + np.power(x, eta) * np.log(x))) / (
                  (np.power(k, eta) + np.power(x, eta)) ** 2)
        grad[1] = - (theta * np.power(x, eta) * np.power(k, eta - 1) * eta) / (
        (np.power(k, eta) + np.power(x, eta)) ** 2)
        grad[2] = (np.power(x, eta)) / (np.power(k, eta) + np.power(x, eta))
        grad[3] = 1.0
        return grad

    @staticmethod
    def log_power(x, a, b, c):
        x = float(x)
        a = float(a)
        b = float(b)
        c = float(c)

        return a / (1.0 + np.power((x / np.exp(b)), c))

    @staticmethod
    def grad_log_power(x, a, b, c):
        x = float(x)
        a = float(a)
        b = float(b)
        c = float(c)

        grad = np.zeros(3)

        grad[0] = 1.0 / (1.0 + np.power((x / np.exp(b)), c))
        grad[1] = (a * np.power((x / np.exp(b)), c) * c) / ((1.0 + np.power((x / np.exp(b)), c)) ** 2)
        grad[2] = (- a * np.power((x / np.exp(b)), c) * (np.log(x) - b)) / (
        (1.0 + np.power((x / np.exp(b)), c)) ** 2)

        return grad

    @staticmethod
    def pow_4(x, a, b, c, alpha):
        x = float(x)
        a = float(a)
        b = float(b)
        c = float(c)
        alpha = float(alpha)

        return c - np.power(a * x + b, -1.0 * alpha)

    @staticmethod
    def grad_pow_4(x, a, b, c, alpha):
        x = float(x)
        a = float(a)
        b = float(b)
        c = float(c)
        alpha = float(alpha)

        grad = np.zeros(4)

        grad[0] = alpha * x * np.power(a * x + b, -1.0 * alpha) / (a * x + b)
        grad[1] = alpha * np.power(a * x + b, -1.0 * alpha) / (a * x + b)
        grad[2] = 1.0
        grad[3] = np.power(a * x + b, -1.0 * alpha) * np.log(a * x + b)

        return grad

    @staticmethod
    def mmf(x, alpha, beta, k, delta):
        x = float(x)
        alpha = float(alpha)
        beta = float(beta)
        k = float(k)
        delta = float(delta)

        return alpha - ((alpha - beta) / (1.0 + np.power(k * x, delta)))

    @staticmethod
    def grad_mmf(x, alpha, beta, k, delta):
        x = float(x)
        alpha = float(alpha)
        beta = float(beta)
        k = float(k)
        delta = float(delta)

        grad = np.zeros(4)

        grad[0] = 1.0 - 1.0 / ((1.0 + np.power(k * x, delta)))
        grad[1] = 1.0 / ((1.0 + np.power(k * x, delta)))
        grad[2] = (alpha - beta) * np.power(k * x, delta) * delta * (1.0 / k) / (
        ((1.0 + np.power(k * x, delta))) ** 2)
        grad[3] = (alpha - beta) * np.power(k * x, delta) * np.log(k * x) / (
        ((1.0 + np.power(k * x, delta))) ** 2)

        return grad

    @staticmethod
    def exp_4(x, a, b, c, alpha):
        x = float(x)
        a = float(a)
        b = float(b)
        c = float(c)
        alpha = float(alpha)

        return c - np.exp(b - a * np.power(x, alpha))

    @staticmethod
    def grad_exp_4(x, a, b, c, alpha):
        x = float(x)
        a = float(a)
        b = float(b)
        c = float(c)
        alpha = float(alpha)

        grad = np.zeros(4)

        grad[0] = - np.exp(b - a * np.power(x, alpha)) * (- np.power(x, alpha))
        grad[1] = - np.exp(b - a * np.power(x, alpha))
        grad[2] = 1.0
        grad[3] = a * np.power(x, alpha) * np.log(x) * np.exp(b - a * np.power(x, alpha))

        return grad

    @staticmethod
    def janoschek(x, alpha, beta, k, delta):
        x = float(x)
        alpha = float(alpha)
        beta = float(beta)
        k = float(k)
        delta = float(delta)

        return alpha - (alpha - beta) * np.exp(-k * np.power(x, delta))

    @staticmethod
    def grad_janoschek(x, alpha, beta, k, delta):
        x = float(x)
        alpha = float(alpha)
        beta = float(beta)
        k = float(k)
        delta = float(delta)

        grad = np.zeros(4)

        grad[0] = 1.0 - np.exp(-k * np.power(x, delta))
        grad[1] = np.exp(-k * np.power(x, delta))
        grad[2] = (alpha - beta) * np.exp(-k * np.power(x, delta)) * np.power(x, delta)
        grad[3] = (alpha - beta) * np.exp(-k * np.power(x, delta)) * k * np.power(x, delta) * np.log(x)

        return grad

    @staticmethod
    def weibull(x, alpha, beta, k, delta):
        x = float(x)
        alpha = float(alpha)
        beta = float(beta)
        k = float(k)
        delta = float(delta)

        return alpha - (alpha - beta) * np.exp(- np.power(k * x, delta))


    @staticmethod
    def grad_weibull(x, alpha, beta, k, delta):
        x = float(x)
        alpha = float(alpha)
        beta = float(beta)
        k = float(k)
        delta = float(delta)

        grad = np.zeros(4)

        grad[0] = 1.0 - np.exp(- np.power(k * x, delta))
        grad[1] = np.exp(- np.power(k * x, delta))
        grad[2] = (alpha - beta) * np.exp(- np.power(k * x, delta)) * np.power(k * x, delta) * delta * (
        1.0 / k)
        grad[3] = (alpha - beta) * np.exp(- np.power(k * x, delta)) * np.power(k * x, delta) * np.log(
            k * x)

        return grad

    @staticmethod
    def ilog2(x, a, c):
        x = float(x)
        a = float(a)
        c = float(c)
        return c - (a / np.log(x + 1))

    @staticmethod
    def grad_ilog2(x, a, c):
        x = float(x)
        a = float(a)
        c = float(c)

        grad = np.zeros(2)
        grad[0] = -1.0 / (np.log(x + 1))
        grad[1] = 1.0
        return grad
