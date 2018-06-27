from __future__ import absolute_import

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit, leastsq, fmin_bfgs, fmin_l_bfgs_b, nnls

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

    def log_prob(self, x, sigma, historical_data):
        """
        log_prob for only one starting point
        Historical data of only one starting point.

        x: [float], x[0:len(list_functions)] are the weights
        historical_data: [float]
        """

        total_iterations = self.total_iterations

        if sigma < 0:
            return -np.inf

        bounds = []

        index = 0
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

        n_functions = len(self.list_functions)
        weights = np.array(x[0:n_functions])
        params = x[n_functions:]
        n_iterations = len(historical_data)


        for i in range(n_functions, len(x)):
            if bounds[i - n_functions][0] is not None and x[i] < bounds[i - n_functions][0]:
                print "bounds"
                print i - n_functions
                print bounds[i - n_functions][0], x[i]
                return -np.inf
            if bounds[i - n_functions][1] is not None and x[i] > bounds[i - n_functions][1]:
                print "bounds"
                print i - n_functions
                print bounds[i - n_functions][0], x[i]
                return -np.inf
        if np.any(weights < 0):
            print "weights"
            return - np.inf

        if self.weighted_combination(1, weights, params) >= self.weighted_combination(total_iterations,
                                                                            weights, params):
            return - np.inf

        if self.lower is not None and self.weighted_combination(1, weights, params) < self.lower:
            print "no lower"

            val = - np.inf
        if self.upper is not None \
                and self.weighted_combination(self.total_iterations, weights, params) > self.upper:
            print "no upper"
            val = -np.inf


        val = 0.0
        for index, y in enumerate(historical_data):
            mean = self.weighted_combination(index + 1, weights, params)
            val += np.log(norm.pdf(y, loc=mean, scale=sigma))

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

    def get_starting_values_mle(self, historical_data):

        bounds = []

        index = 0
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


        st_params = []
        st_weights = []
        for function_n in self.list_functions:
            params_names = self.parameters_functions[function_n]
            result = self.mle_params_per_function(historical_data, function_n, choose_sigma=True,
                                             choose_weights=True)
            st_params += list(result[0][0:len(params_names)])
            st_weights += list([result[0][-1]])

        print st_params

        for i in range(len(st_params)):
            if bounds[i][0] is not None and st_params[i] < bounds[i][0]:
                st_params[i] = bounds[i][0] + 0.1
            if bounds[i][1] is not None and st_params[i] > bounds[i][1]:
                st_params[i] = bounds[i][1] - 0.1



        x = range(1, len(historical_data) + 1)
        evaluations = np.array(
            [self.weighted_combination(t, st_weights / np.sum(st_weights), st_params) for t in x])
        sigma_sq = np.mean((evaluations - historical_data) ** 2)

        start_params = list(st_weights / np.sum(st_weights)) + list(st_params) + [
            np.sqrt(sigma_sq)]
        start_params = np.array(start_params)

        return start_params
        #return start_params, st_params, st_weights / np.sum(st_weights), sigma_sq

    def mle_params_per_function(self, historical_data, function_name, choose_sigma=False,
                                choose_weights=False):
        """
        log_prob for only one starting point

        :param historical_data: [float]
        """
        historical_data = np.array(historical_data)
        n = len(historical_data)
        x = range(1, n + 1)
        function = self.functions[function_name]

        def objective(params):

            param_tmp = {}
            params_names = self.parameters_functions[function_name]
            for ind, name in enumerate(params_names):
                param_tmp[name] = params[ind]

            if choose_sigma:
                sigma_sq = params[len(params_names)] ** 2
            else:
                #             evaluations = np.array([function(t, **param_tmp) for t in x])
                #             sigma_sq = np.mean((evaluations - historical_data) ** 2)
                sigma_sq = 10.0
            if choose_sigma:
                args_lp = list(params[0:-1])
            else:
                args_lp = list(params)

            if choose_sigma:
                add = len(param_tmp) + 1
            else:
                add = len(param_tmp)
            weight = 1.0
            if choose_weights:
                weight = params[add]

            val = self.log_prob_per_function(param_tmp, np.sqrt(sigma_sq), historical_data,
                                        function_name, weight=weight)

            return -1.0 * val

        def gradient(params):
            param_tmp = {}
            params_names = self.parameters_functions[function_name]
            for ind, name in enumerate(params_names):
                param_tmp[name] = params[ind]

            if choose_sigma:
                sigma_sq = params[len(param_tmp)] ** 2
            else:
                #             evaluations = np.array([function(t, **param_tmp) for t in x])
                #             sigma_sq = np.mean((evaluations - historical_data) ** 2)
                sigma_sq = 10.0
            if choose_sigma:
                args_lp = list(params[0:-1])
            else:
                args_lp = list(params)

            if choose_sigma:
                add = len(param_tmp) + 1
            else:
                add = len(param_tmp)
            weight = 1.0
            if choose_weights:
                weight = params[add]

            val = self.gradient_llh_function(param_tmp, np.sqrt(sigma_sq), historical_data,
                                        function_name, choose_sigma=choose_sigma, weight=weight,
                                        choose_weight=choose_weights)

            for t in val:
                if t != t:
                    print "grad"
                    print param_tmp
                    print params
                    print sigma_sq
                    print function_name
                    df
            return -1.0 * val

        n_params_function = len(self.parameters_functions[function_name])

        if not choose_weights:
            if not choose_sigma:
                params_st = np.ones(n_params_function)
            else:
                params_st = np.ones(n_params_function + 1)
                params_st[-1] = 0.5
        else:
            if not choose_sigma:
                params_st = np.ones(n_params_function + 1)
                params_st[-1] = 1.0
            else:
                params_st = np.ones(n_params_function + 2)
                params_st[n_params_function] = 0.5
                params_st[-1] = 1.0

        bounds = []

        if self.default_values[function_name] is not None:
            params_st[0:len(self.default_values[function_name])] = self.default_values[function_name]

        if self.min_values[function_name] is not None:
            bd = []
            for t in self.min_values[function_name]:
                bd += [[t, None]]
            bounds += bd

            if choose_sigma:
                bounds += [[0.00000001, None]]

            if choose_weights:
                bounds += [[0., None]]

        if self.max_values[function_name] is not None:
            bd = []
            for it, t in enumerate(self.max_values[function_name]):
                if t is not None and len(bounds) > 0:
                    bounds[it][1] = t
                elif len(bounds) == 0:
                    bd += [[None, t]]

            if len(bounds) == 0:
                bounds += bd
                if choose_sigma:
                    bounds += [[0.00000001, None]]
                if choose_weights:
                    bounds += [[0., None]]

        if self.max_values[function_name] is None and self.min_values[function_name] is None:
            bounds = len(self.default_values[function_name]) * [[None, None]]
            if choose_sigma:
                bounds += [[0.00000001, None]]
            if choose_weights:
                bounds += [[0., None]]

        popt, fval, info = fmin_l_bfgs_b(objective,
                                         fprime=gradient,
                                         x0=params_st,
                                         bounds=bounds,
                                         approx_grad=False)
        if not choose_sigma:
            param_tmp = {}
            params_names = self.parameters_functions[function_name]
            for ind, name in enumerate(params_names):
                param_tmp[name] = popt[ind]
            evaluations = np.array([function(t, **param_tmp) for t in x])
            sigma_sq = np.sqrt(np.mean((evaluations - historical_data) ** 2))
            return popt, fval, info, sigma_sq
        else:
            param_tmp = {}
            params_names = self.parameters_functions[function_name]
            for ind, name in enumerate(params_names):
                param_tmp[name] = popt[ind]

            evaluations = np.array([function(t, **param_tmp) for t in x])
            return popt, fval, info, evaluations


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


    def log_prob_per_function(self, x, sigma, historical_data, function_name,
                              weight=1.0, L=None, U=None):
        """
        log_prob for only one starting point
        Historical data of only one starting point.

        x: dictionary with arguments of function
        historical_data: [float]
        """
        total_iterations = self.total_iterations
        function = self.functions[function_name]
        params = x
        # n_iterations = len(historical_data)
        # params_names = self.parameters_functions[function_name]

        dom_x = range(1, len(historical_data) + 1)
        evaluations = np.zeros(len(historical_data))
        for i in dom_x:
            evaluations[i - 1] = weight * function(i, **params)

        val = -1.0 * np.sum((evaluations - historical_data) ** 2) / (sigma ** 2) - np.log(sigma ** 2)

        return val


    def gradient_llh_function(self, x, sigma, historical_data, function_name, choose_sigma=True,
                              choose_weight=True, weight=1.0):
        """
        Gradient of the llh respect to a specific function.

        :param function: str
        """
        function = self.functions[function_name]
        gradient_function = self.gradients_functions[function_name]
        params = x
        n_iterations = len(historical_data)

        bounds = []

        if not choose_weight:
            if choose_sigma:
                gradient = np.zeros(len(x) + 1)
            else:
                gradient = np.zeros(len(x))
        else:
            if choose_sigma:
                gradient = np.zeros(len(x) + 2)
            else:
                gradient = np.zeros(len(x) + 1)
        evaluations = np.zeros(len(historical_data))

        dom_x = range(1, len(historical_data) + 1)
        gradient_theta = np.zeros(len(x))
        for i in dom_x:
            evaluations[i - 1] = weight * function(i, **params)
            gradient_theta += weight * gradient_function(i, **params) * (
            evaluations[i - 1] - historical_data[i - 1])

        gradient_theta *= (-2.0 / (sigma ** 2))

        gradient[0: len(x)] = gradient_theta

        if choose_sigma:
            gradient_sigma = 2.0 * np.sum((historical_data - evaluations) ** 2) * \
                             np.power(sigma, 3) - 2.0 / sigma
            gradient[len(x)] = gradient_sigma

        if choose_weight:
            if choose_sigma:
                add = len(x) + 1
            else:
                add = len(x)
            for i in dom_x:
                evaluations[i - 1] = weight * function(i, **params)
                gradient[add] += (-2.0 / (sigma ** 2)) * function(i, **params) * (
                evaluations[i - 1] - historical_data[i - 1])

        return gradient
