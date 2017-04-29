from __future__ import absolute_import

from functools import reduce

from stratified_bayesian_optimization.kernels.abstract_kernel import AbstractKernel


class ProductKernels(AbstractKernel):
    # TODO - Generaliza to more than two kernels, and cover the case where kernels are defined in
    # the same domain.

    def __init__(self, *kernels):
        """

        :param *kernels: ([AbstractKernel])
        """

        name = 'Product_of_'
        dimension = 0
        for kernel in kernels:
            name += kernel.name + '_'
            dimension += kernel.dimension

        super(ProductKernels, self).__init__(name, dimension)


        self.kernels = {}
        self.parameters = {}

        self.names = [kernel.name for kernel in kernels]

        for kernel in kernels:
            self.kernels[kernel.name] = kernel
            self.parameters[kernel.name] = self.kernels[kernel.name].hypers

    @property
    def hypers(self):
        return self.parameters

    def set_parameters(self, parameters):
        """

        :param parameters: {(str) kernel_name: {(str) parameter_name (as in set_parameter function
            of kernel) : ParameterEntity}}
        """

        for name in self.names:
            if name in parameters:
                self.kernels[name].set_parameters(**parameters[name])
                self.parameters[name] = self.kernels[name].hypers

    def cov(self, inputs):
        """

        :param inputs: {(str) kernel_name: np.array(nxd)}
        :return: np.array(nxn)
        """

        return self.cross_cov(inputs, inputs)

    def cross_cov(self, inputs_1, inputs_2):
        """

        :param inputs_1: {(str) kernel_name: np.array(nxd)}
        :param inputs_2: {(str) kernel_name: np.array(mxd')}
        :return: np.array(nxm)
        """

        return reduce(lambda K1, K2: K1 * K2,
                      [self.kernels[name].cross_cov(inputs_1[name], inputs_2[name])
                       for name in self.names])

    def gradient_respect_parameters(self, inputs):
        """
        :param inputs: {(str) kernel_name: np.array(nxd)}
        :return: {
            (str) kernel_name: { (str) parameter_name: np.array(nxn) }
        }
        """
        # TODO - Generalize to more than two kernels

        grad = {}
        cov = {}

        for name in self.names:
            grad[name] = self.kernels[name].gradient_respect_parameters(inputs[name])
            cov[name] = self.kernels[name].cov(inputs[name])

        grad_product = {}

        for i in range(2):
            grad_product[self.names[i]] = {}
            for name_param in self.kernels[self.names[i]].hypers:
                grad_product[self.names[i]][name_param] = \
                    cov[self.names[(i + 1) % 2]] * grad[self.names[i]][name_param]

        return grad_product

    def grad_respect_point(self, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param point: {(str) kernel_name: np.array(1xd)}
        :param inputs: {(str) kernel_name: np.array(nxd)}

        :return: {(str) kernel_name: np.array(nxd)}
        """
        # TODO - Generalize when the gradients share the same inputs
        # TODO - Generalize to more than two kernels

        grad = {}
        cov = {}

        for name in self.names:
            grad[name] = self.kernels[name].grad_respect_point(point[name], inputs[name])
            cov[name] = self.kernels[name].cross_cov(point[name], inputs[name])

        gradient = {}

        for i in range(2):
            gradient[self.names[i]] = grad[self.names[i]] * cov[self.names[(i + 1) % 2]]

        return gradient
