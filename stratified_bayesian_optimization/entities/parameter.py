from __future__ import absolute_import


class ParameterEntity(object):

    def __init__(self, name, value, prior):
        """

        :param name: str
        :param value: np.array
        :param prior: AbstractPrior
        """
        self.name = name
        self.value = value
        self.prior = prior
        self.dimension = len(self.value)

    def set_value(self, value):
        self.value = value

    def log_prior(self):
        return self.prior.logprob(self.value)

    def sample_from_prior(self, n_samples):
        return self.prior.sample(n_samples)
