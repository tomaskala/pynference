import abc

import numpy as np

import utils


# TODO: Consider an exponential family class as PyTorch has.


class Distribution(abc.ABC):
    _constraints = {}
    _support = None

    def __init__(self, batch_shape, rv_shape, check_parameters=True, check_support=True):
        self.batch_shape = batch_shape
        self.rv_shape = rv_shape
        self.check_support = check_support
        
        if check_parameters:
            for parameter, constraint in self._constraints.items():
                if parameter not in self.__dict__:
                    # Useful for cases when the distribution can be parameterized in multiple ways,
                    # e.g. the Gaussian distribution in terms of variance or precision.
                    continue

                parameter_value = getattr(self, parameter)

                if not constraint(parameter_value):
                    raise ValueError(f'Invalid value for {parameter}: {parameter_value}.')

    @property
    def support(self):
        return self._support

    @property
    @abc.abstractmethod
    def mean(self):
        pass

    @property
    @abc.abstractmethod
    def variance(self):
        pass

    def log_prob(self, x):
        self._validate_input(x)
        return self._log_prob(x)

    @abc.abstractmethod
    def _log_prob(self, x):
        pass

    def sample(self, sample_shape, random_state=None):
        random_state = utils.check_random_state(random_state)
        return self._sample(sample_shape, random_state)

    @abc.abstractmethod
    def _sample(self, sample_shape, random_state):
        pass

    def _validate_input(self, x):
        if self.check_support and not np.all(self._support(x)):
            raise ValueError(f'The parameter {x} lies outside the support.')

