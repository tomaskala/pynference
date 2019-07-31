from typing import Dict

from distribution import Distribution, ExponentialFamily
from distribution.constraints import Constraint


class Beta(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {}
    _support: Constraint = None

    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass

    def _log_prob(self, x):
        pass

    def _sample(self, sample_shape, random_state):
        pass

    @property
    def natural_parameter(self):
        pass

    @property
    def log_normalizer(self):
        pass

    def base_measure(self, x):
        pass

    def sufficient_statistic(self, x):
        pass


class Cauchy(Distribution):
    _constraints: Dict[str, Constraint] = {}
    _support: Constraint = None

    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass

    def _log_prob(self, x):
        pass

    def _sample(self, sample_shape, random_state):
        pass


class Exponential(ExponentialFamily):
    pass


class Gamma(ExponentialFamily):
    pass


class InvGamma(ExponentialFamily):
    pass


class Laplace(Distribution):
    pass


class Logistic(Distribution):
    pass


class LogNormal(ExponentialFamily):
    pass


class Normal(ExponentialFamily):
    pass


class Pareto(Distribution):
    pass


class T(Distribution):
    pass


class TruncatedExponential(Distribution):
    pass


class TruncatedNormal(Distribution):
    pass


class Uniform(Distribution):
    pass
