from typing import Dict

import numpy as np

from constraints import Constraint, non_negative, positive, real, zero_one
from distribution import Distribution, ExponentialFamily
from utils import broadcast_shapes


class Beta(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"shape1": positive, "shape2": positive}
    _support: Constraint = zero_one

    def __init__(self, shape1, shape2, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(shape1), np.shape(shape2))
        rv_shape = ()

        self.shape1 = np.broadcast_to(shape1, batch_shape)
        self.shape2 = np.broadcast_to(shape2, batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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
    _constraints: Dict[str, Constraint] = {"loc": real, "scale": positive}
    _support: Constraint = real

    def __init__(self, loc, scale, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(loc), np.shape(scale))
        rv_shape = ()

        self.loc = np.broadcast_to(loc, batch_shape)
        self.scale = np.broadcast_to(scale, batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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
    _constraints: Dict[str, Constraint] = {"rate": positive}
    _support: Constraint = non_negative

    def __init__(self, rate, check_parameters=True, check_support=True):
        batch_shape = np.shape(rate)
        rv_shape = ()

        self.rate = rate

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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


class Gamma(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"shape": positive, "rate": positive}
    _support: Constraint = positive

    def __init__(self, shape, rate, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(shape), np.shape(rate))
        rv_shape = ()

        self.shape = np.broadcast_to(shape, batch_shape)
        self.rate = np.broadcast_to(rate, batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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


# TODO: Transformed Gamma.
class InverseGamma(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"shape": positive, "scale": positive}
    _support: Constraint = positive

    def __init__(self, shape, scale, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(shape), np.shape(scale))
        rv_shape = ()

        self.shape = np.broadcast_to(shape, batch_shape)
        self.scale = np.broadcast_to(scale, batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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


class Laplace(Distribution):
    _constraints: Dict[str, Constraint] = {"loc": real, "scale": positive}
    _support: Constraint = real

    def __init__(self, loc, scale, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(loc), np.shape(scale))
        rv_shape = ()

        self.loc = np.broadcast_to(loc, batch_shape)
        self.scale = np.broadcast_to(scale, batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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


class Logistic(Distribution):
    _constraints: Dict[str, Constraint] = {"loc": real, "scale": positive}
    _support: Constraint = real

    def __init__(self, loc, scale, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(loc), np.shape(scale))
        rv_shape = ()

        self.loc = np.broadcast_to(loc, batch_shape)
        self.scale = np.broadcast_to(scale, batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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


# TODO: Transformed Normal.
class LogNormal(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"loc": real, "scale": positive}
    _support: Constraint = positive

    def __init__(self, loc, scale, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(loc), np.shape(scale))
        rv_shape = ()

        self.loc = np.broadcast_to(loc, batch_shape)
        self.scale = np.broadcast_to(scale, batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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


class Normal(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"mean": real, "variance": positive}
    _support: Constraint = real

    def __init__(self, mean, variance, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(mean), np.shape(variance))
        rv_shape = ()

        self._mean = np.broadcast_to(mean, batch_shape)
        self._variance = np.broadcast_to(variance, batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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


# TODO: Transformed exponential.
class Pareto(Distribution):
    _constraints: Dict[str, Constraint] = {"scale": positive, "shape": positive}
    _support: Constraint = None

    def __init__(self, scale, shape, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(scale), np.shape(shape))
        rv_shape = ()

        self.scale = np.broadcast_to(scale, batch_shape)
        self.shape = np.broadcast_to(shape, batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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


class T(Distribution):
    _constraints: Dict[str, Constraint] = {
        "df": positive,
        "loc": real,
        "scale": positive,
    }
    _support: Constraint = real

    def __init__(self, df, loc, scale, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(df), np.shape(loc), np.shape(scale))
        rv_shape = ()

        self.df = np.broadcast_to(df, batch_shape)
        self.loc = np.broadcast_to(loc, batch_shape)
        self.scale = np.broadcast_to(scale, batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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


# TODO: Transformed base TruncatedNormal (mean=0, variance=1, lower=0, upper=1).
class TruncatedNormal(Distribution):
    _constraints: Dict[str, Constraint] = {
        "mean": real,
        "variance": positive,
        "lower": real,
        "upper": real,
    }
    _support: Constraint = None

    def __init__(
        self, mean, variance, lower, upper, check_parameters=True, check_support=True
    ):
        batch_shape = broadcast_shapes(
            np.shape(mean), np.shape(variance), np.shape(lower), np.shape(upper)
        )
        rv_shape = ()

        self._mean = np.broadcast_to(mean, batch_shape)
        self._variance = np.broadcast_to(variance, batch_shape)
        self.lower = np.broadcast_to(lower, batch_shape)
        self.upper = np.broadcast_to(upper, batch_shape)

        if not np.all(self.lower < self.upper):
            raise ValueError(
                "All the lower bounds must be strictly lower than the upper bounds."
            )

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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


# TODO: Transformed standard uniform.
class Uniform(Distribution):
    _constraints: Dict[str, Constraint] = {"lower": real, "upper": real}
    _support: Constraint = None

    def __init__(self, lower, upper, check_parameters=True, check_support=True):
        batch_shape = broadcast_shapes(np.shape(lower), np.shape(upper))
        rv_shape = ()

        self.lower = np.broadcast_to(lower, batch_shape)
        self.upper = np.broadcast_to(upper, batch_shape)

        if not np.all(self.lower < self.upper):
            raise ValueError(
                "All the lower bounds must be strictly lower than the upper bounds."
            )

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

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
