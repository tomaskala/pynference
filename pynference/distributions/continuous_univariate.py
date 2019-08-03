from typing import Dict, Tuple

import numpy as np

from pynference.constants import ArrayLike, Parameter, Variate
from pynference.distribution.constraints import (
    Constraint,
    non_negative,
    positive,
    real,
    zero_one,
)
from pynference.distribution.distribution import Distribution, ExponentialFamily
from pynference.distribution.utils import broadcast_shapes


class Beta(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"shape1": positive, "shape2": positive}
    _support: Constraint = zero_one

    def __init__(
        self,
        shape1: Parameter,
        shape2: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return self.shape1 / (self.shape1 + self.shape2)

    @property
    def variance(self) -> Parameter:
        shape_sum = self.shape1 + self.shape2
        numerator = self.shape1 * self.shape2
        denominator = np.power(shape_sum, 2) * (shape_sum + 1.0)
        return numerator / denominator

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass

    @property
    def natural_parameter(self) -> Parameter:
        pass

    @property
    def log_normalizer(self) -> Parameter:
        pass

    def base_measure(self, x: Variate) -> ArrayLike:
        pass

    def sufficient_statistic(self, x: Variate) -> ArrayLike:
        pass


class Cauchy(Distribution):
    _constraints: Dict[str, Constraint] = {"loc": real, "scale": positive}
    _support: Constraint = real

    def __init__(
        self,
        loc: Parameter,
        scale: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return np.full(shape=self.batch_shape, fill_value=np.nan)

    @property
    def variance(self) -> Parameter:
        return np.full(shape=self.batch_shape, fill_value=np.nan)

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass


class Exponential(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"rate": positive}
    _support: Constraint = non_negative

    def __init__(
        self, rate: Parameter, check_parameters: bool = True, check_support: bool = True
    ):
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
    def mean(self) -> Parameter:
        return np.reciprocal(self.rate)

    @property
    def variance(self) -> Parameter:
        return np.reciprocal(np.power(self.rate, 2))

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass

    @property
    def natural_parameter(self) -> Parameter:
        pass

    @property
    def log_normalizer(self) -> Parameter:
        pass

    def base_measure(self, x: Variate) -> ArrayLike:
        pass

    def sufficient_statistic(self, x: Variate) -> ArrayLike:
        pass


class Gamma(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"shape": positive, "rate": positive}
    _support: Constraint = positive

    def __init__(
        self,
        shape: Parameter,
        rate: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return self.shape / self.rate

    @property
    def variance(self) -> Parameter:
        return self.shape / np.power(self.rate, 2)

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass

    @property
    def natural_parameter(self) -> Parameter:
        pass

    @property
    def log_normalizer(self) -> Parameter:
        pass

    def base_measure(self, x: Variate) -> ArrayLike:
        pass

    def sufficient_statistic(self, x: Variate) -> ArrayLike:
        pass


# TODO: Transformed Gamma.
class InverseGamma(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"shape": positive, "scale": positive}
    _support: Constraint = positive

    def __init__(
        self,
        shape: Parameter,
        scale: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return np.where(self.shape > 1.0, self.scale / (self.shape - 1.0), np.nan)

    @property
    def variance(self) -> Parameter:
        return np.where(
            self.shape > 2.0,
            np.power(self.scale, 2)
            / (np.power(self.shape - 1.0, 2) * (self.shape - 2.0)),
            np.nan,
        )

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass

    @property
    def natural_parameter(self) -> Parameter:
        pass

    @property
    def log_normalizer(self) -> Parameter:
        pass

    def base_measure(self, x: Variate) -> ArrayLike:
        pass

    def sufficient_statistic(self, x: Variate) -> ArrayLike:
        pass


class Laplace(Distribution):
    _constraints: Dict[str, Constraint] = {"loc": real, "scale": positive}
    _support: Constraint = real

    def __init__(
        self,
        loc: Parameter,
        scale: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return self.loc

    @property
    def variance(self) -> Parameter:
        return 2.0 * np.power(self.scale, 2)

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass


class Logistic(Distribution):
    _constraints: Dict[str, Constraint] = {"loc": real, "scale": positive}
    _support: Constraint = real

    def __init__(
        self,
        loc: Parameter,
        scale: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return self.loc

    @property
    def variance(self) -> Parameter:
        return np.power(self.scale * np.pi, 2) / 3.0

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass


# TODO: Transformed Normal.
class LogNormal(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"loc": real, "scale": positive}
    _support: Constraint = positive

    def __init__(
        self,
        loc: Parameter,
        scale: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return np.exp(self.loc + np.power(self.scale, 2) / 2.0)

    @property
    def variance(self) -> Parameter:
        scale_squared = np.power(self.scale, 2)
        return (np.exp(scale_squared) - 1.0) * np.exp(2.0 * self.loc + scale_squared)

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass

    @property
    def natural_parameter(self) -> Parameter:
        pass

    @property
    def log_normalizer(self) -> Parameter:
        pass

    def base_measure(self, x: Variate) -> ArrayLike:
        pass

    def sufficient_statistic(self, x: Variate) -> ArrayLike:
        pass


# TODO: Allow parameterizing in terms of mean & precision instead.
class Normal(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"mean": real, "variance": positive}
    _support: Constraint = real

    def __init__(
        self,
        mean: Parameter,
        variance: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return self._mean

    @property
    def variance(self) -> Parameter:
        return self._variance

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass

    @property
    def natural_parameter(self) -> Parameter:
        pass

    @property
    def log_normalizer(self) -> Parameter:
        pass

    def base_measure(self, x: Variate) -> ArrayLike:
        pass

    def sufficient_statistic(self, x: Variate) -> ArrayLike:
        pass


# TODO: Transformed exponential.
class Pareto(Distribution):
    _constraints: Dict[str, Constraint] = {"scale": positive, "shape": positive}
    _support: Constraint = None

    def __init__(
        self,
        scale: Parameter,
        shape: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return np.where(
            self.shape > 1.0, self.shape * self.scale / (self.shape - 1.0), np.inf
        )

    @property
    def variance(self) -> Parameter:
        return np.where(
            self.shape > 2.0,
            np.power(self.scale, 2)
            * self.shape
            / (np.power(self.shape - 1.0, 2) * (self.shape - 2.0)),
            np.inf,
        )

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass


class T(Distribution):
    _constraints: Dict[str, Constraint] = {
        "df": positive,
        "loc": real,
        "scale": positive,
    }
    _support: Constraint = real

    def __init__(
        self,
        df: Parameter,
        loc: Parameter,
        scale: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return np.where(self.df > 1, self.loc, np.nan)

    @property
    def variance(self) -> Parameter:
        variance = np.where(
            self.df > 2, np.power(self.scale, 2) * self.df / (self.df - 2.0), np.inf
        )
        return np.where(self.df > 1, variance, np.nan)

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass


# TODO: Transformed base TruncatedNormal (loc=0, scale=1, lower=0, upper=1).
class TruncatedNormal(Distribution):
    _constraints: Dict[str, Constraint] = {
        "loc": real,
        "scale": positive,
        "lower": real,
        "upper": real,
    }
    _support: Constraint = None

    def __init__(
        self,
        loc: Parameter,
        scale: Parameter,
        lower: Parameter,
        upper: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        batch_shape = broadcast_shapes(
            np.shape(loc), np.shape(scale), np.shape(lower), np.shape(upper)
        )
        rv_shape = ()

        self.loc = np.broadcast_to(loc, batch_shape)
        self.scale = np.broadcast_to(scale, batch_shape)
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
    def mean(self) -> Parameter:
        pass

    @property
    def variance(self) -> Parameter:
        pass

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass


# TODO: Transformed standard uniform.
class Uniform(Distribution):
    _constraints: Dict[str, Constraint] = {"lower": real, "upper": real}
    _support: Constraint = None

    def __init__(
        self,
        lower: Parameter,
        upper: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
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
    def mean(self) -> Parameter:
        return (self.lower + self.upper) / 2.0

    @property
    def variance(self) -> Parameter:
        return np.power(self.upper - self.lower, 2) / 12.0

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Tuple[int], random_state) -> Variate:
        pass
