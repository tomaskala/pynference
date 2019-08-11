from typing import Dict, Tuple

import numpy as np
from numpy.random import RandomState
from scipy.special import betaln, gammaln, ndtr

from pynference.constants import ArrayLike, Parameter, Shape, Variate
from pynference.distributions.constraints import (
    Constraint,
    interval,
    non_negative,
    positive,
    real,
    zero_one,
)
from pynference.distributions.distribution import (
    Distribution,
    ExponentialFamily,
    TransformedDistribution,
)
from pynference.distributions.transformations import (
    AffineTransformation,
    ExpTransformation,
    PowerTransformation,
)
from pynference.distributions.utils import broadcast_shapes


# TODO: Just promote shapes instead of broadcasting, this works in most distributions.
# TODO: When testing, do not forget to test transformed distribution batch and rv shapes!
# TODO: Truncated normal.
# TODO: Unit tests.
# TODO: Test batch constraints (many real numbers, many positive numbers, ...)
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

        self._gamma1 = Gamma(self.shape1, 1.0)
        self._gamma2 = Gamma(self.shape2, 1.0)

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
        return (
            (self.shape1 - 1.0) * np.log(x)
            + (self.shape2 - 1.0) * np.log1p(-x)
            - betaln(self.shape1, self.shape2)
        )

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        x = self._gamma1.sample(sample_shape, random_state)
        y = self._gamma2.sample(sample_shape, random_state)
        return x / (x + y)

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return self.shape1, self.shape2

    @property
    def log_normalizer(self) -> Parameter:
        return betaln(self.shape1, self.shape2)

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.reciprocal(x * (1.0 - x))

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return np.log(x), np.log1p(-x)


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
        normalizer = -np.log(np.pi) - np.log(self.scale)
        return -np.log1p(np.power((x - self.loc) / self.scale, 2)) + normalizer

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_cauchy(sample_shape + self.batch_shape)
        return self.loc + self.scale * epsilon


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
        return np.log(self.rate) - self.rate * x

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_exponential(sample_shape + self.batch_shape)
        return epsilon / self.rate

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return (-self.rate,)

    @property
    def log_normalizer(self) -> Parameter:
        return -np.log(self.rate)

    def base_measure(self, x: Variate) -> ArrayLike:
        return 1.0

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return (x,)


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
        normalizer = self.shape * np.log(self.rate) - gammaln(self.shape)
        return (self.shape - 1.0) * np.log(x) - self.rate * x + normalizer

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_gamma(
            self.shape, sample_shape + self.batch_shape
        )
        return epsilon / self.rate

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return self.shape - 1.0, -self.rate

    @property
    def log_normalizer(self) -> Parameter:
        return gammaln(self.shape) - self.shape * np.log(self.rate)

    def base_measure(self, x: Variate) -> ArrayLike:
        return 1.0

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return np.log(x), x


class InverseGamma(TransformedDistribution, ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"shape": positive, "scale": positive}

    def __init__(
        self,
        shape: Parameter,
        scale: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        base_distribution = Gamma(
            shape=shape,
            rate=scale,
            check_parameters=check_parameters,
            check_support=check_support,
        )
        transformation = PowerTransformation(power=-1.0)

        self.shape = base_distribution.shape
        self.scale = base_distribution.rate

        super().__init__(
            base_distribution=base_distribution,
            transformation=transformation,
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

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return -self.shape - 1.0, -self.scale

    @property
    def log_normalizer(self) -> Parameter:
        return gammaln(self.shape) - self.shape * np.log(self.scale)

    def base_measure(self, x: Variate) -> ArrayLike:
        return 1.0

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return np.log(x), np.reciprocal(x)


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
        return -np.abs(x - self.loc) / self.scale - np.log(2.0) - np.log(self.scale)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        x = random_state.random(sample_shape + self.batch_shape)
        y = random_state.random(sample_shape + self.batch_shape)
        epsilon = np.log(x) - np.log(y)
        return self.loc + self.scale * epsilon


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
        standardized_var = (x - self.loc) / self.scale
        return (
            -standardized_var
            - np.log(self.scale)
            - 2.0 * np.log1p(np.exp(-standardized_var))
        )

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        x = random_state.random(sample_shape + self.batch_shape)
        return self.loc + self.scale * (np.log(x) - np.log1p(-x))


class LogNormal(TransformedDistribution, ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"loc": real, "scale": positive}

    def __init__(
        self,
        loc: Parameter,
        scale: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        base_distribution = Normal(
            mean=loc,
            variance=np.power(loc, 2),
            check_parameters=check_parameters,
            check_support=check_support,
        )
        transformation = ExpTransformation()

        self.loc = base_distribution.loc
        self.scale = base_distribution.scale

        super().__init__(
            base_distribution=base_distribution,
            transformation=transformation,
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

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return (
            self.loc / np.power(self.scale, 2),
            -np.reciprocal(2.0 * np.power(self.scale, 2)),
        )

    @property
    def log_normalizer(self) -> Parameter:
        return np.power(self.loc, 2) / (2.0 * np.power(self.scale, 2)) + np.log(
            self.scale
        )

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.reciprocal(np.sqrt(2.0 * np.pi) * x)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        log_x = np.log(x)
        return log_x, np.power(log_x, 2)


# TODO: Allow parameterizing in terms of mean & precision instead.
# TODO: Possibly also mean & std. In this case, modify lognormal appropriately.
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

        self._std = np.sqrt(self._variance)

    @property
    def mean(self) -> Parameter:
        return self._mean

    @property
    def variance(self) -> Parameter:
        return self._variance

    @property
    def loc(self) -> Parameter:
        return self._mean

    @property
    def scale(self) -> Parameter:
        return self._std

    def _log_prob(self, x: Variate) -> ArrayLike:
        normalizer = -0.5 * (np.log(2.0) + np.log(np.pi) + np.log(self._variance))
        return -np.power(x - self._mean, 2) / (2.0 * self._variance) + normalizer

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_normal(sample_shape + self.batch_shape)
        return self._mean + self._std * epsilon

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return self._mean / self._variance, -np.reciprocal(2.0 * self._variance)

    @property
    def log_normalizer(self) -> Parameter:
        return np.power(self._mean, 2) / (2.0 * self._variance) + np.log(self._std)

    def base_measure(self, x: Variate) -> ArrayLike:
        return 1.0 / np.sqrt(2.0 * np.pi)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return x, np.power(x, 2)


class Pareto(TransformedDistribution):
    _constraints: Dict[str, Constraint] = {"scale": positive, "shape": positive}

    def __init__(
        self,
        scale: Parameter,
        shape: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        batch_shape = broadcast_shapes(np.shape(scale), np.shape(shape))

        self.scale = np.broadcast_to(scale, batch_shape)
        self.shape = np.broadcast_to(shape, batch_shape)

        base_distribution = Exponential(
            rate=self.shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )
        transformation = [
            ExpTransformation(),
            AffineTransformation(loc=0.0, scale=self.scale),
        ]

        super().__init__(
            base_distribution=base_distribution,
            transformation=transformation,
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
        standardized_var = (x - self.loc) / self.scale
        normalizer = (
            gammaln((self.df + 1.0) / 2.0)
            - 0.5 * np.log(self.df)
            - 0.5 * np.log(np.pi)
            - gammaln(self.df / 2.0)
            - np.log(self.scale)
        )
        return (
            -((self.df + 1.0) / 2.0) * np.log1p(np.power(standardized_var, 2) / self.df)
            + normalizer
        )

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_t(self.df, sample_shape + self.batch_shape)
        return self.loc + self.scale * epsilon


class _StandardTruncatedNormal(Distribution):
    def __init__(
        self,
        batch_shape: Shape,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        rv_shape = ()

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        raise NotImplementedError()

    @property
    def variance(self) -> Parameter:
        raise NotImplementedError()

    def _log_prob(self, x: Variate) -> ArrayLike:
        raise NotImplementedError()

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        raise NotImplementedError()


class TruncatedNormal(Distribution):
    _constraints: Dict[str, Constraint] = {
        "loc": real,
        "scale": positive,
        "lower": real,
        "upper": real,
    }

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

        self._alpha = (self.lower - self.loc) / self.scale
        self._beta = (self.upper - self.loc) / self.scale
        self._Z = ndtr(self._beta) - ndtr(self._alpha)
        self._phi_alpha = self._phi(self._alpha)
        self._phi_beta = self._phi(self._beta)

    @property
    def support(self) -> Constraint:
        return interval(lower=self.lower, upper=self.upper)

    @property
    def mean(self) -> Parameter:
        return self.loc + (self._phi_alpha - self._phi_beta) * self.scale / self._Z

    @property
    def variance(self) -> Parameter:
        fst = (self._alpha * self._phi_alpha - self._beta * self._phi_beta) / self._Z
        snd = np.power((self._phi_alpha - self._phi_beta) / self._Z, 2)
        return np.power(self.scale, 2) * (1.0 + fst - snd)

    def _log_prob(self, x: Variate) -> ArrayLike:
        xi = (x - self.loc) / self.scale
        return self._log_phi(xi) - np.log(self.scale) - np.log(self._Z)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        pass  # TODO

    def _phi(self, x: Variate) -> ArrayLike:
        return np.exp(-np.power(x, 2) / 2.0) / np.sqrt(2.0 * np.pi)

    def _log_phi(self, x: Variate) -> ArrayLike:
        return -np.power(x, 2) / 2.0 - np.log(2.0) / 2.0 - np.log(np.pi) / 2.0


class _StandardUniform(Distribution):
    _support: Constraint = zero_one

    def __init__(
        self,
        batch_shape: Shape,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        rv_shape = ()

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        return np.full(shape=self.batch_shape, fill_value=0.5)

    @property
    def variance(self) -> Parameter:
        return np.full(shape=self.batch_shape, fill_value=1.0 / 12.0)

    def _log_prob(self, x: Variate) -> ArrayLike:
        batch_shape = broadcast_shapes(self.batch_shape, np.shape(x))
        return np.zeros(batch_shape)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        return random_state.random(sample_shape + self.batch_shape)


class Uniform(TransformedDistribution):
    _constraints: Dict[str, Constraint] = {"lower": real, "upper": real}

    def __init__(
        self,
        lower: Parameter,
        upper: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        batch_shape = broadcast_shapes(np.shape(lower), np.shape(upper))

        self.lower = np.broadcast_to(lower, batch_shape)
        self.upper = np.broadcast_to(upper, batch_shape)

        if not np.all(self.lower < self.upper):
            raise ValueError(
                "All the lower bounds must be strictly lower than the upper bounds."
            )

        base_distribution = _StandardUniform(
            batch_shape=batch_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )
        transformation = AffineTransformation(
            loc=self.lower, scale=self.upper - self.lower
        )

        super().__init__(
            base_distribution=base_distribution,
            transformation=transformation,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        return (self.lower + self.upper) / 2.0

    @property
    def variance(self) -> Parameter:
        return np.power(self.upper - self.lower, 2) / 12.0
