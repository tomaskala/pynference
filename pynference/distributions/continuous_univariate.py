from typing import Dict, Tuple

import numpy as np
from numpy.random import RandomState
from scipy.special import betaln, gammaln, ndtr, ndtri

from pynference.constants import ArrayLike, Parameter, Shape, Variate
from pynference.distributions.constraints import (
    Constraint,
    interval,
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
from pynference.distributions.utils import broadcast_shapes, promote_shapes


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

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.shape1, self.shape2 = promote_shapes(shape1, shape2)

    @property
    def mean(self) -> Parameter:
        return self.shape1 / (self.shape1 + self.shape2)

    @property
    def variance(self) -> Parameter:
        shape_sum = self.shape1 + self.shape2
        numerator = self.shape1 * self.shape2
        denominator = np.square(shape_sum) * (shape_sum + 1.0)
        return numerator / denominator

    def _log_prob(self, x: Variate) -> ArrayLike:
        return (
            (self.shape1 - 1.0) * np.log(x)
            + (self.shape2 - 1.0) * np.log1p(-x)
            - betaln(self.shape1, self.shape2)
        )

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        x = random_state.standard_gamma(self.shape1, sample_shape + self.batch_shape)
        y = random_state.standard_gamma(self.shape2, sample_shape + self.batch_shape)
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

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.loc, self.scale = promote_shapes(loc, scale)

    @property
    def mean(self) -> Parameter:
        return np.full(shape=self.batch_shape, fill_value=np.nan)

    @property
    def variance(self) -> Parameter:
        return np.full(shape=self.batch_shape, fill_value=np.nan)

    def _log_prob(self, x: Variate) -> ArrayLike:
        normalizer = -np.log(np.pi) - np.log(self.scale)
        return -np.log1p(np.square((x - self.loc) / self.scale)) + normalizer

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_cauchy(sample_shape + self.batch_shape)
        return self.loc + self.scale * epsilon


class Exponential(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"rate": positive}
    _support: Constraint = positive

    def __init__(
        self, rate: Parameter, check_parameters: bool = True, check_support: bool = True
    ):
        batch_shape = np.shape(rate)
        rv_shape = ()

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.rate = rate

    @property
    def mean(self) -> Parameter:
        return np.reciprocal(self.rate)

    @property
    def variance(self) -> Parameter:
        return np.reciprocal(np.square(self.rate))

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

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.shape, self.rate = promote_shapes(shape, rate)

    @property
    def mean(self) -> Parameter:
        return self.shape / self.rate

    @property
    def variance(self) -> Parameter:
        return self.shape / np.square(self.rate)

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

        super().__init__(
            base_distribution=base_distribution,
            transformation=transformation,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.shape = base_distribution.shape
        self.scale = base_distribution.rate

    @property
    def mean(self) -> Parameter:
        return np.where(self.shape > 1.0, self.scale / (self.shape - 1.0), np.nan)

    @property
    def variance(self) -> Parameter:
        return np.where(
            self.shape > 2.0,
            np.square(self.scale) / np.square(self.shape - 1.0) / (self.shape - 2.0),
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

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.loc, self.scale = promote_shapes(loc, scale)

    @property
    def mean(self) -> Parameter:
        return self.loc

    @property
    def variance(self) -> Parameter:
        return 2.0 * np.square(self.scale)

    def _log_prob(self, x: Variate) -> ArrayLike:
        return -np.abs(x - self.loc) / self.scale - np.log(2.0) - np.log(self.scale)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        x = random_state.random_sample(sample_shape + self.batch_shape)
        y = random_state.random_sample(sample_shape + self.batch_shape)
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

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.loc, self.scale = promote_shapes(loc, scale)

    @property
    def mean(self) -> Parameter:
        return self.loc

    @property
    def variance(self) -> Parameter:
        return np.square(self.scale * np.pi) / 3.0

    def _log_prob(self, x: Variate) -> ArrayLike:
        standardized_var = (x - self.loc) / self.scale
        return (
            -standardized_var
            - np.log(self.scale)
            - 2.0 * np.log1p(np.exp(-standardized_var))
        )

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        x = random_state.random_sample(sample_shape + self.batch_shape)
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
            std=scale,
            check_parameters=check_parameters,
            check_support=check_support,
        )
        transformation = ExpTransformation()

        super().__init__(
            base_distribution=base_distribution,
            transformation=transformation,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.loc = base_distribution.loc
        self.scale = base_distribution.scale

    @property
    def mean(self) -> Parameter:
        return np.exp(self.loc + 0.5 * np.square(self.scale))

    @property
    def variance(self) -> Parameter:
        scale_squared = np.square(self.scale)
        return np.expm1(scale_squared) * np.exp(2.0 * self.loc + scale_squared)

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return (
            self.loc / np.square(self.scale),
            -np.reciprocal(2.0 * np.square(self.scale)),
        )

    @property
    def log_normalizer(self) -> Parameter:
        return np.square(self.loc) / (2.0 * np.square(self.scale)) + np.log(self.scale)

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.reciprocal(np.sqrt(2.0 * np.pi) * x)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        log_x = np.log(x)
        return log_x, np.square(log_x)


class Normal(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {
        "_mean": real,
        "_variance": positive,
        "_precision": positive,
        "_std": positive,
    }
    _support: Constraint = real

    def __init__(
        self,
        mean: Parameter,
        variance: Parameter = None,
        precision: Parameter = None,
        std: Parameter = None,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        if (variance is not None) + (precision is not None) + (std is not None) != 1:
            raise ValueError(
                "Provide exactly one of the variance, precision or standard deviation parameters."
            )

        if variance is not None:
            batch_shape = broadcast_shapes(np.shape(mean), np.shape(variance))
        elif precision is not None:
            batch_shape = broadcast_shapes(np.shape(mean), np.shape(precision))
        else:
            batch_shape = broadcast_shapes(np.shape(mean), np.shape(std))

        rv_shape = ()

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        if variance is not None:
            self._mean, self._variance = promote_shapes(mean, variance)
            self._precision = np.reciprocal(self._variance)
            self._std = np.sqrt(self._variance)
        elif precision is not None:
            self._mean, self._precision = promote_shapes(mean, precision)
            self._variance = np.reciprocal(self._precision)
            self._std = np.sqrt(self._variance)
        else:
            self._mean, self._std = promote_shapes(mean, std)
            self._variance = np.square(self._std)
            self._precision = np.reciprocal(self._variance)

    @property
    def mean(self) -> Parameter:
        return self._mean

    @property
    def variance(self) -> Parameter:
        return self._variance

    @property
    def precision(self) -> Parameter:
        return self._precision

    @property
    def std(self) -> Parameter:
        return self._std

    @property
    def loc(self) -> Parameter:
        return self._mean

    @property
    def scale(self) -> Parameter:
        return self._std

    def _log_prob(self, x: Variate) -> ArrayLike:
        normalizer = -0.5 * (np.log(2.0) + np.log(np.pi) + np.log(self._variance))
        return -np.square(x - self._mean) * 0.5 * self._precision + normalizer

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_normal(sample_shape + self.batch_shape)
        return self._mean + self._std * epsilon

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return self._mean * self._precision, -0.5 * self._precision

    @property
    def log_normalizer(self) -> Parameter:
        return np.square(self._mean) * 0.5 * self._precision + np.log(self._std)

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.power(2.0 * np.pi, -0.5)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return x, np.square(x)


class Pareto(TransformedDistribution):
    _constraints: Dict[str, Constraint] = {"scale": positive, "shape": positive}

    def __init__(
        self,
        scale: Parameter,
        shape: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        base_distribution = Exponential(
            rate=shape, check_parameters=check_parameters, check_support=check_support
        )
        transformation = [
            ExpTransformation(),
            AffineTransformation(loc=0.0, scale=scale),
        ]

        super().__init__(
            base_distribution=base_distribution,
            transformation=transformation,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.scale, self.shape = promote_shapes(scale, shape)

    @property
    def mean(self) -> Parameter:
        return np.where(
            self.shape > 1.0, self.shape * self.scale / (self.shape - 1.0), np.inf
        )

    @property
    def variance(self) -> Parameter:
        return np.where(
            self.shape > 2.0,
            np.square(self.scale)
            * self.shape
            / (np.square(self.shape - 1.0) * (self.shape - 2.0)),
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

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.df = np.broadcast_to(df, batch_shape)
        self.loc, self.scale = promote_shapes(loc, scale)

    @property
    def mean(self) -> Parameter:
        return np.where(self.df > 1, self.loc, np.nan)

    @property
    def variance(self) -> Parameter:
        variance = np.square(self.scale) * self.df / (self.df - 2.0)
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
            -((self.df + 1.0) / 2.0) * np.log1p(np.square(standardized_var) / self.df)
            + normalizer
        )

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_t(self.df, sample_shape + self.batch_shape)
        return self.loc + self.scale * epsilon


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
        if not np.all(lower < upper):
            raise ValueError(
                "All the lower bounds must be strictly lower than the upper bounds."
            )

        batch_shape = broadcast_shapes(
            np.shape(loc), np.shape(scale), np.shape(lower), np.shape(upper)
        )
        rv_shape = ()

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.loc, self.scale, self.lower, self.upper = promote_shapes(
            loc, scale, lower, upper
        )

        self._alpha = (self.lower - self.loc) / self.scale
        self._beta = (self.upper - self.loc) / self.scale
        self._Z = ndtr(self._beta) - ndtr(self._alpha)
        self._phi_alpha = self._phi(self._alpha)
        self._phi_beta = self._phi(self._beta)
        self._Phi_alpha = ndtr(self._alpha)

        self._standard_uniform = _StandardUniform(
            batch_shape=batch_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def support(self) -> Constraint:
        return interval(lower=self.lower, upper=self.upper)

    @property
    def mean(self) -> Parameter:
        return self.loc + (self._phi_alpha - self._phi_beta) * self.scale / self._Z

    @property
    def variance(self) -> Parameter:
        fst = (self._alpha * self._phi_alpha - self._beta * self._phi_beta) / self._Z
        snd = np.square((self._phi_alpha - self._phi_beta) / self._Z)
        return np.square(self.scale) * (1.0 + fst - snd)

    def _log_prob(self, x: Variate) -> ArrayLike:
        xi = (x - self.loc) / self.scale
        return self._log_phi(xi) - np.log(self.scale) - np.log(self._Z)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = self._standard_uniform.sample(sample_shape, random_state)
        return ndtri(self._Phi_alpha + epsilon * self._Z) * self.scale + self.loc

    def _phi(self, x: Variate) -> ArrayLike:
        return np.exp(-np.square(x) / 2.0) / np.sqrt(2.0 * np.pi)

    def _log_phi(self, x: Variate) -> ArrayLike:
        return -np.square(x) / 2.0 - np.log(2.0) / 2.0 - np.log(np.pi) / 2.0


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
        return random_state.random_sample(sample_shape + self.batch_shape)


class Uniform(TransformedDistribution):
    _constraints: Dict[str, Constraint] = {"lower": real, "upper": real}

    def __init__(
        self,
        lower: Parameter,
        upper: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        if not np.all(lower < upper):
            raise ValueError(
                "All the lower bounds must be strictly lower than the upper bounds."
            )

        batch_shape = broadcast_shapes(np.shape(lower), np.shape(upper))

        base_distribution = _StandardUniform(
            batch_shape=batch_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )
        transformation = AffineTransformation(loc=lower, scale=upper - lower)

        super().__init__(
            base_distribution=base_distribution,
            transformation=transformation,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.lower, self.upper = promote_shapes(lower, upper)

    @property
    def support(self) -> Constraint:
        return interval(self.lower, self.upper)

    @property
    def mean(self) -> Parameter:
        return (self.lower + self.upper) / 2.0

    @property
    def variance(self) -> Parameter:
        return np.square(self.upper - self.lower) / 12.0
