from typing import Dict, Tuple

import numpy as np
from numpy.random import RandomState
from scipy.special import binom, gamma, gammaln

from pynference.constants import ArrayLike, Parameter, Shape, Variate
from pynference.distributions.constraints import (
    Constraint,
    integer,
    integer_interval,
    non_negative_integer,
    positive,
    real,
    zero_one,
    zero_one_integer,
)
from pynference.distributions.distribution import Distribution, ExponentialFamily
from pynference.distributions.utils import (
    broadcast_shapes,
    log_binomial_coefficient,
    promote_shapes,
    sum_last,
)


class Bernoulli(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"p": zero_one}
    _support: Constraint = zero_one_integer

    def __init__(
        self, p: Parameter, check_parameters: bool = True, check_support: bool = True
    ):
        batch_shape = np.shape(p)
        rv_shape = ()

        self.p = p

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        return self.p

    @property
    def variance(self) -> Parameter:
        return self.p * (1.0 - self.p)

    def _log_prob(self, x: Variate) -> ArrayLike:
        return x * np.log(self.p) + (1.0 - x) * np.log1p(-self.p)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        return random_state.binomial(
            n=1, p=self.p, size=sample_shape + self.batch_shape
        )

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return (np.log(self.p) - np.log1p(-self.p),)

    @property
    def log_normalizer(self) -> Parameter:
        return -np.log1p(-self.p)

    def base_measure(self, x: Variate) -> ArrayLike:
        return 1.0

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return (x,)


class Binomial(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"n": non_negative_integer, "p": zero_one}

    def __init__(
        self,
        n: Parameter,
        p: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        batch_shape = broadcast_shapes(np.shape(n), np.shape(p))
        rv_shape = ()

        self.n, self.p = promote_shapes(n, p)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def support(self) -> Constraint:
        return integer_interval(0, self.n, include_lower=True, include_upper=True)

    @property
    def mean(self) -> Parameter:
        return self.n * self.p

    @property
    def variance(self) -> Parameter:
        return self.n * self.p * (1.0 - self.p)

    def _log_prob(self, x: Variate) -> ArrayLike:
        return (
            log_binomial_coefficient(self.n, x)
            + x * np.log(self.p)
            + (self.n - x) * np.log1p(-self.p)
        )

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        return random_state.binomial(
            n=self.n, p=self.p, size=sample_shape + self.batch_shape
        )

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return (np.log(self.p) - np.log1p(-self.p),)

    @property
    def log_normalizer(self) -> Parameter:
        return -self.n * np.log1p(-self.p)

    def base_measure(self, x: Variate) -> ArrayLike:
        return binom(self.n, x)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return (x,)


class Dirac(Distribution):
    _constraints: Dict[str, Constraint] = {"x": real}
    _support: Constraint = real

    def __init__(
        self,
        x: Parameter,
        rv_dim=0,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        batch_dim = np.ndim(x) - rv_dim
        batch_shape = np.shape(x)[:batch_dim]
        rv_shape = np.shape(x)[batch_dim:]

        self.x = x

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        return self.x

    @property
    def variance(self) -> Parameter:
        return np.zeros(shape=self.batch_shape + self.rv_shape)

    def _log_prob(self, x: Variate) -> ArrayLike:
        log_prob = np.log(x == self.x)
        return sum_last(log_prob, len(self.rv_shape))

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        shape = sample_shape + self.batch_shape + self.rv_shape
        return np.broadcast_to(self.x, shape)


class DiscreteUniform(Distribution):
    _constraints: Dict[str, Constraint] = {"lower": integer, "upper": integer}

    def __init__(
        self,
        lower: Parameter,
        upper: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        batch_shape = broadcast_shapes(np.shape(lower), np.shape(upper))
        rv_shape = ()

        self.lower, self.upper = promote_shapes(lower, upper)

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
    def support(self) -> Constraint:
        return integer_interval(
            self.lower, self.upper, include_lower=True, include_upper=True
        )

    @property
    def mean(self) -> Parameter:
        return (self.lower + self.upper) / 2.0

    @property
    def variance(self) -> Parameter:
        return (np.square(self.upper - self.lower + 1.0) - 1.0) / 12.0

    def _log_prob(self, x: Variate) -> ArrayLike:
        return -np.log(self.upper - self.lower + 1.0)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        return random_state.randint(
            low=self.lower,
            high=self.upper + 1,
            size=sample_shape + self.batch_shape,
            dtype=self.lower.dtype,
        )


class Geometric(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"p": zero_one}
    _support: Constraint = non_negative_integer

    def __init__(
        self, p: Parameter, check_parameters: bool = True, check_support: bool = True
    ):
        batch_shape = np.shape(p)
        rv_shape = ()

        self.p = p

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        return (1.0 - self.p) / self.p

    @property
    def variance(self) -> Parameter:
        return (1.0 - self.p) / np.square(self.p)

    def _log_prob(self, x: Variate) -> ArrayLike:
        return x * np.log1p(-self.p) + np.log(self.p)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        return random_state.geometric(p=self.p, size=sample_shape + self.batch_shape)

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return (np.log1p(-self.p),)

    @property
    def log_normalizer(self) -> Parameter:
        return -np.log(self.p)

    def base_measure(self, x: Variate) -> ArrayLike:
        return 1.0

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return (x,)


class NegativeBinomial(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"r": positive, "p": zero_one}
    _support: Constraint = non_negative_integer

    def __init__(
        self,
        r: Parameter,
        p: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        batch_shape = broadcast_shapes(np.shape(r), np.shape(p))
        rv_shape = ()

        self.r, self.p = promote_shapes(r, p)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        return self.r * self.p / (1.0 - self.p)

    @property
    def variance(self) -> Parameter:
        return self.r * self.p / np.square(1.0 - self.p)

    def _log_prob(self, x: Variate) -> ArrayLike:
        return (
            log_binomial_coefficient(self.r + x - 1, x)
            + self.r * np.log1p(-self.p)
            + x * np.log(self.p)
        )

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        return random_state.negative_binomial(
            n=self.r, p=1.0 - self.p, size=sample_shape + self.batch_shape
        )

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return (np.log(self.p),)

    @property
    def log_normalizer(self) -> Parameter:
        return -self.r * np.log1p(-self.p)

    def base_measure(self, x: Variate) -> ArrayLike:
        return binom(x + self.r - 1.0, x)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return (x,)


class Poisson(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"rate": positive}
    _support: Constraint = non_negative_integer

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
        return self.rate

    @property
    def variance(self) -> Parameter:
        return self.rate

    def _log_prob(self, x: Variate) -> ArrayLike:
        return x * np.log(self.rate) - gammaln(x + 1) - self.rate

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        return random_state.poisson(lam=self.rate, size=sample_shape + self.batch_shape)

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return (np.log(self.rate),)

    @property
    def log_normalizer(self) -> Parameter:
        return self.rate

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.reciprocal(gamma(x + 1.0))

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return (x,)
