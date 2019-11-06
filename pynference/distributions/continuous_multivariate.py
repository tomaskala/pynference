from typing import Dict, Tuple

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln

from pynference.constants import ArrayLike, Parameter, Shape, Variate
from pynference.distributions.constraints import (
    Constraint,
    positive,
    simplex
)

from pynference.distributions.distribution import (
    Distribution,
    ExponentialFamily,
    TransformedDistribution,
)

from pynference.distributions.utils import broadcast_shapes, promote_shapes


class Dirichlet(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"concentration": positive}
    _support: Constraint = simplex

    def __init__(
        self,
        concentration: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        if np.ndim(concentration) < 1:
            raise ValueError("The concentration parameter must be at least 1-dimensional.")

        batch_shape = concentration.shape[:-1]
        rv_shape = concentration.shape[-1:]

        self.concentration = concentration
        self.concentration_sum = np.sum(concentration, axis=-1, keepdims=True)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support
        )

    @property
    def mean(self) -> Parameter:
        return self.concentration / self.concentration_sum

    @property
    def variance(self) -> Parameter:
        mean = self.mean
        return mean * (1.0 - mean) / (self.concentration_sum + 1.0)

    def _log_prob(self, x: Variate) -> ArrayLike:
        normalizer = np.sum(gammaln(self.concentration), axis=-1) - gammaln(np.sum(self.concentration, axis=-1))
        return np.sum((self.concentration - 1.0) * np.log(x), axis=-1) - normalizer

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_gamma(self.concentration, sample_shape + self.batch_shape + self.rv_shape)
        return epsilon / np.sum(epsilon, axis=-1, keepdims=True)

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return self.concentration,

    @property
    def log_normalizer(self) -> Parameter:
        return np.sum(gammaln(self.concentration), axis=-1) - gammaln(np.sum(self.concentration, axis=-1))

    def base_measure(self, x: Variate) -> ArrayLike:
        return 1.0 / np.product(x, axis=-1)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return np.log(x),


class InverseWishart(TransformedDistribution):
    pass


class MultivariateNormal(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {}
    _support: Constraint = None

    def __init__(
        self,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        pass

    @property
    def mean(self) -> Parameter:
        pass

    @property
    def variance(self) -> Parameter:
        pass

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        pass

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        pass

    @property
    def log_normalizer(self) -> Parameter:
        pass

    def base_measure(self, x: Variate) -> ArrayLike:
        pass

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        pass


class MultivariateT(Distribution):
    _constraints: Dict[str, Constraint] = {}
    _support: Constraint = None

    def __init__(
        self,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        pass

    @property
    def mean(self) -> Parameter:
        pass

    @property
    def variance(self) -> Parameter:
        pass

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        pass


class Wishart(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {}
    _support: Constraint = None

    def __init__(
        self,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        pass

    @property
    def mean(self) -> Parameter:
        pass

    @property
    def variance(self) -> Parameter:
        pass

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        pass

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        pass

    @property
    def log_normalizer(self) -> Parameter:
        pass

    def base_measure(self, x: Variate) -> ArrayLike:
        pass

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        pass
