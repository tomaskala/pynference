import abc
from typing import Dict, List, Tuple, Union

import jax.numpy as np
from jax.random import PRNGKey

from pynference.constants import ArrayLike, Parameter, Shape, Variate
from pynference.distributions.constraints import Constraint
from pynference.distributions.transformations import Transformation
from pynference.distributions.utils import sum_last


class Distribution(abc.ABC):
    _constraints: Dict[str, Constraint] = {}
    _support: Constraint = None  # type: ignore

    def __init__(
        self,
        batch_shape: Shape,
        rv_shape: Shape,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        self.batch_shape = batch_shape
        self.rv_shape = rv_shape
        self.check_parameters = check_parameters
        self.check_support = check_support

    def __call__(self, key: PRNGKey, sample_shape: Shape = ()) -> Variate:
        return self.sample(key=key, sample_shape=sample_shape)

    def __setattr__(self, name, value):
        if (
            hasattr(self, "check_parameters")
            and self.check_parameters
            and name in self._constraints
        ):
            constraint = self._constraints[name]
            self._check_parameter(constraint, name, value)

        super().__setattr__(name, value)

    def _check_parameter(self, constraint, name, value):
        if not np.all(constraint(value)):
            raise ValueError(
                f"Invalid value for {name}: {value}. The parameter must satisfy the constraint '{constraint}'."
            )

    @property
    def support(self) -> Constraint:
        return self._support

    @property
    @abc.abstractmethod
    def mean(self) -> Parameter:
        pass

    @property
    @abc.abstractmethod
    def variance(self) -> Parameter:
        pass

    def log_prob(self, x: Variate) -> ArrayLike:
        if self.check_support:
            self._validate_input(x)

        return self._log_prob(x)

    @abc.abstractmethod
    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def sample(self, sample_shape: Shape, key: PRNGKey) -> Variate:
        return self._sample(sample_shape=sample_shape, key=key)

    @abc.abstractmethod
    def _sample(self, sample_shape: Shape, key: PRNGKey) -> Variate:
        pass

    def _validate_input(self, x: Variate):
        if not np.all(self.support(x)):
            raise ValueError(f"The variate {x} lies outside the support.")


class ExponentialFamily(Distribution):
    """
    Exponential family distribution of the form
        h(x) \exp{(\eta^T t(x) - a(\eta))},
    where h(x) is the base measure;
          \eta is the natural parameter;
          t(x) is the sufficient statistic;
          a(\eta) is the log-normalizer.

    The natural parameters and sufficient statistics are given as tuples instead
    of arrays since the batch dimensions of the individual parameters may differ.
    """

    @property
    @abc.abstractmethod
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        pass

    @property
    @abc.abstractmethod
    def log_normalizer(self) -> Parameter:
        pass

    @abc.abstractmethod
    def base_measure(self, x: Variate) -> ArrayLike:
        pass

    @abc.abstractmethod
    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        pass


class TransformedDistribution(Distribution):
    def __init__(
        self,
        base_distribution: Distribution,
        transformation: Union[Transformation, List[Transformation]],
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        if isinstance(transformation, Transformation):
            transformation = [transformation]

        self.base_distribution: Distribution
        self.transformation: List[Transformation]

        if isinstance(base_distribution, TransformedDistribution):
            self.base_distribution = base_distribution.base_distribution
            self.transformation = base_distribution.transformation + transformation
        else:
            self.base_distribution = base_distribution
            self.transformation = transformation

        # Register batch and random variable shapes.
        base_shape = base_distribution.batch_shape + base_distribution.rv_shape
        max_shape = max(
            [len(base_distribution.rv_shape)] + [t.rv_dim for t in transformation]
        )

        batch_shape = base_shape[: len(base_shape) - max_shape]
        rv_shape = base_shape[len(base_shape) - max_shape :]

        # Register the support after transformation.
        domain = base_distribution.support

        for t in transformation:
            t.domain = domain
            domain = t.codomain

        self._support = domain

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    def _log_prob(self, x: Variate) -> ArrayLike:
        # Transformation theorem in the log domain.
        log_prob = 0.0
        rv_dim = len(self.rv_shape)
        y = x

        for t in reversed(self.transformation):
            x = t.inverse(y)
            dim_diff = rv_dim - t.rv_dim

            log_det = t.log_abs_J(x, y)
            sum_log_det = sum_last(log_det, dim_diff)
            log_prob -= sum_log_det

            y = x

        dim_diff = rv_dim - len(self.base_distribution.rv_shape)
        log_prob += sum_last(self.base_distribution.log_prob(y), dim_diff)
        return log_prob

    def _sample(self, sample_shape: Shape, key: PRNGKey) -> Variate:
        epsilon = self.base_distribution.sample(key=key, sample_shape=sample_shape)

        for t in self.transformation:
            epsilon = t(epsilon)

        return epsilon
