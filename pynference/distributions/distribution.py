import abc
from typing import Dict, Iterable, Tuple, Union

import numpy as np
from numpy.random import RandomState

import pynference.distributions.utils as utils
from pynference.constants import ArrayLike, Parameter, Shape, Variate
from pynference.distribution.transformations import Transformation
from pynference.distributions.constraints import Constraint


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
        self.check_support = check_support

        if check_parameters:
            for parameter, constraint in self._constraints.items():
                if parameter not in self.__dict__:
                    # Useful for cases when the distribution can be parameterized in multiple ways,
                    # e.g. the Gaussian distribution in terms of variance or precision.
                    continue

                parameter_value = getattr(self, parameter)

                if not np.all(constraint(parameter_value)):
                    raise ValueError(
                        f"Invalid value for {parameter}: {parameter_value}."
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
        self._validate_input(x)
        return self._log_prob(x)

    @abc.abstractmethod
    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def sample(self, sample_shape: Shape = (), random_state=None) -> Variate:
        random_state = utils.check_random_state(random_state)
        return self._sample(sample_shape, random_state)

    @abc.abstractmethod
    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        pass

    def _validate_input(self, x: Variate):
        if self.check_support and not np.all(self.support(x)):
            raise ValueError(f"The parameter {x} lies outside the support.")


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
        transformation: Union[Transformation, Iterable[Transformation]],
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        if isinstance(transformation, Transformation):
            transformation = [transformation]

        if isinstance(base_distribution, TransformedDistribution):
            self.base_distribution = base_distribution.base_distribution
            self.transformation = base_distribution.transformation + transformation
        else:
            self.base_distribution = base_distribution
            self.transformation = transformation

        super().__init__(
            batch_shape=batch_shape,  # TODO
            rv_shape=rv_shape,  # TODO
            check_parameters=check_parameters,
            check_support=check_support,
        )

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass  # TODO

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = self.base_distribution.sample(sample_shape, random_state)

        for t in self.transformation:
            epsilon = t(epsilon)

        return epsilon
