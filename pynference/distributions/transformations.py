import abc

import numpy as np

from pynference.constants import ArrayLike, Variate
from pynference.distribution.constraints import (
    Constraint,
    Interval,
    Real,
    RealVector,
    positive,
    real,
    real_vector,
)


class Transformation(abc.ABC):
    @property
    @abc.abstractmethod
    def domain(self) -> Constraint:
        pass

    @property
    @abc.abstractmethod
    def codomain(self) -> Constraint:
        pass

    @property
    @abc.abstractmethod
    def rv_dim(self) -> int:
        pass

    @abc.abstractmethod
    def __call__(self, x: Variate) -> Variate:
        pass

    @abc.abstractmethod
    def inverse(self, y: Variate) -> Variate:
        pass

    @abc.abstractmethod
    def log_abs_J(self, x: Variate, y: Variate) -> Variate:
        pass


class AffineTransformation(Transformation):
    def __init__(self, loc: ArrayLike, scale: ArrayLike, domain=real):
        if scale <= 0.0:
            raise ValueError("The scale parameter must be positive.")

        self.loc = loc
        self.scale = scale
        self._domain = domain

    @property
    def domain(self) -> Constraint:
        return self._domain

    @property
    def codomain(self) -> Constraint:
        if isinstance(self._domain, Real):
            return real
        elif isinstance(self._domain, RealVector):
            return real_vector
        elif isinstance(self._domain, Interval):
            return Interval(self(self._domain.lower), self(self._domain.upper))
        else:
            raise NotImplementedError()

    @property
    def rv_dim(self) -> int:
        if isinstance(self._domain, Real):
            return 0
        else:
            return 1

    def __call__(self, x: Variate) -> Variate:
        return self.loc + self.scale * x

    def inverse(self, y: Variate) -> Variate:
        return (y - self.loc) / self.scale

    @abc.abstractmethod
    def log_abs_J(self, x: Variate, y: Variate) -> Variate:
        pass  # TODO


class ExpTransformation(Transformation):
    def __init__(self, domain=real):
        self._domain = domain

    @property
    def domain(self) -> Constraint:
        return self._domain

    @property
    def codomain(self) -> Constraint:
        if isinstance(self._domain, Real):
            return positive
        elif isinstance(self._domain, Interval):
            return Interval(self(self._domain.lower), self(self._domain.upper))
        else:
            raise NotImplementedError()

    @property
    def rv_dim(self) -> int:
        return 0

    def __call__(self, x: Variate) -> Variate:
        return np.exp(x)

    def inverse(self, y: Variate) -> Variate:
        return np.log(y)

    @abc.abstractmethod
    def log_abs_J(self, x: Variate, y: Variate) -> Variate:
        pass  # TODO


class PowerTransformation(Transformation):
    def __init__(self, power: ArrayLike):
        self.power = power

    @property
    def domain(self) -> Constraint:
        return positive

    @property
    def codomain(self) -> Constraint:
        return positive

    @property
    def rv_dim(self) -> int:
        return 0

    def __call__(self, x: Variate) -> Variate:
        return np.power(x, self.power)

    def inverse(self, y: Variate) -> Variate:
        return np.power(y, np.reciprocal(self.power))

    @abc.abstractmethod
    def log_abs_J(self, x: Variate, y: Variate) -> Variate:
        pass  # TODO
