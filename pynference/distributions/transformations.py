import abc
import numbers

import numpy as np

from pynference.constants import ArrayLike, Variate
from pynference.distributions.constraints import (
    Constraint,
    Interval,
    Real,
    RealVector,
    positive,
    real,
    real_vector,
)
from pynference.distributions.utils import sum_last


class Transformation(abc.ABC):
    def __init__(self, domain: Constraint):
        self.domain = domain

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
        """
        Calculate the logarithm of the absolute value of the Jacobian dy/dx.
        """


class AffineTransformation(Transformation):
    def __init__(self, loc: ArrayLike, scale: ArrayLike, domain=real):
        if np.any(scale <= 0.0):
            raise ValueError("The scale parameter must be positive.")

        super().__init__(domain=domain)
        self.loc = loc
        self.scale = scale

    @property
    def codomain(self) -> Constraint:
        if isinstance(self.domain, Real):
            return real
        elif isinstance(self.domain, RealVector):
            return real_vector
        elif isinstance(self.domain, Interval):
            return Interval(self(self.domain.lower), self(self.domain.upper))
        else:
            raise NotImplementedError()

    @property
    def rv_dim(self) -> int:
        if isinstance(self.domain, RealVector):
            return 1
        else:
            return 0

    def __call__(self, x: Variate) -> Variate:
        return self.loc + self.scale * x

    def inverse(self, y: Variate) -> Variate:
        return (y - self.loc) / self.scale

    def log_abs_J(self, x: Variate, y: Variate) -> Variate:
        result = np.log(np.abs(self.scale))

        if isinstance(self.scale, numbers.Number):
            result = np.full(shape=np.shape(x), fill_value=result)

        return sum_last(result, self.rv_dim)


class ExpTransformation(Transformation):
    def __init__(self, domain=real):
        super().__init__(domain=domain)

    @property
    def codomain(self) -> Constraint:
        if isinstance(self.domain, Real):
            return positive
        elif isinstance(self.domain, Interval):
            return Interval(self(self.domain.lower), self(self.domain.upper))
        else:
            raise NotImplementedError()

    @property
    def rv_dim(self) -> int:
        return 0

    def __call__(self, x: Variate) -> Variate:
        return np.exp(x)

    def inverse(self, y: Variate) -> Variate:
        return np.log(y)

    def log_abs_J(self, x: Variate, y: Variate) -> Variate:
        return x


class PowerTransformation(Transformation):
    def __init__(self, power: ArrayLike):
        super().__init__(domain=positive)

        self.power = power

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

    def log_abs_J(self, x: Variate, y: Variate) -> Variate:
        # y = x^n
        # dy/dx = n*x^(n-1) = n*x^n/x = n*y/x
        return np.log(np.abs(self.power * y / x))
