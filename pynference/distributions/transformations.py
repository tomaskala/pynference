import abc
from typing import Callable, Dict, Optional, Sequence, Type, Union

import numpy as np
from scipy.special import expit, logit

from pynference.constants import ArrayLike, Variate
from pynference.distributions.constraints import (
    Constraint,
    Interval,
    Real,
    RealVector,
    interval,
    lower_cholesky,
    positive,
    positive_definite,
    positive_vector,
    real,
    real_vector,
    simplex,
    zero_one,
)
from pynference.distributions.utils import sum_last


def _clipped_expit(x: Variate) -> Variate:
    finfo = np.finfo(np.dtype(x))
    return np.clip(expit(x), a_min=finfo.tiny, a_max=1.0 - finfo.eps)


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


class ComposeTransformation(Transformation):
    def __init__(self, transformations: Sequence[Transformation]):
        if not transformations:
            raise ValueError("No transformations given.")

        super().__init__(domain=transformations[0].domain)
        self.transformations = transformations

    @property
    def codomain(self) -> Constraint:
        return self.transformations[-1].codomain

    @property
    def rv_dim(self) -> int:
        return max(t.rv_dim for t in self.transformations)

    def __call__(self, x: Variate) -> Variate:
        for t in self.transformations:
            x = t(x)

        return x

    def inverse(self, y: Variate) -> Variate:
        for t in reversed(self.transformations):
            y = t.inverse(y)

        return y

    def log_abs_J(self, x: Variate, y: Variate) -> Variate:
        result = 0.0

        for t in self.transformations[:-1]:
            y_tmp = t(x)
            dim_diff = self.rv_dim - t.rv_dim

            log_det = t.log_abs_J(x, y_tmp)
            result += sum_last(log_det, dim_diff)

        t = self.transformations[-1]
        dim_diff = self.rv_dim - t.rv_dim

        log_det = t.log_abs_J(x, y)
        result += sum_last(log_det, dim_diff)

        return result


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
        log_det = np.log(np.abs(self.scale))
        return sum_last(np.broadcast_to(log_det, np.shape(x)), self.rv_dim)


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


class IdentityTransformation(Transformation):
    def __init__(self, rv_dim: int = 0, domain=real):
        super().__init__(domain=domain)
        self._rv_dim = rv_dim

    @property
    def codomain(self) -> Constraint:
        return self.domain

    @property
    def rv_dim(self) -> int:
        return self._rv_dim

    def __call__(self, x: Variate) -> Variate:
        return x

    def inverse(self, y: Variate) -> Variate:
        return y

    def log_abs_J(self, x: Variate, y: Variate) -> Variate:
        return 0.0


class SigmoidTransformation(Transformation):
    def __init__(self, rv_dim: int = 0, domain=real):
        super().__init__(domain=domain)
        self._rv_dim = rv_dim

    @property
    def codomain(self) -> Constraint:
        return zero_one

    @property
    def rv_dim(self) -> int:
        return self._rv_dim

    def __call__(self, x: Variate) -> Variate:
        return _clipped_expit(x)

    def inverse(self, y: Variate) -> Variate:
        return logit(y)

    def log_abs_J(self, x: Variate, y: Variate) -> Variate:
        abs_x = np.abs(x)
        return -abs_x - 2 * np.log1p(np.exp(-abs_x))


class _ConstraintMapper:
    def __init__(self):
        self._constraints: Dict[
            Type[Constraint], Callable[[Constraint], Transformation]
        ] = {}

    def register(
        self,
        constraint: Union[Constraint, Type[Constraint]],
        factory: Optional[Transformation] = None,
    ):
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        if isinstance(constraint, Constraint):
            constraint = type(constraint)

        self._constraints[constraint] = factory

    def __call__(self, constraint: Constraint) -> Transformation:
        try:
            factory = self._constraints[type(constraint)]
        except KeyError:
            raise NotImplementedError(
                "Bijection to the constraint {} is not implemented.".format(constraint)
            )

        return factory(constraint)


biject_to = _ConstraintMapper()


@biject_to.register(real)
def _transform_to_real(constraint: Constraint) -> Transformation:
    return IdentityTransformation(rv_dim=0)


@biject_to.register(real_vector)
def _transform_to_real_vector(constraint: Constraint) -> Transformation:
    return IdentityTransformation(rv_dim=1)


@biject_to.register(interval)
def _transform_to_interval(constraint: Constraint) -> Transformation:
    loc = constraint.lower
    scale = constraint.upper - constraint.lower

    return ComposeTransformation(
        (
            SigmoidTransformation(),
            AffineTransformation(loc=loc, scale=scale, domain=zero_one),
        )
    )


@biject_to.register(positive)
def _transform_to_positive(constraint: Constraint) -> Transformation:
    return ExpTransformation()


@biject_to.register(zero_one)
def _transform_to_zero_one(constraint: Constraint) -> Transformation:
    return SigmoidTransformation(rv_dim=0)


@biject_to.register(positive_vector)
def _transform_to_positive_vector(constraint: Constraint) -> Transformation:
    return SigmoidTransformation(rv_dim=1)


@biject_to.register(simplex)
def _transform_to_simplex(constraint: Constraint) -> Transformation:
    raise NotImplementedError("Transformation to simplex is not yet implemented.")


@biject_to.register(positive_definite)
def _transform_to_positive_definite(constraint: Constraint) -> Transformation:
    raise NotImplementedError(
        "Transformation to positive definite is not yet implemented."
    )


@biject_to.register(lower_cholesky)
def _transform_to_lower_cholesky(constraint: Constraint) -> Transformation:
    raise NotImplementedError(
        "Transformation to lower Cholesky is not yet implemented."
    )
