import abc
from typing import Union

import numpy as np
import numpy.linalg as la  # Not SciPy, NumPy works for batches of matrices.

from pynference.constants import ArrayLike


class Constraint(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: Union[float, np.ndarray]) -> np.ndarray:
        pass


class Real(Constraint):
    def __call__(self, x: float) -> np.ndarray:
        return np.isfinite(x)

    def __str__(self) -> str:
        return "real"


class Interval(Constraint):
    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        include_lower: bool = False,
        include_upper: bool = False,
    ):
        self.lower = lower
        self.upper = upper
        self.include_lower = include_lower
        self.include_upper = include_upper

    def __call__(self, x: float) -> np.ndarray:
        if self.include_lower and self.include_upper:
            return (self.lower <= x) & (x <= self.upper)
        elif self.include_lower:
            return (self.lower <= x) & (x < self.upper)
        elif self.include_upper:
            return (self.lower < x) & (x <= self.upper)
        else:
            return (self.lower < x) & (x < self.upper)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented

        return (  # type: ignore
            np.all(self.lower == other.lower)
            and np.all(self.upper == other.upper)
            and np.all(self.include_lower == other.include_lower)
            and np.all(self.include_upper == other.include_upper)
        )

    def __str__(self) -> str:
        if self.include_lower and self.include_upper:
            return f"interval[{self.lower}, {self.upper}]"
        elif self.include_lower:
            return f"interval[{self.lower}, {self.upper})"
        elif self.include_upper:
            return f"interval({self.lower}, {self.upper}]"
        else:
            return f"interval({self.lower}, {self.upper})"


class Positive(Interval):
    def __init__(self):
        super().__init__(lower=0.0, upper=np.inf)

    def __str__(self) -> str:
        return "positive"


class NonNegative(Interval):
    def __init__(self):
        super().__init__(lower=0.0, upper=np.inf, include_lower=True)

    def __str__(self) -> str:
        return "non_negative"


class Negative(Interval):
    def __init__(self):
        super().__init__(lower=-np.inf, upper=0.0)

    def __str__(self) -> str:
        return "negative"


class NonPositive(Interval):
    def __init__(self):
        super().__init__(lower=-np.inf, upper=0.0, include_upper=True)

    def __str__(self) -> str:
        return "non_positive"


class ZeroOne(Interval):
    def __init__(self):
        super().__init__(lower=0.0, upper=1.0)

    def __str__(self) -> str:
        return "zero_one"


class Integer(Constraint):
    def __call__(self, x: float) -> np.ndarray:
        return np.equal(np.mod(x, 1), 0)

    def __str__(self) -> str:
        return "integer"


class IntegerInterval(Interval):
    def __call__(self, x: float) -> np.ndarray:
        return super().__call__(x) & (np.equal(np.mod(x, 1), 0))

    def __str__(self) -> str:
        if self.include_lower and self.include_upper:
            return f"integer_interval[{self.lower}, {self.upper}]"
        elif self.include_lower:
            return f"integer_interval[{self.lower}, {self.upper})"
        elif self.include_upper:
            return f"integer_interval({self.lower}, {self.upper}]"
        else:
            return f"integer_interval({self.lower}, {self.upper})"


class PositiveInteger(IntegerInterval):
    def __init__(self):
        super().__init__(lower=0, upper=np.inf)

    def __str__(self) -> str:
        return "positive_integer"


class NonNegativeInteger(IntegerInterval):
    def __init__(self):
        super().__init__(lower=0, upper=np.inf, include_lower=True)

    def __str__(self) -> str:
        return "non_negative_integer"


class NegativeInteger(IntegerInterval):
    def __init__(self):
        super().__init__(lower=-np.inf, upper=0)

    def __str__(self) -> str:
        return "negative_integer"


class NonPositiveInteger(IntegerInterval):
    def __init__(self):
        super().__init__(lower=-np.inf, upper=0, include_upper=True)

    def __str__(self) -> str:
        return "non_positive_integer"


class ZeroOneInteger(IntegerInterval):
    def __init__(self):
        super().__init__(lower=0, upper=1, include_lower=True, include_upper=True)

    def __str__(self) -> str:
        return "binary"


class Simplex(Constraint):
    eps = 1e-6

    def __call__(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x_sum = np.sum(x, axis=-1)
        return np.all(x > 0.0, axis=-1) & (x_sum <= 1.0) & (x_sum > 1.0 - self.eps)

    def __str__(self) -> str:
        return "simplex"


class RealVector(Constraint):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.all(np.isfinite(x), axis=-1)

    def __str__(self) -> str:
        return "real_vector"


class PositiveVector(Constraint):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.all(x > 0, axis=-1)

    def __str__(self) -> str:
        return "positive_vector"


class NonNegativeVector(Constraint):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.all(x >= 0, axis=-1)

    def __str__(self) -> str:
        return "non_negative_vector"


class NegativeVector(Constraint):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.all(x < 0, axis=-1)

    def __str__(self) -> str:
        return "negative_vector"


class NonPositiveVector(Constraint):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.all(x <= 0, axis=-1)

    def __str__(self) -> str:
        return "non_positive_vector"


class PositiveDefinite(Constraint):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        symmetric = np.all(np.all(x == np.swapaxes(x, -2, -1), axis=-1), axis=-1)
        positive = la.eigvalsh(x)[..., 0] > 0.0  # Smallest eigval > 0.
        return symmetric & positive

    def __str__(self) -> str:
        return "positive_definite"


class LowerCholesky(Constraint):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        tril = np.tril(x)
        lower_triangular = np.all(np.reshape(tril == x, x.shape[:-2] + (-1,)), axis=-1)
        positive_diagonal = np.all(np.diagonal(x, axis1=-2, axis2=-1) > 0, axis=-1)
        return lower_triangular & positive_diagonal

    def __str__(self) -> str:
        return "lower_cholesky"


real = Real()
real_vector = RealVector()
positive_vector = PositiveVector()
non_negative_vector = NonNegativeVector()
negative_vector = NegativeVector()
non_positive_vector = NonPositiveVector()
positive = Positive()
non_negative = NonNegative()
negative = Negative()
non_positive = NonPositive()
zero_one = ZeroOne()
interval = Interval
integer = Integer()
integer_interval = IntegerInterval
positive_integer = PositiveInteger()
non_negative_integer = NonNegativeInteger()
negative_integer = NegativeInteger()
non_positive_integer = NonPositiveInteger()
zero_one_integer = ZeroOneInteger()
simplex = Simplex()
positive_definite = PositiveDefinite()
lower_cholesky = LowerCholesky()


__all__ = [
    "real",
    "positive",
    "non_negative",
    "negative",
    "non_positive",
    "zero_one",
    "interval",
    "integer",
    "integer_interval",
    "positive_integer",
    "non_negative_integer",
    "negative_integer",
    "non_positive_integer",
    "zero_one_integer",
    "simplex",
    "real_vector",
    "positive_vector",
    "non_negative_vector",
    "negative_vector",
    "non_positive_vector",
    "positive_definite",
    "lower_cholesky",
]
