import abc
from typing import Union

import numpy as np

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


class RealVector(Constraint):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.all(np.isfinite(x), axis=-1)

    def __str__(self) -> str:
        return "real_vector"


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
            return f"[{self.lower}, {self.upper}]"
        elif self.include_lower:
            return f"[{self.lower}, {self.upper})"
        elif self.include_upper:
            return f"({self.lower}, {self.upper}]"
        else:
            return f"({self.lower}, {self.upper})"


class Positive(Interval):
    def __init__(self):
        super().__init__(lower=0.0, upper=np.inf)


class NonNegative(Interval):
    def __init__(self):
        super().__init__(lower=0.0, upper=np.inf, include_lower=True)


class Negative(Interval):
    def __init__(self):
        super().__init__(lower=-np.inf, upper=0.0)


class NonPositive(Interval):
    def __init__(self):
        super().__init__(lower=-np.inf, upper=0.0, include_upper=True)


real = Real()
real_vector = RealVector()
positive = Positive()
non_negative = NonNegative()
negative = Negative()
non_positive = NonPositive()
zero_one = Interval(0.0, 1.0)
interval = Interval


__all__ = [
    "real",
    "real_vector",
    "positive",
    "non_negative",
    "negative",
    "non_positive",
    "zero_one",
    "interval",
]
