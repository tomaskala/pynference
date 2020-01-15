import abc
from typing import Dict

from numpy.random import RandomState

from pynference.constants import ArrayLike, Sample, Shape
from pynference.distributions.constraints import Constraint


class Model(abc.ABC):
    @property
    @abc.abstractmethod
    def constraints(self) -> Dict[str, Constraint]:
        pass

    @abc.abstractmethod
    def log_prob(self, theta: Sample) -> ArrayLike:
        pass

    @abc.abstractmethod
    def sample(self, sample_shape: Shape, random_state: RandomState) -> Sample:
        pass
