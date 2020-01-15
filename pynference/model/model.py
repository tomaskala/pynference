import abc
from typing import Dict

from pynference.constants import ArrayLike, Sample
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
    def sample(self) -> Sample:
        pass
