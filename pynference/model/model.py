import abc

from pynference.constants import ArrayLike, Sample


class Model(abc.ABC):
    @abc.abstractmethod
    def log_prob(self, theta: Sample) -> ArrayLike:
        pass

    @abc.abstractmethod
    def sample(self) -> Sample:
        pass
