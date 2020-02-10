from abc import ABC, abstractmethod

import torch
import torch.distributions as distributions


class DistributionABC(ABC):
    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.sample(sample_shape)


class DistributionMixin(DistributionABC):
    def __call__(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        if self.has_rsample:  # type: ignore
            return self.rsample(sample_shape)  # type: ignore
        else:
            return self.sample(sample_shape)

    @property
    def event_dim(self) -> int:
        return len(self.event_shape)  # type: ignore

    def shape(self, sample_shape: torch.Size = torch.Size()) -> torch.Size:
        return sample_shape + self.batch_shape + self.event_shape  # type: ignore


class Distribution(distributions.Distribution, DistributionMixin):
    pass
