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


def _broadcast_shapes(*shapes, strict=False):
    reversed_shape = []

    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError(
                    "Shape mismatch: objects cannot be broadcast to a single shape: {}.".format(
                        " vs ".join(map(str, shapes))
                    )
                )

    return tuple(reversed(reversed_shape))
