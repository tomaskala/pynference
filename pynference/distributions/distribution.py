from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.distributions as distributions
import torch.distributions.constraints as constraints


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

    def expand(self, batch_shape, _instance=None):
        return ExpandedDistribution(base_distribution=self, batch_shape=batch_shape)


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


class ExpandedDistribution(Distribution):
    def __init__(self, base_distribution, batch_shape):
        self.base_distribution = base_distribution
        super().__init__(
            batch_shape=base_distribution.batch_shape,
            event_shape=base_distribution.event_shape,
            validate_args=base_distribution.validate_args,
        )
        self._batch_shape, self._expanded_sizes, self._interstitial_sizes = self.expand(
            batch_shape
        )

    @property
    def has_rsample(self):
        return self.base_distribution.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_distribution.has_enumerate_support

    @constraints.dependent_property
    def support(self):
        return self.base_distribution.support

    @property
    def arg_constraints(self):
        return self.base_distribution.arg_constraints

    @property
    def mean(self):
        return self.base_distribution.mean.expand(self._extended_shape())

    @property
    def variance(self):
        return self.base_distribution.variance.expand(self._extended_shape())

    def expand(self, batch_shape, _instance=None):
        new_shape, _, _ = self._broadcast_shape(self.batch_shape, batch_shape)
        return self._broadcast_shape(self.base_distribution.batch_shape, new_shape)

    def _broadcast_shape(self, existing_shape, new_shape):
        if len(new_shape) < len(existing_shape):
            raise ValueError(
                "Cannot broadcast distribution of shape {} to shape {}.".format(
                    existing_shape, new_shape
                )
            )

        reversed_shape = list(reversed(existing_shape))
        expanded_sizes, interstitial_sizes = [], []

        for i, size in enumerate(reversed(new_shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
                expanded_sizes.append((-i - 1, size))
            elif reversed_shape[i] == 1:
                if size != 1:
                    reversed_shape[i] = size
                    interstitial_sizes.append((-i - 1, size))
            elif reversed_shape[i] != size:
                raise ValueError(
                    "Cannot broadcast distribution of shape {} to shape {}.".format(
                        existing_shape, new_shape
                    )
                )

        return (
            tuple(reversed(reversed_shape)),
            OrderedDict(expanded_sizes),
            OrderedDict(interstitial_sizes),
        )

    def sample(self, sample_shape=torch.Size()):
        return self._sample(self.base_distribution.sample, sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self._sample(self.base_distribution.rsample, sample_shape)

    def _sample(self, sample_fun, sample_shape):
        interstitial_dims = tuple(self._interstitial_sizes.keys())
        interstitial_dims = tuple(i - self.event_dim for i in interstitial_dims)
        interstitial_sizes = tuple(self._interstitial_sizes.values())

        expanded_sizes = tuple(self._expanded_sizes.values())
        batch_shape = expanded_sizes + interstitial_sizes

        samples = sample_fun(sample_shape + batch_shape)

        interstitial_idx = len(sample_shape) + len(expanded_sizes)
        interstitial_sample_dims = tuple(
            range(interstitial_idx, interstitial_idx + len(interstitial_sizes))
        )

        for dim1, dim2 in zip(interstitial_dims, interstitial_sample_dims):
            samples = samples.transpose(dim1, dim2)

        return samples.reshape(sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        shape = _broadcast_shapes(
            self.batch_shape, value.shape[: value.dim() - self.event_dim]
        )
        log_prob = self.base_distribution.log_prob(value)
        return log_prob.expand(shape)

    def cdf(self, value):
        shape = _broadcast_shapes(
            self.batch_shape, value.shape[: value.dim() - self.event_dim]
        )
        cdf = self.base_distribution.cdf(value)
        return cdf.expand(shape)

    def icdf(self, value):
        shape = _broadcast_shapes(
            self.batch_shape, value.shape[: value.dim() - self.event_dim]
        )
        icdf = self.base_distribution.icdf(value)
        return icdf.expand(shape)

    def enumerate_support(self, expand=True):
        support = self.base_dist.enumerate_support(expand=expand)
        enumerated_shape = support.shape[:1]
        support = support.reshape(enumerated_shape + (1,) * len(self.batch_shape))

        if expand:
            support = support.expand(enumerated_shape + self.batch_shape)

        return support

    def entropy(self):
        entropy = self.base_distribution.entropy()
        return entropy.expand(self.batch_shape)
