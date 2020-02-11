import math
from numbers import Number

import torch
from torch.distributions import Normal, constraints
from torch.distributions.utils import broadcast_all

from pynference.distributions.distribution import Distribution


class TruncatedNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "low": constraints.dependent,
        "high": constraints.dependent,
    }

    @property
    def mean(self):
        return (
            self.loc
            + (self._phi(self._alpha) - self._phi(self._beta)) / self._Z * self.scale
        )

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        result = 1.0
        result += (
            self._alpha * self._phi(self._alpha) - self._beta * self._phi(self._beta)
        ) / self._Z
        result -= ((self._phi(self._alpha) - self._phi(self._beta)) / self._Z) ** 2
        return result * self.scale ** 2

    def __init__(
        self, loc, scale, low, high, validate_args=None,
    ):
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        batch_shape = self.loc.size()

        super(TruncatedNormal, self).__init__(batch_shape, validate_args=validate_args)

        if self._validate_args and not torch.lt(self.low, self.high).all():
            raise ValueError("Truncated normal is not defined when low >= high.")

        self._normal = Normal(
            loc=self.loc.new_zeros(self.loc.size()),
            scale=self.scale.new_ones(self.scale.size()),
        )
        self._alpha = self._xi(self.low)
        self._beta = self._xi(self.high)

        neg_inf_mask = torch.isinf(self.low) & (self.low < 0.0)
        pos_inf_mask = torch.isinf(self.high) & (self.high > 0.0)

        self._Phi_alpha = self._normal.cdf(self._alpha)
        self._Phi_alpha[neg_inf_mask] = 0.0

        Phi_beta = self._normal.cdf(self._beta)
        Phi_beta[pos_inf_mask] = 1.0

        self._Z = Phi_beta - self._Phi_alpha

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)

        with torch.no_grad():
            uniform = torch.rand(shape)
            return self.icdf(uniform)

    def cdf(self, x):
        return (self._normal.cdf(self._xi(x)) - self._Phi_alpha) / self._Z

    def entropy(self):
        result = math.log(2.0 * math.pi)
        result += 1.0
        result += (
            math.log(self.scale)
            if isinstance(self.scale, Number)
            else torch.log(self.scale)
        )
        result += (
            math.log(self._Z) if isinstance(self._Z, Number) else torch.log(self._Z)
        )
        result += (
            self._alpha * self._phi(self._alpha) - self._beta * self._phi(self._beta)
        ) / (2.0 * self._Z)
        return result

    def icdf(self, x):
        return self.loc + self.scale * self._normal.icdf(self._Z * x + self._Phi_alpha)

    def log_prob(self, x):
        result = self._normal.log_prob(self._xi(x))
        result -= (
            math.log(self.scale)
            if isinstance(self.scale, Number)
            else torch.log(self.scale)
        )
        result -= (
            math.log(self._Z) if isinstance(self._Z, Number) else torch.log(self._Z)
        )
        return result

    def _phi(self, x):
        result = x.new_zeros(x.size())
        result[torch.isfinite(x)] = torch.exp(-(x[torch.isfinite(x)] ** 2) / 2.0)
        return result / math.sqrt(2.0 * math.pi)

    def _xi(self, x):
        return (x - self.loc) / self.scale

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.low, self.high)
