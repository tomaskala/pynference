import math

import torch
from torch.distributions import Normal, constraints
from torch.distributions.utils import broadcast_all

from pynference.distributions.constraints import generalized_interval
from pynference.distributions.distribution import Distribution
from truncated_normal_cpp import sample_truncated_normal


class TruncatedNormal(Distribution):
    TRIM = 30

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "low": constraints.dependent,
        "high": constraints.dependent,
    }

    @property
    def mean(self):
        phi_a = self._phi(self._alpha)
        phi_b = self._phi(self._beta)

        # Two-sided truncation.
        A = phi_a / self._Z
        B = phi_b / self._Z
        result = A - B

        # Right-sided truncation.
        a_inf = torch.isinf(self._alpha)
        B = phi_b / self._Phi(self._beta)
        result[a_inf] = -B[a_inf]

        # Left-sided truncation.
        b_inf = torch.isinf(self._beta)
        A = phi_a / self._Phi(-self._alpha)
        result[b_inf] = A[b_inf]

        # No truncation at all.
        result[a_inf & b_inf] = 0.0

        result = (self._phi(self._alpha) - self._phi(self._beta)) / self._Z
        return self.loc + self.scale * result

    @property
    def variance(self):
        phi_a = self._phi(self._alpha)
        phi_b = self._phi(self._beta)

        # Two-sided truncation.
        A = phi_a / self._Z
        B = phi_b / self._Z
        phi = A - B
        result = (self._alpha - phi) * A - (self._beta - phi) * B

        # Right-sided truncation.
        a_inf = torch.isinf(self._alpha)
        B = phi_b / self._Phi(self._beta)
        result[a_inf] = (-B * (self._beta - B))[a_inf]

        # Left-sided truncation.
        b_inf = torch.isinf(self._beta)
        A = phi_a / self._Phi(-self._alpha)
        result[b_inf] = (A * (self._alpha - A))[b_inf]

        # No truncation at all.
        result[a_inf & b_inf] = 0.0
        return self.scale ** 2 * (1.0 + result)

    def __init__(
        self, loc, scale, low, high, validate_args=None,
    ):
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        batch_shape = self.loc.size()

        super(TruncatedNormal, self).__init__(batch_shape, validate_args=validate_args)

        if self._validate_args and not torch.lt(self.low, self.high).all():
            raise ValueError("Truncated normal is not defined when low >= high.")

        self._alpha = self._xi(self.low)
        self._beta = self._xi(self.high)

        self._Phi_alpha = self._Phi(self._alpha)
        self._Phi_alpha[self._alpha < -self.TRIM] = 0.0
        self._Phi_alpha[self._alpha > self.TRIM] = 1.0
        self._Z = self._normalizer()

    def _log_phi(self, x):
        return -(x ** 2) / 2.0 - math.log(math.sqrt(2.0 * math.pi))

    def _Phi(self, x):
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _Phi_inv(self, q):
        return torch.erfinv(2.0 * q - 1.0) * math.sqrt(2.0)

    def _normalizer(self):
        Z = torch.empty_like(self._alpha)
        use_cdf = self._alpha <= 0.0

        cdf_a = self._Phi(self._alpha[use_cdf])
        cdf_b = self._Phi(self._beta[use_cdf])
        sf_a = self._Phi(-self._alpha[~use_cdf])
        sf_b = self._Phi(-self._beta[~use_cdf])

        Z[use_cdf] = cdf_b - cdf_a
        Z[~use_cdf] = sf_a - sf_b
        Z[self._alpha > self.TRIM] = 0.0
        Z[self._beta < -self.TRIM] = 0.0

        return torch.max(Z, torch.zeros_like(Z))

    def sample(self, sample_shape=torch.Size()):
        samples = sample_truncated_normal(self.loc, self.scale, self.low, self.high)
        print(samples)
        return samples
        # shape = self._extended_shape(sample_shape)

        # with torch.no_grad():
            # uniform = torch.rand(shape)
            # return self.icdf(uniform)

    def cdf(self, x):
        return (self._Phi(self._xi(x)) - self._Phi_alpha) / self._Z

    def entropy(self):
        phi_a = self._phi(self._alpha)
        phi_b = self._phi(self._beta)

        # Two-sided truncation.
        A = phi_a / self._Z / 2.0
        B = phi_b / self._Z / 2.0
        result = A * self._alpha - B * self._beta

        # Right-sided truncation.
        a_inf = torch.isinf(self._alpha)
        B = phi_b / self._Phi(self._beta) / 2.0
        result[a_inf] = -(B * self._beta)[a_inf]

        # Left-sided truncation.
        b_inf = torch.isinf(self._beta)
        A = phi_a / self._Phi(-self._alpha) / 2.0
        result[b_inf] = (A * self._alpha)[b_inf]

        # No truncation at all.
        result[a_inf & b_inf] = 0.0
        return (
            result
            + 0.5 * math.log(2.0 * math.pi)
            + 0.5
            + torch.log(self.scale)
            + torch.log(self._Z)
        )

    def icdf(self, x):
        return self.loc + self.scale * self._Phi_inv(self._Z * x + self._Phi_alpha)

    def log_prob(self, x):
        return self._log_phi(self._xi(x)) - torch.log(self.scale) - torch.log(self._Z)

    def _phi(self, x):
        return torch.exp(-(x ** 2) / 2.0) / math.sqrt(2.0 * math.pi)

    def _xi(self, x):
        return (x - self.loc) / self.scale

    @constraints.dependent_property
    def support(self):
        return generalized_interval(self.low, self.high)
