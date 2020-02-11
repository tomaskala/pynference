import math
from numbers import Number

import torch
from scipy.special import log_ndtr
from torch.distributions import Normal, constraints
from torch.distributions.utils import broadcast_all

from pynference.distributions.distribution import Distribution


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
        B = phi_b / self._normal.cdf(self._beta)
        result[a_inf] = -B[a_inf]

        # Left-sided truncation.
        b_inf = torch.isinf(self._beta)
        A = phi_a / self._normal.cdf(-self._alpha)
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
        B = phi_b / self._normal.cdf(self._beta)
        result[a_inf] = (-B * (self._beta - B))[a_inf]

        # Left-sided truncation.
        b_inf = torch.isinf(self._beta)
        A = phi_a / self._normal.cdf(-self._alpha)
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

        self._normal = Normal(
            loc=torch.zeros_like(self.loc), scale=torch.ones_like(self.scale),
        )
        self._alpha = self._xi(self.low)
        self._beta = self._xi(self.high)

        self._Phi_alpha, self._Z, self._log_Z = self._get_constants()

    def _get_constants(self):
        # Phi(alpha)
        Phi_alpha = self._normal.cdf(self._alpha)
        Phi_alpha[self._alpha < -self.TRIM] = 0.0
        Phi_alpha[self._alpha > self.TRIM] = 1.0

        # Z
        Z = torch.zeros_like(self._alpha)
        use_cdf = (self._alpha <= 0.0)

        cdf_a = self._normal.cdf(self._alpha)
        cdf_b = self._normal.cdf(self._beta)
        sf_a = self._normal.cdf(-self._alpha)
        sf_b = self._normal.cdf(-self._beta)

        Z[use_cdf] = cdf_b[use_cdf] - cdf_a[use_cdf]
        Z[~use_cdf] = sf_a[~use_cdf] - sf_b[~use_cdf]
        Z[self._alpha > self.TRIM] = 0.0
        Z[self._beta < -self.TRIM] = 0.0
        Z = torch.max(Z, torch.zeros_like(Z))

        # log(Z)
        log_Z = torch.zeros_like(Z)
        use_lcdf = (self._beta < 0.0) | (
            torch.abs(self._alpha) >= torch.abs(self._beta)
        )

        lcdf_a = self._log_Phi(self._alpha)
        lcdf_b = self._log_Phi(self._beta)
        lsf_a = self._log_Phi(-self._alpha)
        lsf_b = self._log_Phi(-self._beta)

        log_Z[use_lcdf] = lcdf_b[use_lcdf] + torch.log1p(
            -torch.exp(lcdf_a[use_lcdf] - lcdf_b[use_lcdf])
        )
        log_Z[~use_lcdf] = lsf_a[~use_lcdf] + torch.log1p(
            -torch.exp(lsf_b[~use_lcdf] - lsf_a[~use_lcdf])
        )

        within_range = (self._alpha <= self.TRIM) & (self._beta >= -self.TRIM)
        within_range_pos = within_range & (self._alpha > 0.0)
        within_range_neg = within_range & (self._alpha <= 0.0)

        log_Z[within_range_pos] = cdf_b[within_range_pos] - cdf_a[within_range_pos]
        log_Z[within_range_neg] = sf_a[within_range_neg] - sf_b[within_range_neg]
        log_Z[within_range] = torch.log(
            torch.max(log_Z[within_range], torch.zeros_like(log_Z[within_range]))
        )

        return Phi_alpha, Z, log_Z

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)

        with torch.no_grad():
            uniform = torch.rand(shape)
            return self.icdf(uniform)

    def cdf(self, x):
        return (self._normal.cdf(self._xi(x)) - self._Phi_alpha) / self._Z

    def entropy(self):
        result = math.log(2.0 * math.pi) + 1.0 + torch.log(self.scale) + self._log_Z
        result += (
            self._alpha * self._phi(self._alpha) - self._beta * self._phi(self._beta)
        ) / (2.0 * self._Z)
        return result

    def icdf(self, x):
        return self.loc + self.scale * self._normal.icdf(self._Z * x + self._Phi_alpha)

    def log_prob(self, x):
        return self._normal.log_prob(self._xi(x)) - torch.log(self.scale) - self._log_Z

    def _phi(self, x):
        return torch.exp(-x ** 2 / 2.0) / math.sqrt(2.0 * math.pi)

    def _log_Phi(self, x):
        return log_ndtr(x)

    def _xi(self, x):
        return (x - self.loc) / self.scale

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.low, self.high)
