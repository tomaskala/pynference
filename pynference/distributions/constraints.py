import torch
from torch.distributions import biject_to, Transform
from torch.distributions.constraints import Constraint, real


__all__ = ["generalized_interval"]


class _GeneralizedInterval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value):
        return (self.lower_bound <= value) & (value <= self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={}, upper_bound={})'.format(self.lower_bound, self.upper_bound)
        return fmt_string


generalized_interval = _GeneralizedInterval


class GeneralizedIntervalTransform(Transform):
    domain = real
    codomain = generalized_interval

    def __init__(self, lower_bound, upper_bound, cache_size=0):
        super(GeneralizedIntervalTransform, self).__init__(cache_size=cache_size)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        lower_finite = torch.isfinite(lower_bound)
        upper_finite = torch.isfinite(upper_bound)

        self.lower_finite = lower_finite & ~upper_finite
        self.upper_finite = upper_finite & ~lower_finite
        self.both_finite = self.lower_finite & self.upper_finite

    def _call(self, x):
        out = torch.empty_like(x)

        # (-inf,inf)
        out.copy_(x)

        # (-inf,a]
        loc = self.upper_bound[self.upper_finite]
        scale = -1.0
        out[self.upper_finite] = loc + scale * x[self.upper_finite].exp()

        # [b,inf)
        loc = self.lower_bound[self.lower_finite]
        scale = 1.0
        out[self.lower_finite] = loc + scale * x[self.lower_finite].exp()

        # [a,b]
        loc = self.lower_bound[self.both_finite]
        scale = (self.upper_bound - self.lower_bound)[self.both_finite]
        out[self.both_finite] = loc + scale * x[self.both_finite]

        return out

    def _inverse(self, y):
        out = torch.empty_like(y)

        # (-inf,inf)
        out.copy_(y)

        # (-inf,a]
        loc = self.upper_bound[self.upper_finite]
        scale = -1.0
        out[self.upper_finite] = ((y[self.upper_finite] - loc) / scale).log()

        # [b,inf)
        loc = self.lower_bound[self.lower_finite]
        scale = 1.0
        out[self.lower_finite] = ((y[self.lower_finite] - loc) / scale).log()

        # [a,b]
        loc = self.lower_bound[self.both_finite]
        scale = (self.upper_bound - self.lower_bound)[self.both_finite]
        out[self.both_finite] = (y[self.both_finite] - loc) / scale

        return out

    def log_abs_det_jacobian(self, x, y):
        # (-inf,inf)
        out = torch.zeros_like(x)

        # (-inf,a]
        out[self.upper_finite] = x[self.upper_finite]

        # [b,inf)
        out[self.lower_finite] = x[self.lower_finite]

        # [a,b]
        scale = (self.upper_bound - self.lower_bound)[self.both_finite]
        out[self.both_finite] = torch.log(torch.abs(scale))

        return out


@biject_to.register(generalized_interval)
def _biject_to_generalized_interval(constraint):
    return GeneralizedIntervalTransform(constraint.lower_bound, constraint.upper_bound)
