import torch.distributions as distributions

from pynference.distributions.distribution import DistributionMixin

__all__ = []


for name, dist in distributions.__dict__.items():
    if not isinstance(dist, type):
        continue

    if not issubclass(dist, distributions.Distribution):
        continue

    if dist is distributions.Distribution:
        continue

    if name not in locals():
        patched_dist = type(name, (dist, DistributionMixin), {})
        patched_dist.__module__ = __name__
        locals()[name] = patched_dist

    __all__.append(name)
