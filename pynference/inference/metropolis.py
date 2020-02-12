import abc
from typing import Any, Callable, Dict, List, Tuple, Type

import torch
from torch.distributions import Cauchy, Laplace, Normal, Transform, Uniform

from pynference.constants import Sample, Shape
from pynference.inference.utils import initialize_model

__all__ = ["Metropolis"]


class Proposal(abc.ABC):
    def __init__(self, scale):
        self.scale = scale

    @abc.abstractmethod
    def __call__(self, shape: Shape):
        pass


class CauchyProposal(Proposal):
    def __call__(self, shape: Shape):
        return Cauchy(loc=torch.zeros(shape), scale=self.scale).sample()


class LaplaceProposal(Proposal):
    def __call__(self, shape: Shape):
        return Laplace(loc=torch.zeros(shape), scale=self.scale).sample()


class NormalProposal(Proposal):
    def __call__(self, shape: Shape):
        return Normal(loc=torch.zeros(shape), scale=self.scale).sample()


class UniformProposal(Proposal):
    def __call__(self, shape: Shape):
        return Uniform(
            low=torch.full(shape, -self.scale / 2.0),
            high=torch.full(shape, self.scale / 2.0),
        ).sample()


class Metropolis:
    _proposal_map: Dict[str, Type[Proposal]] = {
        "cauchy": CauchyProposal,
        "laplace": LaplaceProposal,
        "normal": NormalProposal,
        "uniform": UniformProposal,
    }

    def __init__(
        self,
        model,
        n_samples: int,
        proposal: str,
        proposal_scale: float = 1.0,
        init_strategy: str = "uniform",
        tune: bool = True,
        tune_interval: int = 100,
        track_stats: bool = True,
    ):
        self.model = model
        self.n_samples = n_samples

        if proposal in self._proposal_map:
            self.proposal = self._proposal_map[proposal](scale=proposal_scale)
        else:
            raise ValueError("Unknown proposal type: {}.".format(proposal))

        self.init_strategy = init_strategy
        self.tune = tune
        self.tune_interval = tune_interval
        self.track_stats = track_stats

        self.accepted = 0
        self.stats: List[Dict[str, Any]] = []

        self._accepted_since_tune = 0
        self._scaling = 1.0
        self._steps_until_tune = tune_interval

    def run(self, *args, **kwargs) -> List[Sample]:
        theta, potential_energy, transformations = initialize_model(
            self.model, self.init_strategy, *args, **kwargs
        )
        theta_constrained = {k: transformations[k](v) for k, v in theta.items()}
        samples = []

        for i in range(self.n_samples):
            theta, theta_constrained, stats = self.step(
                theta, theta_constrained, potential_energy, transformations
            )
            samples.append(theta)

            if self.track_stats:
                self.stats.append(stats)

            if i > 0 and i % 100 == 0:
                print(
                    "Done {}/{} samples ({:.2f} %). Accepted {} samples ({:.2f} %).".format(
                        i,
                        self.n_samples,
                        i / self.n_samples * 100.0,
                        self.accepted,
                        self.accepted / i * 100.0,
                    )
                )

        return [
            {k: transformations[k](v) for k, v in sample.items()} for sample in samples
        ]

    def step(
        self,
        theta: Sample,
        theta_constrained: Sample,
        potential_energy: Callable[[Sample], torch.Tensor],
        transformations: Dict[str, Transform],
    ) -> Tuple[Sample, Sample, Dict[str, Any]]:
        if self._steps_until_tune == 0 and self.tune:
            self._tune_scaling()
            self._accepted_since_tune = 0
            self._steps_until_tune = self.tune_interval

        theta_prop = {}
        theta_prop_constrained = {}

        for name, param in theta.items():
            theta_prop[name] = param + self.proposal(param.shape) * self._scaling
            theta_prop_constrained[name] = transformations[name](theta_prop[name])

        acceptance_ratio = potential_energy(theta_constrained) - potential_energy(
            theta_prop_constrained
        )

        if (
            torch.isfinite(acceptance_ratio)
            and torch.log(torch.rand(1)) < acceptance_ratio
        ):
            accepted = True
            theta = theta_prop
            theta_constrained = theta_prop_constrained
        else:
            accepted = False

        self.accepted += accepted
        self._accepted_since_tune += accepted
        self._steps_until_tune -= 1

        stats = {
            "acceptance_ratio": acceptance_ratio,
            "accepted": accepted,
            "scaling": self._scaling,
        }

        return theta, theta_constrained, stats

    def _tune_scaling(self):
        acceptance_rate = self._accepted_since_tune / self.tune_interval

        if acceptance_rate < 0.001:
            # Reduce by 90 percent.
            self._scaling *= 0.1
        elif acceptance_rate < 0.05:
            # Reduce by 50 percent.
            self._scaling *= 0.5
        elif acceptance_rate < 0.2:
            # Reduce by ten percent.
            self._scaling *= 0.9
        elif acceptance_rate > 0.95:
            # Increase by factor of ten.
            self._scaling *= 10.0
        elif acceptance_rate > 0.75:
            # Increase by double.
            self._scaling *= 2.0
        elif acceptance_rate > 0.5:
            # Increase by ten percent.
            self._scaling *= 1.1
