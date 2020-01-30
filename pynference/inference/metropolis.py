import abc
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from numpy.random import RandomState

from pynference.constants import Sample, Shape, Variate
from pynference.inference.initializers import init_to_prior, initialize
from pynference.inference.utils import (
    get_model_transformations,
    potential_energy,
    transform_parameters,
)
from pynference.utils import check_random_state

__all__ = ["Metropolis"]


class Proposal(abc.ABC):
    def __init__(self, scale, random_state: RandomState):
        self.scale = scale
        self.random_state = check_random_state(random_state)

    @abc.abstractmethod
    def __call__(self, shape: Shape) -> Variate:
        pass


class CauchyProposal(Proposal):
    def __call__(self, shape: Shape) -> Variate:
        return self.random_state.standard_cauchy(shape) * self.scale


class LaplaceProposal(Proposal):
    def __call__(self, shape: Shape) -> Variate:
        return self.random_state.laplace(scale=self.scale, size=shape)


class NormalProposal(Proposal):
    def __call__(self, shape: Shape) -> Variate:
        return self.random_state.normal(scale=self.scale, size=shape)


class UniformProposal(Proposal):
    def __init__(self, scale, random_state: RandomState):
        super().__init__(scale=scale, random_state=random_state)
        self.scale /= 2

    def __call__(self, shape: Shape) -> Variate:
        return self.random_state.uniform(low=-self.scale, high=self.scale, size=shape)


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
        scale_init: float = 1.0,
        init=init_to_prior(),
        tune: bool = True,
        tune_interval: int = 100,
        track_stats: bool = True,
        random_state: RandomState = None,
    ):
        self.model = model
        self.n_samples = n_samples

        if proposal in self._proposal_map:
            self.proposal = self._proposal_map[proposal](
                scale=scale_init, random_state=random_state
            )
        else:
            raise ValueError("Unknown proposal type: {}.".format(proposal))

        self.init = init
        self.tune = tune
        self.tune_interval = tune_interval
        self.track_stats = track_stats
        self.random_state = check_random_state(random_state)

        self.accepted = 0
        self.stats: List[Dict[str, Any]] = []

        self._transformations = None
        self._accepted_since_tune = 0
        self._scaling = 1.0
        self._steps_until_tune = tune_interval

    def run(self, *args, **kwargs) -> List[Sample]:
        self._transformations = get_model_transformations(
            self.model, self.random_state, *args, **kwargs
        )
        samples = []
        theta = initialize(self.model, self.init, self.random_state, *args, **kwargs)

        for i in range(self.n_samples):
            theta, stats = self.step(theta, *args, **kwargs)
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
            transform_parameters(sample, self._transformations) for sample in samples
        ]

    def step(self, theta: Sample, *args, **kwargs) -> Tuple[Sample, Dict[str, Any]]:
        if self._steps_until_tune == 0 and self.tune:
            self._tune_scaling()
            self._accepted_since_tune = 0
            self._steps_until_tune = self.tune_interval

        theta_prop = {}

        for name, param in theta.items():
            theta_prop[name] = param + self.proposal(np.shape(param)) * self._scaling

        # Reversed order than usual because the potential energy is minus log_prob.
        acceptance_ratio = potential_energy(
            self.model, self._transformations, theta, *args, **kwargs
        ) - potential_energy(
            self.model, self._transformations, theta_prop, *args, **kwargs
        )

        if (
            np.isfinite(acceptance_ratio)
            and np.log(self.random_state.rand()) < acceptance_ratio
        ):
            accepted = True
            theta = theta_prop
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

        return theta, stats

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
