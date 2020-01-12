import abc
from typing import Dict, List, Tuple, Type

import numpy as np
from numpy.random import RandomState

from pynference.constants import Sample, Shape, Variate
from pynference.model import Model
from pynference.utils import check_random_state


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


# TODO: ArrayOrdering + DictToArrayBijection from PyMC?
# TODO: Or not if Jax is used. Apparently, it is no longer needed there.
class Metropolis:
    _proposal_map: Dict[str, Type[Proposal]] = {
        "cauchy": CauchyProposal,
        "laplace": LaplaceProposal,
        "normal": NormalProposal,
        "uniform": UniformProposal,
    }

    def __init__(
        self,
        model: Model,
        n_samples: int,
        proposal: str,
        scale_init: float = 1.0,
        tune: bool = True,
        tune_interval: int = 100,
        random_state: RandomState = None,
    ):
        self.model = model
        self.n_samples = n_samples

        if proposal in self._proposal_map:
            self.proposal = self._proposal_map[proposal](
                scale=scale_init, random_state=random_state
            )

        self.tune = tune
        self.tune_interval = tune_interval
        self.random_state = check_random_state(random_state)

        self.accepted = 0

        self.accepted_since_tune = 0
        self.scaling = 1.0
        self.steps_until_tune = tune_interval

    def run(self) -> List[Sample]:
        samples = []
        theta = self.initialize()

        # TODO: Unconstrain theta.

        for i in range(self.n_samples):
            if self.steps_until_tune == 0 and self.tune:
                self.tune_scaling()
                self.accepted_since_tune = 0
                self.steps_until_tune = self.tune_interval

            theta, accepted = self.step(theta)
            samples.append(theta)

            self.accepted += accepted
            self.accepted_since_tune += accepted
            self.steps_until_tune -= 1

            print(
                "Done sample {}/{}. Accepted {} samples.".format(
                    i + 1, self.n_samples, self.accepted
                )
            )

        # TODO: Constrain samples.

        return samples

    def initialize(self) -> Sample:
        return self.model.sample()  # TODO: Pass random state?

    def step(self, theta: Sample) -> Tuple[Sample, bool]:
        theta_prop = {}

        for name, param in theta.items():
            theta_prop[name] = param + self.proposal(np.shape(param)) * self.scaling

        acceptance_ratio = self.model.log_prob(theta_prop) - self.model.log_prob(theta)

        # TODO: Return some stats to be reported in the `run` method.
        if (
            np.isfinite(acceptance_ratio)
            and np.log(self.random_state.rand()) < acceptance_ratio
        ):
            return theta_prop, True
        else:
            return theta, False

    # TODO: Add underscores to internal methods. The same for interval variables.
    def tune_scaling(self):
        acceptance_rate = self.accepted_since_tune / self.tune_interval

        if acceptance_rate < 0.001:
            # Reduce by 90 percent.
            self.scaling *= 0.1
        elif acceptance_rate < 0.05:
            # Reduce by 50 percent.
            self.scaling *= 0.5
        elif acceptance_rate < 0.2:
            # Reduce by ten percent.
            self.scaling *= 0.9
        elif acceptance_rate > 0.95:
            # Increase by factor of ten.
            self.scaling *= 10.0
        elif acceptance_rate > 0.75:
            # Increase by double.
            self.scaling *= 2.0
        elif acceptance_rate > 0.5:
            # Increase by ten percent.
            self.scaling *= 1.1
