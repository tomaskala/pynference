import abc
from typing import Any, Dict, List, Tuple, Type

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


# TODO: Constraints
# ----------------
# 1. Implement a function `biject_to(constraint)` which, given a constraint, returns
#    a transformation object. The transformation (some are already implemented) allow
#    to transform, inverse-transform and calculate log_abs_det_J. Take inspiration in
#    https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py
# 2. Each model would return a dictionary {parameter_name: constraint}. Before sampling,
#    assemble the transformations & do inverse (unconstraining) transforms. After sampling,
#    do forward (constraining) transforms.
# NOTE: It would be be much simpler to return a dictionary {parameter: distribution} or
# something like that. Currently impossible, needs higher probabilistic programming
# constructs. For now, stick with the above formulation even if it seems stupid in
# certain parts.

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
        self.stats: List[Dict[str, Any]] = []

        self._accepted_since_tune = 0
        self._scaling = 1.0
        self._steps_until_tune = tune_interval

    def run(self) -> List[Sample]:
        samples = []
        theta = self.initialize()

        # TODO: Unconstrain theta.

        for i in range(self.n_samples):
            theta, stats = self.step(theta)
            samples.append(theta)
            self.stats.append(stats)

        # TODO: Constrain samples.

        return samples

    def initialize(self) -> Sample:
        return self.model.sample()  # TODO: Pass random state?

    def step(self, theta: Sample) -> Tuple[Sample, Dict[str, Any]]:
        if self._steps_until_tune == 0 and self.tune:
            self._tune_scaling()
            self._accepted_since_tune = 0
            self._steps_until_tune = self.tune_interval

        theta_prop = {}

        for name, param in theta.items():
            theta_prop[name] = param + self.proposal(np.shape(param)) * self._scaling

        acceptance_ratio = self.model.log_prob(theta_prop) - self.model.log_prob(theta)

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
            "mh_log_ratio": acceptance_ratio,
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
