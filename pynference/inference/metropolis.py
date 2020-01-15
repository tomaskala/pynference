import abc
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from numpy.random import RandomState

from pynference.constants import ArrayLike, Sample, Shape, Variate
from pynference.distributions.transformations import Transformation, biject_to
from pynference.model.model import Model
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


def get_model_transformations(model: Model) -> Dict[str, Transformation]:
    return {
        name: biject_to(constraint) for name, constraint in model.constraints.items()
    }


def transform_parameters(
    parameters: Sample,
    transformations: Dict[str, Transformation],
    inverse: bool = False,
) -> Sample:
    """
    Apply the given transformations to the sampled parameters. If a parameter without
    a defined transformation is present, it is left untransformed.

    If inverse is False, the forward transformation (unconstrained -> constraint) is applied.
    Otherwise, the inverse transformation (constraint -> unconstrained) is applied.
    :param parameters: dictionary of parameters
    :param transformations: dictionary of transformations
    :param inverse: whether to apply inverse or forward transformations
    """
    if inverse:
        return {
            k: transformations[k].inverse(v) if k in transformations else v
            for k, v in parameters.items()
        }
    else:
        return {
            k: transformations[k](v) if k in transformations else v
            for k, v in parameters.items()
        }


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

        self.accepted = 0  # TODO: This can be removed because of stats.
        self.stats: List[Dict[str, Any]] = []

        self._accepted_since_tune = 0
        self._scaling = 1.0
        self._steps_until_tune = tune_interval

    def run(self) -> List[Sample]:
        samples = []
        transformations = get_model_transformations(self.model)

        # Sample and unconstrain the initial theta.
        theta = self.initialize()
        theta = transform_parameters(theta, transformations, inverse=True)

        for i in range(self.n_samples):
            theta, stats = self.step(theta, transformations)
            samples.append(theta)
            self.stats.append(stats)

            if i > 0 and i % 100 == 0:
                print(
                    "Done {}/{} samples. Accepted {} samples.".format(
                        i, self.n_samples, self.accepted
                    )
                )

        # Constrain samples.
        return [
            transform_parameters(sample, transformations, inverse=False)
            for sample in samples
        ]

    # TODO: Pass n_samples to allow mean/median/... initializations.
    def initialize(self) -> Sample:
        return self.model.sample()  # TODO: Pass random state?

    def step(
        self, theta: Sample, transformations: Dict[str, Transformation]
    ) -> Tuple[Sample, Dict[str, Any]]:
        if self._steps_until_tune == 0 and self.tune:
            self._tune_scaling()
            self._accepted_since_tune = 0
            self._steps_until_tune = self.tune_interval

        theta_prop = {}

        for name, param in theta.items():
            theta_prop[name] = param + self.proposal(np.shape(param)) * self._scaling

        acceptance_ratio = self._log_prob(theta_prop, transformations) - self._log_prob(
            theta, transformations
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
            "mh_log_ratio": acceptance_ratio,
            "accepted": accepted,
            "scaling": self._scaling,
        }

        return theta, stats

    def _log_prob(
        self, theta: Sample, transformations: Dict[str, Transformation]
    ) -> ArrayLike:
        theta_constrained = transform_parameters(theta, transformations, inverse=False)
        return self.model.log_prob(theta_constrained)

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
