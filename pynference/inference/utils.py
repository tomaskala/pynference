from functools import partial
from typing import Callable, Dict

import numpy as np
from numpy.random import RandomState

from pynference.constants import Sample
from pynference.distributions import Uniform
from pynference.distributions.transformations import Transformation, biject_to
from pynference.model.model import Model

__all__ = [
    "init_to_mean",
    "init_to_prior",
    "init_to_uniform",
    "get_model_transformations",
    "transform_parameters",
]


def _init_to_prior(model: Model, random_state: RandomState) -> Sample:
    return model.sample(sample_shape=(), random_state=random_state)


def _init_to_mean(model: Model, random_state: RandomState, n_samples: int) -> Sample:
    thetas = model.sample(sample_shape=(n_samples,), random_state=random_state)
    return {name: np.mean(theta, axis=0) for name, theta in thetas.items()}


def _init_to_uniform(model: Model, random_state: RandomState, radius: float) -> Sample:
    theta = {}
    uniform = Uniform(lower=-radius, upper=radius)

    # Hack to get the variable shapes. Once more complex probabilistic
    # programming constructs are implemented, this can be refined.
    dummy_sample = model.sample(sample_shape=(), random_state=random_state)

    for name, constraint in model.constraints.items():
        transformation = biject_to(constraint)
        uniform_sample = uniform.sample(
            sample_shape=np.shape(dummy_sample[name]), random_state=random_state
        )
        theta[name] = transformation.inverse(uniform_sample)

    return theta


def init_to_prior() -> Callable[[Model, RandomState], Sample]:
    return _init_to_prior


def init_to_mean(n_samples: int = 20) -> Callable[[Model, RandomState], Sample]:
    return partial(_init_to_mean, n_samples=n_samples)


def init_to_uniform(radius: float = 2.0) -> Callable[[Model, RandomState], Sample]:
    return partial(_init_to_uniform, radius=radius)


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
