from typing import Dict

import numpy as np
from numpy.random import RandomState

from pynference.constants import Sample
from pynference.distributions.transformations import Transformation
from pynference.infrastructure import Seed, Substitute, Trace

__all__ = [
    "get_model_transformations",
    "potential_energy",
    "log_prob",
    "transform_parameters",
]


def get_model_transformations(
    model, random_state: RandomState, *args, **kwargs
) -> Dict[str, Transformation]:
    model = Seed(model, random_state)
    trace = Trace(model)
    return trace.transformations(*args, **kwargs)


def potential_energy(
    model, transformations: Dict[str, Transformation], theta: Sample, *args, **kwargs
) -> float:
    theta_constrained = transform_parameters(theta, transformations, inverse=False)
    log_p = log_prob(model, theta_constrained, *args, **kwargs)

    for name, transformation in transformations.items():
        log_p += np.sum(transformation.log_abs_J(theta[name], theta_constrained[name]))

    return -log_p


def log_prob(model, theta_constrained: Sample, *args, **kwargs) -> float:
    model = Substitute(model, condition=theta_constrained)
    trace = Trace(model)
    return trace.log_prob(*args, **kwargs)


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
