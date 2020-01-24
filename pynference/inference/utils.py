from functools import partial
from typing import Any, Callable, Dict, Tuple

import numpy as np
from numpy.random import RandomState

from pynference.constants import Sample, Shape, Variate
from pynference.distributions import TransformedDistribution, Uniform
from pynference.distributions.transformations import Transformation, biject_to
from pynference.infrastructure import (
    Block,
    Message,
    MessageType,
    Seed,
    Substitute,
    Trace,
    sample,
)

__all__ = [
    "init_to_mean",
    "init_to_prior",
    "init_to_uniform",
    "get_model_transformations",
    "initialize",
    "prepare_metropolis_functions",
    "log_prob",
    "transform_parameters",
]


def _sample_from_message(message: Message, sample_shape: Shape) -> Variate:
    if message.message_type is MessageType.SAMPLE:
        if isinstance(message.fun, TransformedDistribution):
            dist = message.fun.base_distribution
        else:
            dist = message.fun  # type: ignore

        return sample(
            "_initialization",
            dist,
            sample_shape=sample_shape + message.kwargs["sample_shape"],
        )
    else:
        raise ValueError(
            "Can only initialize a message of type SAMPLE but {} was given.".format(
                message.message_type
            )
        )


def _init_to_prior(message: Message) -> Variate:
    return _sample_from_message(message, sample_shape=())


def _init_to_mean(message: Message, n_samples: int) -> Variate:
    samples = _sample_from_message(message, sample_shape=(n_samples,))
    return np.mean(samples, axis=0)


def _init_to_uniform(message: Message, radius: float) -> Variate:
    if message.message_type is MessageType.SAMPLE:
        if isinstance(message.fun, TransformedDistribution):
            dist = message.fun.base_distribution
        else:
            dist = message.fun  # type: ignore

        dummy = sample(
            "_initialization", dist, sample_shape=message.kwargs["sample_shape"]
        )
        transformation = biject_to(dist.support)

        uniform_sample = sample(
            "_uniform",
            Uniform(lower=-radius, upper=radius),
            sample_shape=np.shape(transformation.inverse(dummy)),
        )
        return transformation(uniform_sample)
    else:
        raise ValueError(
            "Can only initialize a message of type SAMPLE but {} was given.".format(
                message.message_type
            )
        )


def init_to_prior() -> Callable[[Message], Variate]:
    return _init_to_prior


def init_to_mean(n_samples: int = 20) -> Callable[[Message], Variate]:
    return partial(_init_to_mean, n_samples=n_samples)


def init_to_uniform(radius: float = 2.0) -> Callable[[Message], Variate]:
    return partial(_init_to_uniform, radius=radius)


def get_model_transformations(
    model, random_state: RandomState, *args, **kwargs
) -> Dict[str, Transformation]:
    model = Seed(model, random_state)
    trace = Trace(model)
    return trace.transformations(*args, **kwargs)


# TODO: Change the following:
# 1. make create_log_prob work on unconstrained parameters
# 2. make metropolis nicely constrain the sampled parameters at the end


def initialize(
    model,
    initializer: Callable[[Message], Variate],
    random_state: RandomState,
    *args,
    **kwargs
) -> Sample:
    model = Substitute(model, substitution=Block(Seed(initializer, random_state)))
    trace = Trace(model).trace(*args, **kwargs)

    constrained = {}
    transformations = {}

    for name, message in trace.items():
        if message.message_type is MessageType.SAMPLE and not message.is_observed:
            constrained[name] = message.value
            transformations[name] = biject_to(message.fun.support)

    # Unconstrain the parameters.
    return transform_parameters(constrained, transformations, inverse=True)


def prepare_metropolis_functions(
    model, random_state: RandomState, *args, **kwargs
) -> Tuple[Callable[[Sample, Any, Any], float], Callable[[Sample], Sample]]:
    transformations = get_model_transformations(model, random_state, *args, **kwargs)

    log_prob_function = partial(log_prob, model=model)
    transformation_function = partial(
        transform_parameters, transformations=transformations, inverse=False
    )

    return log_prob_function, transformation_function


def log_prob(theta: Sample, model, *args, **kwargs) -> float:
    # The `theta` is assumed to be constrained to the model support.
    model = Substitute(model, base_distribution_condition=theta)
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
