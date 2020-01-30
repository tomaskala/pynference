from functools import partial
from typing import Callable

import numpy as np
from numpy.random import RandomState

from pynference.constants import Sample, Shape, Variate
from pynference.distributions import TransformedDistribution, Uniform
from pynference.distributions.transformations import biject_to
from pynference.inference.utils import transform_parameters
from pynference.infrastructure import (
    Block,
    Message,
    MessageType,
    Seed,
    Substitute,
    Trace,
    sample,
)

__all__ = ["initialize", "init_to_mean", "init_to_prior", "init_to_uniform"]


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
