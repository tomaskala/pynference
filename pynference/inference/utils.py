from typing import Callable, Dict, Tuple

import torch
from torch.distributions import Transform, Uniform, biject_to

from pynference.constants import Sample
from pynference.infrastructure import Condition, MessageType, Trace

__all__ = ["initialize_model"]


def initialize_model(
    model, init_strategy: str, *args, **kwargs
) -> Tuple[Sample, Callable[[Sample], float], Dict[str, Transform]]:
    trace = Trace(model).trace(*args, **kwargs)
    dummy_samples = {}
    transformations = {}

    for name, message in trace.items():
        if message.message_type is MessageType.SAMPLE and not message.is_observed:
            dummy_samples[name] = message.value.detach()
            transformations[name] = biject_to(message.fun.support)

    potential_energy = _get_potential_energy_fun(
        model, transformations, *args, **kwargs
    )
    initial_samples = _get_initial_samples(
        model, init_strategy, dummy_samples, transformations, *args, **kwargs
    )

    return initial_samples, potential_energy, transformations


def _get_potential_energy_fun(
    model, transformations: Dict[str, Transform], *args, **kwargs
) -> Callable[[Sample], float]:
    def _potential_energy(theta: Sample) -> float:
        nonlocal model, transformations, args, kwargs

        theta_constrained = {k: transformations[k](v) for k, v in theta.items()}
        log_p = _log_prob(model, theta_constrained, *args, **kwargs)

        for name, transformation in transformations.items():
            log_p += torch.sum(
                transformation.log_abs_det_jacobian(
                    theta[name], theta_constrained[name]
                )
            )

        return -log_p

    return _potential_energy


def _log_prob(model, theta_constrained: Sample, *args, **kwargs) -> float:
    model = Condition(model, condition=theta_constrained)
    trace = Trace(model)
    return trace.log_prob(*args, **kwargs)


def _get_initial_samples(
    model,
    init_strategy: str,
    dummy_samples: Sample,
    transformations: Dict[str, Transform],
    *args,
    **kwargs
) -> Sample:
    if init_strategy == "prior":
        trace = Trace(model).trace(*args, **kwargs)
        samples = {name: trace[name].value.detach() for name in dummy_samples}
        return {
            name: transformations[name].inv(sample) for name, sample in samples.items()
        }
    elif init_strategy == "uniform":
        radius = 2.0
        return {
            name: Uniform(
                sample.new_full(sample.shape, -radius),
                sample.new_full(sample.shape, radius),
            ).sample()
            for name, sample in dummy_samples.items()
        }
    else:
        raise ValueError(
            "Unrecognized initialization strategy: {}. Only `prior` "
            "and `uniform` are supported.".format(init_strategy)
        )
