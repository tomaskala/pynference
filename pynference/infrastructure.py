import abc
import collections
import random
from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Union

import numpy as np
import torch

from pynference.constants import Parameter, Shape, Variate
from pynference.distributions.constraints import real
from pynference.distributions.distribution import Distribution, TransformedDistribution
from pynference.distributions.transformations import (
    ComposeTransformation,
    Transformation,
    biject_to,
)

__all__ = [
    "sample",
    "MessageType",
    "Trace",
    "Replay",
    "Block",
    "Seed",
    "Condition",
    "Substitute",
]


class MessageType(Enum):
    PARAM = auto()
    SAMPLE = auto()


@dataclass
class Message:
    message_type: MessageType
    name: str
    fun: Callable[..., Variate]
    value: Optional[Variate]
    is_observed: bool
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    block: bool = field(default=False)


_MESSENGER_STACK: List["Messenger"] = []


class Messenger(abc.ABC):
    def __init__(self, fun: Callable[..., Variate]):
        self.fun = fun

    def __enter__(self):
        _MESSENGER_STACK.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        assert _MESSENGER_STACK[-1] is self
        _MESSENGER_STACK.pop()

    def __call__(self, *args, **kwargs) -> Variate:
        with self:
            if callable(self.fun):
                return self.fun(*args, **kwargs)
            else:
                return self.fun.sample(*args, **kwargs)

    def process_message(self, message: Message):
        pass

    def postprocess_message(self, message: Message):
        pass


def _apply_stack(message: Message) -> Message:
    for i, messenger in enumerate(reversed(_MESSENGER_STACK)):
        messenger.process_message(message)

        if message.block:
            break

    if message.value is None:
        if callable(message.fun):
            message.value = message.fun(*message.args, **message.kwargs)
        else:
            message.value = message.fun.sample(*message.args, **message.kwargs)

    for messenger in _MESSENGER_STACK[-i - 1 :]:
        messenger.postprocess_message(message)

    return message


def sample(
    name: str,
    dist: Distribution,
    observation: Optional[Variate] = None,
    sample_shape: Shape = (),
) -> Variate:
    if not _MESSENGER_STACK:
        return dist(sample_shape=sample_shape)
    else:
        message = Message(
            message_type=MessageType.SAMPLE,
            name=name,
            fun=dist,
            kwargs={"sample_shape": sample_shape},
            value=observation,
            is_observed=observation is not None,
        )
        message = _apply_stack(message)
        return message.value


class Trace(Messenger):
    def __init__(self, fun: Callable[..., Variate]):
        super().__init__(fun=fun)
        self._trace: OrderedDict[str, Message] = collections.OrderedDict()

    def __enter__(self):
        super().__enter__()
        self._trace = collections.OrderedDict()
        return self._trace

    def trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self._trace

    def log_prob(self, *args, **kwargs) -> float:
        trace = self.trace(*args, **kwargs)
        log_prob = 0.0

        for name, message in trace.items():
            if message.message_type is MessageType.SAMPLE:
                # TODO: Memoize inside messages?
                log_prob += torch.sum(message.fun.log_prob(message.value))

        return log_prob

    def transformations(self, *args, **kwargs) -> Dict[str, Transformation]:
        trace = self.trace(*args, **kwargs)
        inv_transforms = {}

        for name, message in trace.items():
            if message.message_type is MessageType.SAMPLE and not message.is_observed:
                # TODO: Add a condition for isinstance transformed distribution.
                # TODO: If yes, replay_model=True and only biject to the base
                # TODO: distribution support. Then use base_distribution_condition
                # TODO: in inference/utils/log_prob.
                inv_transforms[name] = biject_to(message.fun.support)
            elif message.message_type is MessageType.PARAM:
                constraint = kwargs.pop("constraint", real)
                transformation = biject_to(constraint)
                inv_transforms[name] = transformation

        return inv_transforms

    def postprocess_message(self, message: Message):
        if message.message_type != MessageType.SAMPLE:
            raise ValueError("Only sample sites can be registered to a trace.")

        if message.name in self._trace:
            raise ValueError("The sample sites must have unique names.")

        self._trace[message.name] = copy(message)


class Replay(Messenger):
    def __init__(self, fun: Callable[..., Variate], trace: OrderedDict[str, Message]):
        super().__init__(fun=fun)
        self.trace = trace

    def process_message(self, message: Message):
        if message.name in self.trace and message.message_type is MessageType.SAMPLE:
            message.value = self.trace[message.name].value


class Block(Messenger):
    def __init__(
        self,
        fun: Callable[..., Variate],
        hide_predicate: Callable[[Message], bool] = lambda message: True,
    ):
        super().__init__(fun=fun)
        self.hide_predicate = hide_predicate

    def process_message(self, message: Message):
        if self.hide_predicate(message):
            message.block = True


class Seed(Messenger):
    def __init__(self, fun: Callable[..., Variate], random_seed: Union[int, None]):
        super().__init__(fun=fun)
        self.random_seed = random_seed
        self.old_state: Dict[str, Any] = {}

    def __enter__(self):
        self.old_state = {
            "torch": torch.get_rng_state(),
            "random": random.getstate(),
            "numpy": np.random.get_state(),
        }

        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_rng_state(self.old_state["torch"])
        random.setstate(self.old_state["random"])
        np.random.set_state(self.old_state["numpy"])


class Condition(Messenger):
    def __init__(
        self,
        fun: Callable[..., Variate],
        condition: Optional[Dict[str, Parameter]] = None,
        substitution: Optional[Callable[[OrderedDict[str, Message]], Parameter]] = None,
    ):
        if (condition is not None) + (substitution is not None) != 1:
            raise ValueError(
                "Provide exactly one of the condition dictionary or substitution function."
            )

        super().__init__(fun=fun)
        self.condition = condition
        self.substitution = substitution

    def process_message(self, message: Message):
        if message.message_type is MessageType.SAMPLE:
            if message.is_observed:
                raise ValueError(
                    "Cannot condition an already observed sample site {}.".format(
                        message.name
                    )
                )

            if self.condition is not None:
                if message.name in self.condition:
                    value = self.condition[message.name]
            else:
                value = self.substitution(message)  # type: ignore

            if value is not None:
                message.value = value
                message.is_observed = True


class Substitute(Messenger):
    def __init__(
        self,
        fun: Callable[..., Variate],
        condition: Optional[Dict[str, Parameter]] = None,
        base_distribution_condition: Optional[Dict[str, Parameter]] = None,
        substitution: Optional[Callable[[OrderedDict[str, Message]], Parameter]] = None,
    ):
        if (condition is not None) + (base_distribution_condition is not None) + (
            substitution is not None
        ) != 1:
            raise ValueError(
                "Provide exactly one of the condition dictionary, base "
                "distribution condition dictionary or substitution function."
            )

        super().__init__(fun=fun)
        self.condition = condition
        self.base_distribution_condition = base_distribution_condition
        self.substitution = substitution

    def process_message(self, message: Message):
        if (
            message.message_type is MessageType.SAMPLE
            or message.message_type is MessageType.PARAM
        ):
            if self.condition is not None:
                if message.name in self.condition:
                    message.value = self.condition[message.name]
            else:
                if self.substitution is not None:
                    base_value = self.substitution(message)
                else:
                    base_value = self.base_distribution_condition.get(  # type: ignore
                        message.name, None
                    )

                if base_value is not None:
                    if message.message_type is MessageType.SAMPLE:
                        if isinstance(message.fun, TransformedDistribution):
                            for t in message.fun.transformation:
                                base_value = t(base_value)

                        message.value = base_value
                    else:
                        constraint = message.kwargs.pop("constraint", real)
                        transformation = biject_to(constraint)

                        if isinstance(transformation, ComposeTransformation):
                            skip_first = ComposeTransformation(
                                transformation.transformations[1:]
                            )
                            message.value = skip_first(base_value)
                        else:
                            message.value = base_value
