import abc
import collections
import random
from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Union

import numpy as np
import torch
from torch.distributions import Distribution

from pynference.constants import Parameter, Variate

__all__ = [
    "sample",
    "MessageType",
    "Trace",
    "Replay",
    "Block",
    "Seed",
    "Condition",
]


class MessageType(Enum):
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
    log_prob_sum: Union[float, None] = field(default=None)


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
    *args,
    **kwargs
) -> Variate:
    if not _MESSENGER_STACK:
        if callable(dist):
            return dist(*args, **kwargs)
        else:
            return dist.sample(*args, **kwargs)
    else:
        message = Message(
            message_type=MessageType.SAMPLE,
            name=name,
            fun=dist,
            kwargs={},
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
                if message.log_prob_sum is None:
                    message.log_prob_sum = torch.sum(
                        message.fun.log_prob(message.value)
                    )

                log_prob += message.log_prob_sum

        return log_prob

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
        name = message.name

        if name in self.trace and message.message_type is MessageType.SAMPLE:
            trace_message = self.trace[name]

            if message.is_observed:
                return

            if (
                trace_message.message_type is not MessageType.SAMPLE
                or trace_message.is_observed
            ):
                raise ValueError(
                    "The message under the name {} must be a sample site "
                    "in the replayed trace.".format(name)
                )

            message.value = trace_message.value


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
    def __init__(self, fun: Callable[..., Variate], condition: Dict[str, Parameter]):
        super().__init__(fun=fun)
        self.condition = condition

    def process_message(self, message: Message):
        if message.message_type is MessageType.SAMPLE:
            name = message.name

            if name in self.condition:
                message.value = self.condition[name]
                message.is_observed = message.value is not None
