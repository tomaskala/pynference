import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from numpy.random import RandomState

from pynference.constants import Shape, Variate
from pynference.distributions.distribution import Distribution


class MessageType(Enum):
    PARAM = auto()
    SAMPLE = auto()


@dataclass
class Message:
    message_type: MessageType
    name: str
    dist: Distribution
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    value: Variate


_MESSENGER_STACK: List["Messenger"] = []


class Messenger(abc.ABC):
    def __init__(self, fun: Optional[Callable[..., Variate]] = None):
        self.fun = fun

    def __enter__(self):
        _MESSENGER_STACK.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        assert _MESSENGER_STACK[-1] is self
        _MESSENGER_STACK.pop(self)

    def __call__(self, *args, **kwargs) -> Variate:
        with self:
            return self.fun(*args, **kwargs)

    def process_message(self, message: Message):
        pass

    def postprocess_message(self, message: Message):
        pass


def _apply_stack(message: Message) -> Message:
    pass


def sample(
    name: str,
    dist: Distribution,
    observed: Optional[Variate] = None,
    sample_shape: Shape = (),
    random_state: RandomState = None,
) -> Variate:
    if not _MESSENGER_STACK:
        return dist(sample_shape=sample_shape, random_state=random_state)
    else:
        message = Message(
            message_type=MessageType.SAMPLE,
            name=name,
            dist=dist,
            kwargs={"sample_shape": sample_shape, "random_state": random_state},
            value=observed,
        )
        message = _apply_stack(message)
        return message.value
