import abc
import collections
from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

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
    value: Variate
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    stop: bool = field(default=False)


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
    for i, messenger in enumerate(reversed(_MESSENGER_STACK)):
        messenger.process_message(message)

        if message.stop:
            break

    if message.value is None:
        message.value = message.dist(*message.args, **message.kwargs)

    for messenger in _MESSENGER_STACK[-i - 1 :]:
        messenger.postprocess_message(message)

    return message


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


class Trace(Messenger):
    def __init__(self, fun: Optional[Callable[..., Variate]] = None):
        super().__init__(fun=fun)
        self._trace = None

    def __enter__(self):
        super().__enter__()
        self._trace = collections.OrderedDict()
        return self._trace

    def trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self._trace

    def postprocess_message(self, message: Message):
        if message.message_type != MessageType.SAMPLE:
            raise ValueError("Only sample sites can be registered to a trace.")

        if message.name in self._trace:
            raise ValueError("The sample sites must have unique names.")

        self._trace[message.name] = copy(message)


class Replay(Messenger):
    def __init__(self, fun: Callable[..., Variate], trace: OrderedDict):
        super().__init__(fun=fun)
        self.trace = trace

    def process_message(self, message: Message):
        if message.name in self.trace and message.message_type is MessageType.SAMPLE:
            message.value = self.trace[message.name].value
