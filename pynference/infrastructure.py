import abc
import collections
from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

from numpy.random import RandomState

from pynference.constants import Parameter, Shape, Variate
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
    is_observed: bool
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
    observation: Optional[Variate] = None,
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
            value=observation,
            is_observed=observation is not None,
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
            message.stop = True


class Seed(Messenger):
    def __init__(self, fun: Callable[..., Variate], random_state: RandomState):
        super().__init__(fun=fun)
        self.random_state = random_state

    def process_message(self, message: Message):
        if (
            message.message_type is MessageType.SAMPLE
            and not message.is_observed
            and message.kwargs["random_state"] is None
        ):
            message.kwargs["random_state"] = self.random_state


class Condition(Messenger):
    def __init__(
        self,
        fun: Callable[..., Variate],
        condition: Optional[Dict[str, Parameter]] = None,
        substitution: Optional[Callable[OrderedDict[str, Message]], Parameter] = None,
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
                value = self.substitution(message)

            if value is not None:
                message.value = value
                message.is_observed = True
