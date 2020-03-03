import abc
import random
from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from numbers import Number
from typing import (  # type: ignore
    Any,
    Callable,
    Dict,
    List,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)

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
    "Plate",
    "Mask",
    "Enumerate",
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
    log_prob_sum: Union[torch.Tensor, None] = field(default=None)
    conditional_independence_stack: Tuple[
        "ConditionalIndependenceStackFrame", ...  # noqa W504
    ] = field(default_factory=tuple)
    mask: Union[torch.Tensor, None] = field(default=None)
    done: bool = field(default=False)


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
            return self.fun(*args, **kwargs)

    def process_message(self, message: Message):
        pass

    def postprocess_message(self, message: Message):
        pass


def _apply_stack(message: Message) -> Message:
    for i, messenger in enumerate(reversed(_MESSENGER_STACK)):
        messenger.process_message(message)

        if message.block:
            break

    if not message.done and not message.is_observed and message.value is None:
        message.value = message.fun(*message.args, **message.kwargs)

    message.done = True

    for messenger in _MESSENGER_STACK[-i - 1 :]:
        messenger.postprocess_message(message)

    return message


def sample(
    name: str,
    dist: Distribution,
    observation: Optional[Variate] = None,
    *args,
    **kwargs
):
    if not _MESSENGER_STACK:
        return dist(*args, **kwargs)  # type: ignore
    else:
        message = Message(
            message_type=MessageType.SAMPLE,
            name=name,
            fun=dist,  # type: ignore
            kwargs={},
            value=observation,
            is_observed=observation is not None,
        )
        message = _apply_stack(message)
        return message.value


class Trace(Messenger):
    def __init__(self, fun: Callable[..., Variate]):
        super().__init__(fun=fun)
        self._trace = OrderedDict[str, Message]()

    def __enter__(self):
        super().__enter__()
        self._trace = OrderedDict[str, Message]()
        return self._trace

    def trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self._trace

    def log_prob(self, *args, **kwargs) -> torch.Tensor:
        trace = self.trace(*args, **kwargs)
        log_prob = 0.0

        for name, message in trace.items():
            if message.message_type is MessageType.SAMPLE:
                if message.log_prob_sum is None:
                    message_log_prob = message.fun.log_prob(message.value)

                    if message.mask is not None:
                        message_log_prob = torch.where(
                            message.mask, message_log_prob, message_log_prob.new_zeros()
                        )

                    message.log_prob_sum = torch.sum(message_log_prob)

                log_prob += message.log_prob_sum

        return log_prob  # type: ignore

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
        super().__enter__()

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
        return super().__exit__(exc_type, exc_value, traceback)


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


class ConditionalIndependenceStackFrame:
    def __init__(self, name, size, dim, counter):
        self.name = name
        self.size = size
        self.dim = dim
        self.counter = counter

    @property
    def vectorized(self):
        return self.dim is not None

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return type(self) == type(other) and self._key == other._key

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def _key(self):
        return (
            self.name,
            self.size if isinstance(self.size, Number) else self.size.item(),
            self.dim,
            self.counter,
        )


class _DimensionAllocator:
    def __init__(self):
        self._stack: List[Union[str, None]] = []
        self._used_names: Set[str] = set()

    def allocate(self, name: str, dim: Optional[int] = None):
        if name in self._used_names:
            raise ValueError("A plate with the name {} already exists.".format(name))

        if dim is not None and dim >= 0:
            raise ValueError(
                "The dimension to be allocated is expected to be a negative index from the right."
            )

        if dim is None:
            dim = -1

            while -dim <= len(self._stack) and self._stack[-1 - dim] is not None:
                dim -= 1

        while dim < -len(self._stack):
            self._stack.append(None)

        if self._stack[-1 - dim] is not None:
            raise ValueError(
                "The plates {} and {} collide at dimension {}.".format(
                    name, self._stack[-1 - dim], dim
                )
            )

        self._stack[-1 - dim] = name
        self._used_names.add(name)
        return dim

    def free(self, name: str, dim: int):
        free_idx = -1 - dim
        assert self._stack[free_idx] == name
        assert name in self._used_names

        self._stack[free_idx] = None
        self._used_names.remove(name)

        while self._stack and self._stack[-1] is None:
            self._stack.pop()


_DIMENSION_ALLOCATOR = _DimensionAllocator()


class Plate(Messenger):
    def __init__(self, name, size, dim=None):
        self.name = name
        self.size = size
        self.dim = dim
        self.counter = 0

        self._vectorized = None
        self._indices = None

    @property
    def indices(self):
        if self._indices is None:
            self._indices = torch.arange(self.size, dtype=torch.long)

        return self._indices

    def process_message(self, message: Message):
        frame = ConditionalIndependenceStackFrame(
            name=self.name, size=self.size, dim=self.dim, counter=self.counter
        )
        message.conditional_independence_stack = (
            frame,
        ) + message.conditional_independence_stack

        if message.message_type is MessageType.SAMPLE:
            batch_shape = getattr(message.fun, "batch_shape", None)

            if batch_shape is not None:
                expanded_batch_shape = [
                    None if size == 1 else size for size in batch_shape
                ]

                for f in message.conditional_independence_stack:
                    if f.dim is None or f.size == -1:
                        continue

                    assert f.dim < 0
                    expanded_batch_shape = [None] * (
                        -f.dim - len(expanded_batch_shape)
                    ) + expanded_batch_shape

                    if (
                        expanded_batch_shape[f.dim] is not None
                        and expanded_batch_shape[f.dim] != f.size
                    ):
                        raise ValueError(
                            "Mismatched shapes in plate {} at site {} and dimension {}. {} != {}.".format(
                                f.name,
                                message.name,
                                f.dim,
                                f.size,
                                expanded_batch_shape[f.dim],
                            )
                        )

                    expanded_batch_shape[f.dim] = f.size

                    for i in range(-len(expanded_batch_shape) + 1, 1):
                        if expanded_batch_shape[i] is None:
                            expanded_batch_shape[i] = (
                                batch_shape[i] if len(batch_shape) >= -i else 1
                            )

                    message.fun = message.fun.expand(expanded_batch_shape)

    def __enter__(self):
        if self._vectorized is not False:
            self._vectorized = True

        if self._vectorized is True:
            self.dim = _DIMENSION_ALLOCATOR.allocate(self.name, self.dim)

        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._vectorized is True:
            _DIMENSION_ALLOCATOR.free(self.name, self.dim)

        return super().__exit__(exc_type, exc_value, traceback)

    def __iter__(self):
        if self._vectorized or self.dim is not None:
            raise ValueError("Attempting to iterate a vectorized plate.")

        self._vectorized = False

        for i in self.indices:
            self.counter += 1

            with self:
                yield i if isinstance(i, Number) else i.item()


class Mask(Messenger):
    def __init__(self, mask):
        if not (
            isinstance(mask, bool)
            or (isinstance(mask, torch.Tensor) and mask.dtype == torch.bool)
        ):
            raise ValueError("The mask must be either a boolean or a boolean tensor.")

        if isinstance(mask, bool):
            mask = torch.tensor(mask)

        self.mask = mask

    def _process_message(self, message: Message):
        message.mask = self.mask if message.mask is None else self.mask & message.mask


class Enumerate(Messenger):
    def __init__(self):
        pass

    def _process_message(self, message: Message):
        if message.done or message.is_observed:
            return

        dist = message.fun

        if dist.has_enumerate_support:
            value = dist.enumerate_support()

            # TODO: Speed up the categorical distribution.

            message.value = value
            message.done = True
