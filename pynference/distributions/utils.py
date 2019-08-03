import numbers
from typing import Tuple

import numpy as np
from numpy.random import RandomState


def check_random_state(seed) -> RandomState:
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError(f"Invalid random seed {seed}.")


def broadcast_shapes(*shapes: Tuple[int]) -> Tuple[int]:
    if len(shapes) == 1:
        return shapes[0]

    ndim = max(len(shape) for shape in shapes)
    shapes = np.array([(1,) * (ndim - len(shape)) + shape for shape in shapes])

    min_shape = np.min(shapes, axis=0)
    max_shape = np.max(shapes, axis=0)

    result_shape = np.where(min_shape == 0, 0, max_shape)

    if not np.all((shapes == result_shape) | (shapes == 1)):
        raise ValueError(
            f"Incompatible shapes for broadcasting: {tuple(map(tuple, shapes))}."
        )

    return tuple(result_shape)  # type: ignore
