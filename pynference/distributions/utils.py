from typing import Iterable

import numpy as np
from scipy.special import gammaln

from pynference.constants import ArrayLike, Parameter, Shape, Variate


def broadcast_shapes(*shapes: Shape) -> Shape:
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

    return tuple(result_shape)


def log_binomial_coefficient(n: Parameter, k: Variate) -> ArrayLike:
    log_numerator = gammaln(n + 1)
    log_denominator = gammaln(k + 1) + gammaln(n - k + 1)
    return log_numerator - log_denominator


def promote_shapes(*arrays: np.ndarray, shape: Shape = ()) -> Iterable[np.ndarray]:
    if len(arrays) < 2 and not shape:
        return arrays
    else:
        shapes = [np.shape(array) for array in arrays]
        n_dims = len(broadcast_shapes(shape, *shapes))
        return [
            np.reshape(array, (1,) * (n_dims - len(s)) + s)
            if len(s) < n_dims
            else array
            for array, s in zip(arrays, shapes)
        ]


def sum_last(array: np.ndarray, k: int) -> ArrayLike:
    return np.sum(array, axis=tuple(range(-k, 0)))
