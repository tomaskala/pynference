import numpy as np

from pynference.constants import Shape


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
