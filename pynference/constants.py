from typing import Tuple, Union

import numpy as np

# Return type of a probability evaluation.
ArrayLike = Union[float, np.ndarray]

# Parameters of probability distributions.
Parameter = Union[float, np.ndarray]

# Return type of distribution sampling.
Variate = Union[float, np.ndarray]

# Either an empty tuple (scalar) or a tuple of dimensions (array).
Shape = Union[Tuple[()], Tuple[int]]
