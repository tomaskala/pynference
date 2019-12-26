from typing import Tuple, Union

import numpy as np

# Return type of a probability evaluation.
ArrayLike = Union[float, np.ndarray]

# Parameters of probability distributions.
Parameter = np.ndarray

# Return type of distribution sampling.
Variate = np.ndarray

# Either an empty tuple (scalar) or a tuple of dimensions (array).
Shape = Tuple[int, ...]
