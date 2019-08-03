from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]  # Return type of probability a evaluation.
Parameter = Union[float, np.ndarray]  # Parameters of probability distributions.
Variate = Union[float, np.ndarray]  # Return type of distribution sampling.
