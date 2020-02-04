from typing import Dict, Tuple, Union

import torch

# Return type of a probability evaluation.
ArrayLike = Union[float, torch.Tensor]

# Parameters of probability distributions.
Parameter = torch.Tensor

# Return type of distribution sampling.
Variate = torch.Tensor

# Output of inference algorithms. The log-prob of
# these is evaluated within probabilistic models.
Sample = Dict[str, torch.Tensor]

# Either an empty tuple (scalar) or a tuple of dimensions (array).
Shape = Tuple[int, ...]
