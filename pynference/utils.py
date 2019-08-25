import numbers

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
