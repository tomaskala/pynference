import os
import random

import numpy as np
from jax.config import config

config.update("jax_platform_name", "cpu")  # noqa: E702


def pytest_runtest_setup(item):
    if "JAX_ENABLE_x64" in os.environ:
        config.update("jax_enable_x64", True)

    random.seed(123)
    np.random.seed(123)
