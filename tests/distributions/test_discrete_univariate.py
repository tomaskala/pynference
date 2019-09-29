
from typing import Dict, List, Optional, Union

import numpy as np
import pytest
import scipy.stats as stats
from pytest import approx, raises

from pynference.constants import Parameter, Shape
from pynference.distributions import (
    Bernoulli,
    Binomial,
    Dirac,
    DiscreteUniform,
    Geometric,
    NegativeBinomial,
    Poisson,
)
from pynference.distributions.constraints import interval, positive
from pynference.utils import check_random_state


def generate(
    random_state: np.random.RandomState,
    shape: Union[None, Shape],
    positive: Optional[Union[str, List[str]]] = None,
    real: Optional[Union[str, List[str]]] = None,
    lower: Optional[Union[str, List[str]]] = None,
    upper: Optional[Union[str, List[str]]] = None,
    integral: Optional[Union[str, List[str]]] = None,
    **limits: float,
) -> Dict[str, Parameter]:
    parameters = {}

    if limits is None:
        limits = {}

    limits_default = {
        "positive_low": 0.001,
        "positive_high": 10.0,
        "real_low": -10.0,
        "real_high": 10.0,
        "lower_low": -10.0,
        "lower_high": 10.0,
        "upper_low": 20.0,
        "upper_high": 30.0,
        "integral_low": 2,
        "integral_high": 10,
    }

    limits = {**limits_default, **limits}

    if positive is not None:
        if isinstance(positive, str):
            positive = [positive]

        for p in positive:
            parameters[p] = random_state.uniform(
                low=limits["positive_low"], high=limits["positive_high"], size=shape
            )

    if real is not None:
        if isinstance(real, str):
            real = [real]

        for p in real:
            parameters[p] = random_state.uniform(
                low=limits["real_low"], high=limits["real_high"], size=shape
            )

    if lower is not None:
        if isinstance(lower, str):
            lower = [lower]

        for p in lower:
            parameters[p] = random_state.uniform(
                low=limits["lower_low"], high=limits["lower_high"], size=shape
            )

    if upper is not None:
        if isinstance(upper, str):
            upper = [upper]

        for p in upper:
            parameters[p] = random_state.uniform(
                low=limits["upper_low"], high=limits["upper_high"], size=shape
            )

    if integral is not None:
        if isinstance(integral, str):
            integral = [integral]

        for p in integral:
            parameters[p] = random_state.randint(low=limits["integral_low"], high=limits["integral_high"] + 1, size=shape)

    return parameters


class TestBroadcasting:
    random_state = check_random_state(123)

    distributions = {
        Bernoulli: ["p"],
        Binomial: ["n", "p"],
        Dirac: ["x"],
        DiscreteUniform: ["lower", "upper"],
        Geometric: ["p"],
        NegativeBinomial: ["r", "p"],
        Poisson: ["rate"],
    }

    n_samples = 100
    atol = 1e-6
    rtol = 1e-6

    def test_broadcasting(self):
        for distribution_cls, params in self.distributions.items():
            # Try setting each parameter to X and the others to aranges.
            # X == one distribution gets [[0.1]], the other [[0.1, 0.1], [0.1, 0.1]].
            fst_params = {}
            snd_params = {}

            for p1 in params:
                fst_params[p1] = np.full(shape=(2, 2), fill_value=0.1)
                snd_params[p1] = np.array(0.1).reshape(1, 1)

                for p2 in params:
                    if p1 == p2:
                        continue

                    fst_params[p2] = np.arange(0.2, 0.6, 0.1, dtype=float).reshape(2, 2)
                    snd_params[p2] = np.arange(0.2, 0.6, 0.1, dtype=float).reshape(2, 2)

                # Hack to make all lower bounds < upper bounds.
                if (
                    "lower" in fst_params
                    and "upper" in fst_params
                    and "lower" in snd_params
                    and "upper" in snd_params
                ):
                    if np.all(fst_params["lower"] >= fst_params["upper"]):
                        fst_params["lower"], fst_params["upper"] = (
                            fst_params["upper"],
                            fst_params["lower"],
                        )

                        fst_params["upper"] += 5.0

                    if np.all(snd_params["lower"] >= snd_params["upper"]):
                        snd_params["lower"], snd_params["upper"] = (
                            snd_params["upper"],
                            snd_params["lower"],
                        )

                        snd_params["upper"] += 5.0

                    fst_params["lower"] = np.floor(fst_params["lower"]).astype(int)
                    snd_params["lower"] = np.floor(snd_params["lower"]).astype(int)
                    fst_params["upper"] = np.ceil(fst_params["upper"]).astype(int)
                    snd_params["upper"] = np.ceil(snd_params["upper"]).astype(int)

                # Hack to make all `n` parameter integral.
                if "n" in fst_params and "n" in snd_params:
                    fst_params["n"] = fst_params["n"].astype(int)
                    snd_params["n"] = snd_params["n"].astype(int)

                fst = distribution_cls(**fst_params)
                snd = distribution_cls(**snd_params)

                samples = fst.sample(
                    sample_shape=(self.n_samples,), random_state=self.random_state
                )

                assert fst.log_prob(samples) == approx(
                    snd.log_prob(samples), rel=self.rtol, abs=self.atol
                ), f"log_prob of {fst}"
