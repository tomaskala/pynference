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
from pynference.utils import check_random_state


def generate(
    random_state: np.random.RandomState,
    shape: Union[None, Shape],
    positive: Optional[Union[str, List[str]]] = None,
    real: Optional[Union[str, List[str]]] = None,
    lower: Optional[Union[str, List[str]]] = None,
    upper: Optional[Union[str, List[str]]] = None,
    integral: Optional[Union[str, List[str]]] = None,
    integral_lower: Optional[Union[str, List[str]]] = None,
    integral_upper: Optional[Union[str, List[str]]] = None,
    zero_one: Optional[Union[str, List[str]]] = None,
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
        "integral_lower_low": -10,
        "integral_lower_high": 10,
        "integral_upper_low": 20,
        "integral_upper_high": 30,
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
            parameters[p] = random_state.randint(
                low=limits["integral_low"], high=limits["integral_high"] + 1, size=shape
            )

    if integral_lower is not None:
        if isinstance(integral_lower, str):
            integral_lower = [integral_lower]

        for p in integral_lower:
            parameters[p] = random_state.randint(
                low=limits["integral_lower_low"],
                high=limits["integral_lower_high"],
                size=shape,
            )

    if integral_upper is not None:
        if isinstance(integral_upper, str):
            integral_upper = [integral_upper]

        for p in integral_upper:
            parameters[p] = random_state.randint(
                low=limits["integral_upper_low"],
                high=limits["integral_upper_high"],
                size=shape,
            )

    if zero_one is not None:
        if isinstance(zero_one, str):
            zero_one = [zero_one]

            for p in zero_one:
                parameters[p] = random_state.rand(*shape)

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

                # Hack to make all `n` parameter positive integers.
                if "n" in fst_params and "n" in snd_params:
                    fst_params["n"] = fst_params["n"].astype(int) + 1
                    snd_params["n"] = snd_params["n"].astype(int) + 1

                fst = distribution_cls(**fst_params)
                snd = distribution_cls(**snd_params)

                samples = fst.sample(
                    sample_shape=(self.n_samples,), random_state=self.random_state
                )

                assert fst.log_prob(samples) == approx(
                    snd.log_prob(samples), rel=self.rtol, abs=self.atol
                ), f"log_prob of {fst}"


class TestExponentialFamilies:
    random_state = check_random_state(123)

    distributions = {
        Bernoulli: generate(random_state, shape=(), zero_one="p"),
        Binomial: generate(random_state, shape=(), integral="n", zero_one="p"),
        Geometric: generate(random_state, shape=(), zero_one="p"),
        NegativeBinomial: generate(random_state, shape=(), positive="r", zero_one="p"),
        Poisson: generate(random_state, shape=(), positive="rate"),
    }

    n_samples = 100
    atol = 1e-6
    rtol = 1e-6

    def test_base_measure_positive_within_support(self):
        for distribution_cls, parameters in self.distributions.items():
            distribution = distribution_cls(**parameters)

            samples = distribution.sample(
                sample_shape=(self.n_samples,), random_state=self.random_state
            )

            assert np.all(
                distribution.base_measure(samples) > 0
            ), f"base measure of {distribution}"

    def test_log_probs_equal(self):
        for distribution_cls, parameters in self.distributions.items():
            distribution = distribution_cls(**parameters)

            samples = distribution.sample(
                sample_shape=(self.n_samples,), random_state=self.random_state
            )

            h_x = distribution.base_measure(samples)
            eta = distribution.natural_parameter
            t_x = distribution.sufficient_statistic(samples)
            a_eta = distribution.log_normalizer

            dot_product = sum(np.dot(e, t) for e, t in zip(eta, t_x))
            expected_log_prob = np.log(h_x) + dot_product - a_eta

            assert distribution.log_prob(samples) == approx(
                expected_log_prob, rel=self.rtol, abs=self.atol
            ), f"log_prob of {distribution}"


class TestFirstTwoMoments:
    random_state = check_random_state(123)

    distributions = {
        Bernoulli: (
            generate(random_state, shape=(), zero_one="p"),
            generate(random_state, shape=(2,), zero_one="p"),
            generate(random_state, shape=(2, 3), zero_one="p"),
        ),
        Binomial: (
            generate(random_state, shape=(), integral="n", zero_one="p"),
            generate(random_state, shape=(2,), integral="n", zero_one="p"),
            generate(random_state, shape=(2, 3), integral="n", zero_one="p"),
        ),
        Dirac: (
            generate(random_state, shape=(), real="x"),
            generate(random_state, shape=(2,), real="x"),
            generate(random_state, shape=(2, 3), real="x"),
        ),
        DiscreteUniform: (
            generate(
                random_state, shape=(), integral_lower="lower", integral_upper="upper"
            ),
            generate(
                random_state, shape=(2,), integral_lower="lower", integral_upper="upper"
            ),
            generate(
                random_state,
                shape=(2, 3),
                integral_lower="lower",
                integral_upper="upper",
            ),
        ),
        Geometric: (
            generate(random_state, shape=(), zero_one="p"),
            generate(random_state, shape=(2,), zero_one="p"),
            generate(random_state, shape=(2, 3), zero_one="p"),
        ),
        NegativeBinomial: (
            generate(random_state, shape=(), positive="r", zero_one="p"),
            generate(random_state, shape=(2,), positive="r", zero_one="p"),
            generate(random_state, shape=(2, 3), positive="r", zero_one="p"),
        ),
        Poisson: (
            generate(random_state, shape=(), positive="rate"),
            generate(random_state, shape=(2,), positive="rate"),
            generate(random_state, shape=(2, 3), positive="rate"),
        ),
    }

    n_samples = 200000
    atol = 1e-4
    rtol = 0.75

    def test_mean_and_variance(self):
        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(self.n_samples,), random_state=self.random_state
                )

                assert np.mean(samples, axis=0) == approx(
                    distribution.mean, rel=self.rtol, abs=self.atol
                ), f"mean of {distribution}"

                assert np.var(samples, axis=0) == approx(
                    distribution.variance, rel=self.rtol, abs=self.atol
                ), f"variance of {distribution}"


class TestLogProb:
    random_state = check_random_state(123)

    # Purposefully not testing the Dirac distribution since it does not exist in scipy.
    distributions = {
        Bernoulli: generate(random_state, shape=(), zero_one="p"),
        Binomial: generate(random_state, shape=(), integral="n", zero_one="p"),
        DiscreteUniform: generate(
            random_state, shape=(), integral_lower="lower", integral_upper="upper"
        ),
        Geometric: generate(random_state, shape=(), zero_one="p"),
        NegativeBinomial: generate(random_state, shape=(), positive="r", zero_one="p"),
        Poisson: generate(random_state, shape=(), positive="rate"),
    }

    dist2scipy = {
        Bernoulli: lambda dist: stats.bernoulli(p=dist.p),
        Binomial: lambda dist: stats.binom(n=dist.n, p=dist.p),
        DiscreteUniform: lambda dist: stats.randint(
            low=dist.lower, high=dist.upper + 1
        ),
        Geometric: lambda dist: stats.nbinom(n=1, p=dist.p),
        NegativeBinomial: lambda dist: stats.nbinom(n=dist.r, p=1.0 - dist.p),
        Poisson: lambda dist: stats.poisson(mu=dist.rate),
    }

    n_samples = 100
    atol = 1e-6
    rtol = 1e-6

    def test_log_prob(self):
        for distribution_cls, parameters in self.distributions.items():
            distribution = distribution_cls(**parameters)

            if distribution_cls not in self.dist2scipy:
                continue
            scipy_distribution = self.dist2scipy[distribution_cls](distribution)

            samples = distribution.sample(
                sample_shape=(self.n_samples,), random_state=self.random_state
            )
            assert distribution.log_prob(samples) == approx(
                scipy_distribution.logpmf(samples), rel=self.rtol, abs=self.atol
            ), f"log_prob of {distribution}"


class TestParameterConstraints:
    def test_bernoulli(self):
        with raises(ValueError, match=r".*zero_one.*"):
            Bernoulli(p=-1.0)
            Bernoulli(p=0.0)
            Bernoulli(p=1.0)
            Bernoulli(p=2.0)

    def test_binomial(self):
        with raises(ValueError, match=r".*positive_integer.*"):
            Binomial(n=-1, p=0.5)
            Binomial(n=0, p=0.5)
            Binomial(n=-1.5, p=0.5)
            Binomial(n=3.1, p=0.5)
            Binomial(n=10.0, p=0.5)

        with raises(ValueError, match=r".*zero_one.*"):
            Binomial(n=10, p=-1.0)
            Binomial(n=10, p=0.0)
            Binomial(n=10, p=1.0)
            Binomial(n=10, p=2.0)

    def test_dirac(self):
        with raises(ValueError, match=r".*real.*"):
            Dirac(x=np.nan)
            Dirac(x=-np.inf)
            Dirac(x=np.inf)

    @pytest.mark.filterwarnings("ignore", category=RuntimeWarning)
    def test_discrete_uniform(self):
        with raises(ValueError, match=r".*integer.*"):
            DiscreteUniform(lower=-10.0, upper=10.0)
            DiscreteUniform(lower=-10.5, upper=10.5)

        with raises(ValueError, match=r".*strictly lower.*"):
            DiscreteUniform(lower=-10, upper=-20)
            DiscreteUniform(lower=10, upper=10)
            DiscreteUniform(lower=20, upper=10)

    def test_geometric(self):
        with raises(ValueError, match=r".*zero_one.*"):
            Geometric(p=-1.0)
            Geometric(p=0.0)
            Geometric(p=1.0)
            Geometric(p=2.0)

    def test_negative_binomial(self):
        with raises(ValueError, match=r".*positive.*"):
            NegativeBinomial(r=-1.0, p=0.5)
            NegativeBinomial(r=0.0, p=0.5)

        with raises(ValueError, match=r".*zero_one.*"):
            NegativeBinomial(r=10, p=-1.0)
            NegativeBinomial(r=10, p=0.0)
            NegativeBinomial(r=10, p=1.0)
            NegativeBinomial(r=10, p=2.0)

    def test_poisson(self):
        with raises(ValueError, match=r".*positive.*"):
            Poisson(rate=-1.0)
            Poisson(rate=0.0)


class TestSamplingShapes:
    random_state = check_random_state(123)

    distributions = {
        Bernoulli: (
            generate(random_state, shape=(), zero_one="p"),
            generate(random_state, shape=(2,), zero_one="p"),
            generate(random_state, shape=(2, 3), zero_one="p"),
        ),
        Binomial: (
            generate(random_state, shape=(), integral="n", zero_one="p"),
            generate(random_state, shape=(2,), integral="n", zero_one="p"),
            generate(random_state, shape=(2, 3), integral="n", zero_one="p"),
        ),
        Dirac: (
            generate(random_state, shape=(), real="x"),
            generate(random_state, shape=(2,), real="x"),
            generate(random_state, shape=(2, 3), real="x"),
        ),
        DiscreteUniform: (
            generate(
                random_state, shape=(), integral_lower="lower", integral_upper="upper"
            ),
            generate(
                random_state, shape=(2,), integral_lower="lower", integral_upper="upper"
            ),
            generate(
                random_state,
                shape=(2, 3),
                integral_lower="lower",
                integral_upper="upper",
            ),
        ),
        Geometric: (
            generate(random_state, shape=(), zero_one="p"),
            generate(random_state, shape=(2,), zero_one="p"),
            generate(random_state, shape=(2, 3), zero_one="p"),
        ),
        NegativeBinomial: (
            generate(random_state, shape=(), positive="r", zero_one="p"),
            generate(random_state, shape=(2,), positive="r", zero_one="p"),
            generate(random_state, shape=(2, 3), positive="r", zero_one="p"),
        ),
        Poisson: (
            generate(random_state, shape=(), positive="rate"),
            generate(random_state, shape=(2,), positive="rate"),
            generate(random_state, shape=(2, 3), positive="rate"),
        ),
    }

    def test_sampling_shapes_0d(self):
        batch_shapes = [(), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(), random_state=self.random_state
                )
                assert (
                    samples.shape == batch_shapes[i]
                ), f"sampling shape of {distribution}"

    def test_sampling_shapes_1d(self):
        batch_shapes = [(), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(100,), random_state=self.random_state
                )
                assert (
                    samples.shape == (100,) + batch_shapes[i]
                ), f"sampling shape of {distribution}"

    def test_sampling_shapes_2d(self):
        batch_shapes = [(), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(10, 10), random_state=self.random_state
                )
                assert (
                    samples.shape == (10, 10) + batch_shapes[i]
                ), f"sampling shape of {distribution}"

    def test_sampling_shapes_3d(self):
        batch_shapes = [(), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(10, 10, 2), random_state=self.random_state
                )
                assert (
                    samples.shape == (10, 10, 2) + batch_shapes[i]
                ), f"sampling shape of {distribution}"
