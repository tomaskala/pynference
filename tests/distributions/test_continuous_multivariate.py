from typing import Dict, List, Optional, Union

import numpy as np
import scipy.stats as stats
from pytest import approx, raises

from pynference.constants import Parameter, Shape
from pynference.distributions import Dirichlet
from pynference.utils import check_random_state


def generate(
    random_state: np.random.RandomState,
    dim: int,
    shape: Shape,
    positive: Optional[Union[str, List[str]]] = None,
    real_vector: Optional[Union[str, List[str]]] = None,
    positive_vector: Optional[Union[str, List[str]]] = None,
    positive_definite_matrix: Optional[Union[str, List[str]]] = None,
    lower_triangular_matrix: Optional[Union[str, List[str]]] = None,
    **limits: float,
) -> Dict[str, Parameter]:
    def min_max_transform(x, low, high):
        x_std = (x - np.min(x, axis=(-2, -1), keepdims=True)) / (
            np.max(x, axis=(-2, -1), keepdims=True)
            - np.min(x, axis=(-2, -1), keepdims=True)
        )
        return x_std * (high - low) + low

    parameters = {}

    if limits is None:
        limits = {}

    limits_default = {
        "positive_low": 0.001,
        "positive_high": 10.0,
        "real_vector_low": -10.0,
        "real_vector_high": 10.0,
        "positive_vector_low": 0.001,
        "positive_vector_high": 10.0,
        "positive_definite_matrix_low": 0.001,
        "positive_definite_matrix_high": 10.0,
        "lower_triangular_matrix_low": 1.0,
        "lower_triangular_matrix_high": 3.0,
    }

    limits = {**limits_default, **limits}

    if positive is not None:
        if isinstance(positive, str):
            positive = [positive]

        for p in positive:
            parameters[p] = random_state.uniform(
                low=limits["positive_low"], high=limits["positive_high"], size=shape
            )

    if real_vector is not None:
        if isinstance(real_vector, str):
            real_vector = [real_vector]

        for p in real_vector:
            parameters[p] = random_state.uniform(
                low=limits["real_vector_low"],
                high=limits["real_vector_high"],
                size=shape + (dim,),
            )

    if positive_vector is not None:
        if isinstance(positive_vector, str):
            positive_vector = [positive_vector]

        for p in positive_vector:
            parameters[p] = random_state.uniform(
                low=limits["positive_vector_low"],
                high=limits["positive_vector_high"],
                size=shape + (dim,),
            )

    if positive_definite_matrix is not None:
        if isinstance(positive_definite_matrix, str):
            positive_definite_matrix = [positive_definite_matrix]

        for p in positive_definite_matrix:
            w = stats.wishart.rvs(
                df=dim + 1, scale=np.eye(dim), size=shape, random_state=random_state
            )
            parameters[p] = min_max_transform(
                w,
                low=limits["positive_definite_matrix_low"],
                high=limits["positive_definite_matrix_high"],
            )

    if lower_triangular_matrix is not None:
        if isinstance(lower_triangular_matrix, str):
            lower_triangular_matrix = [lower_triangular_matrix]

        for p in lower_triangular_matrix:
            w = stats.wishart.rvs(
                df=dim + 1, scale=np.eye(dim), size=shape, random_state=random_state
            )
            w_scaled = min_max_transform(
                w,
                low=limits["lower_triangular_matrix_low"],
                high=limits["lower_triangular_matrix_high"],
            )
            parameters[p] = np.linalg.cholesky(w_scaled)

    return parameters


class TestBroadcasting:
    random_state = check_random_state(123)

    distributions = {
        Dirichlet: ["concentration"],
    }

    n_samples = 100
    atol = 1e-6
    rtol = 1e-6

    def test_broadcasting(self):
        for distribution_cls, params in self.distributions.items():
            # Try setting each parameter to X and the others to aranges.
            # X == one distribution gets [[1.0]], the other [[1.0, 1.0], [1.0, 1.0]].
            fst_params = {}
            snd_params = {}

            for p1 in params:
                fst_params[p1] = np.full(shape=(2, 2), fill_value=1.0)
                snd_params[p1] = np.array(1.0).reshape(1, 1)

                for p2 in params:
                    if p1 == p2:
                        continue

                    fst_params[p2] = np.arange(2, 6, dtype=float).reshape(2, 2)
                    snd_params[p2] = np.arange(2, 6, dtype=float).reshape(2, 2)

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

                        fst_params["upper"] += 1.0

                    if np.all(snd_params["lower"] >= snd_params["upper"]):
                        snd_params["lower"], snd_params["upper"] = (
                            snd_params["upper"],
                            snd_params["lower"],
                        )

                        snd_params["upper"] += 1.0

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
        Dirichlet: generate(
            random_state, dim=5, shape=(), positive_vector="concentration"
        ),
    }

    n_samples = 20000
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

            # TODO: Write like this (using matmul instead of dot and reversing
            # TODO: arguments) in other tests as well.
            dot_product = sum(np.matmul(t, e) for e, t in zip(eta, t_x))
            expected_log_prob = np.log(h_x) + dot_product - a_eta

            assert distribution.log_prob(samples) == approx(
                expected_log_prob, rel=self.rtol, abs=self.atol
            ), f"log_prob of {distribution}"


class TestFirstTwoMoments:
    random_state = check_random_state(123)

    distributions = {
        Dirichlet: (
            generate(random_state, dim=5, shape=(), positive_vector="concentration"),
            generate(random_state, dim=5, shape=(2,), positive_vector="concentration"),
            generate(
                random_state, dim=5, shape=(2, 3), positive_vector="concentration"
            ),
            generate(random_state, dim=10, shape=(), positive_vector="concentration"),
            generate(random_state, dim=10, shape=(2,), positive_vector="concentration"),
            generate(
                random_state, dim=10, shape=(2, 3), positive_vector="concentration"
            ),
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

    distributions = {
        Dirichlet: generate(
            random_state, dim=5, shape=(), positive_vector="concentration"
        ),
    }

    dist2scipy = {
        Dirichlet: lambda dist: stats.dirichlet(alpha=dist.concentration),
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

            if distribution_cls is Dirichlet:
                # For some reason, the SciPy Dirichlet distribution expects
                # the variates in a reversed shape.
                scipy_result = scipy_distribution.logpdf(samples.T)
            else:
                scipy_result = scipy_distribution.logpdf(samples)

            assert distribution.log_prob(samples) == approx(
                scipy_result, rel=self.rtol, abs=self.atol
            ), f"log_prob of {distribution}"


class TestParameterConstraints:
    def test_dirichlet(self):
        with raises(ValueError, match=r".*positive_vector.*"):
            Dirichlet(concentration=np.array([1.0, 2.0, 0.0]))
            Dirichlet(concentration=np.array([1.0, -2.0, 3.0]))
            Dirichlet(concentration=np.array([0.00001, 0.1, 10.0, 100.0, 5.0]))
            Dirichlet(concentration=np.array([1.0, 2.0, np.nan, 2.1]))


class TestSamplingShapes:
    random_state = check_random_state(123)

    distributions = {
        Dirichlet: (
            (
                generate(
                    random_state, dim=5, shape=(), positive_vector="concentration"
                ),
                5,
            ),
            (
                generate(
                    random_state, dim=5, shape=(2,), positive_vector="concentration"
                ),
                5,
            ),
            (
                generate(
                    random_state, dim=5, shape=(2, 3), positive_vector="concentration"
                ),
                5,
            ),
            (
                generate(
                    random_state, dim=10, shape=(), positive_vector="concentration"
                ),
                10,
            ),
            (
                generate(
                    random_state, dim=10, shape=(2,), positive_vector="concentration"
                ),
                10,
            ),
            (
                generate(
                    random_state, dim=10, shape=(2, 3), positive_vector="concentration"
                ),
                10,
            ),
        ),
    }

    def test_sampling_shapes_0d(self):
        batch_shapes = [(), (2,), (2, 3), (), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, (parameters, dim) in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(), random_state=self.random_state
                )
                assert samples.shape == batch_shapes[i] + (
                    dim,
                ), f"sampling shape of {distribution}"

    def test_sampling_shapes_1d(self):
        batch_shapes = [(), (2,), (2, 3), (), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, (parameters, dim) in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(100,), random_state=self.random_state
                )
                assert samples.shape == (100,) + batch_shapes[i] + (
                    dim,
                ), f"sampling shape of {distribution}"

    def test_sampling_shapes_2d(self):
        batch_shapes = [(), (2,), (2, 3), (), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, (parameters, dim) in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(10, 10), random_state=self.random_state
                )
                assert samples.shape == (10, 10) + batch_shapes[i] + (
                    dim,
                ), f"sampling shape of {distribution}"

    def test_sampling_shapes_3d(self):
        batch_shapes = [(), (2,), (2, 3), (), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, (parameters, dim) in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(10, 10, 2), random_state=self.random_state
                )
                assert samples.shape == (10, 10, 2) + batch_shapes[i] + (
                    dim,
                ), f"sampling shape of {distribution}"


class TestTransformedDistributions:
    random_state = check_random_state(123)

    distributions = {}

    supports = {}

    def test_supports(self):
        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                assert distribution.support == self.supports[type(distribution)](
                    distribution
                ), f"support of {distribution}"

    def test_shapes(self):
        batch_shapes = [(), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                assert (
                    distribution.batch_shape == batch_shapes[i]
                ), f"batch_shape of {distribution}"
                assert distribution.rv_shape == (), f"rv_shape of {distribution}"
