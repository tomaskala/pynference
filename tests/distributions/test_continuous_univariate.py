from typing import Dict, List, Optional, Union

import numpy as np
from pytest import approx

from pynference.constants import Parameter, Shape
from pynference.distributions import (
    Beta,
    Cauchy,
    Exponential,
    Gamma,
    InverseGamma,
    Laplace,
    Logistic,
    LogNormal,
    Normal,
    Pareto,
    T,
    TruncatedNormal,
    Uniform,
)
from pynference.utils import check_random_state

random_state = check_random_state(123)


def generate(
    shape: Union[None, Shape],
    positive: Optional[Union[str, List[str]]] = None,
    real: Optional[Union[str, List[str]]] = None,
    lower: Optional[Union[str, List[str]]] = None,
    upper: Optional[Union[str, List[str]]] = None,
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

    return parameters


class TestExponentialFamilies:
    distributions = [Beta, Exponential, Gamma, InverseGamma, LogNormal, Normal]


class TestFirstTwoMoments:
    distributions = {
        Beta: (
            generate(shape=(), positive=["shape1", "shape2"]),
            generate(shape=(2,), positive=["shape1", "shape2"]),
            generate(shape=(2, 3), positive=["shape1", "shape2"]),
        ),
        Exponential: (
            generate(shape=(), positive="rate"),
            generate(shape=(2,), positive="rate"),
            generate(shape=(2, 3), positive="rate"),
        ),
        Gamma: (
            generate(shape=(), positive=["shape", "rate"]),
            generate(shape=(2,), positive=["shape", "rate"]),
            generate(shape=(2, 3), positive=["shape", "rate"]),
        ),
        InverseGamma: (
            generate(shape=(), positive=["shape", "scale"]),
            generate(shape=(2,), positive=["shape", "scale"]),
            generate(shape=(2, 3), positive=["shape", "scale"]),
        ),
        Laplace: (
            generate(shape=(), positive="scale", real="loc"),
            generate(shape=(2,), positive="scale", real="loc"),
            generate(shape=(2, 3), positive="scale", real="loc"),
        ),
        Logistic: (
            generate(shape=(), positive="scale", real="loc"),
            generate(shape=(2,), positive="scale", real="loc"),
            generate(shape=(2, 3), positive="scale", real="loc"),
        ),
        LogNormal: (
            generate(shape=(), positive="scale", real="loc", positive_high=3.0),
            generate(shape=(2,), positive="scale", real="loc", positive_high=3.0),
            generate(shape=(2, 2), positive="scale", real="loc", positive_high=3.0),
        ),
        Normal: (
            generate(shape=(), positive="std", real="mean"),
            generate(shape=(2,), positive="std", real="mean"),
            generate(shape=(2, 3), positive="std", real="mean"),
        ),
        Pareto: (
            generate(shape=(), positive=["scale", "shape"], positive_low=2.1),
            generate(shape=(2,), positive=["scale", "shape"], positive_low=2.1),
            generate(shape=(2, 3), positive=["scale", "shape"], positive_low=2.1),
        ),
        TruncatedNormal: (
            generate(
                shape=(), positive="scale", real="loc", lower="lower", upper="upper"
            ),
            generate(
                shape=(2,), positive="scale", real="loc", lower="lower", upper="upper"
            ),
            generate(
                shape=(2, 3), positive="scale", real="loc", lower="lower", upper="upper"
            ),
        ),
        Uniform: (
            generate(shape=(), lower="lower", upper="upper"),
            generate(shape=(2,), lower="lower", upper="upper"),
            generate(shape=(2, 3), lower="lower", upper="upper"),
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
                    sample_shape=(self.n_samples,), random_state=random_state
                )

                assert np.mean(samples, axis=0) == approx(
                    distribution.mean, rel=self.rtol, abs=self.atol
                ), f"mean of {str(distribution)}"

                assert np.var(samples, axis=0) == approx(
                    distribution.variance, rel=self.rtol, abs=self.atol
                ), f"variance of {str(distribution)}"


class TestLogProb:
    pass


class TestSamplingShapes:
    distributions = {
        Beta: (
            generate(shape=(), positive=["shape1", "shape2"]),
            generate(shape=(2,), positive=["shape1", "shape2"]),
            generate(shape=(2, 3), positive=["shape1", "shape2"]),
        ),
        Cauchy: (
            generate(shape=(), positive="scale", real="loc"),
            generate(shape=(2,), positive="scale", real="loc"),
            generate(shape=(2, 3), positive="scale", real="loc"),
        ),
        Exponential: (
            generate(shape=(), positive="rate"),
            generate(shape=(2,), positive="rate"),
            generate(shape=(2, 3), positive="rate"),
        ),
        Gamma: (
            generate(shape=(), positive=["shape", "rate"]),
            generate(shape=(2,), positive=["shape", "rate"]),
            generate(shape=(2, 3), positive=["shape", "rate"]),
        ),
        InverseGamma: (
            generate(shape=(), positive=["shape", "scale"]),
            generate(shape=(2,), positive=["shape", "scale"]),
            generate(shape=(2, 3), positive=["shape", "scale"]),
        ),
        Laplace: (
            generate(shape=(), positive="scale", real="loc"),
            generate(shape=(2,), positive="scale", real="loc"),
            generate(shape=(2, 3), positive="scale", real="loc"),
        ),
        Logistic: (
            generate(shape=(), positive="scale", real="loc"),
            generate(shape=(2,), positive="scale", real="loc"),
            generate(shape=(2, 3), positive="scale", real="loc"),
        ),
        LogNormal: (
            generate(shape=(), positive="scale", real="loc"),
            generate(shape=(2,), positive="scale", real="loc"),
            generate(shape=(2, 3), positive="scale", real="loc"),
        ),
        Normal: (
            generate(shape=(), positive="std", real="mean"),
            generate(shape=(2,), positive="std", real="mean"),
            generate(shape=(2, 3), positive="std", real="mean"),
        ),
        Pareto: (
            generate(shape=(), positive=["scale", "shape"]),
            generate(shape=(2,), positive=["scale", "shape"]),
            generate(shape=(2, 3), positive=["scale", "shape"]),
        ),
        T: (
            generate(shape=(), positive=["df", "scale"], real="loc"),
            generate(shape=(2,), positive=["df", "scale"], real="loc"),
            generate(shape=(2, 3), positive=["df", "scale"], real="loc"),
        ),
        TruncatedNormal: (
            generate(
                shape=(), positive="scale", real="loc", lower="lower", upper="upper"
            ),
            generate(
                shape=(2,), positive="scale", real="loc", lower="lower", upper="upper"
            ),
            generate(
                shape=(2, 3), positive="scale", real="loc", lower="lower", upper="upper"
            ),
        ),
        Uniform: (
            generate(shape=(), lower="lower", upper="upper"),
            generate(shape=(2,), lower="lower", upper="upper"),
            generate(shape=(2, 3), lower="lower", upper="upper"),
        ),
    }

    def test_sampling_shapes_0d(self):
        batch_shapes = [(), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                print(parameters)
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(), random_state=random_state
                )
                assert samples.shape == batch_shapes[i]

    def test_sampling_shapes_1d(self):
        batch_shapes = [(), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(100,), random_state=random_state
                )
                assert samples.shape == (100,) + batch_shapes[i]

    def test_sampling_shapes_2d(self):
        batch_shapes = [(), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(10, 10), random_state=random_state
                )
                assert samples.shape == (10, 10) + batch_shapes[i]

    def test_sampling_shapes_3d(self):
        batch_shapes = [(), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, parameters in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(10, 10, 2), random_state=random_state
                )
                assert samples.shape == (10, 10, 2) + batch_shapes[i]


class TestTransformedDistributions:
    distributions = [InverseGamma, LogNormal, Pareto, Uniform]
