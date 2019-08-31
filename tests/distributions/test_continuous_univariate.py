from typing import Dict, List, Optional, Union

import numpy as np
import scipy.stats as stats
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
from pynference.distributions.constraints import interval, positive
from pynference.utils import check_random_state


def generate(
    random_state: np.random.RandomState,
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


# TODO: Pass one parameter as array of parameters, the other parameter as a scalar and check that i broadcasts. This should check whether shape promoting instead of proper broadcasting works.
class TestBroadcasting:
    random_state = check_random_state(123)


class TestExponentialFamilies:
    random_state = check_random_state(123)

    distributions = {
        Beta: generate(random_state, shape=(), positive=["shape1", "shape2"]),
        Exponential: generate(random_state, shape=(), positive="rate"),
        Gamma: generate(random_state, shape=(), positive=["shape", "rate"]),
        InverseGamma: generate(random_state, shape=(), positive=["shape", "scale"]),
        LogNormal: generate(random_state, shape=(), positive="scale", real="loc"),
        Normal: generate(random_state, shape=(), positive="std", real="mean"),
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

            dot_product = sum(np.dot(e, t) for e, t in zip(eta, t_x))
            expected_log_prob = np.log(h_x) + dot_product - a_eta

            assert distribution.log_prob(samples) == approx(
                expected_log_prob, rel=self.rtol, abs=self.atol
            ), f"log_prob of {distribution}"


class TestFirstTwoMoments:
    random_state = check_random_state(123)

    distributions = {
        Beta: (
            generate(random_state, shape=(), positive=["shape1", "shape2"]),
            generate(random_state, shape=(2,), positive=["shape1", "shape2"]),
            generate(random_state, shape=(2, 3), positive=["shape1", "shape2"]),
        ),
        Exponential: (
            generate(random_state, shape=(), positive="rate"),
            generate(random_state, shape=(2,), positive="rate"),
            generate(random_state, shape=(2, 3), positive="rate"),
        ),
        Gamma: (
            generate(random_state, shape=(), positive=["shape", "rate"]),
            generate(random_state, shape=(2,), positive=["shape", "rate"]),
            generate(random_state, shape=(2, 3), positive=["shape", "rate"]),
        ),
        InverseGamma: (
            generate(random_state, shape=(), positive=["shape", "scale"]),
            generate(random_state, shape=(2,), positive=["shape", "scale"]),
            generate(random_state, shape=(2, 3), positive=["shape", "scale"]),
        ),
        Laplace: (
            generate(random_state, shape=(), positive="scale", real="loc"),
            generate(random_state, shape=(2,), positive="scale", real="loc"),
            generate(random_state, shape=(2, 3), positive="scale", real="loc"),
        ),
        Logistic: (
            generate(random_state, shape=(), positive="scale", real="loc"),
            generate(random_state, shape=(2,), positive="scale", real="loc"),
            generate(random_state, shape=(2, 3), positive="scale", real="loc"),
        ),
        LogNormal: (
            generate(
                random_state, shape=(), positive="scale", real="loc", positive_high=3.0
            ),
            generate(
                random_state,
                shape=(2,),
                positive="scale",
                real="loc",
                positive_high=3.0,
            ),
            generate(
                random_state,
                shape=(2, 2),
                positive="scale",
                real="loc",
                positive_high=3.0,
            ),
        ),
        Normal: (
            generate(random_state, shape=(), positive="std", real="mean"),
            generate(random_state, shape=(2,), positive="std", real="mean"),
            generate(random_state, shape=(2, 3), positive="std", real="mean"),
        ),
        Pareto: (
            generate(
                random_state, shape=(), positive=["scale", "shape"], positive_low=2.1
            ),
            generate(
                random_state, shape=(2,), positive=["scale", "shape"], positive_low=2.1
            ),
            generate(
                random_state,
                shape=(2, 3),
                positive=["scale", "shape"],
                positive_low=2.1,
            ),
        ),
        TruncatedNormal: (
            generate(
                random_state,
                shape=(),
                positive="scale",
                real="loc",
                lower="lower",
                upper="upper",
            ),
            generate(
                random_state,
                shape=(2,),
                positive="scale",
                real="loc",
                lower="lower",
                upper="upper",
            ),
            generate(
                random_state,
                shape=(2, 3),
                positive="scale",
                real="loc",
                lower="lower",
                upper="upper",
            ),
        ),
        Uniform: (
            generate(random_state, shape=(), lower="lower", upper="upper"),
            generate(random_state, shape=(2,), lower="lower", upper="upper"),
            generate(random_state, shape=(2, 3), lower="lower", upper="upper"),
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
        Beta: generate(random_state, shape=(), positive=["shape1", "shape2"]),
        Cauchy: generate(random_state, shape=(), positive="scale", real="loc"),
        Exponential: generate(random_state, shape=(), positive="rate"),
        Gamma: generate(random_state, shape=(), positive=["shape", "rate"]),
        InverseGamma: generate(random_state, shape=(), positive=["shape", "scale"]),
        Laplace: generate(random_state, shape=(), positive="scale", real="loc"),
        Logistic: generate(random_state, shape=(), positive="scale", real="loc"),
        LogNormal: generate(random_state, shape=(), positive="scale", real="loc"),
        Normal: generate(random_state, shape=(), positive="std", real="mean"),
        Pareto: generate(random_state, shape=(), positive=["scale", "shape"]),
        T: generate(random_state, shape=(), positive=["df", "scale"], real="loc"),
        TruncatedNormal: generate(
            random_state,
            shape=(),
            positive="scale",
            real="loc",
            lower="lower",
            upper="upper",
        ),
        Uniform: generate(random_state, shape=(), lower="lower", upper="upper"),
    }

    dist2scipy = {
        Beta: lambda dist: stats.beta(a=dist.shape1, b=dist.shape2),
        Cauchy: lambda dist: stats.cauchy(loc=dist.loc, scale=dist.scale),
        Exponential: lambda dist: stats.expon(scale=np.reciprocal(dist.rate)),
        Gamma: lambda dist: stats.gamma(a=dist.shape, scale=np.reciprocal(dist.rate)),
        InverseGamma: lambda dist: stats.invgamma(a=dist.shape, scale=dist.scale),
        Laplace: lambda dist: stats.laplace(loc=dist.loc, scale=dist.scale),
        Logistic: lambda dist: stats.logistic(loc=dist.loc, scale=dist.scale),
        LogNormal: lambda dist: stats.lognorm(scale=np.exp(dist.loc), s=dist.scale),
        Normal: lambda dist: stats.norm(loc=dist.mean, scale=dist.std),
        Pareto: lambda dist: stats.pareto(b=dist.shape, scale=dist.scale),
        T: lambda dist: stats.t(df=dist.df, loc=dist.loc, scale=dist.scale),
        TruncatedNormal: lambda dist: stats.truncnorm(
            a=(dist.lower - dist.loc) / dist.scale,
            b=(dist.upper - dist.loc) / dist.scale,
            loc=dist.loc,
            scale=dist.scale,
        ),
        Uniform: lambda dist: stats.uniform(
            loc=dist.lower, scale=dist.upper - dist.lower
        ),
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
                scipy_distribution.logpdf(samples), rel=self.rtol, abs=self.atol
            ), f"log_prob of {distribution}"


class TestParameterConstraints:
    random_state = check_random_state(123)


class TestSamplingShapes:
    random_state = check_random_state(123)

    distributions = {
        Beta: (
            generate(random_state, shape=(), positive=["shape1", "shape2"]),
            generate(random_state, shape=(2,), positive=["shape1", "shape2"]),
            generate(random_state, shape=(2, 3), positive=["shape1", "shape2"]),
        ),
        Cauchy: (
            generate(random_state, shape=(), positive="scale", real="loc"),
            generate(random_state, shape=(2,), positive="scale", real="loc"),
            generate(random_state, shape=(2, 3), positive="scale", real="loc"),
        ),
        Exponential: (
            generate(random_state, shape=(), positive="rate"),
            generate(random_state, shape=(2,), positive="rate"),
            generate(random_state, shape=(2, 3), positive="rate"),
        ),
        Gamma: (
            generate(random_state, shape=(), positive=["shape", "rate"]),
            generate(random_state, shape=(2,), positive=["shape", "rate"]),
            generate(random_state, shape=(2, 3), positive=["shape", "rate"]),
        ),
        InverseGamma: (
            generate(random_state, shape=(), positive=["shape", "scale"]),
            generate(random_state, shape=(2,), positive=["shape", "scale"]),
            generate(random_state, shape=(2, 3), positive=["shape", "scale"]),
        ),
        Laplace: (
            generate(random_state, shape=(), positive="scale", real="loc"),
            generate(random_state, shape=(2,), positive="scale", real="loc"),
            generate(random_state, shape=(2, 3), positive="scale", real="loc"),
        ),
        Logistic: (
            generate(random_state, shape=(), positive="scale", real="loc"),
            generate(random_state, shape=(2,), positive="scale", real="loc"),
            generate(random_state, shape=(2, 3), positive="scale", real="loc"),
        ),
        LogNormal: (
            generate(random_state, shape=(), positive="scale", real="loc"),
            generate(random_state, shape=(2,), positive="scale", real="loc"),
            generate(random_state, shape=(2, 3), positive="scale", real="loc"),
        ),
        Normal: (
            generate(random_state, shape=(), positive="std", real="mean"),
            generate(random_state, shape=(2,), positive="std", real="mean"),
            generate(random_state, shape=(2, 3), positive="std", real="mean"),
        ),
        Pareto: (
            generate(random_state, shape=(), positive=["scale", "shape"]),
            generate(random_state, shape=(2,), positive=["scale", "shape"]),
            generate(random_state, shape=(2, 3), positive=["scale", "shape"]),
        ),
        T: (
            generate(random_state, shape=(), positive=["df", "scale"], real="loc"),
            generate(random_state, shape=(2,), positive=["df", "scale"], real="loc"),
            generate(random_state, shape=(2, 3), positive=["df", "scale"], real="loc"),
        ),
        TruncatedNormal: (
            generate(
                random_state,
                shape=(),
                positive="scale",
                real="loc",
                lower="lower",
                upper="upper",
            ),
            generate(
                random_state,
                shape=(2,),
                positive="scale",
                real="loc",
                lower="lower",
                upper="upper",
            ),
            generate(
                random_state,
                shape=(2, 3),
                positive="scale",
                real="loc",
                lower="lower",
                upper="upper",
            ),
        ),
        Uniform: (
            generate(random_state, shape=(), lower="lower", upper="upper"),
            generate(random_state, shape=(2,), lower="lower", upper="upper"),
            generate(random_state, shape=(2, 3), lower="lower", upper="upper"),
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


class TestTransformedDistributions:
    random_state = check_random_state(123)

    distributions = {
        InverseGamma: (
            generate(random_state, shape=(), positive=["shape", "scale"]),
            generate(random_state, shape=(2,), positive=["shape", "scale"]),
            generate(random_state, shape=(2, 3), positive=["shape", "scale"]),
        ),
        LogNormal: (
            generate(random_state, shape=(), positive="scale", real="loc"),
            generate(random_state, shape=(2,), positive="scale", real="loc"),
            generate(random_state, shape=(2, 3), positive="scale", real="loc"),
        ),
        Pareto: (
            generate(random_state, shape=(), positive=["scale", "shape"]),
            generate(random_state, shape=(2,), positive=["scale", "shape"]),
            generate(random_state, shape=(2, 3), positive=["scale", "shape"]),
        ),
        Uniform: (
            generate(random_state, shape=(), lower="lower", upper="upper"),
            generate(random_state, shape=(2,), lower="lower", upper="upper"),
            generate(random_state, shape=(2, 3), lower="lower", upper="upper"),
        ),
    }

    supports = {
        InverseGamma: lambda d: positive,
        LogNormal: lambda d: positive,
        Pareto: lambda d: interval(d.scale, np.inf),
        Uniform: lambda d: interval(d.lower, d.upper),
    }

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
