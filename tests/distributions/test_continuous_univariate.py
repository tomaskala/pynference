from typing import Dict, List, Optional, Union

import jax.numpy as np
import jax.random as random
import pytest
import scipy.stats as stats
from numpy.testing import assert_allclose
from pytest import raises

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


def generate(
    key: random.PRNGKey,
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
            parameters[p] = random.uniform(
                key,
                minval=limits["positive_low"],
                maxval=limits["positive_high"],
                shape=shape,
            )

    if real is not None:
        if isinstance(real, str):
            real = [real]

        for p in real:
            parameters[p] = random.uniform(
                key, minval=limits["real_low"], maxval=limits["real_high"], shape=shape
            )

    if lower is not None:
        if isinstance(lower, str):
            lower = [lower]

        for p in lower:
            parameters[p] = random.uniform(
                key,
                minval=limits["lower_low"],
                maxval=limits["lower_high"],
                shape=shape,
            )

    if upper is not None:
        if isinstance(upper, str):
            upper = [upper]

        for p in upper:
            parameters[p] = random.uniform(
                key,
                minval=limits["upper_low"],
                maxval=limits["upper_high"],
                shape=shape,
            )

    return parameters


key = random.PRNGKey(123)

DISTRIBUTIONS_WITH_MOMENTS = [
    Beta(**generate(key, shape=(), positive=["shape1", "shape2"])),
    Beta(**generate(key, shape=(2,), positive=["shape1", "shape2"])),
    Beta(**generate(key, shape=(2, 3), positive=["shape1", "shape2"])),
    Exponential(**generate(key, shape=(), positive="rate")),
    Exponential(**generate(key, shape=(2,), positive="rate")),
    Exponential(**generate(key, shape=(2, 3), positive="rate")),
    Gamma(**generate(key, shape=(), positive=["shape", "rate"])),
    Gamma(**generate(key, shape=(2,), positive=["shape", "rate"])),
    Gamma(**generate(key, shape=(2, 3), positive=["shape", "rate"])),
    InverseGamma(
        **generate(key, shape=(), positive=["shape", "scale"], positive_low=2.0)
    ),
    InverseGamma(
        **generate(key, shape=(2,), positive=["shape", "scale"], positive_low=2.0)
    ),
    InverseGamma(
        **generate(key, shape=(2, 3), positive=["shape", "scale"], positive_low=2.0)
    ),
    Laplace(**generate(key, shape=(), positive="scale", real="loc")),
    Laplace(**generate(key, shape=(2,), positive="scale", real="loc")),
    Laplace(**generate(key, shape=(2, 3), positive="scale", real="loc")),
    Logistic(**generate(key, shape=(), positive="scale", real="loc")),
    Logistic(**generate(key, shape=(2,), positive="scale", real="loc")),
    Logistic(**generate(key, shape=(2, 3), positive="scale", real="loc")),
    LogNormal(
        **generate(key, shape=(), positive="scale", real="loc", positive_high=2.0)
    ),
    LogNormal(
        **generate(key, shape=(2,), positive="scale", real="loc", positive_high=2.0,)
    ),
    LogNormal(
        **generate(key, shape=(2, 2), positive="scale", real="loc", positive_high=2.0,)
    ),
    Normal(**generate(key, shape=(), positive="std", real="mean")),
    Normal(**generate(key, shape=(2,), positive="std", real="mean")),
    Normal(**generate(key, shape=(2, 3), positive="std", real="mean")),
    Pareto(**generate(key, shape=(), positive=["scale", "shape"], positive_low=2.1)),
    Pareto(**generate(key, shape=(2,), positive=["scale", "shape"], positive_low=2.1)),
    Pareto(
        **generate(key, shape=(2, 3), positive=["scale", "shape"], positive_low=2.1,)
    ),
    TruncatedNormal(
        **generate(
            key, shape=(), positive="scale", real="loc", lower="lower", upper="upper",
        )
    ),
    TruncatedNormal(
        **generate(
            key, shape=(2,), positive="scale", real="loc", lower="lower", upper="upper",
        )
    ),
    TruncatedNormal(
        **generate(
            key,
            shape=(2, 3),
            positive="scale",
            real="loc",
            lower="lower",
            upper="upper",
        )
    ),
    Uniform(**generate(key, shape=(), lower="lower", upper="upper")),
    Uniform(**generate(key, shape=(2,), lower="lower", upper="upper")),
    Uniform(**generate(key, shape=(2, 3), lower="lower", upper="upper")),
]

DISTRIBUTIONS = DISTRIBUTIONS_WITH_MOMENTS + [
    Cauchy(**generate(key, shape=(), positive="scale", real="loc")),
    Cauchy(**generate(key, shape=(2,), positive="scale", real="loc")),
    Cauchy(**generate(key, shape=(2, 3), positive="scale", real="loc")),
    T(**generate(key, shape=(), positive=["df", "scale"], real="loc")),
    T(**generate(key, shape=(2,), positive=["df", "scale"], real="loc")),
    T(**generate(key, shape=(2, 3), positive=["df", "scale"], real="loc")),
]


class TestBroadcasting:
    key = random.PRNGKey(123)

    distributions = {
        Beta: ["shape1", "shape2"],
        Cauchy: ["loc", "scale"],
        Exponential: ["rate"],
        Gamma: ["shape", "rate"],
        InverseGamma: ["shape", "scale"],
        Laplace: ["loc", "scale"],
        Logistic: ["loc", "scale"],
        LogNormal: ["loc", "scale"],
        Normal: ["mean", "std"],
        Pareto: ["scale", "shape"],
        T: ["df", "loc", "scale"],
        TruncatedNormal: ["loc", "scale", "lower", "upper"],
        Uniform: ["lower", "upper"],
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

                samples = fst.sample(sample_shape=(self.n_samples,), key=self.key)

                assert_allclose(
                    fst.log_prob(samples),
                    snd.log_prob(samples),
                    atol=self.atol,
                    rtol=self.rtol,
                    err_msg=f"log_prob of {fst}",
                )


class TestExponentialFamilies:
    key = random.PRNGKey(123)

    distributions = {Beta, Exponential, Gamma, InverseGamma, LogNormal, Normal}

    n_samples = 100
    atol = 1e-6
    rtol = 1e-6

    def test_base_measure_positive_within_support(self):
        for distribution in filter(
            lambda dist: type(dist) in self.distributions and dist.batch_shape == (),
            DISTRIBUTIONS,
        ):
            samples = distribution.sample(sample_shape=(self.n_samples,), key=self.key)

            assert np.all(
                distribution.base_measure(samples) > 0
            ), f"base measure of {distribution}"

    def test_log_probs_equal(self):
        for distribution in filter(
            lambda dist: type(dist) in self.distributions and dist.batch_shape == (),
            DISTRIBUTIONS,
        ):
            samples = distribution.sample(sample_shape=(self.n_samples,), key=self.key)

            h_x = distribution.base_measure(samples)
            eta = distribution.natural_parameter
            t_x = distribution.sufficient_statistic(samples)
            a_eta = distribution.log_normalizer

            dot_product = sum(e * t for e, t in zip(eta, t_x))
            expected_log_prob = np.log(h_x) + dot_product - a_eta

            assert_allclose(
                distribution.log_prob(samples),
                expected_log_prob,
                atol=self.atol,
                rtol=self.rtol,
                err_msg=f"log_prob of {distribution}",
            )


class TestFirstTwoMoments:
    key = random.PRNGKey(123)
    n_samples = 20000
    atol = 1e-4
    rtol = 0.75

    def test_mean_and_variance(self):
        for distribution in DISTRIBUTIONS_WITH_MOMENTS:
            samples = distribution.sample(sample_shape=(self.n_samples,), key=self.key)

            assert_allclose(
                np.mean(samples, axis=0),
                distribution.mean,
                atol=self.atol,
                rtol=self.rtol,
                err_msg=f"mean of {distribution}",
            )
            assert_allclose(
                np.std(samples, axis=0),
                np.sqrt(distribution.variance),
                atol=self.atol,
                rtol=self.rtol,
                err_msg=f"variance of {distribution}",
            )


class TestLogProb:
    key = random.PRNGKey(123)

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
        for distribution in filter(lambda dist: dist.batch_shape == (), DISTRIBUTIONS):
            if type(distribution) not in self.dist2scipy:
                continue
            scipy_distribution = self.dist2scipy[type(distribution)](distribution)

            samples = distribution.sample(sample_shape=(self.n_samples,), key=self.key)
            assert_allclose(
                distribution.log_prob(samples),
                scipy_distribution.logpdf(samples),
                atol=self.atol,
                rtol=self.rtol,
                err_msg=f"log_prob of {distribution}",
            )


class TestParameterConstraints:
    def test_beta(self):
        with raises(ValueError, match=r".*positive.*"):
            Beta(shape1=-1.0, shape2=1.0)
        with raises(ValueError, match=r".*positive.*"):
            Beta(shape1=1.0, shape2=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            Beta(shape1=0.0, shape2=1.0)
        with raises(ValueError, match=r".*positive.*"):
            Beta(shape1=1.0, shape2=0.0)

    def test_cauchy(self):
        with raises(ValueError, match=r".*real.*"):
            Cauchy(loc=np.nan, scale=1.0)
        with raises(ValueError, match=r".*real.*"):
            Cauchy(loc=-np.inf, scale=1.0)
        with raises(ValueError, match=r".*real.*"):
            Cauchy(loc=np.inf, scale=1.0)

        with raises(ValueError, match=r".*positive.*"):
            Cauchy(loc=1.0, scale=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            Cauchy(loc=1.0, scale=0.0)

    def test_exponential(self):
        with raises(ValueError, match=r".*positive.*"):
            Exponential(rate=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            Exponential(rate=0.0)

    def test_gamma(self):
        with raises(ValueError, match=r".*positive.*"):
            Gamma(shape=-1.0, rate=1.0)
        with raises(ValueError, match=r".*positive.*"):
            Gamma(shape=1.0, rate=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            Gamma(shape=0.0, rate=1.0)
        with raises(ValueError, match=r".*positive.*"):
            Gamma(shape=1.0, rate=0.0)

    def test_inverse_gamma(self):
        with raises(ValueError, match=r".*positive.*"):
            InverseGamma(shape=-1.0, scale=1.0)
        with raises(ValueError, match=r".*positive.*"):
            InverseGamma(shape=1.0, scale=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            InverseGamma(shape=0.0, scale=1.0)
        with raises(ValueError, match=r".*positive.*"):
            InverseGamma(shape=1.0, scale=0.0)

    def test_laplace(self):
        with raises(ValueError, match=r".*real.*"):
            Laplace(loc=np.nan, scale=1.0)
        with raises(ValueError, match=r".*real.*"):
            Laplace(loc=-np.inf, scale=1.0)
        with raises(ValueError, match=r".*real.*"):
            Laplace(loc=np.inf, scale=1.0)

        with raises(ValueError, match=r".*positive.*"):
            Laplace(loc=1.0, scale=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            Laplace(loc=1.0, scale=0.0)

    def test_logistic(self):
        with raises(ValueError, match=r".*real.*"):
            Logistic(loc=np.nan, scale=1.0)
        with raises(ValueError, match=r".*real.*"):
            Logistic(loc=-np.inf, scale=1.0)
        with raises(ValueError, match=r".*real.*"):
            Logistic(loc=np.inf, scale=1.0)

        with raises(ValueError, match=r".*positive.*"):
            Logistic(loc=1.0, scale=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            Logistic(loc=1.0, scale=0.0)

    def test_log_normal(self):
        with raises(ValueError, match=r".*real.*"):
            LogNormal(loc=np.nan, scale=1.0)
        with raises(ValueError, match=r".*real.*"):
            LogNormal(loc=-np.inf, scale=1.0)
        with raises(ValueError, match=r".*real.*"):
            LogNormal(loc=np.inf, scale=1.0)

        with raises(ValueError, match=r".*positive.*"):
            LogNormal(loc=1.0, scale=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            LogNormal(loc=1.0, scale=0.0)

    def test_normal(self):
        with raises(ValueError, match=r".*real.*"):
            Normal(mean=np.nan, std=1.0)
        with raises(ValueError, match=r".*real.*"):
            Normal(mean=-np.inf, std=1.0)
        with raises(ValueError, match=r".*real.*"):
            Normal(mean=np.inf, std=1.0)

        with raises(ValueError, match=r".*positive.*"):
            Normal(mean=1.0, std=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            Normal(mean=1.0, std=0.0)

    def test_pareto(self):
        with raises(ValueError, match=r".*positive.*"):
            Pareto(scale=-1.0, shape=1.0)
        with raises(ValueError, match=r".*positive.*"):
            Pareto(scale=1.0, shape=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            Pareto(scale=0.0, shape=1.0)
        with raises(ValueError, match=r".*positive.*"):
            Pareto(scale=1.0, shape=0.0)

    def test_t(self):
        with raises(ValueError, match=r".*positive.*"):
            T(df=-1.0, loc=1.0, scale=1.0)
        with raises(ValueError, match=r".*positive.*"):
            T(df=0.0, loc=1.0, scale=1.0)

        with raises(ValueError, match=r".*real.*"):
            T(df=1.0, loc=np.nan, scale=1.0)
        with raises(ValueError, match=r".*real.*"):
            T(df=1.0, loc=np.inf, scale=1.0)
        with raises(ValueError, match=r".*real.*"):
            T(df=1.0, loc=-np.inf, scale=1.0)

        with raises(ValueError, match=r".*positive.*"):
            T(df=1.0, loc=1.0, scale=-1.0)
        with raises(ValueError, match=r".*positive.*"):
            T(df=1.0, loc=1.0, scale=0.0)

    def test_truncated_normal(self):
        with raises(ValueError, match=r".*real.*"):
            TruncatedNormal(loc=np.nan, scale=1.0, lower=1.0, upper=2.0)
        with raises(ValueError, match=r".*real.*"):
            TruncatedNormal(loc=np.inf, scale=1.0, lower=1.0, upper=2.0)
        with raises(ValueError, match=r".*real.*"):
            TruncatedNormal(loc=-np.inf, scale=1.0, lower=1.0, upper=2.0)

        with raises(ValueError, match=r".*positive.*"):
            TruncatedNormal(loc=1.0, scale=-1.0, lower=1.0, upper=2.0)
        with raises(ValueError, match=r".*positive.*"):
            TruncatedNormal(loc=1.0, scale=0.0, lower=1.0, upper=2.0)

        with raises(ValueError, match=r".*real.*"):
            TruncatedNormal(loc=1.0, scale=1.0, lower=-np.inf, upper=1.0)
        with raises(ValueError, match=r".*real.*"):
            TruncatedNormal(loc=1.0, scale=1.0, lower=1.0, upper=np.inf)

        with raises(ValueError, match=r".*strictly lower.*"):
            TruncatedNormal(loc=1.0, scale=1.0, lower=np.nan, upper=1.0)
        with raises(ValueError, match=r".*strictly lower.*"):
            TruncatedNormal(loc=1.0, scale=1.0, lower=1.0, upper=np.nan)
        with raises(ValueError, match=r".*strictly lower.*"):
            TruncatedNormal(loc=1.0, scale=1.0, lower=1.0, upper=1.0)
        with raises(ValueError, match=r".*strictly lower.*"):
            TruncatedNormal(loc=1.0, scale=1.0, lower=2.0, upper=1.0)
        with raises(ValueError, match=r".*strictly lower.*"):
            TruncatedNormal(loc=1.0, scale=1.0, lower=np.inf, upper=np.inf)
        with raises(ValueError, match=r".*strictly lower.*"):
            TruncatedNormal(loc=1.0, scale=1.0, lower=-np.inf, upper=-np.inf)

    @pytest.mark.filterwarnings("ignore", category=RuntimeWarning)
    def test_uniform(self):
        with raises(ValueError, match=r".*real.*"):
            Uniform(lower=-np.inf, upper=1.0)
        with raises(ValueError, match=r".*real.*"):
            Uniform(lower=1.0, upper=np.inf)

        with raises(ValueError, match=r".*strictly lower.*"):
            Uniform(lower=np.nan, upper=1.0)
        with raises(ValueError, match=r".*strictly lower.*"):
            Uniform(lower=1.0, upper=np.nan)
        with raises(ValueError, match=r".*strictly lower.*"):
            Uniform(lower=1.0, upper=1.0)
        with raises(ValueError, match=r".*strictly lower.*"):
            Uniform(lower=2.0, upper=1.0)
        with raises(ValueError, match=r".*strictly lower.*"):
            Uniform(lower=np.inf, upper=np.inf)
        with raises(ValueError, match=r".*strictly lower.*"):
            Uniform(lower=-np.inf, upper=-np.inf)


class TestTransformedDistributions:
    key = random.PRNGKey(123)

    distributions = {
        InverseGamma: (
            generate(key, shape=(), positive=["shape", "scale"]),
            generate(key, shape=(2,), positive=["shape", "scale"]),
            generate(key, shape=(2, 3), positive=["shape", "scale"]),
        ),
        LogNormal: (
            generate(key, shape=(), positive="scale", real="loc"),
            generate(key, shape=(2,), positive="scale", real="loc"),
            generate(key, shape=(2, 3), positive="scale", real="loc"),
        ),
        Pareto: (
            generate(key, shape=(), positive=["scale", "shape"]),
            generate(key, shape=(2,), positive=["scale", "shape"]),
            generate(key, shape=(2, 3), positive=["scale", "shape"]),
        ),
        Uniform: (
            generate(key, shape=(), lower="lower", upper="upper"),
            generate(key, shape=(2,), lower="lower", upper="upper"),
            generate(key, shape=(2, 3), lower="lower", upper="upper"),
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
