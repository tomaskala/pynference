import inspect
import math
from collections import namedtuple
from numbers import Number

import pytest
import torch
from scipy import stats
from torch.distributions import constraints
from torch.testing import assert_allclose

from pynference.distributions import TruncatedNormal

Dist = namedtuple("Dist", ["dist", "scipy_instance", "args"])


SCIPY_DISTS = {
    TruncatedNormal: lambda loc, scale, low, high: stats.truncnorm(
        a=(low - loc) / scale, b=(high - loc) / scale, loc=loc, scale=scale
    )
}


def create_dist(dist_cls, *args):
    scipy_dist = SCIPY_DISTS.get(dist_cls, None)
    scipy_instance = scipy_dist(
        *[arg if isinstance(arg, Number) else arg.numpy() for arg in args]
    )
    return Dist(dist_cls, scipy_instance, args)


DISTRIBUTIONS = [
    create_dist(
        TruncatedNormal,
        torch.Tensor([0.0]),
        torch.Tensor([1.0]),
        torch.Tensor([-1.0]),
        torch.Tensor([1.0]),
    ),
    create_dist(TruncatedNormal, 0.0, 1.0, -1.0, 1.0),
    create_dist(TruncatedNormal, torch.zeros((3,)), 1.0, -1.0, 1.0),
    create_dist(
        TruncatedNormal,
        torch.zeros((3,)),
        torch.ones((1,)),
        -torch.ones((3,)),
        torch.ones((3,)),
    ),
    create_dist(
        TruncatedNormal,
        torch.Tensor([3.0, -2.0]),
        torch.Tensor([0.5, 1.5]),
        torch.Tensor([1.0, -3.0]),
        torch.Tensor([5.0, 1.0]),
    ),
    create_dist(
        TruncatedNormal,
        torch.Tensor([0.0]),
        torch.Tensor([1.0]),
        torch.Tensor([float("-inf")]),
        torch.Tensor([1.0]),
    ),
    create_dist(
        TruncatedNormal,
        torch.Tensor([0.0]),
        torch.Tensor([1.0]),
        torch.Tensor([-1.0]),
        torch.Tensor([float("inf")]),
    ),
    create_dist(
        TruncatedNormal,
        torch.Tensor([0.0]),
        torch.Tensor([1.0]),
        torch.Tensor([float("-inf")]),
        torch.Tensor([float("inf")]),
    ),
    create_dist(
        TruncatedNormal,
        torch.Tensor([0.0]),
        torch.Tensor([1.0]),
        torch.Tensor([-3.0, float("-inf")]),
        torch.Tensor([float("inf"), 1.0]),
    ),
    create_dist(
        TruncatedNormal,
        torch.Tensor([0.0]),
        torch.Tensor([1.0]),
        torch.Tensor([-1.0, 4.0]),
        torch.Tensor([2.0, float("inf")]),
    ),
    create_dist(
        TruncatedNormal,
        torch.Tensor([0.0]),
        torch.Tensor([1.0]),
        torch.Tensor([-1.0, float("-inf")]),
        torch.Tensor([1.0, float("inf")]),
    ),
]


def generate_within(constraint, size):
    eps = 1e-3

    if isinstance(constraint, constraints._Real):
        return torch.normal(mean=torch.zeros(size), std=1.0)
    elif (
        isinstance(constraint, constraints._GreaterThan)
        and constraint.lower_bound == 0.0
    ):
        return torch.rand(size) + eps
    elif isinstance(constraint, constraints._Dependent):
        return torch.normal(mean=torch.zeros(size), std=1.0)
    else:
        raise NotImplementedError(
            "Generating values within {} is not implemented.".format(constraint)
        )


def generate_outside(constraint, size):
    if isinstance(constraint, constraints._Real):
        return torch.full(size, float("NaN"))
    elif (
        isinstance(constraint, constraints._GreaterThan)
        and constraint.lower_bound == 0.0
    ):
        return -torch.rand(size)
    elif isinstance(constraint, constraints._Dependent):
        return torch.normal(mean=torch.zeros(size), std=1.0)
    else:
        raise NotImplementedError(
            "Generating values outside {} is not implemented.".format(constraint)
        )


@pytest.fixture(autouse=True)
def reset_rng_state():
    torch.manual_seed(123)


@pytest.mark.parametrize("dist, scipy_instance, params", DISTRIBUTIONS)
def test_constraints(dist, scipy_instance, params):
    dist_args = [param.name for param in inspect.signature(dist).parameters.values()]
    within = []
    outside = []

    low_index = None
    high_index = None

    for i in range(len(params)):
        arg_name = dist_args[i]
        arg_constraint = dist.arg_constraints[arg_name]
        arg_value = params[i]
        arg_shape = () if isinstance(arg_value, Number) else arg_value.size()

        parameter_within = generate_within(arg_constraint, arg_shape)
        parameter_outside = generate_outside(arg_constraint, arg_shape)

        within.append(parameter_within)
        outside.append(parameter_outside)

        if arg_name == "low":
            low_index = i
        elif arg_name == "high":
            high_index = i

    # Ensure that all "low" args are < "high" args
    # in `within` and vice versa in `outside`.
    if low_index is not None and high_index is not None:
        within[low_index] = -torch.abs(within[low_index])
        within[high_index] = torch.abs(within[low_index])

        outside[low_index] = torch.abs(outside[low_index])
        outside[high_index] = -torch.abs(outside[low_index])

    with pytest.raises(ValueError):
        dist(*outside, validate_args=True)

    try:
        dist(*within)
    except ValueError:
        pytest.fail("{} failed on parameters within the boundaries.".format(dist))


@pytest.mark.parametrize("dist, scipy_instance, params", DISTRIBUTIONS)
def test_moments(dist, scipy_instance, params):
    n_samples = 50000
    rtol = 1e-1

    dist_instance = dist(*params)
    samples = dist_instance.sample((n_samples,))

    assert_allclose(dist_instance.mean, samples.mean(0), atol=1e-2, rtol=rtol)
    assert_allclose(dist_instance.stddev, samples.std(0), atol=1e-1, rtol=rtol)

    if not scipy_instance:
        pytest.skip("No corresponding SciPy distribution.")

    if dist is TruncatedNormal and any(
        math.isinf(param) if isinstance(param, Number) else torch.isinf(param).any()
        for param in params
    ):
        return

    scipy_mean = scipy_instance.mean()
    scipy_var = scipy_instance.var()

    assert_allclose(dist_instance.mean, scipy_mean, atol=1e-2, rtol=rtol)
    assert_allclose(dist_instance.variance, scipy_var, atol=1e-2, rtol=rtol)


# TODO: Test entropy (compare to scipy).


@pytest.mark.parametrize("dist, scipy_instance, params", DISTRIBUTIONS)
@pytest.mark.parametrize("sample_shape", [(), (200,), (10, 20)])
def test_prob(dist, scipy_instance, params, sample_shape):
    atol = 1e-2
    rtol = 5e-2

    if not scipy_instance:
        pytest.skip("No corresponding SciPy distribution.")

    dist_instance = dist(*params)
    samples = dist_instance.sample(sample_shape)

    dist_log_prob = dist_instance.log_prob(samples)
    scipy_log_prob = scipy_instance.logpdf(samples.numpy())
    assert_allclose(dist_log_prob, scipy_log_prob, atol=atol, rtol=rtol)

    dist_cdf = dist_instance.cdf(samples)
    scipy_cdf = scipy_instance.cdf(samples.numpy())
    assert_allclose(dist_cdf, scipy_cdf, atol=atol, rtol=rtol)

    if sample_shape == ():
        uniform_samples = torch.rand(sample_shape)
        dist_icdf = dist_instance.icdf(uniform_samples)
        scipy_icdf = scipy_instance.ppf(uniform_samples)
        assert_allclose(dist_icdf, scipy_icdf, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dist, scipy_instance, params", DISTRIBUTIONS)
@pytest.mark.parametrize("sample_shape", [(), (200,), (10, 20)])
def test_icdf(dist, scipy_instance, params, sample_shape):
    atol = 1e-5
    rtol = 5e-2

    dist_instance = dist(*params)
    samples = dist_instance.sample(sample_shape)

    cdf = dist_instance.cdf(samples)
    icdf = dist_instance.icdf(cdf)

    assert_allclose(icdf, samples, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dist, scipy_instance, params", DISTRIBUTIONS)
@pytest.mark.parametrize("sample_shape", [(), (200,), (10, 20)])
def test_sampling_shapes(dist, scipy_instance, params, sample_shape):
    pass
