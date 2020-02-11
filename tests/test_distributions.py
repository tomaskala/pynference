from collections import namedtuple
from numbers import Number
import inspect

import pytest
import torch
from torch.distributions import constraints
from scipy import stats

from pynference.distributions import TruncatedNormal

Dist = namedtuple("Dist", ["dist", "scipy_dist", "args"])


SCIPY_DISTS = {
    TruncatedNormal: lambda loc, scale, low, high: stats.truncnorm(
        a=(low - loc) / scale, b=(high - loc) / scale, loc=loc, scale=scale
    )
}


def create_dist(dist_cls, *args):
    scipy_dist = SCIPY_DISTS.get(dist_cls, None)
    return Dist(dist_cls, scipy_dist, args)


# TODO: Handle the case when either bound is infinite.
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
    elif isinstance(constraint, constraints._GreaterThan) and constraint.lower_bound == 0.0:
        return torch.rand(size) + eps
    elif isinstance(constraint, constraints._Dependent):
        return torch.normal(mean=torch.zeros(size), std=1.0)
    else:
        raise NotImplementedError("Generating values within {} is not implemented.".format(constraint))


def generate_outside(constraint, size):
    if isinstance(constraint, constraints._Real):
        return torch.full(size, float("NaN"))
    elif isinstance(constraint, constraints._GreaterThan) and constraint.lower_bound == 0.0:
        return -torch.rand(size)
    elif isinstance(constraint, constraints._Dependent):
        return torch.normal(mean=torch.zeros(size), std=1.0)
    else:
        raise NotImplementedError("Generating values outside {} is not implemented.".format(constraint))


@pytest.fixture(autouse=True)
def reset_rng_state():
    torch.manual_seed(123)


@pytest.mark.parametrize("dist, scipy_dist, params", DISTRIBUTIONS)
def test_constraints(dist, scipy_dist, params):
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

    print(outside)

    with pytest.raises(ValueError):
        dist(*outside, validate_args=True)

    try:
        dist(*within)
    except ValueError:
        pytest.fail("{} failed on parameters within the boundaries.".format(dist))


@pytest.mark.parametrize("dist, scipy_dist, params", DISTRIBUTIONS)
def test_moments(dist, scipy_dist, params):
    pass


@pytest.mark.parametrize("dist, scipy_dist, params", DISTRIBUTIONS)
@pytest.mark.parametrize("sample_shape", [(), (2,), (2, 3)])
def test_log_prob(dist, scipy_dist, params, sample_shape):
    pass


@pytest.mark.parametrize("dist, scipy_dist, params", DISTRIBUTIONS)
@pytest.mark.parametrize("sample_shape", [(), (2,), (2, 3)])
def test_sampling_shapes(dist, scipy_dist, params, sample_shape):
    pass
