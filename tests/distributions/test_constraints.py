import numpy as np

from pynference.distributions import constraints


def test_real_within():
    real = constraints.Real()
    x = 4.0
    y = 1000000.0
    z = np.pi

    assert real(x)
    assert real(y)
    assert real(z)


def test_real_outside():
    real = constraints.Real()
    x = np.inf
    y = -np.inf
    z = np.nan

    assert not real(x)
    assert not real(y)
    assert not real(z)


def test_real_vector_within():
    real_vector = constraints.RealVector()
    x = np.array([-100.0, 0.0, 12000.0, np.sqrt(12), -np.pi])

    assert real_vector(x)


def test_real_vector_outside():
    real_vector = constraints.RealVector()
    x = np.array([-100.0, 0.0, 12000.0, np.sqrt(12), np.nan, -np.pi])
    y = np.array([-100.0, 0.0, -np.inf, 12000.0, np.sqrt(12), -np.pi])

    assert not real_vector(x)
    assert not real_vector(y)


def test_open_interval():
    lower = 4.0
    upper = 21.0
    interval = constraints.Interval(
        lower, upper, include_lower=False, include_upper=False
    )
    x = 10.0
    y = 10000.0

    assert interval(x)
    assert not interval(y)
    assert not interval(lower)
    assert not interval(upper)


def test_lower_open_interval():
    lower = 4.0
    upper = 21.0
    interval = constraints.Interval(
        lower, upper, include_lower=False, include_upper=True
    )
    x = 10.0
    y = 10000.0

    assert interval(x)
    assert not interval(y)
    assert not interval(lower)
    assert interval(upper)


def test_upper_open_interval():
    lower = 4.0
    upper = 21.0
    interval = constraints.Interval(
        lower, upper, include_lower=True, include_upper=False
    )
    x = 10.0
    y = 10000.0

    assert interval(x)
    assert not interval(y)
    assert interval(lower)
    assert not interval(upper)


def test_closed_interval():
    lower = 4.0
    upper = 21.0
    interval = constraints.Interval(
        lower, upper, include_lower=True, include_upper=True
    )
    x = 10.0
    y = 10000.0

    assert interval(x)
    assert not interval(y)
    assert interval(lower)
    assert interval(upper)


def test_positive_within():
    positive = constraints.Positive()
    x = 9.0
    y = 128.0

    assert positive(x)
    assert positive(y)


def test_positive_outside():
    positive = constraints.Positive()
    x = -9.0
    y = -128.0
    z = 0.0

    assert not positive(x)
    assert not positive(y)
    assert not positive(z)


def test_nonnegative_within():
    nonnegative = constraints.NonNegative()
    x = 9.0
    y = 128.0
    z = 0.0

    assert nonnegative(x)
    assert nonnegative(y)
    assert nonnegative(z)


def test_nonnegative_outside():
    nonnegative = constraints.NonNegative()
    x = -1e-6
    y = -1000.0

    assert not nonnegative(x)
    assert not nonnegative(y)


def test_negative_within():
    negative = constraints.Negative()
    x = -9.0
    y = -128.0

    assert negative(x)
    assert negative(y)


def test_negative_outside():
    negative = constraints.Negative()
    x = 9.0
    y = 128.0
    z = 0.0

    assert not negative(x)
    assert not negative(y)
    assert not negative(z)


def test_nonpositive_within():
    nonpositive = constraints.NonPositive()
    x = -9.0
    y = -128.0
    z = 0.0

    assert nonpositive(x)
    assert nonpositive(y)
    assert nonpositive(z)


def test_nonpositive_outside():
    nonpositive = constraints.NonPositive()
    x = 1e-6
    y = 1000.0

    assert not nonpositive(x)
    assert not nonpositive(y)


def test_integer_true():
    integer = constraints.Integer()
    x = 10
    y = -2
    z = 0

    assert integer(x)
    assert integer(y)
    assert integer(z)


def test_integer_like():
    integer = constraints.Integer()
    x = 10.0
    y = -2.0
    z = 0.0

    assert integer(x)
    assert integer(y)
    assert integer(z)


def test_integer_not():
    integer = constraints.Integer()
    x = 10.01
    y = -2.00000000001
    z = 0.15

    assert not integer(x)
    assert not integer(y)
    assert not integer(z)


def test_open_interval_integer():
    lower = 4
    upper = 21
    interval = constraints.IntegerInterval(
        lower, upper, include_lower=False, include_upper=False
    )
    x = 10.0
    y = 10000.0

    assert interval(x)
    assert not interval(y)
    assert not interval(lower)
    assert not interval(upper)


def test_lower_open_interval_integer():
    lower = 4
    upper = 21
    interval = constraints.IntegerInterval(
        lower, upper, include_lower=False, include_upper=True
    )
    x = 10.0
    y = 10000.0

    assert interval(x)
    assert not interval(y)
    assert not interval(lower)
    assert interval(upper)


def test_upper_open_interval_integer():
    lower = 4
    upper = 21
    interval = constraints.IntegerInterval(
        lower, upper, include_lower=True, include_upper=False
    )
    x = 10.0
    y = 10000.0
    z = 10.5

    assert interval(x)
    assert not interval(y)
    assert not interval(z)
    assert interval(lower)
    assert not interval(upper)


def test_closed_interval_integer():
    lower = 4
    upper = 21
    interval = constraints.IntegerInterval(
        lower, upper, include_lower=True, include_upper=True
    )
    x = 10.0
    y = 10000.0
    z = 10.05

    assert interval(x)
    assert not interval(y)
    assert not interval(z)
    assert interval(lower)
    assert interval(upper)


def test_positive_within_integer():
    positive = constraints.PositiveInteger()
    x = 9.0
    y = 128.0
    z = 1.4

    assert positive(x)
    assert positive(y)
    assert not positive(z)


def test_positive_outside_integer():
    positive = constraints.PositiveInteger()
    x = -9.0
    y = -128.0
    z = 0.2

    assert not positive(x)
    assert not positive(y)
    assert not positive(z)


def test_nonnegative_within_integer():
    nonnegative = constraints.NonNegativeInteger()
    x = 9.0
    y = 128.0
    z = 0.0
    a = 1.3

    assert nonnegative(x)
    assert nonnegative(y)
    assert nonnegative(z)
    assert not nonnegative(a)


def test_nonnegative_outside_integer():
    nonnegative = constraints.NonNegativeInteger()
    x = -1e-6
    y = -1000.0
    z = -0.08

    assert not nonnegative(x)
    assert not nonnegative(y)
    assert not nonnegative(z)


def test_negative_within_integer():
    negative = constraints.NegativeInteger()
    x = -9.0
    y = -128.0
    z = -11.1

    assert negative(x)
    assert negative(y)
    assert not negative(z)


def test_negative_outside_integer():
    negative = constraints.NegativeInteger()
    x = 9.0
    y = 128.0
    z = 0.1

    assert not negative(x)
    assert not negative(y)
    assert not negative(z)


def test_nonpositive_within_integer():
    nonpositive = constraints.NonPositiveInteger()
    x = -9.0
    y = -128.0
    z = 0.0
    a = -99.8

    assert nonpositive(x)
    assert nonpositive(y)
    assert nonpositive(z)
    assert not nonpositive(a)


def test_nonpositive_outside_integer():
    nonpositive = constraints.NonPositiveInteger()
    x = 1e-6
    y = 1000.0
    z = 0.001

    assert not nonpositive(x)
    assert not nonpositive(y)
    assert not nonpositive(z)
