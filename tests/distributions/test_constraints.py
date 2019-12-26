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


def test_simplex_within():
    simplex = constraints.Simplex()

    x = np.arange(1, 10, dtype=float)
    x /= np.sum(x)

    y = np.linspace(0.01, 1.0, 20)
    y /= np.sum(y)

    assert simplex(x)
    assert simplex(y)


def test_simplex_outside():
    simplex = constraints.Simplex()

    x = np.linspace(0.0, 1.0, 20)
    x /= np.sum(x)

    y = np.linspace(0.0, 1.0, 20)
    y /= np.sum(x)
    y[0] = -0.0001

    z = np.linspace(0.0, 0.5, 25)

    assert not simplex(x)
    assert not simplex(y)
    assert not simplex(z)


def test_positive_vector_within():
    positive_vector = constraints.PositiveVector()

    x = np.array([0.1, 1.0, 2.0])
    y = np.array([1, 2, 3])

    assert positive_vector(x)
    assert positive_vector(y)


def test_positive_vector_outside():
    positive_vector = constraints.PositiveVector()

    x = np.array([-0.00001, 10.0])
    y = np.array([2.0, 0.0])
    z = np.array([-np.inf, 10.0, 23.0])

    assert not positive_vector(x)
    assert not positive_vector(y)
    assert not positive_vector(z)


def test_non_negative_vector_within():
    non_negative_vector = constraints.NonNegativeVector()

    x = np.array([0.1, 1.0, 2.0])
    y = np.array([1, 2, 3])
    z = np.array([2.0, 0.0])

    assert non_negative_vector(x)
    assert non_negative_vector(y)
    assert non_negative_vector(z)


def test_non_negative_vector_outside():
    non_negative_vector = constraints.NonNegativeVector()

    x = np.array([-0.00001, 10.0])
    y = np.array([-np.inf, 10.0, 23.0])

    assert not non_negative_vector(x)
    assert not non_negative_vector(y)


def test_negative_vector_within():
    negative_vector = constraints.NegativeVector()

    x = np.array([-0.1, -1.0, -2.0])
    y = np.array([-1, -2, -3])

    assert negative_vector(x)
    assert negative_vector(y)


def test_negative_vector_outside():
    negative_vector = constraints.NegativeVector()

    x = np.array([0.00001, -10.0])
    y = np.array([-2.0, 0.0])
    z = np.array([np.inf, -10.0, -23.0])

    assert not negative_vector(x)
    assert not negative_vector(y)
    assert not negative_vector(z)


def test_non_positive_vector_within():
    non_positive_vector = constraints.NonPositiveVector()

    x = np.array([-0.1, -1.0, -2.0])
    y = np.array([-1, -2, -3])
    z = np.array([-2.0, 0.0])

    assert non_positive_vector(x)
    assert non_positive_vector(y)
    assert non_positive_vector(z)


def test_non_positive_vector_outside():
    non_positive_vector = constraints.NonPositiveVector()

    x = np.array([0.00001, -10.0])
    y = np.array([np.inf, -10.0, -23.0])

    assert not non_positive_vector(x)
    assert not non_positive_vector(y)


def test_positive_definite_within():
    positive_definite = constraints.PositiveDefinite()

    x = np.array([[1.0, 0.0, 3.0], [3.0, 2.0, 0.0], [1.0, 1.0, 1.0]])
    x = x.T @ x

    y = np.array([[1.0, 6.0, 0.0], [6.0, 7.0, 2.0], [0.0, 2.0, 3.0]])
    y = y.T @ y

    assert positive_definite(x)
    assert positive_definite(y)


def test_positive_definite_outside():
    positive_definite = constraints.PositiveDefinite()

    x = np.arange(1, 10, dtype=float).reshape(3, 3)
    y = np.eye(4)
    y[1, 1] = -0.01

    assert not positive_definite(x)
    assert not positive_definite(y)


def test_lower_cholesky_within():
    lower_cholesky = constraints.LowerCholesky()

    x = np.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 2.0, 1.0]])

    y = np.eye(4)

    z = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [9.9, 2.1, 3.1, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert lower_cholesky(x)
    assert lower_cholesky(y)
    assert lower_cholesky(z)


def test_lower_cholesky_outside():
    lower_cholesky = constraints.LowerCholesky()

    x = np.array([[1.0, 1.0, 0.0], [2.0, -3.0, 0.0], [4.0, 2.0, 1.0]])

    y = np.eye(4)
    y[2, 2] = -0.01

    z = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 2.0],
            [9.9, 2.1, 3.1, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert not lower_cholesky(x)
    assert not lower_cholesky(y)
    assert not lower_cholesky(z)
