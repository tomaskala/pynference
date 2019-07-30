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
