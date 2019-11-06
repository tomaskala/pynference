import numpy as np
import pytest
from pytest import approx
from scipy.special import factorial

from pynference.distributions.utils import (
    arraywise_diagonal,
    broadcast_shapes,
    log_binomial_coefficient,
    promote_shapes,
    sum_last,
)


def test_broadcast_single_arg():
    x = (4, 3)

    assert broadcast_shapes(x) == (4, 3)


def test_broadcast_2d():
    x = (4, 1)
    y = (4, 3)

    assert broadcast_shapes(x, y) == (4, 3)


def test_broadcast_3d():
    x = (4, 1, 1)
    y = (4, 3, 1)
    z = (4, 3, 2)

    assert broadcast_shapes(x, y, z) == (4, 3, 2)


def test_broadcast_5d():
    x = (1, 4, 5, 1, 1)
    y = (2, 4, 5, 6, 2)

    assert broadcast_shapes(x, y) == (2, 4, 5, 6, 2)


def test_broadcast_correct_order_matters():
    x = (10, 1)
    y = (10, 1)
    z1 = (10, 2)
    z2 = (2, 10)

    assert broadcast_shapes(x, y, z1) == (10, 2)

    with pytest.raises(ValueError):
        broadcast_shapes(x, y, z2)


def test_broadcast_incompatible_richer_shapes():
    x = (2, 5)
    y = (4, 5)

    with pytest.raises(ValueError):
        broadcast_shapes(x, y)


def test_broadcast_require_explicit_ones():
    x = (4,)
    y = (4, 3)
    z = (4, 3, 2)

    with pytest.raises(ValueError):
        broadcast_shapes(x, y, z)


def test_broadcast_incompatible1():
    x = (4, 5)
    y = (2, 6)

    with pytest.raises(ValueError):
        broadcast_shapes(x, y)


def test_broadcast_incompatible2():
    x = (1, 2, 3)
    y = (3, 2)

    with pytest.raises(ValueError):
        broadcast_shapes(x, y)


def test_promote_shapes_no_extra_shape():
    x = np.arange(12).reshape(4, 3)
    y = np.arange(3)
    z = np.arange(3).reshape(1, 1, 3)

    xx, yy, zz = promote_shapes(x, y, z)

    assert xx.shape == (1, 4, 3)
    assert yy.shape == (1, 1, 3)
    assert zz.shape == (1, 1, 3)

    assert np.all(xx[0] == x)
    assert np.all(yy[0, 0] == y)
    assert np.all(zz == z)


def test_promote_shapes_extra_shape():
    x = np.arange(12).reshape(4, 3)
    y = np.arange(3)
    z = np.arange(3).reshape(1, 1, 3)

    xx, yy, zz = promote_shapes(x, y, z, shape=(1, 1, 1, 4, 3))

    assert xx.shape == (1, 1, 1, 4, 3)
    assert yy.shape == (1, 1, 1, 1, 3)
    assert zz.shape == (1, 1, 1, 1, 3)

    assert np.all(xx[0, 0, 0] == x)
    assert np.all(yy[-1] == y)
    assert np.all(zz[-1] == z)


def test_sum_last1():
    array = np.arange(24).reshape(3, 8)
    assert np.all(sum_last(array, 1) == np.array([28, 92, 156]))


def test_sum_last2():
    array = np.arange(48).reshape(3, 8, 2)
    assert np.all(sum_last(array, 1) == np.arange(1, 97, 4).reshape(3, 8))
    assert np.all(sum_last(array, 2) == np.array([120, 376, 632]))


def test_log_binomial_coefficient1():
    n = 10
    k = 5
    binomial_coef = factorial(n, exact=True) / (
        factorial(k, exact=True) * factorial(n - k, exact=True)
    )

    assert log_binomial_coefficient(n, k) == approx(
        np.log(binomial_coef), rel=1e-5, abs=1e-5
    )


def test_log_binomial_coefficient2():
    n = 6
    k = 3
    binomial_coef = factorial(n, exact=True) / (
        factorial(k, exact=True) * factorial(n - k, exact=True)
    )

    assert log_binomial_coefficient(n, k) == approx(
        np.log(binomial_coef), rel=1e-5, abs=1e-5
    )


def test_arraywise_diagonal1():
    single = np.array([4.0])
    assert arraywise_diagonal(single) == approx(np.array([[4.0]]), rel=1e-5, abs=1e-5)


def test_arraywise_diagonal2():
    batch_single = np.array([[4.0], [5.0], [6.0]])
    assert arraywise_diagonal(batch_single) == approx(
        np.array([[[4.0]], [[5.0]], [[6.0]]]), rel=1e-5, abs=1e-5
    )


def test_arraywise_diagonal3():
    diagonal = np.arange(3.0) + 1.0
    assert arraywise_diagonal(diagonal) == approx(np.diag(diagonal), rel=1e-5, abs=1e-5)


def test_arraywise_diagonal4():
    batch_diagonal = np.arange(6.0).reshape(2, 3) + 1.0
    expected = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
        ]
    )
    assert arraywise_diagonal(batch_diagonal) == approx(expected, rel=1e-5, abs=1e-5)


def test_arraywise_diagonal5():
    arange = np.arange(8.0)
    ones = np.ones(shape=(8, 4))
    large_diagonal = arange.reshape(-1, 1) * ones

    expected = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
            ],
            [
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
            ],
            [
                [4.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            [
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
            ],
            [
                [6.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 0.0, 0.0],
                [0.0, 0.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 6.0],
            ],
            [
                [7.0, 0.0, 0.0, 0.0],
                [0.0, 7.0, 0.0, 0.0],
                [0.0, 0.0, 7.0, 0.0],
                [0.0, 0.0, 0.0, 7.0],
            ],
        ]
    )

    assert arraywise_diagonal(large_diagonal) == approx(expected, rel=1e-5, abs=1e-5)
