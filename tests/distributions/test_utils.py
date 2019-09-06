import numpy as np
import pytest

from pynference.distributions.utils import broadcast_shapes, promote_shapes, sum_last


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
