import numpy as np
import pytest

from pynference.distributions.utils import broadcast_shapes, sum_last


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


def test_sum_last1():
    array = np.arange(24).reshape(3, 8)
    assert np.all(sum_last(array, 1) == np.array([28, 92, 156]))


def test_sum_last2():
    array = np.arange(48).reshape(3, 8, 2)
    assert np.all(sum_last(array, 1) == np.arange(1, 97, 4).reshape(3, 8))
    assert np.all(sum_last(array, 2) == np.array([120, 376, 632]))
