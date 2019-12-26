from typing import Dict, List, Optional, Union

import numpy as np
import scipy.stats as stats
from pytest import approx, raises

from pynference.constants import Parameter, Shape
from pynference.distributions import Dirichlet, MultivariateNormal
from pynference.utils import check_random_state


def generate(
    random_state: np.random.RandomState,
    dim: int,
    shape: Shape,
    positive: Optional[Union[str, List[str]]] = None,
    real_vector: Optional[Union[str, List[str]]] = None,
    positive_vector: Optional[Union[str, List[str]]] = None,
    positive_definite_matrix: Optional[Union[str, List[str]]] = None,
    lower_triangular_matrix: Optional[Union[str, List[str]]] = None,
    **limits: float,
) -> Dict[str, Parameter]:
    def min_max_transform(x, low, high):
        x_std = (x - np.min(x, axis=(-2, -1), keepdims=True)) / (
            np.max(x, axis=(-2, -1), keepdims=True)
            - np.min(x, axis=(-2, -1), keepdims=True)
        )
        return x_std * (high - low) + low

    parameters = {}

    if limits is None:
        limits = {}

    limits_default = {
        "positive_low": 0.001,
        "positive_high": 10.0,
        "real_vector_low": -10.0,
        "real_vector_high": 10.0,
        "positive_vector_low": 0.001,
        "positive_vector_high": 10.0,
        "positive_definite_matrix_low": 0.1,
        "positive_definite_matrix_high": 5.0,
        "lower_triangular_matrix_low": 1.0,
        "lower_triangular_matrix_high": 2.0,
    }

    limits = {**limits_default, **limits}

    if positive is not None:
        if isinstance(positive, str):
            positive = [positive]

        for p in positive:
            parameters[p] = random_state.uniform(
                low=limits["positive_low"], high=limits["positive_high"], size=shape
            )

    if real_vector is not None:
        if isinstance(real_vector, str):
            real_vector = [real_vector]

        for p in real_vector:
            parameters[p] = random_state.uniform(
                low=limits["real_vector_low"],
                high=limits["real_vector_high"],
                size=shape + (dim,),
            )

    if positive_vector is not None:
        if isinstance(positive_vector, str):
            positive_vector = [positive_vector]

        for p in positive_vector:
            parameters[p] = random_state.uniform(
                low=limits["positive_vector_low"],
                high=limits["positive_vector_high"],
                size=shape + (dim,),
            )

    if positive_definite_matrix is not None:
        if isinstance(positive_definite_matrix, str):
            positive_definite_matrix = [positive_definite_matrix]

        for p in positive_definite_matrix:
            w = stats.wishart.rvs(
                df=dim + 1,
                scale=np.eye(dim),
                size=shape if shape != () else 1,
                random_state=random_state,
            )
            parameters[p] = min_max_transform(
                w,
                low=limits["positive_definite_matrix_low"],
                high=limits["positive_definite_matrix_high"],
            )

    if lower_triangular_matrix is not None:
        if isinstance(lower_triangular_matrix, str):
            lower_triangular_matrix = [lower_triangular_matrix]

        for p in lower_triangular_matrix:
            w = stats.wishart.rvs(
                df=dim + 1,
                scale=np.eye(dim),
                size=shape if shape != () else 1,
                random_state=random_state,
            )
            w_scaled = min_max_transform(
                w,
                low=limits["lower_triangular_matrix_low"],
                high=limits["lower_triangular_matrix_high"],
            )
            parameters[p] = np.linalg.cholesky(w_scaled)

    return parameters


class TestBroadcasting:
    random_state = check_random_state(123)

    n_samples = 100
    atol = 1e-6
    rtol = 1e-6

    def test_dirichlet(self):
        fst = Dirichlet(concentration=np.array([[1.0, 1.0], [1.0, 1.0]]))
        snd = Dirichlet(concentration=np.array(1.0).reshape(1, 1))

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )

    def test_mvn_scalar1(self):
        # scalar, scalar
        fst = MultivariateNormal(mean=1.0, variance=np.array([1.0]))
        snd = MultivariateNormal(mean=np.array(1.0).reshape(1, 1), variance=1.0)

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == snd.batch_shape == (1,)
        assert fst.rv_shape == snd.rv_shape == (1,)

        # scalar, vector
        fst = MultivariateNormal(mean=1.0, variance=np.ones(shape=(2, 2)))
        snd = MultivariateNormal(mean=1.0, variance=np.ones(shape=(1, 2)))

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 2)
        assert fst.rv_shape == snd.rv_shape == (1,)

        # scalar, matrix
        fst = MultivariateNormal(mean=1.0, variance=np.ones(shape=(2, 2, 2)))
        snd = MultivariateNormal(mean=1.0, variance=np.ones(shape=(1, 1, 2)))

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2, 2)
        assert snd.batch_shape == (1, 1, 2)
        assert fst.rv_shape == snd.rv_shape == (1,)

        # vector, scalar
        fst = MultivariateNormal(mean=np.ones(shape=(2, 2)), variance=1.0)
        snd = MultivariateNormal(mean=np.ones(shape=(1, 2)), variance=1.0)

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (1,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)), variance=np.ones(shape=(2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)), variance=np.ones(shape=(2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)), variance=np.ones(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)), variance=np.ones(shape=(2, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2, 2)
        assert snd.batch_shape == (2, 2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, scalar
        fst = MultivariateNormal(mean=np.ones(shape=(2, 2, 2)), variance=1.0)
        snd = MultivariateNormal(mean=np.ones(shape=(1, 1, 2)), variance=1.0)

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 1)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)), variance=np.ones(shape=(2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)), variance=np.ones(shape=(2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)), variance=np.ones(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)), variance=np.ones(shape=(2, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2, 2)
        assert snd.batch_shape == (2, 2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

    def test_mvn_scalar2(self):
        # scalar, scalar
        fst = MultivariateNormal(mean=1.0, precision=np.array([1.0]))
        snd = MultivariateNormal(mean=np.array(1.0).reshape(1, 1), precision=1.0)

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == snd.batch_shape == (1,)
        assert fst.rv_shape == snd.rv_shape == (1,)

        # scalar, vector
        fst = MultivariateNormal(mean=1.0, precision=np.ones(shape=(2, 2)))
        snd = MultivariateNormal(mean=1.0, precision=np.ones(shape=(1, 2)))

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 2)
        assert fst.rv_shape == snd.rv_shape == (1,)

        # scalar, matrix
        fst = MultivariateNormal(mean=1.0, precision=np.ones(shape=(2, 2, 2)))
        snd = MultivariateNormal(mean=1.0, precision=np.ones(shape=(1, 1, 2)))

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2, 2)
        assert snd.batch_shape == (1, 1, 2)
        assert fst.rv_shape == snd.rv_shape == (1,)

        # vector, scalar
        fst = MultivariateNormal(mean=np.ones(shape=(2, 2)), precision=1.0)
        snd = MultivariateNormal(mean=np.ones(shape=(1, 2)), precision=1.0)

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (1,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)), precision=np.ones(shape=(2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)), precision=np.ones(shape=(2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)), precision=np.ones(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)), precision=np.ones(shape=(2, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2, 2)
        assert snd.batch_shape == (2, 2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, scalar
        fst = MultivariateNormal(mean=np.ones(shape=(2, 2, 2)), precision=1.0)
        snd = MultivariateNormal(mean=np.ones(shape=(1, 1, 2)), precision=1.0)

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 1)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)), precision=np.ones(shape=(2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)), precision=np.ones(shape=(2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)), precision=np.ones(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)), precision=np.ones(shape=(2, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2, 2)
        assert snd.batch_shape == (2, 2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

    def test_mvn_vector1(self):
        # scalar, vector
        fst = MultivariateNormal(mean=1.0, variance_diag=np.ones(shape=(2, 2)))
        snd = MultivariateNormal(mean=1.0, variance_diag=np.ones(shape=(1, 2)))

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (1,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # scalar, matrix
        fst = MultivariateNormal(mean=1.0, variance_diag=np.ones(shape=(2, 2, 2)))
        snd = MultivariateNormal(mean=1.0, variance_diag=np.ones(shape=(1, 1, 2)))

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 1)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)), variance_diag=np.ones(shape=(2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)), variance_diag=np.ones(shape=(2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (2,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)), variance_diag=np.ones(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)), variance_diag=np.ones(shape=(2, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)), variance_diag=np.ones(shape=(2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)), variance_diag=np.ones(shape=(2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)), variance_diag=np.ones(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)), variance_diag=np.ones(shape=(2, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

    def test_mvn_vector2(self):
        # scalar, vector
        fst = MultivariateNormal(mean=1.0, precision_diag=np.ones(shape=(2, 2)))
        snd = MultivariateNormal(mean=1.0, precision_diag=np.ones(shape=(1, 2)))

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (1,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # scalar, matrix
        fst = MultivariateNormal(mean=1.0, precision_diag=np.ones(shape=(2, 2, 2)))
        snd = MultivariateNormal(mean=1.0, precision_diag=np.ones(shape=(1, 1, 2)))

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 1)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)), precision_diag=np.ones(shape=(2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)), precision_diag=np.ones(shape=(2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (2,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)), precision_diag=np.ones(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)), precision_diag=np.ones(shape=(2, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)), precision_diag=np.ones(shape=(2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)), precision_diag=np.ones(shape=(2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)), precision_diag=np.ones(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)), precision_diag=np.ones(shape=(2, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

    def _multidimensional_eye(self, shape):
        out = np.zeros(shape=shape)
        out[..., np.arange(out.shape[-1]), np.arange(out.shape[-1])] = 1.0
        return out

    def test_mvn_matrix1(self):
        # scalar, vector
        fst = MultivariateNormal(
            mean=1.0, covariance_matrix=self._multidimensional_eye(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=1.0, covariance_matrix=self._multidimensional_eye(shape=(1, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (1,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # scalar, matrix
        fst = MultivariateNormal(
            mean=1.0, covariance_matrix=self._multidimensional_eye(shape=(2, 2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=1.0, covariance_matrix=self._multidimensional_eye(shape=(1, 1, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 1)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)),
            covariance_matrix=self._multidimensional_eye(shape=(2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)),
            covariance_matrix=self._multidimensional_eye(shape=(2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (2,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)),
            covariance_matrix=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)),
            covariance_matrix=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)),
            covariance_matrix=self._multidimensional_eye(shape=(2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)),
            covariance_matrix=self._multidimensional_eye(shape=(2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)),
            covariance_matrix=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)),
            covariance_matrix=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

    def test_mvn_matrix2(self):
        # scalar, vector
        fst = MultivariateNormal(
            mean=1.0, precision_matrix=self._multidimensional_eye(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=1.0, precision_matrix=self._multidimensional_eye(shape=(1, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (1,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # scalar, matrix
        fst = MultivariateNormal(
            mean=1.0, precision_matrix=self._multidimensional_eye(shape=(2, 2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=1.0, precision_matrix=self._multidimensional_eye(shape=(1, 1, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 1)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)),
            precision_matrix=self._multidimensional_eye(shape=(2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)),
            precision_matrix=self._multidimensional_eye(shape=(2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (2,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)),
            precision_matrix=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)),
            precision_matrix=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)),
            precision_matrix=self._multidimensional_eye(shape=(2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)),
            precision_matrix=self._multidimensional_eye(shape=(2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)),
            precision_matrix=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)),
            precision_matrix=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

    def test_mvn_matrix3(self):
        # scalar, vector
        fst = MultivariateNormal(
            mean=1.0, cholesky_tril=self._multidimensional_eye(shape=(2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=1.0, cholesky_tril=self._multidimensional_eye(shape=(1, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (1,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # scalar, matrix
        fst = MultivariateNormal(
            mean=1.0, cholesky_tril=self._multidimensional_eye(shape=(2, 2, 2, 2))
        )
        snd = MultivariateNormal(
            mean=1.0, cholesky_tril=self._multidimensional_eye(shape=(1, 1, 2, 2))
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 1)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)),
            cholesky_tril=self._multidimensional_eye(shape=(2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)),
            cholesky_tril=self._multidimensional_eye(shape=(2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2,)
        assert snd.batch_shape == (2,)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # vector, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2)),
            cholesky_tril=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 2)),
            cholesky_tril=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, vector
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)),
            cholesky_tril=self._multidimensional_eye(shape=(2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)),
            cholesky_tril=self._multidimensional_eye(shape=(2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (1, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)

        # matrix, matrix
        fst = MultivariateNormal(
            mean=np.ones(shape=(2, 2, 2)),
            cholesky_tril=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )
        snd = MultivariateNormal(
            mean=np.ones(shape=(1, 1, 2)),
            cholesky_tril=self._multidimensional_eye(shape=(2, 2, 2, 2)),
        )

        samples = fst.sample(
            sample_shape=(self.n_samples,), random_state=self.random_state
        )

        assert fst.log_prob(samples) == approx(
            snd.log_prob(samples), rel=self.rtol, abs=self.atol
        )
        assert fst.batch_shape == (2, 2)
        assert snd.batch_shape == (2, 2)
        assert fst.rv_shape == snd.rv_shape == (2,)


class TestExponentialFamilies:
    random_state = check_random_state(123)

    distributions = {
        Dirichlet: [
            generate(random_state, dim=5, shape=(), positive_vector="concentration"),
            generate(random_state, dim=5, shape=(4,), positive_vector="concentration"),
            generate(
                random_state, dim=5, shape=(4, 3), positive_vector="concentration"
            ),
        ],
        MultivariateNormal: [
            generate(
                random_state, dim=5, shape=(), real_vector="mean", positive="variance"
            ),
            generate(
                random_state, dim=5, shape=(4,), real_vector="mean", positive="variance"
            ),
            generate(
                random_state,
                dim=5,
                shape=(4, 3),
                real_vector="mean",
                positive="variance",
            ),
            generate(
                random_state, dim=5, shape=(), real_vector="mean", positive="precision"
            ),
            generate(
                random_state,
                dim=5,
                shape=(4,),
                real_vector="mean",
                positive="precision",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4, 3),
                real_vector="mean",
                positive="precision",
            ),
            generate(
                random_state,
                dim=5,
                shape=(),
                real_vector="mean",
                positive_vector="variance_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4,),
                real_vector="mean",
                positive_vector="variance_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4, 3),
                real_vector="mean",
                positive_vector="variance_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(),
                real_vector="mean",
                positive_vector="precision_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4,),
                real_vector="mean",
                positive_vector="precision_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4, 3),
                real_vector="mean",
                positive_vector="precision_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(),
                real_vector="mean",
                positive_definite_matrix="covariance_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4,),
                real_vector="mean",
                positive_definite_matrix="covariance_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4, 3),
                real_vector="mean",
                positive_definite_matrix="covariance_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(),
                real_vector="mean",
                positive_definite_matrix="precision_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4,),
                real_vector="mean",
                positive_definite_matrix="precision_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4, 3),
                real_vector="mean",
                positive_definite_matrix="precision_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(),
                real_vector="mean",
                lower_triangular_matrix="cholesky_tril",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4,),
                real_vector="mean",
                lower_triangular_matrix="cholesky_tril",
            ),
            generate(
                random_state,
                dim=5,
                shape=(4, 3),
                real_vector="mean",
                lower_triangular_matrix="cholesky_tril",
            ),
        ],
    }

    n_samples = 100
    atol = 1e-2
    rtol = 1e-2

    def test_base_measure_positive_within_support(self):
        for distribution_cls, p in self.distributions.items():
            if not isinstance(p, list):
                p = [p]

            for parameters in p:
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(self.n_samples,), random_state=self.random_state
                )

                assert np.all(
                    distribution.base_measure(samples) > 0
                ), f"base measure of {distribution}"

    def test_log_probs_equal(self):
        for distribution_cls, p in self.distributions.items():
            if not isinstance(p, list):
                p = [p]

            for parameters in p:
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(self.n_samples,), random_state=self.random_state
                )

                h_x = distribution.base_measure(samples)
                eta = distribution.natural_parameter
                t_x = distribution.sufficient_statistic(samples)
                a_eta = distribution.log_normalizer

                # TODO: Write like this (using elementwise multiplication and sum) in the other tests as well.
                dot_product = sum(np.sum(e * t, axis=-1) for e, t in zip(eta, t_x))
                expected_log_prob = np.log(h_x) + dot_product - a_eta

                assert distribution.log_prob(samples) == approx(
                    expected_log_prob, rel=self.rtol, abs=self.atol
                ), f"log_prob of {distribution}"


class TestFirstTwoMoments:
    random_state = check_random_state(123)

    distributions = {
        Dirichlet: (
            generate(random_state, dim=5, shape=(), positive_vector="concentration"),
            generate(random_state, dim=5, shape=(2,), positive_vector="concentration"),
            generate(
                random_state, dim=5, shape=(2, 3), positive_vector="concentration"
            ),
            generate(random_state, dim=10, shape=(), positive_vector="concentration"),
            generate(random_state, dim=10, shape=(2,), positive_vector="concentration"),
            generate(
                random_state, dim=10, shape=(2, 3), positive_vector="concentration"
            ),
        ),
        MultivariateNormal: (
            generate(
                random_state, dim=5, shape=(), real_vector="mean", positive="variance"
            ),
            generate(
                random_state, dim=5, shape=(2,), real_vector="mean", positive="variance"
            ),
            generate(
                random_state,
                dim=5,
                shape=(2, 3),
                real_vector="mean",
                positive="variance",
            ),
            generate(
                random_state, dim=5, shape=(), real_vector="mean", positive="precision"
            ),
            generate(
                random_state,
                dim=5,
                shape=(2,),
                real_vector="mean",
                positive="precision",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2, 3),
                real_vector="mean",
                positive="precision",
            ),
            generate(
                random_state,
                dim=5,
                shape=(),
                real_vector="mean",
                positive_vector="variance_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2,),
                real_vector="mean",
                positive_vector="variance_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2, 3),
                real_vector="mean",
                positive_vector="variance_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(),
                real_vector="mean",
                positive_vector="precision_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2,),
                real_vector="mean",
                positive_vector="precision_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2, 3),
                real_vector="mean",
                positive_vector="precision_diag",
            ),
            generate(
                random_state,
                dim=5,
                shape=(),
                real_vector="mean",
                positive_definite_matrix="covariance_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2,),
                real_vector="mean",
                positive_definite_matrix="covariance_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2, 3),
                real_vector="mean",
                positive_definite_matrix="covariance_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(),
                real_vector="mean",
                positive_definite_matrix="precision_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2,),
                real_vector="mean",
                positive_definite_matrix="precision_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2, 3),
                real_vector="mean",
                positive_definite_matrix="precision_matrix",
            ),
            generate(
                random_state,
                dim=5,
                shape=(),
                real_vector="mean",
                lower_triangular_matrix="cholesky_tril",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2,),
                real_vector="mean",
                lower_triangular_matrix="cholesky_tril",
            ),
            generate(
                random_state,
                dim=5,
                shape=(2, 3),
                real_vector="mean",
                lower_triangular_matrix="cholesky_tril",
            ),
        ),
    }

    n_samples = 200000
    atol = 1e-2
    rtol = 0.75

    def test_mean_and_variance(self):
        for distribution_cls, parameter_set in self.distributions.items():
            if distribution_cls == MultivariateNormal:
                continue

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

    def _batch_covariance_matrix(self, samples: np.ndarray) -> np.ndarray:
        centered = samples - np.mean(samples, axis=0, keepdims=True)
        batch_outer = np.einsum("j...i, i...k", np.swapaxes(centered, 0, -1), centered)
        return batch_outer / (samples.shape[0] - 1)

    def _multidimensional_diag(self, x):
        return x[..., np.arange(x.shape[-1]), np.arange(x.shape[-1])]

    def test_mean_and_variance_mvn(self):
        from pynference.distributions.continuous_multivariate import (
            _MVNScalar,
            _MVNVector,
            _MVNMatrix,
        )

        parameter_set = self.distributions[MultivariateNormal]

        for i, parameters in enumerate(parameter_set):
            distribution = MultivariateNormal(**parameters)

            samples = distribution.sample(
                sample_shape=(self.n_samples,), random_state=self.random_state
            )

            true_mean = distribution.mean
            empirical_mean = np.mean(samples, axis=0)

            assert empirical_mean == approx(
                true_mean, rel=self.rtol, abs=self.atol
            ), f"mean of {distribution}"

            if type(distribution) is _MVNScalar:
                true_variance = distribution.variance
                empirical_variance = np.mean(np.var(samples, axis=0), axis=-1)

                assert empirical_variance == approx(
                    true_variance, rel=self.rtol, abs=self.atol
                ), f"variance of {distribution}"
            elif type(distribution) is _MVNVector:
                true_variance = distribution.variance
                empirical_variance = np.var(samples, axis=0)

                assert empirical_variance == approx(
                    true_variance, rel=self.rtol, abs=self.atol
                ), f"variance of {distribution}"

                true_covariance = distribution.covariance_matrix

                assert empirical_variance == approx(
                    self._multidimensional_diag(true_covariance),
                    rel=self.rtol,
                    abs=self.atol,
                ), f"variance of {distribution}"
            elif type(distribution) is _MVNMatrix:
                true_covariance = distribution.covariance_matrix
                empirical_covariance = self._batch_covariance_matrix(samples)

                assert empirical_covariance == approx(
                    true_covariance, rel=self.rtol, abs=self.atol
                ), f"covariance of {distribution}"

                true_variance = distribution.variance
                empirical_variance = self._multidimensional_diag(empirical_covariance)

                assert empirical_variance == approx(
                    true_variance, rel=self.rtol, abs=self.atol
                ), f"variance of {distribution}"
            else:
                raise ValueError("This should never happen")


class TestLogProb:
    random_state = check_random_state(123)

    distributions = {
        Dirichlet: generate(
            random_state, dim=5, shape=(), positive_vector="concentration"
        )
    }

    dist2scipy = {Dirichlet: lambda dist: stats.dirichlet(alpha=dist.concentration)}

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

            if distribution_cls is Dirichlet:
                # For some reason, the SciPy Dirichlet distribution expects
                # the variates in a reversed shape.
                scipy_result = scipy_distribution.logpdf(samples.T)
            else:
                scipy_result = scipy_distribution.logpdf(samples)

            assert distribution.log_prob(samples) == approx(
                scipy_result, rel=self.rtol, abs=self.atol
            ), f"log_prob of {distribution}"


class TestParameterConstraints:
    def test_dirichlet(self):
        with raises(ValueError, match=r".*positive_vector.*"):
            Dirichlet(concentration=np.array([1.0, 2.0, 0.0]))
            Dirichlet(concentration=np.array([1.0, -2.0, 3.0]))
            Dirichlet(concentration=np.array([0.00001, 0.1, 10.0, 100.0, 5.0]))
            Dirichlet(concentration=np.array([1.0, 2.0, np.nan, 2.1]))


class TestSamplingShapes:
    random_state = check_random_state(123)

    distributions = {
        Dirichlet: (
            (
                generate(
                    random_state, dim=5, shape=(), positive_vector="concentration"
                ),
                5,
            ),
            (
                generate(
                    random_state, dim=5, shape=(2,), positive_vector="concentration"
                ),
                5,
            ),
            (
                generate(
                    random_state, dim=5, shape=(2, 3), positive_vector="concentration"
                ),
                5,
            ),
            (
                generate(
                    random_state, dim=10, shape=(), positive_vector="concentration"
                ),
                10,
            ),
            (
                generate(
                    random_state, dim=10, shape=(2,), positive_vector="concentration"
                ),
                10,
            ),
            (
                generate(
                    random_state, dim=10, shape=(2, 3), positive_vector="concentration"
                ),
                10,
            ),
        )
    }

    def test_sampling_shapes_0d(self):
        batch_shapes = [(), (2,), (2, 3), (), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, (parameters, dim) in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(), random_state=self.random_state
                )
                assert samples.shape == batch_shapes[i] + (
                    dim,
                ), f"sampling shape of {distribution}"

    def test_sampling_shapes_1d(self):
        batch_shapes = [(), (2,), (2, 3), (), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, (parameters, dim) in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(100,), random_state=self.random_state
                )
                assert samples.shape == (100,) + batch_shapes[i] + (
                    dim,
                ), f"sampling shape of {distribution}"

    def test_sampling_shapes_2d(self):
        batch_shapes = [(), (2,), (2, 3), (), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, (parameters, dim) in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(10, 10), random_state=self.random_state
                )
                assert samples.shape == (10, 10) + batch_shapes[i] + (
                    dim,
                ), f"sampling shape of {distribution}"

    def test_sampling_shapes_3d(self):
        batch_shapes = [(), (2,), (2, 3), (), (2,), (2, 3)]

        for distribution_cls, parameter_set in self.distributions.items():
            for i, (parameters, dim) in enumerate(parameter_set):
                distribution = distribution_cls(**parameters)

                samples = distribution.sample(
                    sample_shape=(10, 10, 2), random_state=self.random_state
                )
                assert samples.shape == (10, 10, 2) + batch_shapes[i] + (
                    dim,
                ), f"sampling shape of {distribution}"


class TestTransformedDistributions:
    random_state = check_random_state(123)

    distributions = {}

    supports = {}

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
