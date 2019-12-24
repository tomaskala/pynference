from typing import Dict, Tuple

import numpy as np
import numpy.linalg as la  # Not SciPy, NumPy works for batches of matrices.
from numpy.random import RandomState
from scipy.linalg import solve_triangular
from scipy.special import gammaln

from pynference.constants import ArrayLike, Parameter, Shape, Variate
from pynference.distributions.constraints import (
    Constraint,
    lower_cholesky,
    positive,
    positive_definite,
    positive_vector,
    real_vector,
    simplex,
)
from pynference.distributions.distribution import (
    Distribution,
    ExponentialFamily,
    TransformedDistribution,
)
from pynference.distributions.utils import (
    arraywise_diagonal,
    broadcast_shapes,
    promote_shapes,
    replicate_along_last_axis,
)

# TODO: Thoroughly test all cases of the multivariate normal distribution.
# TODO: mean: scalar/vector/matrix, variance & precision: scalar/vector/matrix
# TODO: mean: scalar/vector/matrix, variance_diag & precision_diag: vector/matrix
# TODO: mean: scalar/vector/matrix, covariance_matrix & precision_matrix & cholesky_triu: matrix & matrices
# TODO: Are all the np.broadcast_to necessary?
# TODO: Replace np.matmul by @ if they are equivalent.
# TODO: Test the new constraints, including those from discrete distributions.
# TODO: MVN natural parameter is not very memory-efficient now.


class Dirichlet(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"concentration": positive_vector}
    _support: Constraint = simplex

    def __init__(
        self,
        concentration: Parameter,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        if np.ndim(concentration) < 1:
            raise ValueError(
                "The concentration parameter must be at least 1-dimensional."
            )

        batch_shape = np.shape(concentration)[:-1]
        rv_shape = np.shape(concentration)[-1:]

        self.concentration = concentration
        self.concentration_sum = np.sum(concentration, axis=-1, keepdims=True)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        return self.concentration / self.concentration_sum

    @property
    def variance(self) -> Parameter:
        mean = self.mean
        return mean * (1.0 - mean) / (self.concentration_sum + 1.0)

    def _log_prob(self, x: Variate) -> ArrayLike:
        normalizer = np.sum(gammaln(self.concentration), axis=-1) - gammaln(
            np.sum(self.concentration, axis=-1)
        )
        return np.sum((self.concentration - 1.0) * np.log(x), axis=-1) - normalizer

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_gamma(
            self.concentration, sample_shape + self.batch_shape + self.rv_shape
        )
        return epsilon / np.sum(epsilon, axis=-1, keepdims=True)

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        return (self.concentration,)

    @property
    def log_normalizer(self) -> Parameter:
        return np.sum(gammaln(self.concentration), axis=-1) - gammaln(
            np.sum(self.concentration, axis=-1)
        )

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.reciprocal(np.exp(np.sum(np.log(x), axis=-1)))

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return (np.log(x),)


class InverseWishart(TransformedDistribution):
    pass


def cholesky_inverse(matrix: np.ndarray) -> np.ndarray:
    # https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    tril_inv = np.swapaxes(
        la.cholesky(matrix[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
    )
    identity = np.broadcast_to(np.identity(matrix.shape[-1]), tril_inv.shape)
    return la.solve_triangular(tril_inv, identity, lower=True)


class _MVNScalar(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {
        "mean": real_vector,
        "precision": positive,
        "std": positive,
    }
    _support: Constraint = real_vector

    def __init__(
        self,
        mean: Parameter,
        precision: Parameter,
        std: Parameter,
        batch_shape: Shape,
        rv_shape: Shape,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        self._mean = mean
        self._precision = precision
        self._std = std

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        return self._mean

    @property
    def variance(self) -> Parameter:
        variance = np.reciprocal(self._precision)
        return variance

    @property
    def precision(self) -> Parameter:
        return self._precision

    @property
    def covariance_matrix(self) -> Parameter:
        variance = np.reciprocal(self._precision)
        replicated_variance = replicate_along_last_axis(variance, self.rv_shape)
        return arraywise_diagonal(replicated_variance)

    @property
    def precision_matrix(self) -> Parameter:
        replicated_precision = replicate_along_last_axis(self._precision, self.rv_shape)
        return arraywise_diagonal(replicated_precision)

    def _log_prob(self, x: Variate) -> ArrayLike:
        half_log_det = self._half_log_det()
        mahalanobis_squared = self._mahalanobis_squared(x - self._mean)

        normalizer = half_log_det + 0.5 * self.rv_shape[0] * np.log(2.0 * np.pi)
        return -0.5 * mahalanobis_squared - normalizer

    def _half_log_det(self):
        return -0.5 * self.rv_shape[0] * np.log(self._precision)

    def _mahalanobis_squared(self, centered_x: Variate):
        return self._precision * np.sum(np.square(centered_x), axis=-1)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_normal(
            sample_shape + self.batch_shape + self.rv_shape
        )
        return self._mean + self._std * epsilon

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        precision = self.precision_matrix
        return np.matmul(precision, self._mean), -0.5 * precision

    @property
    def log_normalizer(self) -> Parameter:
        half_log_det = self._half_log_det()
        mahalanobis_squared = self._mahalanobis_squared(self._mean)

        return 0.5 * mahalanobis_squared + half_log_det

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.power(2.0 * np.pi, -self.rv_shape[0] / 2.0)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        if np.isscalar(x):
            x = np.expand_dims(x, axis=-1)

        batch_outer = np.matmul(
            x[..., np.newaxis], np.swapaxes(x[..., np.newaxis], -2, -1)
        )
        return x, batch_outer


class _MVNVector(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {
        "mean": real_vector,
        "precision_diag": positive_vector,
        "std_diag": positive_vector,
    }
    _support: Constraint = real_vector

    def __init__(
        self,
        mean: Parameter,
        precision_diag: Parameter,
        std_diag: Parameter,
        batch_shape: Shape,
        rv_shape: Shape,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        self._mean = mean
        self._precision_diag = precision_diag
        self._std_diag = std_diag

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        return self._mean

    @property
    def variance(self) -> Parameter:
        variance_diag = np.reciprocal(self._precision_diag)
        return variance_diag

    @property
    def precision(self) -> Parameter:
        return self._precision_diag

    @property
    def covariance_matrix(self) -> Parameter:
        variance_diag = np.reciprocal(self._precision_diag)
        return arraywise_diagonal(variance_diag)

    @property
    def precision_matrix(self) -> Parameter:
        return arraywise_diagonal(self._precision_diag)

    def _log_prob(self, x: Variate) -> ArrayLike:
        half_log_det = self._half_log_det()
        mahalanobis_squared = self._mahalanobis_squared(x - self._mean)

        normalizer = half_log_det + 0.5 * self.rv_shape[0] * np.log(2.0 * np.pi)
        return -0.5 * mahalanobis_squared - normalizer

    def _half_log_det(self):
        return -0.5 * np.sum(np.log(self._precision_diag), axis=-1)

    def _mahalanobis_squared(self, centered_x: Variate):
        return np.sum(self._precision_diag * np.square(centered_x), axis=-1)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_normal(
            sample_shape + self.batch_shape + self.rv_shape
        )
        return self._mean + self._std_diag * epsilon

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        precision = self.precision_matrix
        return np.matmul(precision, self._mean), -0.5 * precision

    @property
    def log_normalizer(self) -> Parameter:
        half_log_det = self._half_log_det()
        mahalanobis_squared = self._mahalanobis_squared(self._mean)

        return 0.5 * mahalanobis_squared + half_log_det

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.power(2.0 * np.pi, -self.rv_shape[0] / 2.0)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        if np.isscalar(x):
            x = np.expand_dims(x, axis=-1)

        batch_outer = np.matmul(
            x[..., np.newaxis], np.swapaxes(x[..., np.newaxis], -2, -1)
        )
        return x, batch_outer


class _MVNMatrix(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {
        "mean": real_vector,
        "cholesky_tril": lower_cholesky,
    }
    _support: Constraint = real_vector

    def __init__(
        self,
        mean: Parameter,
        cholesky_tril: Parameter,
        batch_shape: Shape,
        rv_shape: Shape,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        self._mean = mean
        self._cholesky_tril = cholesky_tril

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    @property
    def mean(self) -> Parameter:
        return self._mean

    @property
    def variance(self) -> Parameter:
        return np.sum(np.square(self._cholesky_tril), axis=-1)

    @property
    def precision(self) -> Parameter:
        return np.reciprocal(np.sum(np.square(self._cholesky_tril), axis=-1))

    @property
    def covariance_matrix(self) -> Parameter:
        return self._cholesky_tril @ self._cholesky_tril.T

    @property
    def precision_matrix(self) -> Parameter:
        cholesky_tril_inv = la.inv(self._cholesky_tril)
        return cholesky_tril_inv.T @ cholesky_tril_inv

    def _log_prob(self, x: Variate) -> ArrayLike:
        half_log_det = self._half_log_det()
        mahalanobis_squared = self._mahalanobis_squared(x - self._mean)

        normalizer = half_log_det + 0.5 * self.rv_shape[0] * np.log(2.0 * np.pi)
        return -0.5 * mahalanobis_squared - normalizer

    def _half_log_det(self):
        return np.sum(
            np.log(np.diagonal(self._cholesky_tril, axis1=-2, axis2=-1)), axis=-1
        )

    def _mahalanobis_squared(self, centered_x: Variate):
        # Source: https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/continuous.py.
        # This procedure handles the case:
        # self._cholesky_tril.shape = (i, 1, n, n), centered_x.shape = (i, j, n),
        # because we do not want to broadcast self._cholesky_tril to the shape (i, j, n, n).

        # Assume that self._cholesky_tril.shape = (i, 1, n, n), centered_x.shape = (..., i, j, n),
        # we are going to make centered_x have shape (..., 1, j,  i, 1, n) to apply batched tril_solve.
        sample_ndim = (
            np.ndim(centered_x) - np.ndim(self._cholesky_tril) + 1
        )  # size of sample_shape
        out_shape = np.shape(centered_x)[:-1]  # shape of output
        # Reshape centered_x with the shape (..., 1, i, j, 1, n).
        centered_x_new_shape = out_shape[:sample_ndim]

        for (sL, sx) in zip(self._cholesky_tril.shape[:-2], out_shape[sample_ndim:]):
            centered_x_new_shape += (sx // sL, sL)

        centered_x_new_shape += (-1,)
        centered_x = np.reshape(centered_x, centered_x_new_shape)
        # Permute centered_x to make it have shape (..., 1, j, i, 1, n).
        permute_dims = (
            tuple(range(sample_ndim))
            + tuple(range(sample_ndim, centered_x.ndim - 1, 2))
            + tuple(range(sample_ndim + 1, centered_x.ndim - 1, 2))
            + (centered_x.ndim - 1,)
        )
        centered_x = np.transpose(centered_x, permute_dims)

        # Reshape to (-1, i, 1, n).
        xt = np.reshape(centered_x, (-1,) + self._cholesky_tril.shape[:-1])
        # Permute to (i, 1, n, -1).
        xt = np.moveaxis(xt, 0, -1)
        solved_triangular = solve_triangular(
            self._cholesky_tril, xt, lower=True
        )  # shape: (i, 1, n, -1)
        M = np.sum(solved_triangular ** 2, axis=-2)  # shape: (i, 1, -1)
        # Permute back to (-1, i, 1).
        M = np.moveaxis(M, -1, 0)
        # Reshape back to (..., 1, j, i, 1).
        M = np.reshape(M, centered_x.shape[:-1])
        # Permute back to (..., 1, i, j, 1).
        permute_inv_dims = tuple(range(sample_ndim))

        for i in range(self._cholesky_tril.ndim - 2):
            permute_inv_dims += (sample_ndim + i, len(out_shape) + i)

        M = np.transpose(M, permute_inv_dims)
        return np.reshape(M, out_shape)

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_normal(
            sample_shape + self.batch_shape + self.rv_shape
        )
        return self._mean + np.squeeze(
            np.matmul(self._cholesky_tril, epsilon[..., np.newaxis]), axis=-1
        )

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        precision = self.precision_matrix
        return np.matmul(precision, self._mean), -0.5 * precision

    @property
    def log_normalizer(self) -> Parameter:
        half_log_det = self._half_log_det()
        mahalanobis_squared = self._mahalanobis_squared(self._mean)

        return 0.5 * mahalanobis_squared + half_log_det

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.power(2.0 * np.pi, -self.rv_shape[0] / 2.0)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        if np.isscalar(x):
            x = np.expand_dims(x, axis=-1)

        batch_outer = np.matmul(
            x[..., np.newaxis], np.swapaxes(x[..., np.newaxis], -2, -1)
        )
        return x, batch_outer


# TODO: Check constraints manually.
def MultivariateNormal(
    mean: Parameter,
    variance: Parameter = None,
    precision: Parameter = None,
    variance_diag: Parameter = None,
    precision_diag: Parameter = None,
    covariance_matrix: Parameter = None,
    precision_matrix: Parameter = None,
    cholesky_tril: Parameter = None,
    check_parameters: bool = True,
    check_support: bool = True,
):
    if (
        (variance is not None)
        + (precision is not None)
        + (variance_diag is not None)
        + (precision_diag is not None)
        + (covariance_matrix is not None)
        + (precision_matrix is not None)
        + (cholesky_tril is not None)
        != 1
    ):
        raise ValueError(
            "Provide exactly one of the variance, precision, diagonal variance, "
            "diagonal precision, covariance matrix, precision matrix or Cholesky "
            "lower triangular part of the covariance matrix parameters."
        )

    if np.isscalar(mean):
        mean = np.expand_dims(mean, axis=-1)

    if variance is not None or precision is not None:
        if variance is not None:
            precision = np.reciprocal(variance)
            std = np.sqrt(variance)
        else:
            std = np.reciprocal(np.sqrt(precision))

        batch_shape = broadcast_shapes(np.shape(mean)[:-1], np.shape(precision))
        rv_shape = np.shape(mean)[-1:]

        mean = np.broadcast_to(mean, batch_shape + rv_shape)
        precision = np.broadcast_to(precision, batch_shape + rv_shape)

        return _MVNScalar(
            mean=mean,
            precision=precision,
            std=std,
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )
    elif variance_diag is not None or precision_diag is not None:
        if variance_diag is not None:
            precision_diag = np.reciprocal(variance_diag)
            std_diag = np.sqrt(variance_diag)
        else:
            std_diag = np.reciprocal(np.sqrt(precision_diag))

        mean, precision_diag = promote_shapes(mean, precision_diag)

        batch_shape = broadcast_shapes(
            np.shape(mean)[:-1], np.shape(precision_diag)[:-1]
        )
        rv_shape = np.shape(precision_diag)[-1:]

        mean = np.broadcast_to(mean, batch_shape + rv_shape)
        precision_diag = np.broadcast_to(precision_diag, batch_shape + rv_shape)

        return _MVNVector(
            mean=mean,
            precision_diag=precision_diag,
            std_diag=std_diag,
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )
    else:
        mean = mean[..., np.newaxis]

        if covariance_matrix is not None:
            mean, covariance_matrix = promote_shapes(mean, covariance_matrix)
            cholesky_tril = la.cholesky(covariance_matrix)
        elif precision_matrix is not None:
            mean, precision_matrix = promote_shapes(mean, precision_matrix)
            cholesky_tril = cholesky_inverse(precision_matrix)
        else:
            mean, cholesky_tril = promote_shapes(mean, cholesky_tril)

        batch_shape = broadcast_shapes(
            np.shape(mean)[:-2], np.shape(cholesky_tril)[:-2]
        )
        rv_shape = np.shape(cholesky_tril)[-1:]

        mean = np.broadcast_to(np.squeeze(mean, axis=-1), batch_shape + rv_shape)

        return _MVNMatrix(
            mean=mean,
            cholesky_tril=cholesky_tril,
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

    super().__init__(
        batch_shape=batch_shape,
        rv_shape=rv_shape,
        check_parameters=check_parameters,
        check_support=check_support,
    )


class MultivariateT(Distribution):
    _constraints: Dict[str, Constraint] = {}
    _support: Constraint = None

    def __init__(self, check_parameters: bool = True, check_support: bool = True):
        pass

    @property
    def mean(self) -> Parameter:
        pass

    @property
    def variance(self) -> Parameter:
        pass

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        pass


class Wishart(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {}
    _support: Constraint = None

    def __init__(self, check_parameters: bool = True, check_support: bool = True):
        pass

    @property
    def mean(self) -> Parameter:
        pass

    @property
    def variance(self) -> Parameter:
        pass

    def _log_prob(self, x: Variate) -> ArrayLike:
        pass

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        pass

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        pass

    @property
    def log_normalizer(self) -> Parameter:
        pass

    def base_measure(self, x: Variate) -> ArrayLike:
        pass

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        pass
