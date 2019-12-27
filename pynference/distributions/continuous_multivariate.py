from typing import Dict, Tuple
from warnings import warn

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

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.concentration = concentration
        self.concentration_sum = np.sum(concentration, axis=-1, keepdims=True)

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


class _MVNScalar(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {
        "_mean": real_vector,
        "_precision": positive,
        "_std": positive,
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
        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self._mean = mean
        self._precision = precision
        self._std = std

    @property
    def mean(self) -> Parameter:
        return self._mean

    @property
    def variance(self) -> Parameter:
        return np.reciprocal(self._precision)

    @property
    def precision(self) -> Parameter:
        return self._precision

    @property
    def covariance_matrix(self) -> Parameter:
        replicated_variance = replicate_along_last_axis(self.variance, self.rv_shape)
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
        return self._mean + self._std[..., np.newaxis] * epsilon

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        warn(
            "The natural parameter for a scalar*eye covariance matrix is not very efficient. "
            "Currently, it builds the entire covariance matrix in memory. Consider optimization.",
            RuntimeWarning,
        )
        precision = self.precision_matrix

        # The matrix is vectorized so that the Frobenius product can be written
        # as an ordinary dot product.
        return (
            np.sum(precision * self._mean[..., np.newaxis], axis=-1),
            (-0.5 * precision).reshape(
                self.batch_shape + (self.rv_shape[0] * self.rv_shape[0],)
            ),
        )

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

        batch_outer = x[..., np.newaxis] @ np.swapaxes(x[..., np.newaxis], -2, -1)

        # The matrix is vectorized so that the Frobenius product can be written
        # as an ordinary dot product.
        return (
            x,
            batch_outer.reshape(
                (-1,) + self.batch_shape + (self.rv_shape[0] * self.rv_shape[0],)
            ),
        )


class _MVNVector(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {
        "_mean": real_vector,
        "_precision_diag": positive_vector,
        "_std_diag": positive_vector,
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
        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self._mean = mean
        self._precision_diag = precision_diag
        self._std_diag = std_diag

    @property
    def mean(self) -> Parameter:
        return self._mean

    @property
    def variance(self) -> Parameter:
        return np.reciprocal(self._precision_diag)

    @property
    def precision(self) -> Parameter:
        return self._precision_diag

    @property
    def covariance_matrix(self) -> Parameter:
        return arraywise_diagonal(self.variance)

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
        warn(
            "The natural parameter for a scalar*eye covariance matrix is not very efficient. "
            "Currently, it builds the entire covariance matrix in memory. Consider optimization.",
            RuntimeWarning,
        )
        precision = self.precision_matrix

        # The matrix is vectorized so that the Frobenius product can be written
        # as an ordinary dot product.
        return (
            np.sum(precision * self._mean[..., np.newaxis], axis=-1),
            (-0.5 * precision).reshape(
                self.batch_shape + (self.rv_shape[0] * self.rv_shape[0],)
            ),
        )

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

        batch_outer = x[..., np.newaxis] @ np.swapaxes(x[..., np.newaxis], -2, -1)

        # The matrix is vectorized so that the Frobenius product can be written
        # as an ordinary dot product.
        return (
            x,
            batch_outer.reshape(
                (-1,) + self.batch_shape + (self.rv_shape[0] * self.rv_shape[0],)
            ),
        )


def _half_log_det(cholesky_tril: np.ndarray):
    return np.sum(np.log(np.diagonal(cholesky_tril, axis1=-2, axis2=-1)), axis=-1)


def _mahalanobis_squared(
    centered_x: Variate, cholesky_tril: np.ndarray, batch_shape: Shape
):
    # Source: https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/continuous.py.
    # This procedure handles the case:
    # cholesky_tril.shape = (i, 1, n, n), centered_x.shape = (i, j, n),
    # because we do not want to broadcast cholesky_tril to the shape (i, j, n, n).

    # Assume that cholesky_tril.shape = (i, 1, n, n), centered_x.shape = (..., i, j, n),
    # we are going to make centered_x have shape (..., 1, j,  i, 1, n) to apply batched tril_solve.
    sample_ndim = (
        np.ndim(centered_x) - np.ndim(cholesky_tril) + 1
    )  # size of sample_shape
    out_shape = np.shape(centered_x)[:-1]  # shape of output
    # Reshape centered_x with the shape (..., 1, i, j, 1, n).
    centered_x_new_shape = out_shape[:sample_ndim]

    for (sL, sx) in zip(cholesky_tril.shape[:-2], out_shape[sample_ndim:]):
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
    xt = np.reshape(centered_x, (-1,) + cholesky_tril.shape[:-1])
    # Permute to (i, 1, n, -1).
    xt = np.moveaxis(xt, 0, -1)

    if batch_shape == ():
        solved_triangular = solve_triangular(
            cholesky_tril, xt, lower=True
        )  # shape: (i, 1, n, -1)
    else:
        solved_triangular = la.solve(cholesky_tril, xt)  # shape: (i, 1, n, -1)

    M = np.sum(solved_triangular ** 2, axis=-2)  # shape: (i, 1, -1)
    # Permute back to (-1, i, 1).
    M = np.moveaxis(M, -1, 0)
    # Reshape back to (..., 1, j, i, 1).
    M = np.reshape(M, centered_x.shape[:-1])
    # Permute back to (..., 1, i, j, 1).
    permute_inv_dims = tuple(range(sample_ndim))

    for i in range(cholesky_tril.ndim - 2):
        permute_inv_dims += (sample_ndim + i, len(out_shape) + i)

    M = np.transpose(M, permute_inv_dims)
    return np.reshape(M, out_shape)


class _MVNMatrix(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {
        "_mean": real_vector,
        "_cholesky_tril": lower_cholesky,
    }
    _support: Constraint = real_vector

    def __init__(
        self,
        mean: Parameter,
        precision_matrix: Parameter,
        cholesky_tril: Parameter,
        batch_shape: Shape,
        rv_shape: Shape,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self._mean = mean
        self._precision_matrix = precision_matrix
        self._cholesky_tril = cholesky_tril

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
        cholesky_tril_T = np.swapaxes(self._cholesky_tril, -1, -2)
        return self._cholesky_tril @ cholesky_tril_T

    @property
    def precision_matrix(self) -> Parameter:
        return self._precision_matrix

    def _log_prob(self, x: Variate) -> ArrayLike:
        half_log_det = _half_log_det(self._cholesky_tril)
        mahalanobis_squared = _mahalanobis_squared(
            x - self._mean, self._cholesky_tril, self.batch_shape
        )

        normalizer = half_log_det + 0.5 * self.rv_shape[0] * np.log(2.0 * np.pi)
        return -0.5 * mahalanobis_squared - normalizer

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        epsilon = random_state.standard_normal(
            sample_shape + self.batch_shape + self.rv_shape
        )
        return self._mean + np.squeeze(
            self._cholesky_tril @ epsilon[..., np.newaxis], axis=-1
        )

    @property
    def _natural_parameter(self) -> Tuple[Parameter, ...]:
        precision = self.precision_matrix

        # The matrix is vectorized so that the Frobenius product can be written
        # as an ordinary dot product.
        return (
            np.sum(precision * self._mean[..., np.newaxis], axis=-1),
            (-0.5 * precision).reshape(
                self.batch_shape + (self.rv_shape[0] * self.rv_shape[0],)
            ),
        )

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        precision = self.precision_matrix
        covariance = self.covariance_matrix
        precision_mean = la.solve(covariance, self._mean)

        # The matrix is vectorized so that the Frobenius product can be written
        # as an ordinary dot product.
        return (
            precision_mean,
            (-0.5 * precision).reshape(
                self.batch_shape + (self.rv_shape[0] * self.rv_shape[0],)
            ),
        )

    @property
    def log_normalizer(self) -> Parameter:
        half_log_det = _half_log_det(self._cholesky_tril)
        mahalanobis_squared = _mahalanobis_squared(
            self._mean, self._cholesky_tril, self.batch_shape
        )

        return 0.5 * mahalanobis_squared + half_log_det

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.power(2.0 * np.pi, -self.rv_shape[0] / 2.0)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        if np.isscalar(x):
            x = np.expand_dims(x, axis=-1)

        batch_outer = x[..., np.newaxis] @ np.swapaxes(x[..., np.newaxis], -2, -1)

        # The matrix is vectorized so that the Frobenius product can be written
        # as an ordinary dot product.
        return (
            x,
            batch_outer.reshape(
                (-1,) + self.batch_shape + (self.rv_shape[0] * self.rv_shape[0],)
            ),
        )


def _precision2cholesky(precision_matrix: np.ndarray, batch_shape: Shape) -> np.ndarray:
    if batch_shape == ():
        tril_inv = np.swapaxes(
            la.cholesky(precision_matrix[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
        )
        identity = np.broadcast_to(
            np.identity(precision_matrix.shape[-1]), tril_inv.shape
        )

        return solve_triangular(tril_inv, identity, lower=True)
    else:
        identity = np.broadcast_to(
            np.identity(precision_matrix.shape[-1]), precision_matrix.shape
        )
        inv = la.solve(precision_matrix, identity)
        return la.cholesky(inv)


def _cholesky2precision(cholesky_tril: np.ndarray, batch_shape: Shape) -> np.ndarray:
    if batch_shape == ():
        identity = np.identity(cholesky_tril.shape[-1])
        cholesky_inv = solve_triangular(cholesky_tril, identity, lower=True)
    else:
        cholesky_inv = la.inv(cholesky_tril)

    cholesky_inv_T = np.swapaxes(cholesky_inv, -2, -1)
    return cholesky_inv_T @ cholesky_inv


# Manually checking constraints within the `MultivariateNormal` function
# since it cannot refer to the `Distribution` ABC before initializing one.
# We need to check the parameters before calculating the Cholesky factors.
def _check_constraint(
    constraint: Constraint, parameter: str, parameter_value: Parameter
):
    if not np.all(constraint(parameter_value)):
        raise ValueError(
            f"Invalid value for {parameter}: {parameter_value}. "
            f"The parameter must satisfy the constraint '{constraint}'."
        )


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
            if check_parameters:
                _check_constraint(positive, "variance", variance)

            precision = np.reciprocal(variance)
            std = np.sqrt(variance)
        else:
            if check_parameters:
                _check_constraint(positive, "precision", precision)

            std = np.reciprocal(np.sqrt(precision))

        batch_shape = broadcast_shapes(np.shape(mean)[:-1], np.shape(precision))
        rv_shape = np.shape(mean)[-1:]

        mean = np.broadcast_to(mean, batch_shape + rv_shape)
        precision = np.broadcast_to(precision, batch_shape)

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
            if check_parameters:
                _check_constraint(positive_vector, "variance_diag", variance_diag)

            precision_diag = np.reciprocal(variance_diag)
            std_diag = np.sqrt(variance_diag)
        else:
            if check_parameters:
                _check_constraint(positive_vector, "precision_diag", precision_diag)

            std_diag = np.reciprocal(np.sqrt(precision_diag))

        batch_shape = broadcast_shapes(
            np.shape(mean)[:-1], np.shape(precision_diag)[:-1]
        )
        rv_shape = np.shape(precision_diag)[-1:]

        mean, precision_diag = promote_shapes(mean, precision_diag)

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
            if check_parameters:
                _check_constraint(
                    positive_definite, "covariance_matrix", covariance_matrix
                )

            cholesky_tril = la.cholesky(covariance_matrix)

        if precision_matrix is not None:
            if check_parameters:
                _check_constraint(
                    positive_definite, "precision_matrix", precision_matrix
                )

            mean, precision_matrix = promote_shapes(mean, precision_matrix)

            batch_shape = broadcast_shapes(
                np.shape(mean)[:-2], np.shape(precision_matrix)[:-2]
            )

            cholesky_tril = _precision2cholesky(precision_matrix, batch_shape)
        else:
            if check_parameters:
                _check_constraint(lower_cholesky, "cholesky_tril", cholesky_tril)

            mean, cholesky_tril = promote_shapes(mean, cholesky_tril)

            batch_shape = broadcast_shapes(
                np.shape(mean)[:-2], np.shape(cholesky_tril)[:-2]
            )

            precision_matrix = _cholesky2precision(cholesky_tril, batch_shape)

        rv_shape = np.shape(cholesky_tril)[-1:]
        mean = np.squeeze(mean, axis=-1)

        return _MVNMatrix(
            mean=mean,
            precision_matrix=precision_matrix,
            cholesky_tril=cholesky_tril,
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )


class MultivariateT(Distribution):
    _constraints: Dict[str, Constraint] = {
        "df": positive,
        "loc": real_vector,
        "_cholesky_tril": lower_cholesky,
    }
    _support: Constraint = real_vector

    def __init__(
        self,
        df: Parameter,
        loc: Parameter,
        scale: Parameter = None,
        cholesky_tril: Parameter = None,
        check_parameters: bool = True,
        check_support: bool = True,
    ):
        if (scale is not None) + (cholesky_tril is not None) != 1:
            raise ValueError(
                "Provide either the scale matrix or its lower"
                "triangular Cholesky decomposition."
            )

        if np.isscalar(loc):
            loc = np.expand_dims(loc, axis=-1)

        loc = loc[..., np.newaxis]

        if scale is not None:
            cholesky_tril = la.cholesky(scale)

        loc, cholesky_tril = promote_shapes(loc, cholesky_tril)

        batch_shape = broadcast_shapes(
            np.shape(df), np.shape(loc)[:-2], np.shape(cholesky_tril)[:-2]
        )
        rv_shape = np.shape(cholesky_tril)[-1:]
        loc = np.squeeze(loc, axis=-1)

        super().__init__(
            batch_shape=batch_shape,
            rv_shape=rv_shape,
            check_parameters=check_parameters,
            check_support=check_support,
        )

        self.df = np.broadcast_to(df, batch_shape)
        self.loc = loc
        self._cholesky_tril = cholesky_tril

    @property
    def mean(self) -> Parameter:
        return np.where(self.df > 1, self.loc, np.nan)

    @property
    def variance(self) -> Parameter:
        scale = self._cholesky_tril @ np.swapaxes(self._cholesky_tril, -2, -1)
        variance = np.where(
            self.df > 2, np.square(scale) * self.df / (self.df - 2.0), np.inf
        )
        return np.where(self.df > 1, variance, np.nan)

    def _log_prob(self, x: Variate) -> ArrayLike:
        p = self.rv_shape[0]
        half_log_det = _half_log_det(self._cholesky_tril)

        normalizer = (
            gammaln((self.df + p) / 2.0)
            - gammaln(self.df / 2.0)
            - (p / 2.0) * (np.log(self.df) + np.log(np.pi))
            - half_log_det
        )

        mahalanobis_squared = _mahalanobis_squared(
            x - self.loc, self._cholesky_tril, self.batch_shape
        )

        return normalizer - ((self.df + p) / 2.0) * np.log1p(
            mahalanobis_squared / self.df
        )

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        std_norm = random_state.standard_normal(
            sample_shape + self.batch_shape + self.rv_shape
        )
        norm = np.squeeze(self._cholesky_tril @ std_norm[..., np.newaxis], axis=-1)
        chi2 = random_state.chisquare(df=self.df, size=sample_shape + self.batch_shape)
        epsilon = norm / chi2[..., np.newaxis]
        return np.sqrt(self.df) * epsilon + self.loc


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
