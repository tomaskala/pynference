from enum import Enum, auto
from typing import Dict, Tuple

import numpy as np
import scipy.linalg as la
from numpy.random import RandomState
from scipy.special import gammaln

from pynference.constants import ArrayLike, Parameter, Shape, Variate
from pynference.distributions.constraints import Constraint, positive, simplex
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


class Dirichlet(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {"concentration": positive}
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


class ScaleType(Enum):
    SCALAR = auto()
    VECTOR = auto()
    MATRIX = auto()


class MultivariateNormal(ExponentialFamily):
    _constraints: Dict[str, Constraint] = {}
    _support: Constraint = None

    def __init__(
        self,
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
            self._scale_type = ScaleType.SCALAR

            if variance is not None:
                precision = np.reciprocal(variance)

            batch_shape = broadcast_shapes(np.shape(mean)[:-1], np.shape(precision))
            rv_shape = np.shape(mean)[-1:]

            self._mean = np.broadcast_to(mean, batch_shape + rv_shape)
            self._precision = np.broadcast_to(precision, batch_shape + rv_shape)
        elif variance_diag is not None or precision_diag is not None:
            self._scale_type = ScaleType.VECTOR

            if variance_diag is not None:
                precision_diag = np.reciprocal(variance_diag)

            mean, precision_diag = promote_shapes(mean, precision_diag)

            batch_shape = broadcast_shapes(
                np.shape(mean)[:-1], np.shape(precision_diag)[:-1]
            )
            rv_shape = np.shape(precision_diag)[-1:]

            self._mean = np.broadcast_to(mean, batch_shape + rv_shape)
            self._precision_diag = np.broadcast_to(
                precision_diag, batch_shape + rv_shape
            )
        else:
            self._scale_type = ScaleType.MATRIX
            mean = mean[..., np.newaxis]

            if covariance_matrix is not None:
                mean, covariance_matrix = promote_shapes(mean, covariance_matrix)
                self._cholesky_tril = la.cholesky(covariance_matrix)
            elif precision_matrix is not None:
                mean, precision_matrix = promote_shapes(mean, precision_matrix)
                self._cholesky_tril = cholesky_inverse(precision_matrix)
            else:
                mean, self._cholesky_tril = promote_shapes(mean, cholesky_tril)

            batch_shape = broadcast_shapes(
                np.shape(mean)[:-2], np.shape(self._cholesky_tril)[:-2]
            )
            rv_shape = np.shape(self._cholesky_tril)[-1:]

            self._mean = np.broadcast_to(
                np.squeeze(mean, axis=-1), batch_shape + rv_shape
            )

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
    def variance(self) -> Parameter:  # TODO: Return variance matrix diagonal.
        if self._scale_type == ScaleType.SCALAR:
            pass
        elif self._scale_type == ScaleType.VECTOR:
            pass
        else:
            pass

    @property
    def precision(self) -> Parameter:  # TODO: Return precision matrix diagonal.
        if self._scale_type == ScaleType.SCALAR:
            pass
        elif self._scale_type == ScaleType.VECTOR:
            pass
        else:
            pass

    @property
    def covariance_matrix(self) -> Parameter:
        if self._scale_type == ScaleType.SCALAR:
            variance = np.reciprocal(self._precision)
            replicated_variance = replicate_along_last_axis(variance, self.rv_shape)
            return arraywise_diagonal(replicated_variance)
        elif self._scale_type == ScaleType.VECTOR:
            variance_diag = np.reciprocal(self._precision_diag)
            return arraywise_diagonal(variance_diag)
        else:
            return self._cholesky_tril @ self._cholesky_tril.T

    @property
    def precision_matrix(self) -> Parameter:
        if self._scale_type == ScaleType.SCALAR:
            replicated_precision = replicate_along_last_axis(
                self._precision, self.rv_shape
            )
            return arraywise_diagonal(replicated_precision)
        elif self._scale_type == ScaleType.VECTOR:
            return arraywise_diagonal(self._precision_diag)
        else:
            cholesky_tril_inv = la.inv(self._cholesky_tril)
            return cholesky_tril_inv.T @ cholesky_tril_inv

    def _log_prob(self, x: Variate) -> ArrayLike:
        if self._scale_type == ScaleType.SCALAR:
            pass
        elif self._scale_type == ScaleType.VECTOR:
            pass
        else:
            pass

    def _sample(self, sample_shape: Shape, random_state: RandomState) -> Variate:
        if self._scale_type == ScaleType.SCALAR:
            pass
        elif self._scale_type == ScaleType.VECTOR:
            pass
        else:
            pass

    @property
    def natural_parameter(self) -> Tuple[Parameter, ...]:
        pass

    @property
    def log_normalizer(self) -> Parameter:
        pass

    def base_measure(self, x: Variate) -> ArrayLike:
        return np.power(2.0 * np.pi, -self.rv_shape[0] / 2.0)

    def sufficient_statistic(self, x: Variate) -> Tuple[ArrayLike, ...]:
        return x, np.outer(x, x)


class MultivariateT(Distribution):
    _constraints: Dict[str, Constraint] = {}
    _support: Constraint = None

    def __init__(
        self, check_parameters: bool = True, check_support: bool = True,
    ):
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

    def __init__(
        self, check_parameters: bool = True, check_support: bool = True,
    ):
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
