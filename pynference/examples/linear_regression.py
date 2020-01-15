"""
isort:skip
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState

from pynference.constants import ArrayLike, Sample, Shape
from pynference.distributions import InverseGamma, MultivariateNormal
from pynference.distributions.constraints import Constraint, positive, real_vector
from pynference.inference import (
    Metropolis,
    init_to_mean,
    init_to_prior,
    init_to_uniform,
)
from pynference.model.model import Model
from pynference.utils import check_random_state


class LinearRegression(Model):
    def __init__(self, X, y, a0, b0, Sigma0):
        self.X = X
        self.y = y
        self.a0 = a0
        self.b0 = b0
        self.Sigma0 = Sigma0

    @property
    def constraints(self) -> Dict[str, Constraint]:
        return {"beta": real_vector, "sigma2": positive}

    def log_prob(self, theta: Sample) -> ArrayLike:
        beta = theta["beta"]
        sigma2 = theta["sigma2"]

        y_mean = self.X @ beta

        p_sigma2 = InverseGamma(shape=self.a0, scale=self.b0)

        p_beta = MultivariateNormal(
            mean=np.zeros(self.Sigma0.shape[0]), covariance_matrix=self.Sigma0
        )

        p_y = MultivariateNormal(mean=y_mean, variance=sigma2)

        return (
            np.sum(p_y.log_prob(self.y))
            + np.sum(p_beta.log_prob(beta))
            + np.sum(p_sigma2.log_prob(sigma2))
        )

    def sample(self, sample_shape: Shape, random_state: RandomState) -> Sample:
        p_sigma2 = InverseGamma(shape=self.a0, scale=self.b0)

        p_beta = MultivariateNormal(
            mean=np.zeros(self.Sigma0.shape[0]), covariance_matrix=self.Sigma0
        )

        return {
            "beta": p_beta.sample(sample_shape=sample_shape, random_state=random_state),
            "sigma2": p_sigma2.sample(
                sample_shape=sample_shape, random_state=random_state
            ),
        }


def main():
    random_state = check_random_state(123)

    n = 100
    X = np.ones(shape=(n, 2))
    X[:, 1] = random_state.normal(scale=2.0, size=n)

    sigma2_true = 0.5
    beta_true = np.array([2.0, 1.0])

    y = X @ beta_true + random_state.normal(scale=np.sqrt(sigma2_true), size=n)

    a0 = 2.0
    b0 = 1.0
    Sigma0 = np.array([[1.0, 0.0], [0.0, 1.0]])

    model = LinearRegression(X, y, a0, b0, Sigma0)

    n_samples = 10000
    proposal = "normal"
    scale_init = 0.05
    init = init_to_uniform()
    tune = True

    metropolis = Metropolis(
        model=model,
        n_samples=n_samples,
        proposal=proposal,
        scale_init=scale_init,
        init=init,
        tune=tune,
        random_state=random_state,
    )
    samples = metropolis.run()

    sigma2s = np.zeros(shape=n_samples)
    betas = np.zeros(shape=(n_samples, 2))

    for i, sample in enumerate(samples):
        sigma2s[i] = sample["sigma2"]
        betas[i] = sample["beta"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 6))
    ax1.set_title(r"$\beta_0$")
    ax1.plot(betas[:, 0])
    ax1.axhline(beta_true[0], color="red")

    ax2.set_title(r"$\beta_1$")
    ax2.plot(betas[:, 1])
    ax2.axhline(beta_true[1], color="red")

    ax3.set_title(r"$\sigma^2$")
    ax3.plot(sigma2s)
    ax3.axhline(sigma2_true, color="red")

    print("mean(sigma^2)")
    print(np.mean(sigma2s))
    print("mean(betas)")
    print(np.mean(betas, axis=0))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
