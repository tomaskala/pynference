"""
isort:skip
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

import matplotlib.pyplot as plt
import numpy as np

from pynference.constants import ArrayLike, Sample
from pynference.distributions import MultivariateNormal
from pynference.inference.metropolis import Metropolis
from pynference.model import Model
from pynference.utils import check_random_state


class LinearRegression(Model):
    def __init__(self, X, y, variance, Sigma0, random_state):
        self.X = X
        self.y = y
        self.variance = variance
        self.Sigma0 = Sigma0
        self.random_state = check_random_state(random_state)

    def log_prob(self, theta: Sample) -> ArrayLike:
        beta = theta["beta"]
        y_mean = self.X @ beta
        p_beta = MultivariateNormal(
            mean=np.zeros(self.Sigma0.shape[0]), covariance_matrix=self.Sigma0
        )
        p_y = MultivariateNormal(mean=y_mean, variance=self.variance)
        return np.sum(p_y.log_prob(self.y)) + np.sum(p_beta.log_prob(beta))

    def sample(self) -> Sample:
        p_beta = MultivariateNormal(
            mean=np.zeros(self.Sigma0.shape[0]), covariance_matrix=self.Sigma0
        )
        return {"beta": p_beta.sample(random_state=self.random_state)}


def main():
    random_state = check_random_state(123)
    variance = 0.5
    n = 100
    X = np.ones(shape=(n, 2))
    X[:, 1] = random_state.normal(scale=2.0)
    beta_true = np.array([2.0, 1.0])
    y = X @ beta_true + random_state.normal(scale=np.sqrt(variance), size=X.shape[0])
    Sigma0 = np.array([[1.0, 0.0], [0.0, 1.0]])

    model = LinearRegression(X, y, variance, Sigma0, random_state)

    n_samples = 10000
    proposal = "normal"
    scale_init = 1.0
    tune = True

    metropolis = Metropolis(
        model=model,
        n_samples=n_samples,
        proposal=proposal,
        scale_init=scale_init,
        tune=tune,
        random_state=random_state,
    )
    samples = metropolis.run()

    betas = np.empty(shape=(n_samples, 2))

    for i, sample in enumerate(samples):
        betas[i] = sample["beta"]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 10))
    ax1.set_title(r"$\beta_0$")
    ax1.plot(betas[:, 0])
    ax1.axhline(beta_true[0], color="red")

    ax2.set_title(r"$\beta_1$")
    ax2.plot(betas[:, 1])
    ax2.axhline(beta_true[1], color="red")

    print(np.mean(betas, axis=0))

    # plt.show()


if __name__ == "__main__":
    main()
