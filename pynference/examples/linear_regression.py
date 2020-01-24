"""
isort:skip
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

import matplotlib.pyplot as plt
import numpy as np

from pynference.distributions import InverseGamma, MultivariateNormal
from pynference.inference import Metropolis, init_to_uniform
from pynference.infrastructure import sample
from pynference.utils import check_random_state


def model(X, y, a0, b0, Sigma0):
    sigma2 = sample("sigma2", InverseGamma(shape=a0, scale=b0))
    beta = sample(
        "beta",
        MultivariateNormal(mean=np.zeros(Sigma0.shape[0]), covariance_matrix=Sigma0),
    )

    y_mean = X @ beta
    sample("y", MultivariateNormal(mean=y_mean, variance=sigma2), observation=y)


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

    n_samples = 10000
    proposal = "normal"
    scale_init = 0.05
    init = init_to_uniform()
    tune = True

    mcmc = Metropolis(
        model=model,
        n_samples=n_samples,
        proposal=proposal,
        scale_init=scale_init,
        init=init,
        tune=tune,
        random_state=random_state,
    )
    samples = mcmc.run(X=X, y=y, a0=a0, b0=b0, Sigma0=Sigma0)

    sigma2s = np.zeros(shape=n_samples)
    betas = np.zeros(shape=(n_samples, 2))

    for i, theta in enumerate(samples):
        sigma2s[i] = theta["sigma2"]
        betas[i] = theta["beta"]

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
