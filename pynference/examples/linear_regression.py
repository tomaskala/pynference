"""
isort:skip
"""
import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

from pynference.inference import Metropolis
from pynference.infrastructure import sample


def model(X, y, a0, b0, Sigma0):
    sigma2 = sample("sigma2", dist.Gamma(concentration=a0, rate=b0))
    beta = sample(
        "beta",
        dist.MultivariateNormal(
            loc=torch.zeros(Sigma0.shape[0]), covariance_matrix=Sigma0
        ),
    )
    sample(
        "y",
        dist.MultivariateNormal(
            loc=X @ beta, precision_matrix=sigma2 * torch.eye(X.shape[0])
        ),
        observation=y,
    )


def main():
    torch.manual_seed(123)

    n = 100
    X = torch.ones(size=(n, 2))
    X[:, 1] = torch.normal(mean=torch.zeros(n), std=2.0)

    sigma2_true = 0.5
    beta_true = torch.tensor([2.0, 1.0])

    eps = torch.normal(mean=torch.zeros(n), std=math.sqrt(sigma2_true))
    y = X @ beta_true + eps

    a0 = 2.0
    b0 = 1.0
    Sigma0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    n_samples = 10000
    proposal = "normal"
    scale_init = 0.05
    init_strategy = "uniform"
    tune = True

    mcmc = Metropolis(
        model=model,
        n_samples=n_samples,
        proposal=proposal,
        scale_init=scale_init,
        init_strategy=init_strategy,
        tune=tune,
    )
    samples = mcmc.run(X=X, y=y, a0=a0, b0=b0, Sigma0=Sigma0)

    sigma2s = torch.zeros(size=(n_samples,))
    betas = torch.zeros(size=(n_samples, 2))

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
    print(torch.mean(sigma2s))
    print("mean(betas)")
    print(torch.mean(betas, axis=0))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
