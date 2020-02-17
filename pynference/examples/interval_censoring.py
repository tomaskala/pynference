import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

import matplotlib.pyplot as plt  # noqa E402
import pandas as pd  # noqa E402
import pyreadr  # noqa E402
import torch  # noqa E402

import pynference.distributions as dist  # noqa E402
from pynference.inference import Metropolis  # noqa E402
from pynference.infrastructure import sample  # noqa E402


def model(X, logL, logU, hypers):
    beta0 = hypers["beta0"]
    Sigma0 = hypers["Sigma0"]
    a0 = hypers["a0"]
    b0 = hypers["b0"]

    beta = sample("beta", dist.MultivariateNormal(loc=beta0, covariance_matrix=Sigma0))
    tau = sample("tau", dist.Gamma(concentration=a0, rate=b0))

    logT = sample(
        "logT",
        dist.TruncatedNormal(
            loc=X @ beta, scale=1.0 / torch.sqrt(tau), low=logL, high=logU
        ),
    )
    assert not torch.isinf(logT).any()
    assert not torch.isnan(logT).any()
    return logT


def load_dataframe(df_path: str, which: str) -> pd.DataFrame:
    # See tandmob_analysis.ipynb for more details on data processing.
    r_data = pyreadr.read_r(df_path)
    data1 = r_data["Data1"]  # Misclassifications.
    data2 = r_data["Data2"]  # Regressors.

    data1["STATUS"] = data1["STATUS"].astype(float)
    data1[["IDNR", "TOOTH"]] = data1["IdTooth"].str.split("_", expand=True)
    data1["IDNR"] = data1["IDNR"].astype(int)
    data1["TOOTH"] = data1["TOOTH"].astype(int)
    data1["EXAMINER"] = data1["EXAMINER"].astype(int)
    data1.drop("IdTooth", axis="columns", inplace=True)
    data1 = data1[["IDNR", "TOOTH", "VISIT", "EXAMINER", "STATUS"]]

    data2["IDNR"] = data2["IDNR"].astype(int)
    data2["TOOTH"] = data2["TOOTH"].astype(int)
    data2.loc[data2["FBEG"].isna(), "FBEG"] = 0.0
    data2.loc[data2["FEND"].isna(), "FEND"] = float("inf")

    if which == "misclassifications":
        return data1
    elif which == "regressors":
        return data2
    elif which == "both":
        return data1, data2

    data_merged = pd.merge(
        data1, data2, how="inner", on=["IDNR", "TOOTH"], validate="many_to_one"
    )

    if which == "merged":
        return data_merged
    elif which == "all":
        return data1, data2, data_merged
    else:
        raise ValueError("Specify which dataset you want.")


def main():
    torch.set_default_dtype(torch.double)
    df_path = str(Path(__file__).parent / Path("./data/Data_20130610.RData"))
    df = load_dataframe(df_path, which="regressors")

    # TODO: Random intercept.

    # Load regressors.
    regressors = ["GIRL", "SEAL", "FREQ.BR"]
    p = len(regressors)
    X = torch.from_numpy(df[regressors].values)

    # Load interval censoring bounds.
    logL = torch.log(torch.from_numpy(df["FBEG"].values))
    logU = torch.log(torch.from_numpy(df["FEND"].values))

    # Set-up priors.
    beta0 = torch.zeros((p,))
    Sigma0 = 1000.0 * torch.eye(p)
    a0 = 1.0
    b0 = 0.005
    hypers = {"beta0": beta0, "Sigma0": Sigma0, "a0": a0, "b0": b0}

    # Run inference.
    n_samples = 10000
    proposal = "normal"
    proposal_scale = 0.05
    init_strategy = "uniform"
    tune = True

    mcmc = Metropolis(
        model=model,
        n_samples=n_samples,
        proposal=proposal,
        proposal_scale=proposal_scale,
        init_strategy=init_strategy,
        tune=tune,
    )
    samples = mcmc.run(X=X, logL=logL, logU=logU, hypers=hypers)

    beta_samples = torch.zeros((n_samples, p))
    tau_samples = torch.zeros((n_samples,))
    logT_samples = torch.zeros((n_samples,))

    for i, theta in enumerate(samples):
        beta_samples[i] = theta["beta"]
        tau_samples[i] = theta["tau"]
        logT_samples[i] = theta["logT"]

    for i in range(p):
        fig, ax = plt.subplots()
        ax.set_title(r"$\beta_{}$ corresponding to {}".format(i + 1, regressors[i]))
        ax.plot(beta_samples[:, i])
        plt.show()

    fig, ax = plt.subplots()
    ax.set_title(r"$\tau$")
    ax.plot(tau_samples)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title(r"$\log{T}$")
    ax.plot(logT_samples)
    plt.show()


if __name__ == "__main__":
    main()
