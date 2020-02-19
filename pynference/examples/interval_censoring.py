import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

import matplotlib.pyplot as plt  # noqa E402
import pandas as pd  # noqa E402
import pyreadr  # noqa E402
import torch  # noqa E402

import pynference.distributions as dist  # noqa E402
from pynference.inference import Metropolis  # noqa E402
from pynference.infrastructure import sample  # noqa E402


def model(X, logL, logU, hypers, n_subjects, n_units_per_subject):
    # Fixed effects: beta ~ N(m_beta, V_beta).
    m_beta = hypers["m_beta"]
    V_beta = hypers["V_beta"]
    beta = sample("beta", dist.MultivariateNormal(loc=m_beta, covariance_matrix=V_beta))

    # Noise term: sigma2_eps_inv ~ Gamma(nu_eps1, nu_eps2).
    nu_eps1 = hypers["nu_eps1"]
    nu_eps2 = hypers["nu_eps2"]
    sigma2_eps_inv = sample("sigma2_eps_inv", dist.Gamma(concentration=nu_eps1, rate=nu_eps2))

    # Random effects mean: mu ~ N(m_mu, s2_mu).
    m_mu = hypers["m_mu"]
    s2_mu = hypers["s2_mu"]
    mu = sample("mu", dist.Normal(loc=m_mu, scale=math.sqrt(s2_mu)))

    # Random effects variance: tau2_inv ~ Gamma(nu_tau1, nu_tau2).
    nu_tau1 = hypers["nu_tau1"]
    nu_tau2 = hypers["nu_tau2"]
    tau2_inv = sample("tau2_inv", dist.Gamma(concentration=nu_tau1, rate=nu_tau2))

    # Random effect: b ~ N(mu, tau2_inv^{-1}).
    mu = mu * torch.ones((n_subjects,))
    tau2_inv = tau2_inv * torch.ones((n_subjects,))
    b = sample("b", dist.Normal(loc=mu, scale=tau2_inv.sqrt().reciprocal()))
    b = torch.repeat_interleave(b, n_units_per_subject)

    # Latent log(event times): logT ~ TN(X @ beta + b, sigma2_eps_inv^{-1}, logL, logU).
    logT = sample(
        "logT",
        dist.TruncatedNormal(
            loc=X @ beta + b, scale=sigma2_eps_inv.sqrt().reciprocal(), low=logL, high=logU
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
    data1.sort_values(by=["IDNR", "TOOTH"], inplace=True)

    data2["IDNR"] = data2["IDNR"].astype(int)
    data2["TOOTH"] = data2["TOOTH"].astype(int)
    data2.loc[data2["FBEG"].isna(), "FBEG"] = 0.0
    data2.loc[data2["FEND"].isna(), "FEND"] = float("inf")
    data2.sort_values(by=["IDNR", "TOOTH"], inplace=True)

    # Drop subjects for which we observe fewer than 4 teeth.
    grouped_by_subject = data2.groupby("IDNR").count()
    not_4_teeth = grouped_by_subject[grouped_by_subject["TOOTH"] != 4].index

    data1.drop(data1[data1["IDNR"].isin(not_4_teeth)].index, inplace=True)
    data2.drop(data2[data2["IDNR"].isin(not_4_teeth)].index, inplace=True)

    if which == "misclassifications":
        return data1
    elif which == "regressors":
        return data2
    elif which == "both":
        return data1, data2

    data_merged = pd.merge(
        data1, data2, how="inner", on=["IDNR", "TOOTH"], validate="many_to_one"
    ).sort_values(by=["IDNR", "TOOTH"])

    if which == "merged":
        return data_merged
    elif which == "all":
        return data1, data2, data_merged
    else:
        raise ValueError("Specify which dataset you want.")


def main():
    torch.manual_seed(941026)
    torch.set_default_dtype(torch.double)
    df_path = str(Path(__file__).parent / Path("./data/Data_20130610.RData"))
    df = load_dataframe(df_path, which="regressors")

    n_subjects = df["IDNR"].nunique()
    n_units_per_subject = 4  # We observe 4 teeth on each subject.

    # Load regressors.
    regressors = ["GIRL", "SEAL", "FREQ.BR"]
    p = len(regressors)
    X = torch.from_numpy(df[regressors].values)

    # Load interval censoring bounds.
    logL = torch.log(torch.from_numpy(df["FBEG"].values))
    logU = torch.log(torch.from_numpy(df["FEND"].values))

    # Set-up priors.
    hypers = {
        "m_beta": torch.zeros((p,)),
        "V_beta": 1000.0 * torch.eye(p),
        "nu_eps1": 1.0,
        "nu_eps2": 0.005,
        "m_mu": 0.0,
        "s2_mu": 100.0,
        "nu_tau1": 1.0,
        "nu_tau2": 0.005,
    }

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
    sampled_theta = mcmc.run(X=X, logL=logL, logU=logU, hypers=hypers, n_subjects=n_subjects, n_units_per_subject=n_units_per_subject)

    # Collect samples.
    samples = defaultdict(list)

    for theta in sampled_theta:
        for k, v in theta.items():
            samples[k].append(v)

    for k in samples:
        samples[k] = torch.stack(samples[k])

    # Plot samples.
    for i in range(p):
        fig, ax = plt.subplots()
        ax.set_title(r"$\beta_{}$ corresponding to {}".format(i + 1, regressors[i]))
        ax.plot(samples["beta"][:, i])
        plt.show()

    fig, ax = plt.subplots()
    ax.set_title(r"$\sigma^{-2}_{\epsilon}$")
    ax.plot(samples["sigma2_eps_inv"])
    plt.show()

    # fig, ax = plt.subplots()
    # ax.set_title(r"$\log{T}$")
    # ax.plot(logT_samples)
    # plt.show()


if __name__ == "__main__":
    main()
