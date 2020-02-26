import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

import matplotlib.pyplot as plt  # noqa E402
import numpy as np  # noqa E402
import pandas as pd  # noqa E402
import pyreadr  # noqa E402
import torch  # noqa E402

import pynference.distributions as dist  # noqa E402
from pynference.inference import Metropolis  # noqa E402
from pynference.infrastructure import sample, Mask, Plate  # noqa E402


# TODO: Misclassification.
# TODO: Optimize truncnorm constraint.
# TODO: Optimize truncnorm normalizing constant.
# TODO: HMC & NUTS.
# TODO: More chains.
# TODO: MCMC diagnostics.


# TODO: Some of the truncnorm calculation could use `torch.where` as in masked.
# TODO: ExpandedDistribution can probably be removed.
def model(X, Y, logL, logU, logv, xi, hypers, N, J, K_max, visit_exists):
    ### Global parameters.
    assert logL.size() == logU.size() == (N, J, 1)
    assert logv.size() == (N, J, K_max)
    assert xi.size() == (N, J, K_max)
    assert visit_exists.size() == (N, J, K_max) and visit_exists.dtype == torch.bool

    # Fixed effects: beta ~ N(m_beta, V_beta).
    m_beta = hypers["m_beta"]
    V_beta = hypers["V_beta"]
    beta = sample("beta", dist.MultivariateNormal(loc=m_beta, covariance_matrix=V_beta))

    # Noise term: sigma2_eps_inv ~ Gamma(nu_eps1, nu_eps2).
    nu_eps1 = hypers["nu_eps1"]
    nu_eps2 = hypers["nu_eps2"]
    sigma2_eps_inv = sample(
        "sigma2_eps_inv", dist.Gamma(concentration=nu_eps1, rate=nu_eps2)
    )

    # Random effects mean: mu ~ N(m_mu, s2_mu).
    m_mu = hypers["m_mu"]
    s2_mu = hypers["s2_mu"]
    mu = sample("mu", dist.Normal(loc=m_mu, scale=math.sqrt(s2_mu)))

    # Random effects variance: tau2_inv ~ Gamma(nu_tau1, nu_tau2).
    nu_tau1 = hypers["nu_tau1"]
    nu_tau2 = hypers["nu_tau2"]
    tau2_inv = sample("tau2_inv", dist.Gamma(concentration=nu_tau1, rate=nu_tau2))

    # TODO: Handle `alpha + eta > 1`.
    # TODO: This is the simplified model M2.
    # Sensitivity: alpha ~ Beta(a0_alpha, a1_alpha).
    a0_alpha = hypers["a0_alpha"]
    a1_alpha = hypers["a1_alpha"]
    alpha = sample("alpha", dist.Beta(a0_alpha, a1_alpha))
    assert alpha.size() == (a0_alpha.size(0),), alpha.size()

    # Specificity: eta ~ Beta(a0_eta, a1_eta).
    a0_eta = hypers["a0_eta"]
    a1_eta = hypers["a1_eta"]
    eta = sample("eta", dist.Beta(a0_eta, a1_eta))
    assert eta.size() == (a0_eta.size(0),), eta.size()

    ### Subject-specific parameters.
    with Plate("subjects", N, dim=-3):
        b = sample("b", dist.Normal(loc=mu, scale=tau2_inv.sqrt().reciprocal()))
        b = b.repeat(1, J, 1)
        assert b.size() == (N, J, 1), b.size()

        # TODO: This could be vectorized.
        x_girl = X[0].unsqueeze(-1)
        x_seal = X[1].unsqueeze(-1)
        x_freqbr = X[2].unsqueeze(-1)

        ### Subject-tooth-specific parameters.
        with Plate("teeth", J, dim=-2):
            loc = x_girl * beta[0] + x_seal * beta[1] + x_freqbr * beta[2] + b
            scale = sigma2_eps_inv.sqrt().reciprocal()
            assert loc.size() == (N, J, 1), loc.size()

            logT = sample(
                "logT", dist.TruncatedNormal(loc=loc, scale=scale, low=logL, high=logU)
            )

            ### Subject-tooth-visit-specific parameters.
            with Plate("visits", K_max, dim=-1):
                alpha = alpha[xi]
                eta = eta[xi]

                logT = logT.expand(-1, -1, K_max)

                assert (
                    logT.size()
                    == logv.size()
                    == alpha.size()
                    == eta.size()
                    == (N, J, K_max)
                )

                with Mask(visit_exists & (logT <= logv)):
                    sample("Y_alpha", dist.Bernoulli(alpha), observation=Y)

                with Mask(visit_exists & (logT > logv)):
                    sample("Y_eta", dist.Bernoulli(1.0 - eta), observation=Y)


def model_old(X, Y, logL, logU, logv, xi, hypers, N, J, K):
    ### Event times model.

    # Fixed effects: beta ~ N(m_beta, V_beta).
    m_beta = hypers["m_beta"]
    V_beta = hypers["V_beta"]
    beta = sample("beta", dist.MultivariateNormal(loc=m_beta, covariance_matrix=V_beta))

    # Noise term: sigma2_eps_inv ~ Gamma(nu_eps1, nu_eps2).
    nu_eps1 = hypers["nu_eps1"]
    nu_eps2 = hypers["nu_eps2"]
    sigma2_eps_inv = sample(
        "sigma2_eps_inv", dist.Gamma(concentration=nu_eps1, rate=nu_eps2)
    )

    # Random effects mean: mu ~ N(m_mu, s2_mu).
    m_mu = hypers["m_mu"]
    s2_mu = hypers["s2_mu"]
    mu = sample("mu", dist.Normal(loc=m_mu, scale=math.sqrt(s2_mu)))

    # Random effects variance: tau2_inv ~ Gamma(nu_tau1, nu_tau2).
    nu_tau1 = hypers["nu_tau1"]
    nu_tau2 = hypers["nu_tau2"]
    tau2_inv = sample("tau2_inv", dist.Gamma(concentration=nu_tau1, rate=nu_tau2))

    # Random effect: b ~ N(mu, tau2_inv^{-1}).
    mu = mu * torch.ones((N,))
    tau2_inv = tau2_inv * torch.ones((N,))
    b = sample("b", dist.Normal(loc=mu, scale=tau2_inv.sqrt().reciprocal()))
    b = torch.repeat_interleave(b, J)

    # Latent log(event times): logT ~ TN(X @ beta + b, sigma2_eps_inv^{-1}, logL, logU).
    logT = sample(
        "logT",
        dist.TruncatedNormal(
            loc=X @ beta + b,
            scale=sigma2_eps_inv.sqrt().reciprocal(),
            low=logL,
            high=logU,
        ),
    )

    # TODO: Remove this once truncnorm is finished.
    assert not torch.isinf(logT).any()
    assert not torch.isnan(logT).any()

    ### Misclassification model.

    a0_alpha = hypers["a0_alpha"]
    a1_alpha = hypers["a1_alpha"]

    a0_eta = hypers["a0_eta"]
    a1_eta = hypers["a1_eta"]

    # TODO: Handle the alpha + eta > 1 constraint.
    alpha = sample("alpha", dist.Beta(a0_alpha, a1_alpha))
    eta = sample("eta", dist.Beta(a0_eta, a1_eta))

    # TODO: Could the entire i plate be vectorized? This would require merging
    # TODO: together the misclassification and event times model, obfuscating
    # TODO: the generative structure a bit, but might be more efficient.

    # TODO: An alternative approach would be to set K[i] = max_j K[j] for all i,
    # TODO: vectorize all three plates and use masked distributions where a diagnosis
    # TODO: does not exist.
    for i in Plate("subjects", N):
        # TODO: Swap the j and k loops?
        for j in Plate("teeth", J):
            for k in Plate("visits", K[i]):
                xi_ik = xi[i, k]

                if (i, j, k) not in Y:
                    # The j-th tooth of the i-th subject was not examined on the k-th visit.
                    continue

                # TODO: The indexing of alpha and eta currently assumes the simplified model M2.
                # TODO: Could this be handled by a masked distribution?
                if logT[i * J + j] <= logv[i, k]:
                    # Observed diagnosis: Y_ijk | T_ij <= v_ik ~ Alt(alpha_{xi_ik}).
                    sample(
                        "Y_({},{},{})".format(i, j, k),
                        dist.Bernoulli(alpha[xi_ik]),
                        observation=Y[i, j, k],
                    )
                else:
                    # Observed diagnosis: Y_ijk | T_ij > v_ik ~ Alt(1 - eta_{xi_ik}).
                    sample(
                        "Y_({},{},{})".format(i, j, k),
                        dist.Bernoulli(1.0 - eta[xi_ik]),
                        observation=Y[i, j, k],
                    )


def load_dataframe(df_path: str, which: str) -> pd.DataFrame:
    # See tandmob_analysis.ipynb for more details on data processing.
    r_data = pyreadr.read_r(df_path)
    data1 = r_data["Data1"]  # Misclassifications.
    data2 = r_data["Data2"]  # Regressors.

    data1["STATUS"] = data1["STATUS"].astype(float)
    data1[["IDNR", "TOOTH"]] = data1["IdTooth"].str.split("_", expand=True)

    # Make zero-indexed.
    data1["IDNR"] = data1["IDNR"].astype(int) - 1
    data1["TOOTH"] = data1["TOOTH"].astype(int)
    data1["TOOTH_RANK"] = data1["TOOTH"].replace({16: 0, 26: 1, 36: 2, 46: 3})

    # Make zero-indexed.
    data1["EXAMINER"] = data1["EXAMINER"].astype(int) - 1
    data1.drop("IdTooth", axis="columns", inplace=True)

    # Ascending rank of each visit time, zero-indexed.
    data1["VISIT_RANK"] = (
        data1.groupby(["IDNR", "TOOTH"]).rank()["VISIT"].astype(int) - 1
    )
    data1 = data1[
        ["IDNR", "TOOTH", "TOOTH_RANK", "VISIT", "VISIT_RANK", "EXAMINER", "STATUS"]
    ]
    data1.sort_values(by=["IDNR", "TOOTH"], inplace=True)

    # Make zero-indexed.
    data2["IDNR"] = data2["IDNR"].astype(int) - 1
    data2["TOOTH"] = data2["TOOTH"].astype(int)
    data2["TOOTH_RANK"] = data2["TOOTH"].replace({16: 0, 26: 1, 36: 2, 46: 3})
    data2.loc[data2["FBEG"].isna(), "FBEG"] = 0.0
    data2.loc[data2["FEND"].isna(), "FEND"] = float("inf")
    data2.sort_values(by=["IDNR", "TOOTH"], inplace=True)

    # Drop subjects for which we observe fewer than 4 teeth.
    grouped_by_subject = data2.groupby("IDNR").count()
    not_4_teeth = grouped_by_subject[grouped_by_subject["TOOTH"] != 4].index

    data1.drop(data1[data1["IDNR"].isin(not_4_teeth)].index, inplace=True)
    data2.drop(data2[data2["IDNR"].isin(not_4_teeth)].index, inplace=True)

    # Ensure consecutively-numbered IDNR.
    N = data2["IDNR"].nunique()
    J = 4
    data2["IDNR"] = np.repeat(np.arange(N, dtype=int), J)

    visits_per_subject = data1.groupby("IDNR").count()["VISIT"]
    data1["IDNR"] = np.repeat(np.arange(N, dtype=int), visits_per_subject)

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
    df_misclassifications, df_regressors = load_dataframe(df_path, which="both")

    N = df_regressors["IDNR"].nunique()  # Number of subjects.
    J = 4  # We observe 4 teeth on each subject.
    Q = 16  # There are 16 examiners in total.

    # Number of visits per subject. Note that there are some visits where not all teeth
    # were examined, hence the max over tooth visit time counts after grouping.
    K_max = df_misclassifications.groupby(["IDNR", "TOOTH"]).count()["VISIT"].max()

    # Load regressors.
    regressors = ["GIRL", "SEAL", "FREQ.BR"]
    p = len(regressors)
    X = np.empty(shape=(p, N, J))

    for i, regressor in enumerate(regressors):
        X[i] = df_regressors[regressor].values.reshape(N, J)

    X = torch.from_numpy(X)

    # Load the potentially misclassified diagnoses.
    # Y[i, j, k] ... diagnosis of the j-th tooth of the i-th subject on the k-th visit
    # if such visit occurred, or an arbitrary value otherwise.
    Y = np.full(shape=(N, J, K_max), fill_value=666.0)
    Y[
        df_misclassifications["IDNR"].values,
        df_misclassifications["TOOTH_RANK"].values,
        df_misclassifications["VISIT_RANK"].values,
    ] = df_misclassifications["STATUS"].values
    Y = torch.from_numpy(Y)

    # Load interval censoring bounds.
    logL = torch.log(
        torch.from_numpy(df_regressors["FBEG"].values.reshape(N, J))
    ).unsqueeze(-1)
    logU = torch.log(
        torch.from_numpy(df_regressors["FEND"].values.reshape(N, J))
    ).unsqueeze(-1)

    # Load visit log-times.
    # logv[i, j, k] ... log(time of the k-th visit of the i-th subject on the j-th
    # tooth) if such visit occurred, an arbitrary value otherwise.
    logv = np.full(shape=(N, J, K_max), fill_value=666.0)
    logv[
        df_misclassifications["IDNR"].values,
        df_misclassifications["TOOTH_RANK"].values,
        df_misclassifications["VISIT_RANK"].values,
    ] = df_misclassifications["VISIT"].values
    logv = torch.from_numpy(logv)
    logv.log_()

    # Load subject-visit examiner indicators.
    # xi[i, j, k] ... id of the examiner at the k-th visit of the j-th tooth of
    # the i-th subject if such visit occurred, or an arbitrary value otherwise.
    xi = np.zeros(shape=(N, J, K_max), dtype=int)
    xi[
        df_misclassifications["IDNR"].values,
        df_misclassifications["TOOTH_RANK"].values,
        df_misclassifications["VISIT_RANK"].values,
    ] = df_misclassifications["EXAMINER"].values
    xi = torch.from_numpy(xi)

    # Load indicators of visits (i, j, k) being defined.
    visit_exists = Y < 666.0

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
        "a0_alpha": torch.ones((Q,)),
        "a1_alpha": torch.ones((Q,)),
        "a0_eta": torch.ones((Q,)),
        "a1_eta": torch.ones((Q,)),
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
    sampled_theta = mcmc.run(
        X=X,
        Y=Y,
        logL=logL,
        logU=logU,
        logv=logv,
        xi=xi,
        hypers=hypers,
        N=N,
        J=J,
        K_max=K_max,
        visit_exists=visit_exists,
    )

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
