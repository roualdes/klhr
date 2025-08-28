import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import bridgestan as bs
from bsmodel import BSModel
from klhr import KLHR
from klhr_sinh import KLHRSINH
from mh import MH
from onlinemoments import OnlineMoments

@click.command()
@click.option("-M", "--iterations", "M", type=int, default=1_000, help="number of iterations")
@click.option("-w", "--warmup", "warmup", type=int, default=100, help="set value from which RMSEs are plot")
@click.option("-v", "--verbose", "verbose", is_flag=True, help="print information during run")
@click.argument("algorithm", type=str)
def main(M, warmup, verbose, algorithm):

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "normal"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    if algorithm == "klhr":
        algo = KLHR(bs_model, warmup = warmup)
    elif algorithm == "klhrsinh":
        algo = KLHRSINH(bs_model, warmup = warmup)
    else:
        print(f"Unknown algorithm {algorithm}")
        print("Available algorithms: klhr or klhrsinh")
        sys.exit(0)

    mh = MH(bs_model, 0.09)

    stats_klhr = {
        "om": OnlineMoments(algo.D),
        "rmse_mean": np.zeros(M),
        "rmse_var": np.zeros(M),
        "log_density": np.zeros(M),
    }

    stats_mh = {
        "om": OnlineMoments(algo.D),
        "rmse_mean": np.zeros(M),
        "rmse_var": np.zeros(M),
        "log_density": np.zeros(M),
    }

    rng = np.random.default_rng()
    mu = np.zeros(algo.D)
    Sigma = np.eye(algo.D)
    log_density_iid = np.zeros(M)

    mdx = np.arange(M)
    for m in mdx:
        theta = algo.draw()
        stats_klhr["om"].update(theta)
        stats_klhr["rmse_mean"][m] = np.sqrt(np.mean( stats_klhr["om"].mean() ** 2) )
        stats_klhr["rmse_var"][m] = np.sqrt(np.mean( (stats_klhr["om"].var() - 1) ** 2 ))
        stats_klhr["log_density"][m] = bs_model.log_density(theta)

        theta = mh.draw()
        stats_mh["om"].update(theta)
        stats_mh["rmse_mean"][m] = np.sqrt(np.mean( stats_mh["om"].mean() ** 2) )
        stats_mh["rmse_var"][m] = np.sqrt(np.mean( (stats_mh["om"].var() - 1) ** 2 ))
        stats_mh["log_density"][m] = bs_model.log_density(theta)

        x = rng.multivariate_normal(mu, Sigma)
        log_density_iid[m] = bs_model.log_density(x)

    if verbose:
        print(f"Acceptance rate: {algo.acceptance_probability}")
        print(f"Minimization failure rate: {algo.minimization_failure_rate}")
        print(f"means: {stats_klhr['om'].mean()}")
        print(f"vars: {stats_klhr['om'].var()}")

    # df = pd.DataFrame.from_dict({
    #     "iteration": mdx + 1,

    #     "rmse_mean_klhr": stats_klhr["rmse_mean"],
    #     "rmse_var_klhr": stats_klhr["rmse_var"],
    #     "log_density_klhr": stats_klhr["log_density"],

    #     "rmse_mean_mh": stats_mh["rmse_mean"],
    #     "rmse_var_mh": stats_mh["rmse_var"],
    #     "log_density_mh": stats_mh["log_density"],

    #     "log_density_iid": log_density_iid})
    # df.to_parquet(source_dir / f"experiments/accuracy/{algorithm}_rmse.parquet")

    plt.clf()
    plt.rc('axes', labelsize = 12)

    origin = 10 ** 2

    plt.plot(mdx[origin:], stats_klhr["rmse_mean"][origin:], label="KLHR: mean",
             linestyle = "dotted", color = "#0072B2", linewidth = 2)
    plt.plot(mdx[origin:], stats_klhr["rmse_var"][origin:], label="KLHR: var",
             linestyle = (0, (1, 5)), color = "#D55E00", linewidth = 2)

    plt.plot(mdx[origin:], stats_mh["rmse_mean"][origin:], label="MH: mean",
             linestyle = "dashed", color = "#009E73", linewidth = 2)
    plt.plot(mdx[origin:], stats_mh["rmse_var"][origin:], label="MH: var",
             linestyle = (0, (5, 5)), color = "#F0E442", linewidth = 2)

    plt.plot([origin, M], [1 / np.sqrt(origin), 1 / np.sqrt(M)],
             linestyle = "solid", color = "black", alpha = 0.2)

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("RMSE")
    plt.xlabel("iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/accuracy/{algorithm}_{warmup}_rmse.png")

    plt.clf()
    plt.rc('axes', labelsize = 12)
    plt.hist(stats_klhr["log_density"][origin:], histtype = "step", density = True, label = "KLHR")
    plt.hist(stats_mh["log_density"][origin:], histtype = "step", density = True, label = "MH")
    plt.hist(log_density_iid[origin:], histtype = "step", density = True, label = "iid")
    plt.xlabel("log_density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/accuracy/{algorithm}_{warmup}_histogram_log_density.png")
    plt.close()

if __name__ == "__main__":
    main()
