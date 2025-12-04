import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as st

import bridgestan as bs
from bsmodel import BSModel
from klhr import KLHR
from klhr_sinh import KLHRSINH
from onlinemoments import OnlineMoments

@click.command()
@click.option("-M", "--iterations", "M", type=int, default=1_000, help="number of iterations")
@click.option("-w", "--warmup", "warmup", type=int, default=100, help="number of warmup iterations")
@click.option("--windowsize", "windowsize", type=int, default=25, help="set window size")
@click.option("--windowscale", "windowscale", type=int, default=2, help="set window scale")
@click.option("-l", "--amnesia", "l", type=int, default=2, help="set the amnesia parameter for OnlinePCA")
@click.option("-J", "J", type=int, default=2, help="number of eigenvectors")
@click.option("-r", "--replication", "rep", type=int, default=0, help="replication number for naming output files")
@click.option("-v", "--verbose", "verbose", is_flag=True, help="print information during run")
@click.option("-s", "--scale_dir_cov", "scale_dir_cov", is_flag=True, help="scale covariance matrix used to select a random direction")
@click.option("-o", "--overrelaxed", "overrelaxed", is_flag=True, help="use overrelaxed proposals in metropolis step")
@click.option("-e1", "--eigen_method_one", "eigen_method_one", is_flag=True, help="Use option one for utilizing eigenvectors to select a direction")
@click.argument("algorithm", type=str)
def main(M, warmup, windowsize, windowscale, l, J, rep, verbose, scale_dir_cov, overrelaxed, eigen_method_one, algorithm):

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "ar1"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    if algorithm == "klhr":
        algo = KLHR(bs_model,
                    warmup = warmup,
                    windowsize = windowsize,
                    windowscale = windowscale,
                    l = l,
                    J = J,
                    scale_dir_cov = scale_dir_cov,
                    overrelaxed = overrelaxed,
                    eigen_method_one = eigen_method_one
                    )
    elif algorithm == "klhr_sinh":
        algo = KLHRSINH(bs_model,
                        warmup = warmup,
                        windowsize = windowsize,
                        windowscale = windowscale,
                        l = l,
                        J = J,
                        scale_dir_cov = scale_dir_cov,
                        overrelaxed = overrelaxed,
                        eigen_method_one = eigen_method_one)
    else:
        print(f"Unknown algorithm {algorithm}")
        print("Available algorithms: klhr or klhr_sinh")
        sys.exit(0)

    mdx = np.arange(M)
    thetas = np.zeros((M, algo.D))
    for m in mdx:
        thetas[m] = algo.draw()

    if verbose:
        print(f"Acceptance rate: {algo.acceptance_probability}")
        print(f"Minimization failure rate: {algo.minimization_failure_rate}")

    # df = pd.DataFrame(thetas, columns = algo.model.parameter_names())
    # df.to_parquet(source_dir / f"experiments/ar1/{algorithm}.parquet")

    v = np.var(thetas[warmup:, :], ddof = 1, axis = 0)
    rmse_var = np.sqrt(np.mean( (v - 1) ** 2))
    m = np.mean(thetas[warmup:, :], axis = 0)
    rmse_mean = np.sqrt(np.mean( m ** 2))
    if verbose:
        print(f"maximum absolute mean: {np.max(np.abs(m)):.4f}")
        print(f"RMSE(mean): {rmse_mean:.4f}")
        print(f"means: {m}")
        print(f"minimum variance: {np.min(v):.4f}")
        print(f"RMSE(var): {rmse_var:.4f}")
        print(f"vars: {v}")

    plt.clf()
    for d in range(algo.D):
        plt.hist(thetas[warmup:, d], histtype = "step", density = True,
                 color = "#0072B2", alpha = 0.1)
    x = np.linspace(-4, 4, 301)
    fx = st.norm().pdf(x)
    plt.plot(x, fx, linestyle = "dashed", color = "#D55E00")
    plt.title(f"RMSE(mean) = {rmse_mean:.4f}, RMSE(var) = {rmse_var:.4f}")
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/ar1/{algorithm}_{windowsize}_{windowscale}_{l}_{J}_{rep:0>2}.png")
    plt.close()

if __name__ == "__main__":
    main()
