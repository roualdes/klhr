import click
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

import bridgestan as bs
from bsmodel import BSModel
from klhr_sinh import KLHRSINH
from klhr import KLHR

@click.command()
@click.option("-M", "--iterations", "M", type=int, default=2_000, help="number of iterations")
@click.option("-w", "--warmup", "warmup", type=int, default=1_000, help="set value from which RMSEs are plot")
@click.option("--windowsize", "windowsize", type=int, default=50, help="set window size")
@click.option("--windowscale", "windowscale", type=int, default=2, help="set window scale")
@click.option("-l", "--amnesia", "l", type=int, default=0, help="set the amnesia parameter for OnlinePCA")
@click.option("-J", "J", type=int, default=2, help="number of eigenvectors")
@click.option("-r", "--replication", "rep", type=int, default=0, help="replication number for naming output files")
@click.option("-v", "--verbose", "verbose", is_flag=True, help="print information during run")
@click.option("-s", "--scale_dir_cov", "scale_dir_cov", is_flag=True, help="scale covariance matrix used to select a random direction")
@click.option("-o", "--overrelaxed", "overrelaxed", is_flag=True, help="use overrelaxed proposals in metropolis step")
@click.option("-e1", "--eigen_method_one", "eigen_method_one", is_flag=True, help="Use option one for utilizing eigenvectors to select a direction")
@click.argument("algorithm", type=str)
def main(M, warmup, windowsize, windowscale, l, J, rep, verbose, scale_dir_cov, overrelaxed, eigen_method_one, algorithm):
    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "earnings"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    if algorithm == "klhr":
        algo = KLHR(bs_model,
                    warmup = warmup,
                    windowsize = windowsize,
                    windowscale = windowscale,
                    J = J,
                    l = l,
                    scale_dir_cov = scale_dir_cov,
                    overrelaxed = overrelaxed,
                    eigen_method_one = eigen_method_one)
    elif algorithm == "klhrsinh":
        algo = KLHRSINH(bs_model,
                        warmup = warmup,
                        windowsize = windowsize,
                        windowscale = windowscale,
                        J = J,
                        l = l,
                        scale_dir_cov = scale_dir_cov,
                        overrelaxed = overrelaxed,
                        eigen_method_one = eigen_method_one)
    else:
        print(f"Unknown algorithm {algorithm}")
        print("Available algorithms: klhr or klhrsinh")
        sys.exit(0)

    draws = algo.sample(M)
    idx = np.arange(M)

    fig, axs = plt.subplots(2, 2, figsize = (14, 6))
    axs[0, 0].plot(idx, draws[:, 0])
    axs[0, 0].set_ylabel(r"$\beta_0$")

    axs[0, 1].plot(idx, draws[:, 1])
    axs[0, 1].set_ylabel(r"$\beta_1$")

    axs[1, 0].plot(idx, draws[:, 2])
    axs[1, 0].set_ylabel(r"$\sigma$")

    axs[1, 1].plot(idx, draws[:, 3])
    axs[1, 1].set_ylabel(r"$s$")

    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/relaxationtime/{algorithm}_{windowsize}_{windowscale}_{l}_{J}_{rep:0>2}.png")
    plt.close()

    if verbose:
        print(np.mean(draws[warmup:, :], axis = 0))
        print(np.std(draws[warmup:, :], axis = 0))

if __name__ == "__main__":
    main()
