import click
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

from models.earnings import Earnings
from nprmodel import NPRModel

import bridgestan as bs
from bsmodel import BSModel
from klhr_sinh import KLHRSINH
from klhr import KLHR

@click.command()
@click.option("-M", "--iterations", "M", type=int, default=2_000, help="number of iterations")
@click.option("-w", "--warmup", "warmup", type=int, default=1_000, help="set value from which RMSEs are plot")
@click.option("--windowsize", "windowsize", type=int, default=50, help="set window size")
@click.option("--windowscale", "windowscale", type=int, default=2, help="set window scale")
@click.option("-l", "--amnesia", "l", type=int, default=2, help="set the amnesia parameter for OnlinePCA")
@click.option("-J", "J", type=int, default=2, help="number of eigenvectors")
@click.option("-r", "--replication", "rep", type=int, default=0, help="replication number for naming output files")
@click.option("-v", "--verbose", "verbose", is_flag=True, help="print information during run")
@click.argument("algorithm", type=str)
def main(M, warmup, windowsize, windowscale, l, J, rep, verbose, algorithm):

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "earnings"
    source_dir = Path(__file__).resolve().parent

    rng = np.random.default_rng(204)
    theta = np.ones(4)#rng.normal(size = 4)
    # print(f"{theta=}")
    print(f"Bridgestan:")
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")
    print(f"bs_model: {bs_model.log_density_gradient(theta, propto=False)}")
    # print("NumPyro")
    # earnings = Earnings()
    # model = NPRModel(earnings.model(), earnings.data())
    # print(f"npr_model: {model.log_density_gradient(theta)}")
    sys.exit(0)

    if algorithm == "klhr":
        algo = KLHR(bs_model,
                    warmup = warmup,
                    windowsize = windowsize,
                    windowscale = windowscale,
                    J = J,
                    l = l)
    elif algorithm == "klhrsinh":
        algo = KLHRSINH(bs_model,
                        warmup = warmup,
                        windowsize = windowsize,
                        windowscale = windowscale,
                        J = J,
                        l = l)
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
