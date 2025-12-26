import click
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

import bridgestan as bs
from bsmodel import BSModel
from klhr_sinh import KLHRSINH
from klhr import KLHR
from klhr_sub_sinh import KLHRSUBSINH
from slice import Slice

@click.command()
@click.option("-M", "--iterations", "M", type=int, default=2_000, help="number of iterations, including warmup")
@click.option("-w", "--warmup", "warmup", type=int, default=1_000, help="number of warmup iterations")
@click.option("-r", "--replication", "rep", type=int, default=0, help="replication number for naming output files")
@click.option("-v", "--verbose", "verbose", is_flag=True, help="print information during run")
@click.argument("algorithm", type=str)
def main(M, warmup, rep, verbose, algorithm):
    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "earnings"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    if algorithm == "klhr":
        algo = KLHR(bs_model, warmup = warmup)
    elif algorithm == "klhr_sinh":
        algo = KLHRSINH(bs_model, warmup = warmup)
    elif algorithm == "klhr_sub_sinh":
        algo = KLHRSUBSINH(bs_model, warmup = warmup)
    elif algorithm == "slice":
        algo = Slice(bs_model, warmup = warmup)
    else:
        print(f"Unknown algorithm {algorithm}")
        print("Available algorithms: klhr, klhr_sinh, klhr_sub_sinh, or slice")
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
    plt.savefig(source_dir / f"experiments/relaxationtime/{algorithm}_{rep:0>2}.png")
    plt.close()

    if verbose:
        print(f"Acceptance rate: {np.round(algo.acceptance_probability, 2)}")
        msjd = np.mean([np.linalg.norm(draws[m+1] - draws[m]) for m in range(M-1)])
        print(f"MSJD: {np.round(msjd, 2)}")
        print(np.mean(draws[warmup:, :], axis = 0))
        print(np.std(draws[warmup:, :], axis = 0))
        if algorithm == "slice":
            print(f"#ld evals: {algo.ld_evals}")
        else:
            print(f"#ldg evals: {algo.grad_evals}")

if __name__ == "__main__":
    main()
