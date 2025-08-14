import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import bridgestan as bs
from bsmodel import BSModel
from klhr import KLHR
from klhr_sinh import KLHRSINH
from onlinemoments import OnlineMoments

@click.command()
@click.option("-M", "--iterations", "M", type=int, default=1_000, help="number of iterations")
@click.option("-w", "--warmup", "warmup", type=int, default=100, help="number of warmup iterations")
@click.option("-v", "--verbose", "verbose", is_flag=True, help="print information during run")
@click.argument("algorithm", type=str)
def main(M, warmup, verbose, algorithm):

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "funnel"
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

    mdx = np.arange(M)
    thetas = np.zeros((M, algo.D))
    for m in mdx:
        thetas[m] = algo.draw()

    if verbose:
        print(f"Acceptance rate: {algo.acceptance_probability}")
        print(f"Minimization failure rate: {algo.minimization_failure_rate}")

    # df = pd.DataFrame(thetas, columns = algo.model.parameter_names())
    # df.to_parquet(source_dir / f"experiments/funnel/{algorithm}.parquet")

    plt.clf()
    plt.scatter(thetas[warmup:, 1], thetas[warmup:, 0], color = "#0072B2", alpha = 0.1)
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/funnel/scatter_{algorithm}.png")

    plt.clf()
    plt.hist(thetas[warmup:, 0], histtype = "step", density = True, linewidth = 2)
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/funnel/histogram_{algorithm}.png")
    plt.close()

if __name__ == "__main__":
    main()
