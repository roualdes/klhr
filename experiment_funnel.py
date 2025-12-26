from pathlib import Path
import sys

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


import bridgestan as bs
from bsmodel import BSModel
from klhr import KLHR
from klhr_sinh import KLHRSINH
from klhr_sub_sinh import KLHRSUBSINH
from slice import Slice
from onlinemoments import OnlineMoments

@click.command()
@click.option("-M", "--iterations", "M", type=int, default=1_000, help="number of iterations")
@click.option("-w", "--warmup", "warmup", type=int, default=0, help="number of warmup iterations")
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

    mdx = np.arange(M)
    draws = algo.sample(M)

    if verbose:
        print(f"Acceptance rate: {algo.acceptance_probability}")
        msjd = np.mean([np.linalg.norm(draws[m+1] - draws[m]) for m in range(M-1)])
        print(f"MSJD: {np.round(msjd, 2)}")
        if algorithm == "slice":
            print(f"#ld evals: {algo.ld_evals}")
        else:
            print(f"#ldg evals: {algo.grad_evals}")

    plt.clf()
    plt.scatter(draws[warmup:, 1], draws[warmup:, 0],
                color = "#0072B2", alpha = 0.1)
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/funnel/scatter_{algorithm}.png")

    plt.clf()
    plt.hist(draws[warmup:, 0], histtype = "step",
             density = True, linewidth = 2)
    Normal = st.norm(loc = 0, scale = 3)
    x = np.linspace(-10, 10, 101)
    plt.plot(x, Normal.pdf(x), color = "#D55E00")
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/funnel/histogram_{algorithm}.png")
    plt.close()

if __name__ == "__main__":
    main()
