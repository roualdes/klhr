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
from klhr_sub_sinh import KLHRSUBSINH
from slice import Slice
from onlinemoments import OnlineMoments

@click.command()
@click.option("-M", "--iterations", "M", type=int, default=1_000, help="number of iterations")
@click.option("-w", "--warmup", "warmup", type=int, default=0, help="number of warmup iterations")
@click.option("-r", "--replication", "rep", type=int, default=1, help="replication number for naming output files")
@click.option("-v", "--verbose", "verbose", is_flag=True, help="print information during run")
@click.argument("algorithm", type=str)
def main(M, warmup, rep, verbose, algorithm):

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "ar1"
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

    v = np.var(draws[warmup:, :], ddof = 1, axis = 0)
    rmse_var = np.sqrt(np.mean( (v - 1) ** 2))
    m = np.mean(draws[warmup:, :], axis = 0)
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
        plt.hist(draws[warmup:, d], histtype = "step",
                 density = True, color = "#0072B2", alpha = 0.1)
    x = np.linspace(-4, 4, 301)
    fx = st.norm().pdf(x)
    plt.plot(x, fx, linestyle = "dashed", color = "#D55E00")
    plt.title(f"RMSE(mean) = {rmse_mean:.4f}, RMSE(var) = {rmse_var:.4f}")
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/ar1/{algorithm}_{rep:0>2}.png")
    plt.close()

if __name__ == "__main__":
    main()
