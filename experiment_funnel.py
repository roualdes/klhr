from pathlib import Path
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

import bridgestan as bs
from bsmodel import BSModel
import db
from klhr import KLHR
from klhr_sinh import KLHRSINH
from klhr_sub_sinh import KLHRSUBSINH
from slice import Slice
from onlinemoments import OnlineMoments

@click.command()
@click.option("-M", "--iterations", "M",
              type=int, default=1_000,
              help="number of iterations")
@click.option("-w", "--warmup", "warmup",
              type=int, default=0,
              help="number of warmup iterations")
@click.option("-s", "--seed", "seed",
              type=int, default=530,
              help="seed to initialize the replications")
@click.option("-f", "--fresh", "start_fresh",
              is_flag=True,
              help="erase database before experiments")
@click.argument("algorithm", type=str)
def main(M, warmup, seed, start_fresh, algorithm):
    dbpath = "experiments.db"
    db.init_funnel(dbpath, start_fresh)

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "funnel"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    algorithms = {
        "klhr": KLHR,
        "klhr_sinh": KLHRSINH,
        "klhr_sub_sinh": KLHRSUBSINH,
        "slice": Slice
    }
    algo = algorithms[algorithm](bs_model,
                                 warmup = warmup,
                                 seed = seed)

    start = time.perf_counter()
    draws = algo.sample(M)
    runtime = time.perf_counter() - start
    mdx = np.arange(M)

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

    msjd = 0.0
    for m in range(M-1):
        d = np.linalg.norm(draws[m+1] - draws[m]) - msjd
        msjd += d / (m + 1)
    ldevals = algo.ld_evals if algorithm == "slice" else algo.grad_evals
    d = {
        "algorithm": algorithm,
        "msjd": msjd,
        "acceptance_rate": algo.acceptance_probability,
        "ld_evals": ldevals,
        "runtime": runtime,
    }
    db.append_df(dbpath, "funnel", d)

if __name__ == "__main__":
    main()
