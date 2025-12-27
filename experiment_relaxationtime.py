from pathlib import Path
import time

import click
import matplotlib.pyplot as plt
import numpy as np

import bridgestan as bs
from bsmodel import BSModel
import db
from klhr_sinh import KLHRSINH
from klhr import KLHR
from klhr_sub_sinh import KLHRSUBSINH
from slice import Slice

@click.command()
@click.option("-M", "--iterations", "M",
              type=int, default=2_000,
              help="number of iterations, including warmup")
@click.option("-w", "--warmup", "warmup",
              type=int, default=1_000,
              help="number of warmup iterations")
@click.option("-r", "--replications", "reps",
              type=int, default=1,
              help="replication number for naming output files")
@click.option("-s", "--seed", "seed",
              type=int, default=204,
              help="seed to initialize the replications")
@click.option("-f", "--fresh", "start_fresh",
              is_flag=True,
              help="erase database before experiments")
@click.option("-t", "--tolerance", "tolerance",
              type = float, default=1e-3)
@click.argument("algorithm", type=str)
def main(M, warmup, reps, seed, start_fresh, tolerance, algorithm):
    dbpath = "experiments.db"
    db.init_relaxationtime(dbpath, start_fresh)

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "earnings"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    algorithms = {
        "klhr": KLHR,
        "klhr_sinh": KLHRSINH,
        "klhr_sub_sinh": KLHRSUBSINH,
        "slice": Slice
    }

    for rep in range(reps):
        seedi = np.random.SeedSequence([seed, rep])
        algo = algorithms[algorithm](bs_model,
                                     warmup = warmup,
                                     seed = seedi,
                                     gtol = tolerance)
        start = time.perf_counter()
        draws = algo.sample(M)
        runtime = time.perf_counter() - start
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

        msjd = 0.0
        for m in range(M-1):
            d = np.linalg.norm(draws[m+1] - draws[m]) - msjd
            msjd += d / (m + 1)
        ldevals = algo.ld_evals if algorithm == "slice" else algo.grad_evals
        d = {
            "algorithm": algorithm,
            "replication": rep,
            "msjd": msjd,
            "acceptance_rate": algo.acceptance_probability,
            "ld_evals": ldevals,
            "runtime": runtime,
        }

        m = np.mean(draws[warmup:, :], axis = 0)
        v = np.var(draws[warmup:, :], ddof = 1, axis = 0)
        varnames = ["b0", "b1", "sigma", "s"]
        d |= {"m" + varnames[i]: mi for i, mi in enumerate(m)}
        d |= {"v" + varnames[i]: vi for i, vi in enumerate(v)}
        db.append_df(dbpath, "relaxationtime", d)

if __name__ == "__main__":
    main()
