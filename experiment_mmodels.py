from pathlib import Path
import time

import click
import matplotlib.pyplot as plt
import numpy as np

import bridgestan as bs
from bsmodel import BSModel
import db
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
@click.argument("algorithm", type=str)
def main(M, warmup, reps, seed, start_fresh, algorithm):
    dbpath = "experiments.db"
    db.init_mmodels(dbpath, start_fresh)

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

    models = [
        "corr-normal",
        "garch",
        "glmm-poisson",
        "hmm",
        "ill-normal",
        "lotka-volterra",
    ]

    for model in models:
    for rep in range(reps):
        seedi = np.random.SeedSequence([seed, rep])
        algo = algorithms[algorithm](bs_model,
                                     warmup = warmup,
                                     seed = seedi)
        start = time.perf_counter()
        draws = algo.sample(M)
        runtime = time.perf_counter() - start
        idx = np.arange(M)

        msjd = 0.0
        for m in range(M-1):
            d = np.linalg.norm(draws[m+1] - draws[m]) - msjd
            msjd += d / (m + 1)
        ldevals = algo.ld_evals if algorithm == "slice" else algo.grad_evals
        d = {
            "algorithm": algorithm,
            "replication": rep,
            "model": model,
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
