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
              type=int, default=123898,
              help="seed to initialize the replications")
@click.argument("algorithm", type=str)
def main(M, warmup, reps, seed, algorithm):
    dbpath = "experiments.db"
    db.init_mmodels(dbpath, False) # if need start fresh, re-run experiment_stan.py -f

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")
    source_dir = Path(__file__).resolve().parent

    algorithms = {
        "klhr": KLHR,
        "klhr_sub_sinh": KLHRSUBSINH,
        "slice": Slice
    }

    models = [
        "corr-normal",
        "garch",
        "rosenbrock",
        "hmm",
        "ill-normal",
        "lotka-volterra",
    ]

    for model in models:
        for rep in range(reps):
            bs_model = BSModel(
                stan_file = source_dir / f"stan/{model}.stan",
                data_file = source_dir / f"stan/{model}.json")

            seedi = np.random.SeedSequence([seed, rep])
            algo = algorithms[algorithm](bs_model,
                                         warmup = warmup,
                                         seed = seedi)
            start = time.perf_counter()
            draws = algo.sample(M)
            runtime = time.perf_counter() - start

            post_warmup_draws = draws[warmup:, :]
            constrained_draws = np.zeros_like(post_warmup_draws)
            N = np.shape(constrained_draws)[0]
            for n in range(N):
                theta = post_warmup_draws[n]
                constrained_draws[n] = algo.model.constrain(theta)

            msjd = 0.0
            for n in range(N-1):
                l1 = constrained_draws[n+1] - constrained_draws[n]
                d = np.linalg.norm(l1) - msjd
                msjd += d / (n + 1)
            ldevals = algo.ld_evals if algorithm == "slice" else algo.grad_evals
            min_ess = np.inf
            num_params = np.shape(draws)[1]
            for d in range(algo.D):
                e = ess(draws[:, d, np.newaxis])
                if e < min_ess:
                    min_ess = e
            d = {
                "algorithm": algorithm,
                "replication": rep,
                "model": model,
                "msjd": msjd,
                "min_ess": min_ess,
                "acceptance_rate": algo.acceptance_probability,
                "ld_evals": ldevals,
                "runtime": runtime,
            }

            varnames = algo.model.parameter_names()
            m = np.mean(constrained_draws, axis = 0)
            m2 = np.mean(constrained_draws ** 2, axis = 0)
            d |= {varnames[i] + "m": mi for i, mi in enumerate(m)}
            d |= {varnames[i] + "s": mi for i, mi in enumerate(m)}
            d |= {varnames[i] + "m2": m2i for i, m2i in enumerate(m2)}
            d |= {varnames[i] + "s2": m2i for i, m2i in enumerate(m2)}
            db.append_df(dbpath, model.replace("-", ""), d)

if __name__ == "__main__":
    main()
