import json
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
from klhr_reflection import KLHRReflection
from klhr_sinh import KLHRSINH
from klhr_sub_sinh import KLHRSUBSINH
from slice import Slice
from onlinemoments import OnlineMoments

@click.command()
@click.option("-M", "--iterations", "M",
              type=int, default=1_000, help="number of iterations")
@click.option("-w", "--warmup", "warmup",
              type=int, default=0, help="number of warmup iterations")
@click.option("-r", "--replications", "reps",
              type=int, default=1,
              help="replication number for naming output files")
@click.option("-s", "--seed", "seed",
              type=int, default=886,
              help="seed to initialize the replications")
@click.option("-f", "--fresh", "start_fresh",
              is_flag=True,
              help="erase database before experiments")
@click.argument("algorithm", type=str)
def main(M, warmup, reps, seed, start_fresh, algorithm):
    dbpath = "experiments.db"
    db.init_ar1(dbpath, start_fresh)
    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "ar1"
    source_dir = Path(__file__).resolve().parent
    stan_model = source_dir / f"stan/{model}.stan"
    stan_data = source_dir / f"stan/{model}.json"

    algorithms = {
        "klhr": KLHR,
        "klhr_sinh": KLHRSINH,
        "klhr_sub_sinh": KLHRSUBSINH,
        "slice": Slice,
        "klhr_reflection": KLHRReflection,
    }

    alphas = 0.1 * np.array([9]) # np.arange(10)
    acceptance_rates = np.zeros_like(alphas)
    for adx, alpha in enumerate(alphas):
        with open(stan_data, "r") as f:
            data = json.load(f)
        data["alpha"] = alpha
        with open(stan_data, "w") as f:
            json.dump(data, f)

        bs_model = BSModel(stan_file = stan_model,
                           data_file = stan_data)

        seedi = np.random.SeedSequence([seed, adx])
        algo = algorithms[algorithm](bs_model,
                                     warmup = warmup,
                                     seed = seedi)
        start = time.perf_counter()
        draws = algo.sample(M)
        runtime = time.perf_counter() - start

        msjd = 0.0
        for m in range(M-1):
            d = np.linalg.norm(draws[m+1] - draws[m])
            msjd += d / (m + 1)
        ldevals = algo.ld_evals if algorithm == "slice" else algo.grad_evals
        d = {
            "algorithm": algorithm,
            "alpha": alpha,
            "msjd": msjd,
            "acceptance_rate": algo.acceptance_probability,
            "ld_evals": ldevals,
            "runtime": runtime,
        }
        acceptance_rates[adx] = algo.acceptance_probability

        m = np.mean(draws[warmup:, :], axis = 0)
        v = np.var(draws[warmup:, :], ddof = 1, axis = 0)
        d |= {
            "m1": m[0],
            "v1": v[0],
            "max_dist_mean": np.max(np.abs(m)),
            "max_dist_var": np.max(np.abs(v - 1)),
            "prop_mean_g0": np.mean(m > 0),
            "prop_var_g1": np.mean(v > 1),
        }
        db.append_df(dbpath, "ar1", d)

        plt.clf()
        for d in range(algo.D):
            plt.hist(draws[warmup:, d], histtype = "step",
                     density = True, color = "#0072B2", alpha = 0.1)
            x = np.linspace(-4, 4, 301)
            fx = st.norm().pdf(x)
            plt.plot(x, fx, linestyle = "dashed", color = "#D55E00")
            plt.tight_layout()
            plt.savefig(source_dir / f"experiments/ar1/{algorithm}_{np.round(alpha, 2)}.png")

    plt.clf()
    plt.scatter(alphas, acceptance_rates, color = "#0072B2")
    plt.xlabel("alpha")
    plt.ylabel("acceptance probability")
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/ar1/acceptance_probabilities.png")
    plt.close()

if __name__ == "__main__":
    main()
