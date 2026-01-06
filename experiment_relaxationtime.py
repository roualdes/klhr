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
from klhr_step import KLHRStep
from slice import Slice

def make_plot(draws, algorithm, rep, source_dir):
    M = np.shape(draws)[0]
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
    plot_name = f"experiments/relaxationtime/{algorithm}_{rep:0>2}.png"
    plot_path = source_dir / plot_name
    plt.savefig(plot_path)
    plt.close()

@click.command()
@click.option("-M", "--iterations", "M", type=int, default=2_000,
              help="number of iterations, including warmup")
@click.option("-w", "--warmup", "warmup", type=int, default=1_000,
              help="number of warmup iterations")
@click.option("-r", "--replications", "reps", type=int, default=1,
              help="replication number for naming output files")
@click.option("-s", "--seed", "seed", type=int, default=204,
              help="seed to initialize the replications")
@click.option("-f", "--fresh", "start_fresh", is_flag=True,
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

    dbpath = "experiments.db"
    tbl = "relaxationtime"
    db.make_table(dbpath, bs_model, tbl, start_fresh)

    algorithms = {
        "klhr": KLHR,
        "klhr_sinh": KLHRSINH,
        "klhr_sub_sinh": KLHRSUBSINH,
        "slice": Slice,
        "klhr_step": KLHRStep,
    }

    if seed < 0:
        seed = np.random.SeedSequence().entropy

    for rep in range(reps):
        seedi = np.random.SeedSequence([seed, rep])
        algo = algorithms[algorithm](bs_model,
                                     warmup = warmup,
                                     seed = seedi)
        start = time.perf_counter()
        draws = algo.sample(M)
        runtime = time.perf_counter() - start

        db.update_table(dbpath,
                        tbl,
                        algorithm,
                        algo,
                        draws,
                        warmup,
                        rep,
                        runtime)
        make_plot(draws, algorithm, rep, source_dir)

if __name__ == "__main__":
    main()
