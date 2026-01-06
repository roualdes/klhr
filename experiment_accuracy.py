from pathlib import Path
import time

import click
import matplotlib.pyplot as plt
import numpy as np

import bridgestan as bs
from bsmodel import BSModel
import db
from klhr import KLHR
from klhr_sinh import KLHRSINH
from klhr_sub_sinh import KLHRSUBSINH
from slice import Slice
from klhr_step import KLHRStep

from mh import MH
from onlinemoments import OnlineMoments

@click.command()
@click.option("-M", "--iterations", "M",
              type=int, default=1_000,
              help="number of iterations")
@click.option("-w", "--warmup", "warmup",
              type=int, default=0,
              help="set value from which RMSEs are plot")
@click.option("-s", "--seed", "seed",
              type=int, default=123,
              help="seed to initialize the replications")
@click.option("-f", "--fresh", "start_fresh",
              is_flag=True,
              help="erase database before experiments")
@click.argument("algorithm", type=str)
def main(M, warmup, seed, start_fresh, algorithm):
    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")
    model = "normal"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    dbpath = "experiments.db"
    tbl = "accuracy"
    db.make_table(dbpath, bs_model, tbl ,start_fresh)

    algorithms = {
        "klhr": KLHR,
        "klhr_sinh": KLHRSINH,
        "klhr_sub_sinh": KLHRSUBSINH,
        "slice": Slice,
        "klhr_step": KLHRStep,
    }

    if seed < 0:
        seed = np.random.SeedSequence().entropy

    algo = algorithms[algorithm](bs_model,
                                 warmup = warmup,
                                 seed = seed)

     start = time.perf_counter()
     klhr_draws = algo.sample(M)
     runtime = time.perf_counter() - start
     mdx = np.arange(M)

     msjd = 0.0
     for m in range(M-1):
         d = np.linalg.norm(klhr_draws[m+1] - klhr_draws[m]) - msjd
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
     db.update_table(dbpath,
                     tbl,
                     algorithm,
                     algo,
                     klhr_draws,
                     warmup,
                     0,
                     runtime)

    # when D = 2 => stepsize = 2.4
    # when D = 100 => stepsize = 0.24
    stepsize = 2.4 if algo.D == 2 else 0.24
    mh = MH(bs_model, stepsize)

    start = time.perf_counter()
     mh_draws = mh.sample(M)
     runtime = time.perf_counter() - start

     msjd = 0.0
     for m in range(M-1):
         d = np.linalg.norm(mh_draws[m+1] - mh_draws[m]) - msjd
         msjd += d / (m + 1)
    ldevals = mh._ld_evals
    d = {
        "algorithm": "mh",
        "msjd": msjd,
        "acceptance_rate": mh.acceptance_probability,
        "ld_evals": mh._ld_evals,
        "runtime": runtime,
    }
    db.append_df(dbpath, "funnel", d)
    # TODO fix this name mismatch
    mh.grad_evals = mh._ld_evals
    db.update_table(dbpath,
                    tbl,
                    "mh",
                    mh,
                    mh_draws,
                    warmup,
                    0,
                    runtime)

    stats_klhr = {
        "om": OnlineMoments(algo.D),
        "rmse_mean": np.zeros(M),
        "rmse_var": np.zeros(M),
        "log_density": np.zeros(M),
    }

    stats_mh = {
        "om": OnlineMoments(algo.D),
        "rmse_mean": np.zeros(M),
        "rmse_var": np.zeros(M),
        "log_density": np.zeros(M),
    }

    mdx = np.arange(M)
    klhr_msjd = 0.0
    mh_msjd = 0.0
    klhr_prev_draw = algo.theta
    mh_prev_draw = mh.theta

    rng = np.random.default_rng()
    mu = np.zeros(algo.D)
    Sigma = np.eye(algo.D)
    log_density_iid = np.zeros(M)

    for m in range(M-1):
        klhr_draw = algo.draw()
        d = np.linalg.norm(klhr_draw - klhr_prev_draw) - klhr_msjd
        klhr_msjd += d / (m + 1)
        klhr_prev_draw = klhr_draw
        mh_draw = mh.draw()
        d = np.linalg.norm(mh_draw - mh_prev_draw) - mh_msjd
        mh_msjd += d / (m + 1)
        mh_prev_draw = mh_draw

        stats_klhr["om"].update(klhr_draw)
        stats_klhr["rmse_mean"][m] = np.sqrt(np.mean( stats_klhr["om"].mean() ** 2) )
        stats_klhr["rmse_var"][m] = np.sqrt(np.mean( (stats_klhr["om"].var() - 1) ** 2 ))
        stats_klhr["log_density"][m] = bs_model.log_density(klhr_draw)

        stats_mh["om"].update(mh_draw)
        stats_mh["rmse_mean"][m] = np.sqrt(np.mean( stats_mh["om"].mean() ** 2) )
        stats_mh["rmse_var"][m] = np.sqrt(np.mean( (stats_mh["om"].var() - 1) ** 2 ))
        stats_mh["log_density"][m] = bs_model.log_density(mh_draw)

        x = rng.multivariate_normal(mu, Sigma)
        log_density_iid[m] = bs_model.log_density(x)

    ldevals = algo.ld_evals if algorithm == "slice" else algo.grad_evals
    d = {
        "algorithm": algorithm,
        "msjd": klhr_msjd,
        "acceptance_rate": algo.acceptance_probability,
        "ld_evals": ldevals,
        "runtime": 0,
    }
    db.append_df(dbpath, "accuracy", d)

    ldevals = mh._ld_evals
    d = {
        "algorithm": "mh",
        "msjd": mh_msjd,
        "acceptance_rate": mh.acceptance_probability,
        "ld_evals": mh._ld_evals,
        "runtime": 0,
    }
    db.append_df(dbpath, "accuracy", d)


    plt.clf()
    plt.rc('axes', labelsize = 12)

    origin = 10 ** 2

    plt.plot(mdx[origin:], stats_klhr["rmse_mean"][origin:],
             label=f"{'KLHR' if algorithm != 'slice' else 'SLICE'}: mean",
             linestyle = "dotted",
             color = "#0072B2",
             linewidth = 2)
    plt.plot(mdx[origin:], stats_klhr["rmse_var"][origin:],
             label=f"{'KLHR' if algorithm != 'slice' else 'SLICE'}: var",
             linestyle = (0, (1, 5)),
             color = "#D55E00",
             linewidth = 2)

    plt.plot(mdx[origin:], stats_mh["rmse_mean"][origin:],
             label="MH: mean",
             linestyle = "dashed",
             color = "#009E73",
             linewidth = 2)
    plt.plot(mdx[origin:], stats_mh["rmse_var"][origin:],
             label="MH: var",
             linestyle = (0, (5, 5)),
             color = "#F0E442",
             linewidth = 2)

    plt.plot([origin, M], [1 / np.sqrt(origin), 1 / np.sqrt(M)],
             linestyle = "solid", color = "black", alpha = 0.2)

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("RMSE")
    plt.xlabel("iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/accuracy/{algorithm}_{warmup}_rmse.png")

    plt.clf()
    plt.rc('axes', labelsize = 12)
    plt.hist(stats_klhr["log_density"][origin:],
             histtype = "step",
             density = True,
             label = f"{'KLHR' if algorithm != 'slice' else 'SLICE'}")

    plt.hist(stats_mh["log_density"][origin:],
             histtype = "step",
             density = True,
             label = "MH")

    plt.hist(log_density_iid[origin:],
             histtype = "step",
             density = True,
             label = "IID")

    plt.xlabel("log_density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(source_dir / f"experiments/accuracy/{algorithm}_{warmup}_histogram_log_density.png")
    plt.close()

if __name__ == "__main__":
    main()
