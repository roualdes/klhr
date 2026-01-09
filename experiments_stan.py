import click
import cmdstanpy as csp
import pandas as pd
import numpy as np
import time

import db

@click.command()
@click.option("-f", "--fresh", "start_fresh",
              is_flag=True,
              help="erase database before experiments")
def main(start_fresh):
    dbpath = "experiments.db"
    db.init_mmodels(dbpath, start_fresh)

    models = [
        "corr-normal",
        "garch",
        "glmm-poisson",
        "hmm",
        "ill-normal",
        "lotka-volterra",
    ]

    for model_name in models:
        model = csp.CmdStanModel(stan_file=f"./stan/{model_name}.stan")
        start = time.perf_counter()
        fit = model.sample(data=f"./stan/{model_name}.json",
                           chains = 1,
                           iter_warmup = 2_000,
                           iter_sampling = 5_000,
                           show_console = False,
                           show_progress = False)
        runtime = time.perf_counter() - start

        df = fit.draws_pd()
        msjd = 0.0
        for n in range(np.shape(df)[0] - 1):
            d = np.linalg.norm(df.iloc[n+1] - df.iloc[n]) - msjd
            msjd += d / (n + 1)
        ldevals = df["n_leapfrog__"].sum()
        draws = df.iloc[:, 10:].values
        m = np.mean(draws, axis = 0)
        s = np.std(draws, ddof = 1, axis = 0)
        draws2 = draws ** 2
        m2 = np.mean(draws2, axis = 0)
        s2 = np.std(draws2, ddof = 1, axis = 0)
        d = {
            "algorithm": "stan",
            "model": model_name,
            "replication": 0,
            "acceptance_rate": 1,
            "msjd": float(msjd),
            "ld_evals": float(ldevals)
        }
        pnames = list(df.columns[10:])
        d |= {pname + "m": m[p] for p, pname in enumerate(pnames)}
        d |= {pname + "s": s[p] for p, pname in enumerate(pnames)}
        d |= {pname + "m2": m2[p] for p, pname in enumerate(pnames)}
        d |= {pname + "s2": s2[p] for p, pname in enumerate(pnames)}
        db.append_df(dbpath, model_name.replace("-", ""), d)

if __name__ == "__main__":
    main()
