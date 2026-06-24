import cmdstanpy as csp
import numpy as np
import time

model_name = "garch"
model = csp.CmdStanModel(stan_file=f"./stan/{model_name}.stan")
start = time.perf_counter()
fit = model.sample(data=f"./stan/{model_name}.json",
                           chains = 1,
                           save_warmup = True,
                           iter_warmup = 15_000,
                           iter_sampling = 15_000)
runtime = time.perf_counter() - start
print(f"time: {runtime}")
df = fit.draws_pd(inc_warmup = True)
print(f"log_density evals: {np.sum(df['n_leapfrog__'])}")
print(f"number draws: {2 * np.shape(df)[0]}")
print(fit.summary())
