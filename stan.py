import cmdstanpy as csp
import numpy as np
import time
import matplotlib.pyplot as plt

model_name = "ssp3nc3r"
model = csp.CmdStanModel(stan_file=f"./stan/{model_name}.stan")
start = time.perf_counter()
fit = model.sample(data=f"./stan/{model_name}.json",
                           chains = 1,
                           save_warmup = True,
                           # iter_warmup = 15_000,
                           # iter_sampling = 15_000
                   )
runtime = time.perf_counter() - start
print(f"time: {runtime}")
df = fit.draws_pd(inc_warmup = True)
print(f"log_density evals: {np.sum(df['n_leapfrog__'])}")
print(f"number draws: {np.shape(df)[0]}")
print(fit.summary())

draws = df.iloc[:, 10:].values
M = np.shape(draws)[0]
warmup = M // 2
idx = np.arange(warmup) + 1

plt.clf()
fig = plt.figure(figsize = (14, 9))
gs = fig.add_gridspec(3, 1)

axs = np.empty(3, dtype = object)
axs[0] = fig.add_subplot(gs[0])
axs[1] = fig.add_subplot(gs[1])
axs[2] = fig.add_subplot(gs[2])

axs[0].plot(idx, draws[warmup:, 0])
axs[1].plot(idx, draws[warmup:, 1])
axs[2].plot(idx, draws[warmup:, 2])

plt.tight_layout()
plt.savefig(f"draws/{model_name}_stan_trace_plot.png")
plt.close()
