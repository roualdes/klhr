import matplotlib.pyplot as plt
import pandas as pd

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, BarkerMH



rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)


# Run NUTS.
kernel = BarkerMH(model, step_size = 0.1) # NUTS(model)
num_warmup = 15_000
num_samples = 15_000
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
            progress_bar=False)
mcmc.warmup(rng_key_, earn = earn, height = height, collect_warmup=True,
            extra_fields=("adapt_state.step_size", "adapt_state.inverse_mass_matrix"))

warmup_samples = mcmc.get_samples()
print(f"warmup_samples shape = {jnp.shape(warmup_samples['b0'])}")

extras = mcmc.get_extra_fields()
step_sizes_over_time = extras["adapt_state.step_size"]                  # shape: (chains, warmup+samples)
# inv_mass_over_time   = extras["adapt_state.inverse_mass_matrix"]
print(f"stepsize = {step_sizes_over_time}")

mcmc.run(rng_key_, earn = earn, height = height,
         extra_fields=("adapt_state.step_size", "adapt_state.inverse_mass_matrix"),
)
mcmc.print_summary()
samples = mcmc.get_samples()

b0 = samples["b0"]
b1 = samples["b1"]
sigma = samples["sigma"]
s = samples["s"]
mdx = jnp.arange(num_samples)

extras = mcmc.get_extra_fields(group_by_chain=True)
step_sizes_over_time = extras["adapt_state.step_size"]                  # shape: (chains, warmup+samples)
inv_mass_over_time   = extras["adapt_state.inverse_mass_matrix"]

print(f"stepsize = {step_sizes_over_time}")

plt.clf()
fig, axs = plt.subplots(2, 2, figsize = (14, 6))
axs[0, 0].plot(mdx, b0)
axs[0, 0].set_ylabel(r"$\beta_0$")

axs[0, 1].plot(mdx, b1)
axs[0, 1].set_ylabel(r"$\beta_1$")

axs[1, 0].plot(mdx, sigma)
axs[1, 0].set_ylabel(r"$\sigma$")

axs[1, 1].plot(mdx, s)
axs[1, 1].set_ylabel(r"$s$")

plt.tight_layout()
plt.savefig("barker_trace.png")
