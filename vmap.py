import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model

from pathlib import Path
from scipy.stats import norm
from scipy.optimize import minimize

from bsmodel import BSModel
import bridgestan as bs
from klhr import KLHR
from klhr_sinh import KLHRSINH

# bs.set_bridgestan_path(Path.home() / "bridgestan")

# bs_model = BSModel(stan_file = "stan/one_exponential.stan",
#                    data_file = "stan/one_exponential.json")

def model():
    x = numpyro.sample("x", dist.Exponential(30.0))

key = jax.random.PRNGKey(seed = 0)
param_info, potential, postprocess, _ = initialize_model(key, model, dynamic_args = True)

rng = np.random.default_rng()
theta0 = rng.normal(size = 1)
rho = rng.normal(size = 1)
rho /= np.linalg.norm(rho)

def logdensity(theta):
    return -potential()(theta)

theta_init = param_info.z
_, unflatten = ravel_pytree(theta_init)

print(f"logdensity(theta) = {logdensity(unflatten(theta0))}")
