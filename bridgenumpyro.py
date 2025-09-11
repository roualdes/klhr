import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model
from typing import Dict, Tuple

class BridgeNumpyro():
    def __init__(self, model, data, seed = None):

        self._model = model
        self._data = data
        if seed is None:
            rng = np.random.default_rng()
            seed = rng.integers(np.iinfo(np.int32).max)
        self._key = jax.random.PRNGKey(seed=seed)
        self._param_info, self._potential, self._postprocess, _ = initialize_model(self._key,
                                                                                   self._model,
                                                                                   model_args = self._data,
                                                                                   dynamic_args = True)
        self._initial_theta_map = self._param_info.z
        self._initial_theta, self._unflatten = ravel_pytree(self._initial_theta_map)
        self._D = jnp.size(self._initial_theta)
        self._potential = jax.jit(self._potential(*self._data))
        self._gradient = jax.jit(jax.grad(self._potential))
        self._hessian = jax.jit(jax.hessian(self._potential))

    def logdensity(self, theta_map: Dict) -> jax.Array:
        return -self._potential(theta_map)

    def gradient(self, theta_map: Dict) -> jax.Array:
        return -self.flatten(self._gradient(theta_map))

    def hessian(self, theta_map: Dict) -> jax.Array:
        D = self.param_unc_num()
        return -self.flatten(self._hessian(theta_map)).reshape(D, D)

    def flatten(self, theta_map: Dict) -> jax.Array:
        theta, _ = jax.flatten_util.ravel_pytree(theta_map)
        return theta

    def unflatten(self, theta: jax.Array) -> Dict:
        return self._unflatten(theta)

    def param_constrain(self, theta_unc: Dict) -> Dict:
        return self._postprocess(*self._data)(theta_unc)

    def param_unc_num(self) -> int:
        return self._D

if __name__ == "__main__":

    def model():
        mu = numpyro.sample("mu", dist.Normal(0., 1.))
        sigma = numpyro.sample("sigma", dist.Exponential(1.))
        y = numpyro.sample("y", dist.Normal(mu, sigma))

    npr_model = BridgeNumpyro(model, ())
    x = jnp.asarray([0.5, 0.1, -0.2])
    theta = npr_model.unflatten(x)

    assert jnp.isclose(npr_model.logdensity(theta), -3.2686370189890988)
    assert jnp.allclose(npr_model.gradient(theta), jnp.asarray([-1.07311153, -0.70399285,  0.57311153]))
    assert jnp.isclose(npr_model.param_unc_num(), 3)
