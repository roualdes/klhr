import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple

from bridgenumpyro import BridgeNumpyro

class NPRModel():
    def __init__(self, model, data, seed = None):
        self.model = BridgeNumpyro(model, data, seed = seed)

    def dim(self):
        return self.model.param_unc_num()

    def _to_map(self, theta: npt.NDArray[np.float64]) -> Dict:
        return self.model.unflatten(jnp.asarray(theta))

    def log_density(self, theta: npt.NDArray[np.float64], **kws) -> float:
        ld = -np.inf
        try:
            ld = self.model.logdensity(self._to_map(theta))
        except Exception as e:
            print(f"error: {e}")
            pass
        return ld

    def log_density_gradient(self, theta: npt.NDArray[np.float64], **kws) -> Tuple[float, npt.NDArray[np.float64]]:
        ld = -np.inf
        grad = np.zeros_like(theta)
        try:
            ld = self.log_density(theta)
            grad = self.model.gradient(self._to_map(theta))
        except Exception as e:
            pass
        return ld, np.asarray(grad)

    def log_density_hessian(self, theta: npt.NDArray[np.float64], **kws) -> Tuple[float, npt.NDArray[np.float64]]:
        ld = -np.inf
        D = self.dim()
        H = np.zeros((D, D))
        try:
            ld = self.log_density(theta)
            H = self.model.hessian(self._to_map(theta))
        except Exception as e:
            pass
        return ld, np.asarray(H)

    def constrain(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.model.param_constrain(self._to_map(theta))

    def parameter_names(self):
        return list(self._initial_theta_map.keys())
