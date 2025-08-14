import numpy as np

import bridgestan as bs

class BSModel():
    def __init__(self, stan_file = "", data_file = "",
                 stepsize = 1.0, warn = False):
        self._stan_file = stan_file
        self._data_file = data_file
        self.model = bs.StanModel(self._stan_file,
                                  data = self._data_file,
                                  make_args=["STAN_THREADS=True"],
                                  warn = warn)

    def log_density(self, theta, **kws):
        ld = -np.inf
        try:
            ld = self.model.log_density(theta, **kws)
        except Exception as e:
            print(f"error: {e}")
            pass
        return ld

    def log_density_gradient(self, theta, **kws):
        ld = -np.inf
        grad = np.zeros_like(theta)
        try:
            ld, grad = self.model.log_density_gradient(theta, **kws)
        except Exception as e:
            pass
        return ld, grad

    def dim(self):
        return self.model.param_unc_num()

    def Hamiltonian(self, theta, rho):
        return -self.log_density(theta) + 0.5 * rho.dot(rho)

    def unconstrain(self, theta):
        return self.model.param_unconstrain(theta)

    def constrain(self, theta):
        return self.model.param_constrain(theta)

    def parameter_names(self):
        return self.model.param_names()
