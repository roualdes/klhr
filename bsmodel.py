import numpy as np

import bridgestan as bs

class BSModel():
    def __init__(self, stan_file = "", data_file = "",
                 stepsize = 1.0, warn = False):
        self._stan_file = stan_file
        self._data_file = data_file
        make_args = ["STAN_THREADS=True",
                     "CXXFLAGS += -march=native",
                     "STANCFLAGS+= --warn-pedantic --O1"]
        self.model = bs.StanModel(self._stan_file,
                                  data = self._data_file,
                                  make_args=make_args,
                                  warn = warn)

    def log_density(self, theta, **kws):
        ld = -np.inf
        try:
            ld = self.model.log_density(theta, **kws)
        except Exception as e:
            pass
        return ld

    # def log_density_batch_1(self, theta, rho, xis, **kws):
    #     ld = self.model.log_density_batch_1(theta, rho, xis, **kws)
    #     ld = np.nan_to_num(ld, nan=-np.inf)
    #     return ld

    # def log_density_batch_2(self, theta, rho1, rho2, xi1s, xi2s, **kws):
    #     ld = self.model.log_density_batch_2(theta,
    #                                         rho1,
    #                                         rho2,
    #                                         xi1s,
    #                                         xi2s,
    #                                         **kws)
    #     ld = np.nan_to_num(ld, nan=-np.inf)
    #     return ld

    def log_density_gradient(self, theta, **kws):
        ld = -np.inf
        grad = np.zeros_like(theta)
        try:
            ld, grad = self.model.log_density_gradient(theta, **kws)
        except Exception as e:
            pass
        return ld, grad

    def log_density_hvp(self, theta, v, **kws):
        ld = -np.inf
        D = self.dim()
        hvp = np.zeros_like(theta)
        try:
            ld, hvp = self.model.log_density_hessian_vector_product(theta, v)
        except Exception as e:
            pass
        return ld, hvp

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
