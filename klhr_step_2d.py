import warnings
import sys

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, root_scalar
from scipy.special import softmax
import scipy.stats as st

from bsmodel import BSModel
from onlinemoments import OnlineMoments
from onlinepca import OnlinePCA
from mcmc import MCMCBase
from windowedadaptation import WindowedAdaptation

class KLHRStep(MCMCBase):
    def __init__(self,
                 bsmodel,
                 theta = None,
                 seed = None,
                 B = 32,
                 J = 2,
                 l = 0,
                 initscale = 0.1,
                 warmup = 1_000,
                 windowsize = 50,
                 windowscale = 2,
                 tol = 1e-10,
                 **kwargs):
        super().__init__(bsmodel, -1, theta = theta, seed = seed)

        self.B = B
        self.J = J
        self.l = l
        self._tol = tol

        self._initscale = initscale
        self._windowedadaptation = \
            WindowedAdaptation(warmup,
                               windowsize = windowsize,
                               windowscale = windowscale)
        self._onlinemoments = OnlineMoments(self.D)
        self._mean = np.zeros(self.D)
        self._cov = np.ones(self.D)
        self._onlinepca = OnlinePCA(self.D, K = self.J, l = self.l)
        self._eigvecs = np.zeros((self.D, self.J + 1))
        self._eigvals = np.ones(self.J + 1)

        self._draw = 0
        self.acceptance_probability = 0
        self.grad_evals = 0

    def _Newton_step(self, g, h, x0, step=1.0, max_step=2**10):
        g0 = g(x0)
        self.grad_evals += 1

        x = x0
        h0 = h(x0)
        self.grad_evals += 2

        if (not np.isfinite(h0)) or h0 <= self._tol:
            direction = -np.sign(g0)
            x = x0 + direction * step
        else:
            dx = np.clip(-g0 / h0, -max_step, max_step)
            if dx == 0:
                dx = -np.sign(g0) * step
            x = x0 + dx
        return g0, x

    def _logp_rho_hess(self, rho):
        rho0, rho1 = rho
        def hess(xi):
            y = xi[0] * rho0 + xi[1] * rho1 + self.theta
            _, hvp0 = self.model.log_density_hvp(y, rho0)
            _, hvp1 = self.model.log_density_hvp(y, rho1)
            # return np.array([[rho1.dot(hvp1), rho2.dot(hvp1)],
            #                  [rho1.dot(hvp2), rho2.dot(hvp2)]])
            return np.array([[rho0.dot(hvp0), 0.0],
                             [0.0, rho1.dot(hvp1)]])
        return hess

    def logp_grad_rho(self, xi, rho):
        rho0, rho1 = rho
        y = xi[0] * rho0 + xi[1] * rho1 + self.theta
        l, g = self.model.log_density_gradient(y)
        return -l, -np.array([g.dot(rho0), g.dot(rho1)])

    def fit_bfgs(self, rho):
        o = minimize(self.logp_grad_rho,
                     self.rng.normal(size = 2) * self._initscale,
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        self.grad_evals += o["nfev"]
        H = self._logp_rho_hess(rho)
        s = -0.5 * np.log(-np.diag(H(o.x)))
        return o.x, np.exp(s)

    def _proposal(self, eta, rho):
        m, s = eta
        rho0, rho1 = rho
        xi_step = (self.rng.uniform(size=2) - 0.5) * s
        step = xi_step[0] * rho0 + xi_step[1] * rho1
        theta_p = step + self.theta
        r = self.model.log_density(theta_p)
        r -= self.model.log_density(self.theta)
        a = np.log(self.rng.uniform()) < np.minimum(0, r)

        self.theta += step * a
        if np.size(self.theta) != 4:
            print("MH step:")
            print(np.size(self.theta))
        d = a - self.acceptance_probability
        self.acceptance_probability += d / self._draw
        lb0, ub0 = m[0] - 10 * s[0], m[0] + 10 * s[0]
        lb1, ub1 = m[1] - 10 * s[1], m[1] + 10 * s[1]
        xi0s = np.linspace(lb0, ub0, self.B)
        xi1s = np.linspace(lb1, ub1, self.B)
        lds = self.model.log_density_batch_2(self.theta,
                                             rho0,
                                             rho1,
                                             xi0s,
                                             xi1s)
        xis = np.stack(np.meshgrid(xi0s, xi1s), -1).reshape(-1, 2)
        return self.rng.choice(xis, p = softmax(lds))

    def _metropolis_step(self, eta, rho):
        xi_prop = self._proposal(eta, rho)
        rho0, rho1 = rho
        self.theta = xi_prop[0] * rho0 + xi_prop[1] * rho1 + self.theta
        if np.size(self.theta) != 4:
            print("update theta")
            print(np.size(self.theta))
        return self.theta

    def _random_direction(self):
        m = self._eigvecs[:, 0]
        S = np.diag(self._cov)
        rho0 = self.rng.multivariate_normal(m, S)
        while True:
            n0 = np.linalg.norm(rho0)
            if n0 > self._tol:
                break
            rho0 = self.rng.multivariate_normal(m, S)
        rho0 /= np.linalg.norm(rho0)

        m = np.zeros_like(rho0)
        rho1 = self.rng.multivariate_normal(m, S)
        while True:
            n1 = np.linalg.norm(rho1)
            if n1 > self._tol:
                break
            rho1 = self.rng.multivariate_normal(m, S)
        rho1 -= rho0.dot(rho1) * rho0
        rho1 /= np.linalg.norm(rho1)
        return rho0, rho1

    def draw(self):
        self._draw += 1
        if self._draw % 1_000 == 0:
            print(self._draw)
        rho = self._random_direction()
        eta = self.fit_bfgs(rho)
        theta = self._metropolis_step(eta, rho)

        if self._windowedadaptation.window_closed(self._draw):
            self._mean = self._onlinemoments.mean()
            self._cov = self._onlinemoments.var()
            self._onlinemoments.reset()
            self._eigvecs[:, :self.J] = self._onlinepca.vectors()
            self._eigvals[:self.J] = self._onlinepca.values()
            self._onlinepca.reset()
        else:
            self._onlinemoments.update(theta)
            self._onlinepca.update(theta - self._mean)
        return theta

if __name__ == "__main__":

    import numpy as np
    from pathlib import Path
    from scipy.differentiate import jacobian

    import bridgestan as bs
    from bsmodel import BSModel
    from klhr_sinh import KLHRSINH
    from klhr import KLHR

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "earnings"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    algo = KLHR(bs_model)

    rng = np.random.default_rng()
    rho = rng.multivariate_normal(np.zeros(algo.D), np.eye(algo.D))
    rho /= np.linalg.norm(rho)

    def f(x):
        def inner(x):
            vf = lambda x: algo.KL(x, rho)[0]
            return np.apply_along_axis(vf, axis=0, arr=x)
        return np.array([inner(x)])

    x = rng.normal(size = 2) * 0.1
    approx_grad = jacobian(f, x)
    grad = algo.KL(x, rho)[1]
    assert np.all(approx_grad.success)
    assert np.allclose(grad, approx_grad.df)
