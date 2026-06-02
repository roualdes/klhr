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
                 B = 128,
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

    def _bracket(self, g, h, x0, step=1.0, grow=2.0, max_iter=50,
                max_jump=2 ** 10):
        g0, x = self._Newton_step(g, h, x0, step=step, max_step=max_jump)
        gx = g(x)
        self.grad_evals += 1
        if g0 * gx < 0:
            if x0 < x:
                a, b = x0, x
            else:
                a, b = x, x0
        else:
            direction = -np.sign(gx)
            a = x
            ga = gx
            s = step

            for _ in range(max_iter):
                b = x + direction * s
                gb = g(b)
                self.grad_evals += 1
                if ga * gb <= 0:
                    a, b = (a, b) if a < b else (b, a)
                    break
                a, ga = b, gb
                s *= grow
            else:
                # TODO need safer exit strategy
                raise RuntimeError("couldn't find bracket")
        return a[0], b[0]

    def _logp_rho_grad(self, rho):
        def grad(xi):
            _, g = self.model.log_density_gradient(xi * rho + self.theta)
            return -g.dot(rho)
        return grad

    def _logp_rho_hess(self, rho):
        def hess(xi):
            theta = xi * rho + self.theta
            _, hvp = self.model.log_density_hvp(theta, rho)
            return rho.dot(hvp)
        return hess

    def _approx_normal_tails(self, m, x, rho):
        theta_m = m * rho + self.theta
        theta_x = x * rho + self.theta
        dx = self.model.log_density(theta_x)
        dx -= self.model.log_density(theta_m)
        self.grad_evals += 2
        s = np.abs(x - m) / np.sqrt(2 * np.abs(dx) + self._tol)
        return s

    def _approx_t_tails(self, m, x, rho, nu = 5):
        theta_m = m * rho + self.theta
        theta_x = x * rho + self.theta
        dx = self.model.log_density(theta_x)
        dx -= self.model.log_density(theta_m)
        self.grad_evals += 2
        s = np.abs(x - m) / np.sqrt(nu * (np.exp(-2 * np.abs(dx)/ (nu + 1)) - 1))
        return s

    def fit_root(self, rho):
        g = self._logp_rho_grad(rho)
        h = self._logp_rho_hess(rho)
        bracket = self._bracket(g, h, np.zeros(1))
        res = root_scalar(g, bracket=bracket, method="toms748",
                          xtol = 1e-8, rtol = 1e-8)
        self.grad_evals += res["function_calls"]
        m = res.root
        h = -h(m)
        self.grad_evals += 2
        s = 0.0
        if np.isfinite(h) and h > self._tol:
            s = -0.5 * np.log(h)
        s = np.exp(s)
        # TODO maybe a try around this
        # experiment with either shrinking the step, 0.1
        # or going to normal
        # d = 1.0
        # s = self._approx_t_tails(m, m + d, rho)
        # if ~np.isfinite(s):
        #     for _ in range(3):
        #         s = self._approx_normal_tails(m, m + d, rho)
        #         if np.isfinite(s):
        #             break
        #         d *= 0.5
        # if ~np.isfinite(s):
        #     s = 1.0
        return np.array([m, s])

    def logp_grad_rho(self, xi, rho):
        l, g = self.model.log_density_gradient(xi * rho + self.theta)
        return -l, -g.dot(rho)

    def fit_bfgs(self, rho):
        g = self._logp_rho_grad(rho)
        H = self._logp_rho_hess(rho)
        x0 = self.rng.normal() * self._initscale
        # _, x = self._Newton_step(g, H, x0)
        o = minimize(self.logp_grad_rho,
                     x0,
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        self.grad_evals += o["nfev"]
        m = o.x[0]
        # s = np.exp(-0.5 * np.log(-H(m)))
        x = 0.1 * (1 + np.abs(m))
        s = self._approx_normal_tails(m, x, rho)
        # hi = o["hess_inv"][0, 0]
        # if hi > self._tol and np.isfinite(hi):
        #     s = 0.5 * np.log(hi)
        # else:
        #     H = self._logp_rho_hess(rho)
        #     nh = -H(o.x[0])
        #     self.grad_evals += 2
        #     if np.isfinite(nh) and nh > self._tol:
        #         s = -0.5 * np.log(nh)
        #     else:
        #         s = 0.0
        return np.array([m, s])

    def _proposal(self, eta, rho):
        m, s = eta
        xi_step = (self.rng.uniform() - 0.5) * s
        r = self.model.log_density(xi_step * rho + self.theta)
        r -= self.model.log_density(self.theta)
        a = np.log(self.rng.uniform()) < np.minimum(0, r)
        self.theta += xi_step * rho * a
        d = a - self.acceptance_probability
        self.acceptance_probability += d / self._draw
        xis = np.linspace(m - 10 * s, m + 10 * s, self.B)
        lds = self.model.log_density_batch_1(self.theta, rho, xis)
        return self.rng.choice(xis, p = softmax(lds))

    # def _proposal(self, eta, rho):
    #     m, s = eta
    #     jitter = self.rng.uniform(0.1, 0.5)
    #     stepsize = s * jitter
    #     ld0 = self.model.log_density(self.theta)
    #     xi_prop = 0.0
    #     lsw = -np.inf
    #     for xi in self._sequence(m, s, stepsize):
    #       ld = self.model.log_density(xi * rho + self.theta) - ld0
    #       lsw = np.logaddexp(ld, lsw)
    #       if not np.isfinite(ld): continue
    #       log_alpha = ld - lsw
    #       if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
    #           xi_prop = xi
    #     return xi_prop

    def _metropolis_step(self, eta, rho):
        xi_prop = self._proposal(eta, rho)
        self.theta += xi_prop * rho
        return self.theta

    def _random_direction(self):
        evals = self._eigvals
        p = evals / np.sum(evals)
        j = self.rng.choice(np.size(p), p = p)
        m = self._eigvecs[:, j]
        S = np.diag(self._cov)
        rho = self.rng.multivariate_normal(m, S)
        return rho / np.linalg.norm(rho)

    def draw(self):
        self._draw += 1
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
