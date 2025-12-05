import sys

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
import scipy.stats as st

from bsmodel import BSModel
from smoother import Smoother
from onlinemoments import OnlineMoments
from onlinepca import OnlinePCA
from mcmc import MCMCBase
from windowedadaptation import WindowedAdaptation

class KLHR(MCMCBase):
    def __init__(self,
                 bsmodel,
                 theta = None,
                 seed = None,
                 N = 8,
                 K = 10,
                 J = 2,
                 l = 4,
                 initscale = 0.1,
                 warmup = 1_000,
                 windowsize = 50,
                 windowscale = 2,
                 tol = 1e-12,
                 grad_clip = 1e15,
                 scale_clip = 600,
                 scale_dir_cov = False,
                 overrelaxed = False,
                 eigen_method_one = True,
                 max_init_tries = 100):
        super().__init__(bsmodel, -1, theta = theta, seed = seed)

        self.N = N
        self.K = K
        self.J = J if J < self.D else self.D - 1
        self.l = l
        self._tol = tol
        self._grad_clip = grad_clip
        self._scale_clip = scale_clip
        self._max_init_tries = max_init_tries

        self.x, self.w = hermgauss(self.N)
        # normalize roots and weights
        self.x *= np.sqrt(2)
        self.w /= np.sqrt(np.pi)

        self._initscale = initscale
        self._windowedadaptation = \
            WindowedAdaptation(warmup,
                               windowsize = windowsize,
                               windowscale = windowscale)
        self._onlinemoments = OnlineMoments(self.D)
        self._mean = np.zeros(self.D)
        self._cov = np.ones(self.D)
        self._scale_dir_cov = scale_dir_cov
        self._overrelaxed = overrelaxed
        self._eigen_method_one = eigen_method_one
        self._onlinemoments_density = OnlineMoments(self.D)
        self._onlinepca = OnlinePCA(self.D, K = self.J, l = self.l)
        if eigen_method_one:
            self._eigvecs = np.zeros((self.D, self.J + 1))
            self._eigvals = np.ones(self.J + 1)
        else:
            self._eigvecs = np.zeros((self.D, self.J))
            self._eigvals = np.ones(self.J)

        self._smoothK = Smoother(self.K)
        self._prev_theta = np.zeros(self.D)
        self._msjd = 0.0

        self._draw = 0
        self.acceptance_probability = 0
        self.grad_evals = 0

        self._initialize()

    def _unpack(self, eta):
        m = eta[0]
        log_mn, log_mx = -self._scale_clip, self._scale_clip
        s = np.exp(np.clip(eta[1], log_mn, log_mx))
        return m, s

    def _initialize(self):
        tries = 0
        while True:
            tries += 1
            init = self.rng.normal(size = self.D) * self._initscale
            l, g = self.model.log_density_gradient(init)
            if np.isfinite(l) and np.isfinite(np.linalg.norm(g)):
                self.theta = init
                break

            if tries >= self._max_init_tries:
                print("failed to initialize")
                sys.exit(1)

    def _logp_grad(self, x):
        l, g = self.model.log_density_gradient(x)
        mn, mx = -self._grad_clip, self._grad_clip
        return l, np.clip(g, mn, mx)

    def KL(self, eta, rho):
        m, s = self._unpack(eta)
        out = 0.0
        grad = np.zeros(2)
        for xn, wn in zip(self.x, self.w):
            y = s * xn + m
            xi = y * rho + self.theta
            logp, grad_logp = self.model.log_density_gradient(xi)
            out += wn * logp
            w_grad_rho = wn * grad_logp.dot(rho)
            grad[0] += w_grad_rho
            grad[1] += w_grad_rho * xn * s
        out += eta[1]
        grad[1] += 1
        return -out, -grad

    def logp_grad_rho(self, xi, rho):
        l, g = self.model.log_density_gradient(xi * rho + self.theta)
        return -l, -g.dot(rho)

    def fit(self, rho):
        o = minimize(self.logp_grad_rho,
                     self.rng.normal() * self._initscale,
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        self.grad_evals += o["nfev"]
        s = o["hess_inv"][0, 0]
        init = np.array([o.x[0], (s > 0) * 0.5 * np.log(s)])
        o = minimize(self.KL,
                     init,
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        self.grad_evals += o["nfev"] * self.N
        return o.x

    def _random_direction(self):
        evals = self._eigvals
        p = evals / np.sum(evals)
        if self._eigen_method_one:
            j = self.rng.choice(np.size(p), p = p)
            m = self._eigvecs[:, j]
        else:
            m = np.sum(evals * self._eigvecs, axis = 1)
        S = np.diag(self._cov)
        rho = self.rng.multivariate_normal(m, S)
        return rho / np.linalg.norm(rho + self._tol)

    def _logq(self, x, eta):
        m, s = self._unpack(eta)
        z = (x - m) / s
        return -np.log(s) - 0.5 * z * z

    def _overrelaxed_proposal(self, eta):
        m, s = self._unpack(eta)
        K = self.K
        Normal = st.norm(m, s)
        u = Normal.cdf(np.array([0]))
        r = st.binom(K, u).rvs()
        up = u
        if r > K - r:
            v = st.beta(K - r + 1, 2 * r - K).rvs()
            up = u * v
        elif r < K - r:
            v = st.beta(r + 1, K - 2 * r).rvs()
            up = 1 - (1 - u) * v
        return Normal.ppf(up)

    def _metropolis_step(self, eta, rho):
        m, s = self._unpack(eta)
        if self._overrelaxed:
            zp = self._overrelaxed_proposal(eta)
        else:
            zp = self.rng.normal(loc = m, scale = s, size = 1)
        thetap = zp * rho + self.theta

        r = self.model.log_density(thetap)
        r -= self.model.log_density(self.theta)
        r += self._logq(0, eta)
        r -= self._logq(zp, eta)

        a = np.log(self.rng.uniform()) < np.minimum(0, r)
        self._prev_theta = self.theta
        self.theta = a * thetap + (1 - a) * self.theta

        d = a - self.acceptance_probability
        self.acceptance_probability += d / self._draw
        return self.theta

    def draw(self):
        self._draw += 1
        rho = self._random_direction()
        etakl = self.fit(rho)
        theta = self._metropolis_step(etakl, rho)

        if self._windowedadaptation.window_closed(self._draw):
            self._mean = self._onlinemoments.mean()
            self._cov = self._onlinemoments.var()
            if self._scale_dir_cov:
                self._cov /= (self._tol + self._onlinemoments_density.var())
            self._onlinemoments_density.reset()
            self._onlinemoments.reset()
            self._eigvecs[:, :self.J] = self._onlinepca.vectors()
            self._eigvals[:self.J] = self._onlinepca.values()
            self._onlinepca.reset()
            K = self._smoothK.optimum()
            self.K = int(np.clip(K, 1, 50)) # TODO needs testing
            self._smoothK.reset()
        else:
            _, g = self.model.log_density_gradient(theta)
            self._onlinemoments_density.update(g)
            self._onlinemoments.update(theta)
            self._onlinepca.update(theta - self._mean)
            msjd = np.linalg.norm(theta - self._prev_theta)
            self._smoothK.update(2 * (msjd > self._msjd) - 1)

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
