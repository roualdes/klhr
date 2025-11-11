from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.integrate import quad
import scipy.stats as st

import sys

from bsmodel import BSModel
from onlinemoments import OnlineMoments
from onlinepca import OnlinePCA
from mcmc import MCMCBase
from windowedadaptation import WindowedAdaptation

class KLHR(MCMCBase):
    def __init__(self, bsmodel, theta = None, seed = None,
                 N = 16, K = 10, J = 2, l = 0, initscale = 0.1,
                 warmup = 1_000, windowsize = 50, windowscale = 2,
                 tol = 1e-8, scale_dir_cov = False,
                 overrelaxed = False, eigen_method_one = True):
        super().__init__(bsmodel, -1, theta = theta, seed = seed)

        self.N = N
        self.K = K
        self.J = J if J < self.D else self.D - 1
        self.l = l
        self.x, self.w = hermgauss(self.N)
        self._tol = tol
        self._mean = np.zeros(self.D)
        self._cov = np.ones(self.D)
        self._eta = None

        self._initscale = initscale
        self._windowedadaptation = WindowedAdaptation(warmup,
                                                      windowsize = windowsize,
                                                      windowscale = windowscale)
        self._onlinemoments = OnlineMoments(self.D)
        self._scale_dir_cov = scale_dir_cov
        self._overrelaxed = overrelaxed
        self._eigen_method_one = eigen_method_one
        self._onlinemoments_density = OnlineMoments(self.D)
        self._onlinepca = OnlinePCA(self.D, K = self.J, l = self.l)
        # self._eigvecs = np.zeros((self.D, self.J + 1))
        # self._eigvals = np.ones(self.J + 1)
        self._eigvecs = np.zeros((self.D, self.J))
        self._eigvals = np.ones(self.J)

        self._draw = 0
        self.acceptance_probability = 0

        # constants
        self._invsqrtpi = 1 / np.sqrt(np.pi)
        self._sqrt2 = np.sqrt(2)

    def _unpack(self, eta):
        m = eta[0]
        s = np.exp(np.clip(eta[1], -650, 650))
        return m, s

    def _logp_grad(self, x):
        x_unc = self.model.unconstrain(x)
        logp, grad = self.model.log_density_gradient(x_unc)
        return logp, np.clip(grad, -1e12, 1e12)

    def KL(self, eta, rho):
        m, s = self._unpack(eta)
        out = 0.0
        grad = np.zeros(2)
        for xn, wn in zip(self.x, self.w):
            y = self._sqrt2 * s * xn + m
            xi = y * rho + self.theta
            logp, grad_logp = self._logp_grad(xi)
            out += wn * logp
            w_grad_logp_rho = wn * grad_logp.dot(rho)
            grad[0] += w_grad_logp_rho
            grad[1] += w_grad_logp_rho * xn * self._sqrt2
        out *= self._invsqrtpi
        out += np.log(eta[1])
        grad *= self._invsqrtpi
        grad[1] += 1 / eta[1]
        return -out, -grad

    def forwardKL(self, eta, rho):
        m, s = self._unpack(eta)
        def f(x):
            y = x # s * x + m
            xi = y * rho + self.theta
            logp = self.model.log_density(xi)
            logq = self._logq(x, eta)
            # print(f"logp = {logp}, logq = {logq}")
            return (logp - logq) * np.exp(np.clip(logp, -700, 700))
        return quad(f, -np.inf, np.inf)[0]

    def fit(self, rho):
        init = self.rng.normal(size = 2) * self._initscale
        o = minimize(self.forwardKL, init, args = (rho,))
        return o.x

    def _random_direction(self):
        evals = self._eigvals
        p = evals / np.sum(evals)
        if self._eigen_method_one:
            j = self.rng.choice(self.J + 1, p = p)
            m = evals[j] * self._eigvecs[:, j]
        else:
            m = np.sum(evals * self._eigvecs, axis = 1)
        S = np.diag(self._cov)
        rho = self.rng.multivariate_normal(m, S)
        return rho / np.linalg.norm(rho)

    def _logq(self, x, eta):
        m, s = self._unpack(eta)
        z = (x - m) / s
        o = -np.log(s) - 0.5 * z * z
        if np.isnan(o) or np.isinf(o):
            return -np.inf
        return o

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
        elif r < self.K - r:
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

        a = self.model.log_density(thetap)
        a -= self.model.log_density(self.theta)
        a += self._logq(0, eta)
        a -= self._logq(zp, eta)

        accept = np.log(self.rng.uniform()) < np.minimum(0, a)
        if accept:
            self.theta = thetap

        d = accept - self.acceptance_probability
        self.acceptance_probability += d / self._draw

        return self.theta

    def draw(self):
        self._draw += 1
        if self._draw % 1000 == 0:
            print(self._draw)
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
        else:
            _, g = self.model.log_density_gradient(theta)
            self._onlinemoments_density.update(g)
            self._onlinemoments.update(theta)
            self._onlinepca.update(theta - self._mean)

        return theta
