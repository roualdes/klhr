from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
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
                 tol = 1e-12, scale_dir_cov = False,
                 overrelaxed = True, eigen_method_one = True):
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

    def _make_loss(self):
        def key(x):
            return x.tobytes()

        @lru_cache()
        def logp_grad(k):
            theta = np.frombuffer(k).reshape(self.D)
            logp, grad = self.model.log_density_gradient(theta)
            return logp, np.clip(grad, -1e12, 1e12)

        def logp_rho(x, rho):
            logp, grad = logp_grad(key(x.reshape(-1) * rho + self.theta))
            return -logp, -grad.dot(rho)

        def KL(eta, rho):
            m, s = self._unpack(eta)
            out = 0.0
            for xn, wn in zip(self.x, self.w):
                y = self._sqrt2 * s * xn + m
                xi = y * rho + self.theta
                logp, _ = logp_grad(key(xi))
                out += wn * logp
            out *= self._invsqrtpi
            out += eta[1]
            return -out

        def gradKL(eta, rho):
            m, s = self._unpack(eta)
            grad = np.zeros(2)
            for xn, wn in  zip(self.x, self.w):
                y = self._sqrt2 * s * xn + m
                xi = y * rho + self.theta
                _, grad_logp = logp_grad(key(xi))
                w_grad_logp_rho = wn * grad_logp.dot(rho)
                grad[0] += w_grad_logp_rho
                grad[1] += w_grad_logp_rho * s * xn * self._sqrt2
            grad *= self._invsqrtpi
            grad[1] += 1
            return -grad

        def hessKL(eta, rho):
            m, s = self._unpack(eta)
            H = np.zeros((2, 2))
            for xn, wn in zip(self.x, self.w):
                y = self._sqrt2 * s * xn + m
                xi = y * rho + self.theta
                _, Hrho = self.model.log_density_hvp(xi, rho)
                Hrho2 = rho.dot(Hrho)
                sq = np.ones((2, 2)) * Hrho2
                sq[[0, 1], [1, 0]] *= self._sqrt2 * xn * s
                sq[1, 1] *= 2 * xn * xn * s * s
                _, grad_logp = logp_grad(key(xi))
                sq[1, 1] += grad_logp.dot(rho) * self._sqrt2 * xn * s
                H += wn * sq
            H *= -self._invsqrtpi
            return H

        def clear_cache():
            logp_grad.cache_clear()

        return logp_rho, KL, gradKL, hessKL, clear_cache

    def _det(self, H):
        a, b, c = H[0,0], H[0,1], H[1,1]
        return a * c - b * b

    def _inv(self, H):
        det = self._det(H)
        if abs(det) < self._tol:
            if det != 0.0:
                det = np.sign(det) * self._tol
            else:
                det = self._tol
        a, b, c = H[0, 0], H[0, 1], H[1, 1]
        invH = np.array([[c, -b],
                         [-b,  a]]) / det
        return invH

    def _posdef(self, H):
        det = self._det(H)
        return np.all(np.diag(H) > 0.0) and (det > 0.0)

    def _safe_inv(self, H, lam=1e-6, max_attempts = 16):
        I = np.eye(2)
        attempt = 0
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            return I
        while not self._posdef(H):
            H += lam * I
            lam *= 10
            attempt += 1
            if attempt > max_attempts:
                return I
        return self._inv(H)

    def fit(self, rho):
        logp_rho, KL, gradKL, hessKL, clear_cache = self._make_loss()
        clear_cache()

        init = self.rng.normal(size = 2)

        o = minimize(KL,
                     init,
                     jac = gradKL,
                     args = (rho,),
                     method = "BFGS",
                     options = {"maxiter": 1})

        H = hessKL(o.x, rho)
        Hinv = self._safe_inv(H)
        o = minimize(KL,
                     o.x,
                     jac = gradKL,
                     args = (rho,),
                     method = "BFGS",
                     options = {"hess_inv0": Hinv})

        return o.x

    def _random_direction(self):
        p = self._eigvals / np.sum(self._eigvals)
        if self._eigen_method_one:
            j = self.rng.choice(np.size(self._eigvals), p = p)
            rho = self.rng.multivariate_normal(self._eigvecs[:, j], np.diag(self._cov))
        else:
            m = np.sum(p * self._eigvecs, axis = 1)
            rho = self.rng.multivariate_normal(m, np.diag(self._cov))
        return rho / np.linalg.norm(rho)

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
