import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
import scipy.stats as st

from bsmodel import BSModel
from onlinemoments import OnlineMoments
from onlinepca import OnlinePCA
from mcmc import MCMCBase
from windowedadaptation import WindowedAdaptation

class KLHR(MCMCBase):
    def __init__(self, bsmodel, theta = None, seed = None,
                 N = 16, K = 10, J = 2, l = 0,
                 initscale = 0.1,
                 warmup = 1_000, windowsize = 25, windowscale = 2,
                 tol = 1e-10, clip_grad = 1e6, tol_grad = 1e12,
                 random = True, scaled = True):
        super().__init__(bsmodel, -1, theta = theta, seed = seed)

        self.N = N
        self.K = K
        self.J = J
        self.l = l
        self.x, self.w = hermgauss(self.N)
        self.tol = tol
        self.clip_grad = clip_grad
        self.tol_grad = tol_grad
        self._mean = np.zeros(self.D)
        self._var = np.ones(self.D)

        self._random = int(random)
        self._Jr = self.J + self._random
        self._scaled = scaled

        self._initscale = initscale
        self._windowedadaptation = WindowedAdaptation(warmup,
                                                      windowsize = windowsize,
                                                      windowscale = windowscale)
        self._onlinemoments = OnlineMoments(self.D)
        self._onlinepca = OnlinePCA(self.D, K = self._Jr - self._random, l = self.l)
        self._eigvecs = np.zeros((self.D, self._Jr))
        self._eigvals = np.ones(self._Jr)

        self._draw = 0
        self.acceptance_probability = 0
        self.minimization_failure_rate = 0

        # constants
        self.invsqrtpi = 1 / np.sqrt(np.pi)
        self.sqrt2 = np.sqrt(2)

    def _unpack(self, eta):
        m = eta[0]
        s = np.exp(eta[1]) + self.tol
        return m, s

    def _logp_grad(self, theta):
        p, g = self.model.log_density_gradient(theta)
        g = np.clip(g, -self.clip_grad, self.clip_grad)
        ng = np.linalg.norm(g)
        if ng > self.tol_grad:
            g *= self.tol_grad / (ng + self.tol)
        return p, g

    def _L(self, eta, rho):
        m, s = self._unpack(eta)
        out = 0.0
        grad = np.zeros(2)
        for xn, wn in zip(self.x, self.w):
            y = self.sqrt2 * s * xn + m
            xi = self._to_rho(y, rho, self.theta)
            logp, grad_logp = self._logp_grad(xi)
            out += wn * logp
            w_grad_logp_rho = wn * grad_logp.dot(rho)
            grad[0] += w_grad_logp_rho
            grad[1] += w_grad_logp_rho * s * xn * self.sqrt2
        out *= self.invsqrtpi
        out += eta[1]
        grad[1] *= self.invsqrtpi
        grad[1] += 1
        return -out, -grad

    def fit(self, rho):
        init = self.rng.normal(size = 2) * self._initscale
        o = minimize(self._L,
                     init,
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        self.minimization_failure_rate += (o.success - self.minimization_failure_rate) / self._draw
        return o.x

    def _random_direction(self):
        p = self._eigvals / np.sum(self._eigvals)
        j = self.rng.choice(self._Jr, p = p)
        rho = self.rng.multivariate_normal(self._eigvecs[:, j], np.diag(self._var))
        return rho / np.linalg.norm(rho)

    def _to_rho(self, x, rho, origin):
        return x * rho + origin

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
        # zp = self.rng.normal(loc = m, scale = s, size = 1)
        zp = self._overrelaxed_proposal(eta)
        thetap = self._to_rho(zp, rho, self.theta)

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
            if self._scaled: self._var = self._onlinemoments.var()
            self._onlinemoments.reset()
            self._eigvecs[:, :self._Jr - self._random] = self._onlinepca.vectors()
            self._eigvals[:self._Jr - self._random] = self._onlinepca.values()
            self._onlinepca.reset()
        else:
            self._onlinemoments.update(theta)
            self._onlinepca.update(theta - self._mean)

        return theta
