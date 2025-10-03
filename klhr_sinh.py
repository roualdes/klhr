import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, root_scalar
import scipy.special as sp
import scipy.stats as st

from bsmodel import BSModel
from onlinemoments import OnlineMoments
from onlinepca import OnlinePCA
from mcmc import MCMCBase
from windowedadaptation import WindowedAdaptation

class KLHRSINH(MCMCBase):
    def __init__(self, bsmodel, theta = None, seed = None,
                 N = 16, K = 10, J = 2, l = 0,
                 initscale = 0.1,
                 warmup = 1_000, windowsize = 50, windowscale = 2,
                 tol = 1e-12, tol_clip = 1e10, tol_grad = 1e2, scale_dir_cov = False):
        super().__init__(bsmodel, -1, theta = theta, seed = seed)

        self.N = N
        self.K = K
        self.J = J
        self.l = l
        self.x, self.w = hermgauss(self.N)
        self._tol = tol
        self._tol_clip = tol_clip
        self._tol_grad = tol_grad
        self._mean = np.zeros(self.D)
        self._cov = np.ones(self.D)

        self._initscale = initscale
        self._windowedadaptation = WindowedAdaptation(warmup,
                                                      windowsize = windowsize,
                                                      windowscale = windowscale)
        self._onlinemoments = OnlineMoments(self.D)
        self._scale_dir_cov = scale_dir_cov
        self._onlinemoments_density = OnlineMoments(self.D) 
        self._onlinepca = OnlinePCA(self.D, K = self.J, l = self.l)
        # self._eigvecs = np.zeros((self.D, self.J + 1))
        # self._eigvals = np.ones(self.J + 1)
        self._eigvecs = np.zeros((self.D, self.J))
        self._eigvals = np.ones(self.J)

        self._draw = 0
        self.acceptance_probability = 0
        self.minimization_failure_rate = 0

        # constants
        self._invsqrtpi = 1 / np.sqrt(np.pi)
        self._log2 = np.log(2)
        self._log2pi = np.log(2 * np.pi)
        self._sqrt2 = np.sqrt(2)

    def _random_direction(self):
        p = self._eigvals / np.sum(self._eigvals)
        # j = self.rng.choice(self.J + 1, p = p)
        # rho = self.rng.multivariate_normal(self._eigvecs[:, j], np.diag(self._var))
        m = np.sum(p * self._eigvecs, axis = 1)
        rho = self.rng.multivariate_normal(m, np.diag(self._cov))
        return rho / np.linalg.norm(rho)

    def _to_rho(self, x, rho, origin):
        return x * rho + origin

    def _unpack(self, eta):
        m = eta[0]
        s = np.exp(np.clip(eta[1], -700, 700)) # + self.tol
        d = np.exp(np.clip(eta[2], -700, 700)) # + self.tol
        e = eta[3]
        return m, s, d, e

    def _T(self, x, eta):
        m, s, d, e = self._unpack(eta)
        return m + s * self._sinh_aed(x, eta)

    def _T_inv(self, x, eta):
        m, s, d, e = self._unpack(eta)
        z = (x - m) / s
        y = (np.arcsinh(z) - e) * d
        y = np.clip(y, -700, 700)
        return np.sinh(y)

    def _CDF(self, x, eta):
        t_inv = self._T_inv(x, eta)
        return sp.ndtr(t_inv)

    def _CDF_inv(self, x, eta):
        phi_inv = sp.ndtri(x)
        return self._T(phi_inv, eta)

    def _overrelaxed_proposal(self, eta):
        m, s, d, e = self._unpack(eta)
        K = self.K
        u = self._CDF(np.array([0]), eta)
        r = st.binom(K, u).rvs()
        up = 0
        if r > K - r:
            v = st.beta(K - r + 1, 2 * r - K).rvs()
            up = u * v
        elif r < K - r:
            v = st.beta(r + 1, K - 2 * r).rvs()
            up = 1 - (1 - u) * v
        elif r == K - r:
            up = u
        return self._CDF_inv(up, eta)

    def _logq(self, x, eta):
        m, s, d, e = self._unpack(eta)
        z = (x - m) / s
        asinhz = np.arcsinh(z)
        dae = d * asinhz - e
        abs_dae = np.abs(dae)
        out = eta[2] - eta[1] - self._log2
        out -= 0.5 * (self._log2pi + np.log1p(z * z) + 0.5 * (np.cosh(2 * dae) - 1))
        out += abs_dae + np.log1p(np.exp(-2 * abs_dae))
        return out

    def _sinh_aed(self, x, eta):
        _, _, d, e = self._unpack(eta)
        y = (np.arcsinh(x) + e) / d
        y = np.clip(y, -700, 700)
        return np.sinh(y)

    def _cosh_aed(self, x, eta):
        _, _, d, e = self._unpack(eta)
        y = (np.arcsinh(x) + e) / d
        y = np.clip(y, -700, 700)
        return np.cosh(y)

    def _logp_grad(self, x):
        f, g = self.model.log_density_gradient(x)
        # g = np.clip(g, -self.clip_grad, self.clip_grad)
        # ng = np.linalg.norm(g)
        # if ng > self.tol_grad:
        #     g *= self.tol_grad / (ng + self.tol)
        return f, np.clip(g, -self._tol_clip, self._tol_clip)

    def _grad_T(self, x, eta):
        m, s, d, e = self._unpack(eta)
        grad = np.zeros(4)
        grad[0] = 1
        asinhx = np.arcsinh(x)
        invd = 1 / d
        aed = (asinhx + e) * invd
        grad[1] = s * self._sinh_aed(x, eta)
        coshaed = self._cosh_aed(x, eta)
        grad[2] = -s * coshaed * aed
        grad[3] = s * coshaed * invd
        return grad

    def _log_cosh_asinh(self, x):
        return 0.5 * np.log1p(x * x)

    def _log_sech_aed(self, x, eta):
        m, s, d, e = self._unpack(eta)
        aed = (np.arcsinh(x) + e) / d
        return -np.abs(aed) - np.log1p(np.exp(-2 * np.abs(aed))) + self._log2

    def _log_abs_jac(self, x, eta):
        out = self._log_sech_aed(x, eta)
        out += eta[2] - eta[1]
        return out

    def _grad_log_abs_jac(self, x, eta):
        m, s, d, e = self._unpack(eta)
        invd = 1 / d
        grad = np.zeros(4)
        grad[1] = -1
        aed = (np.arcsinh(x) + e) * invd
        taed = np.tanh(aed)
        grad[2] = 1 + taed * aed
        grad[3] = -taed * invd
        return grad

    def _L_normal(self, eta, rho):
        m, s, _, _ = self._unpack(eta)
        out = 0.0
        grad = np.zeros(4)
        for xn, wn in zip(self.x, self.w):
            y = self._sqrt2 * s * xn + m
            xi = self._to_rho(y, rho, self.theta)
            logp, grad_logp = self._logp_grad(xi)
            out += wn * logp
            w_grad_logp_rho = wn * grad_logp.dot(rho)
            grad[0] += w_grad_logp_rho
            grad[1] += w_grad_logp_rho * s * xn * self._sqrt2
        out *= self._invsqrtpi
        out += eta[1]
        grad[1] *= self._invsqrtpi
        grad[1] += 1
        return -out, -grad

    def _L(self, eta, rho):
        out = 0.0
        grad = np.zeros(4)
        m, s, d, e = self._unpack(eta)
        invd = 1 / d
        for xn, wn in zip(self.x, self.w):
            y = self._sqrt2 * xn
            log_abs_jac = self._log_abs_jac(y, eta)
            t = self._T(y, eta)
            xi = self._to_rho(t, rho, self.theta)
            logp, grad_logp = self._logp_grad(xi)
            out += wn * (log_abs_jac - logp)
            grad_log_abs_jac = self._grad_log_abs_jac(y, eta)
            grad_T = self._grad_T(y, eta)
            grad += wn * (grad_log_abs_jac - grad_logp.dot(rho) * grad_T)
        out *= self._invsqrtpi
        grad *= self._invsqrtpi
        return out, grad

    def fit(self, rho):
        o = minimize(self._L_normal,
                     self.rng.normal(size = 4) * self._initscale,
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        o = minimize(self._L,
                     np.array([o.x[0], o.x[1], 0.0, 0.0]),
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        # if self._draw > 0:
        #     self.minimization_failure_rate += (o.success - self.minimization_failure_rate) / self._draw
        return o.x

    def _metropolis_step(self, eta, rho):
        # zp = self._T(self.rng.normal(size = 1), eta)
        zp = self._overrelaxed_proposal(eta)
        thetap = self._to_rho(zp, rho, self.theta)

        a = self.model.log_density(thetap) - self.model.log_density(self.theta)
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
                self._cov /= (self._tol + self.online_density.var())
            self._onlinemoments_density.reset()
            self._onlinemoments.reset()
            self._eigvecs[:, :self.J] = self._onlinepca.vectors()
            self._eigvals[:self.J] = self._onlinepca.values()
            self._onlinepca.reset()
        else:
            _, g = self._logp_grad(theta)
            self._onlinemoments_density.update(g)
            self._onlinemoments.update(theta)
            self._onlinepca.update(theta - self._mean)

        return theta
