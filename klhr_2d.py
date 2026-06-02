import warnings

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, minimize_scalar
import scipy.stats as st

from bsmodel import BSModel
from onlinemoments import OnlineMoments
from onlinepca import OnlinePCA
from mcmc import MCMCBase
from windowedadaptation import WindowedAdaptation

class KLHR2D(MCMCBase):
    def __init__(self,
                 bsmodel,
                 theta = None,
                 seed = None,
                 K = 16,
                 J = 1,
                 l = 0,
                 initscale = 0.1,
                 warmup = 1_000,
                 windowsize = 50,
                 windowscale = 2,
                 tol = 1e-10,
                 grad_clip = 1e15,
                 scale_clip = 300,
                 gtol = 1e-3,
                 **kwargs):
        super().__init__(bsmodel, -1, theta = theta, seed = seed)

        self.K = K
        self.J = J
        self.l = l
        self._tol = tol
        self._grad_clip = grad_clip
        self._scale_clip = scale_clip
        self._gtol = gtol

        self.x, self.w = hermgauss(8)
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
        self._onlinepca = OnlinePCA(self.D, K = self.J, l = self.l)
        self._eigvecs = np.zeros((self.D, self.J + 1))
        self._eigvals = np.ones(self.J + 1)

        self._draw = 0
        self.acceptance_probability = 0
        self.grad_evals = 0

    def _invlogit(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = np.exp(x)
            return z / (1.0 + z)

    def _corr(self, r):
        return

    def _unpack(self, eta):
        m0 = eta[0]
        c = self._scale_clip
        s0 = np.exp(np.clip(eta[1], -c, c)) + self._tol
        m1 = eta[2]
        s1 = np.exp(np.clip(eta[3], -c, c)) + self._tol
        rho = 2 * self._invlogit(eta[4]) - 1
        return m0, s0, m1, s1, np.clip(rho, -0.99999, 0.99999)

    def _logp_grad(self, x):
        l, g = self.model.log_density_gradient(x)
        c = self._grad_clip
        return l, np.clip(g, -c, c)

    def _log_abs_det(self, eta):
        _, s0, _, s1, r = self._unpack(eta)
        out = 2 * (np.log(s0) + np.log(s1))
        out += np.log1p(-(r * r))
        return out

    def KL(self, eta, rho):
        m0, s0, m1, s1, r = self._unpack(eta)
        out = 0.0
        grad = np.zeros(5)
        rho0, rho1 = rho
        one_m_r2 = np.maximum(0.0, 1.0 - r * r)
        tr = np.sqrt(one_m_r2 + self._tol)
        for xi, wi in zip(self.x, self.w):
            zi = s0 * xi + m0
            for xj, wj in zip(self.x, self.w):
                zj = m1 + r * s1 * xi + s1 * tr * xj
                y = zi * rho0 + zj * rho1 + self.theta
                logp, grad_logp = self._logp_grad(y)
                wij = wi * wj
                out += wij * logp
                grad_rho0 = wij * grad_logp.dot(rho0)
                grad_rho1 = wij * grad_logp.dot(rho1)
                grad[0] += grad_rho0
                grad[1] += grad_rho0 * xi * s0
                grad[2] += grad_rho1
                j3 = r * xi + tr * xj
                grad[3] += grad_rho1 * j3 * s1
                j4 = one_m_r2 * xi - r * xj * tr
                grad[4] += grad_rho1 * s1 * 0.5 * j4
        out += self._log_abs_det(eta)
        grad[1] += 2
        grad[3] += 2
        grad[4] -= np.tanh(0.5 * eta[4])
        return -out, -grad

    def logp_grad_rho(self, xi, rho):
        rho0, rho1 = rho
        y = xi[0] * rho0 + xi[1] * rho1 + self.theta
        l, g = self.model.log_density_gradient(y)
        return -l, -np.array([g.dot(rho0), g.dot(rho1)])

    def fit(self, rho):
        o = minimize(self.logp_grad_rho,
                     self.rng.normal(size = 2) * self._initscale,
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        self.grad_evals += o["nfev"]
        h0 = o["hess_inv"][0, 0]
        s0 = 0.0
        if h0 > 0 and np.isfinite(h0):
            s0 = 0.5 * np.log(h0)
        h1 = o["hess_inv"][1, 1]
        s1 = 0.0
        if h1 > 0 and np.isfinite(h1):
            s1 = 0.5 * np.log(h1)

        init = np.array([o.x[0], s0, o.x[1], s1, 0.0])
        o = minimize(self.KL,
                     init,
                     args = (rho,),
                     jac = True,
                     method = "BFGS",
                     options = {"gtol": self._gtol})
        self.grad_evals += o["nfev"] * 8
        return o.x

    def _random_direction(self):
        S = np.diag(self._cov)
        rho0 = self.rng.multivariate_normal(self._eigvecs[:, 0], S)
        while True:
            n0 = np.linalg.norm(rho0)
            if n0 < self._tol:
                rho0 = self.rng.multivariate_normal(self._eigvecs[:, 0], S)
            else:
                break
        rho0 /= np.linalg.norm(rho0)
        rho1 = self.rng.multivariate_normal(np.zeros_like(rho0), S)
        while True:
            n1 = np.linalg.norm(rho1)
            if n1 < self._tol:
                rho1 = self.rng.multivariate_normal(np.zeros_like(rho0), S)
            else:
                break
        rho1 -= rho0.dot(rho1) * rho0
        rho1 /= np.linalg.norm(rho1)
        return rho0, rho1

    def _sigma(self, eta):
        _, s0, _, s1, r = self._unpack(eta)
        s02 = s0 * s0
        s12 = s1 * s1
        od = r * s0 * s1
        return np.array([[s02, od], [od, s12]])

    def _metropolis_step(self, eta, rho):
        m0, _, m1, _, _ = self._unpack(eta)
        m = np.array([m0, m1])
        S = self._sigma(eta)
        mvn = st.multivariate_normal(m, S)
        xi = mvn.rvs()
        rho0, rho1 = rho
        thetap = xi[0] * rho0 + xi[1] * rho1 + self.theta

        r = self.model.log_density(thetap)
        r -= self.model.log_density(self.theta)
        r += mvn.logpdf(np.zeros_like(xi))
        r -= mvn.logpdf(xi)

        self.grad_evals += 2

        a = np.log(self.rng.uniform()) < np.minimum(0, r)
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
    from klhr_2d import KLHR2D

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "earnings"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    algo = KLHR2D(bs_model, seed = 678)
    rho = algo._random_direction()

    def f(x):
        def inner(x):
            vf = lambda x: algo.KL(x, rho)[0]
            return np.apply_along_axis(vf, axis=0, arr=x)
        return np.array([inner(x)])

    x = algo.rng.normal(size = 5) * 0.1
    approx_grad = jacobian(f, x)
    grad = algo.KL(x, rho)[1]
    assert np.all(approx_grad.success)
    assert np.allclose(grad, approx_grad.df)
