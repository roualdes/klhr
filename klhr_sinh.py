import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
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
                 warmup = 1_000,
                 windowsize = 50,
                 windowscale = 2,
                 tol = 1e-10,
                 scale_dir_cov = False,
                 overrelaxed = True,
                 eigen_method_one = False):
        super().__init__(bsmodel, -1, theta = theta, seed = seed)

        self.N = N
        self.K = K
        self.J = J
        self.l = l
        self.x, self.w = hermgauss(self.N)
        self._tol = tol
        self._mean = np.zeros(self.D)
        self._cov = np.ones(self.D)

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
        self._eigvecs = np.zeros((self.D, self.J + 1))
        self._eigvals = np.ones(self.J + 1)
        # self._eigvecs = np.zeros((self.D, self.J))
        # self._eigvals = np.ones(self.J)

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
        if self._eigen_method_one:
            j = self.rng.choice(np.size(p), p = p)
            m = evals[j] * self._eigvecs[:, j]
        else:
            m = np.sum(p * self._eigvecs, axis = 1)
        S = np.diag(self._cov)
        rho = self.rng.multivariate_normal(m, S)
        return rho / np.linalg.norm(rho + self._tol)

    def _to_rho(self, x, rho, origin):
        return x * rho + origin

    def _unpack(self, eta):
        m = eta[0]
        s = np.exp(np.clip(eta[1], -650, 650))
        d = np.exp(np.clip(eta[2], -650, 650))
        e = eta[3]
        return m, s, d, e

    def _T(self, x, eta):
        m, s, d, e = self._unpack(eta)
        return m + s * self._sinh_aed(x, eta)

    def _T_inv(self, x, eta):
        m, s, d, e = self._unpack(eta)
        z = (x - m) / s
        y = (np.arcsinh(z) - e) * d
        y = np.clip(y, -650, 650)
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
        asinhz = np.arcsinh(np.clip(z, -650, 650))
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
        y = np.clip(y, -650, 650)
        return np.cosh(y)

    def _logp_grad(self, x):
        logp, grad = self.model.log_density_gradient(x)
        return logp, np.clip(grad, -1e15, 1e15)

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

    def KLnormal(self, eta, rho):
        m, s, _, _ = self._unpack(eta)
        out = 0.0
        grad = np.zeros(4)
        for xn, wn in zip(self.x, self.w):
            y = self._sqrt2 * s * xn + m
            xi = y * rho + self.theta
            logp, grad_logp = self._logp_grad(xi)
            out += wn * logp
            w_grad_logp_rho = wn * grad_logp.dot(rho)
            grad[0] += w_grad_logp_rho
            grad[1] += w_grad_logp_rho * xn * self._sqrt2 * s
        out *= self._invsqrtpi
        out += eta[1]
        grad *= self._invsqrtpi
        grad[1] += 1
        return np.log(-out), grad/out

    def KL(self, eta, rho):
        out = 0.0
        grad = np.zeros(4)
        for xn, wn in zip(self.x, self.w):
            y = self._sqrt2 * xn
            t = self._T(y, eta)
            xi = t * rho + self.theta
            logp, grad_logp = self._logp_grad(xi)
            log_abs_jac = self._log_abs_jac(y, eta)
            out += wn * (log_abs_jac - logp)
            grad_log_abs_jac = self._grad_log_abs_jac(y, eta)
            grad_T = self._grad_T(y, eta)
            grad += wn * (grad_log_abs_jac - grad_logp.dot(rho) * grad_T)
        out *= self._invsqrtpi
        grad *= self._invsqrtpi
        return np.log(out), grad/out

    def fit(self, rho):
        init = self.rng.normal(size = 4) * self._initscale
        o = minimize(self.KL, init, args = (rho,),
                     jac = True, method = "BFGS")
        return o.x

    def _metropolis_step(self, eta, rho):
        m, s, d, e = self._unpack(eta)
        if self._overrelaxed:
            zp = self._overrelaxed_proposal(eta)
        else:
            zp = self._T(self.rng.normal(loc = m, scale = s, size = 1), eta)
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
        if self._draw % 1_000 == 0:
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
            _, g = self._logp_grad(theta)
            self._onlinemoments_density.update(g)
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

    algo = KLHRSINH(bs_model)

    rng = np.random.default_rng()
    rho = rng.multivariate_normal(np.zeros(algo.D), np.eye(algo.D))
    rho /= np.linalg.norm(rho)

    def f(x):
        def inner(x):
            vf = lambda x: algo._T(0, x)
            return np.apply_along_axis(vf, axis=0, arr=x)
        return np.array([inner(x)])

    x = rng.normal(size = 4) * 0.1
    approx_grad = jacobian(f, x)
    grad = algo._grad_T(0, x)
    assert np.all(approx_grad.success)
    assert np.allclose(grad, approx_grad.df)

    def g(x):
        def inner(x):
            vf = lambda x: algo._log_abs_jac(0, x)
            return np.apply_along_axis(vf, axis=0, arr=x)
        return np.array([inner(x)])

    x = rng.normal(size = 4) * 0.1
    approx_grad = jacobian(g, x)
    grad = algo._grad_log_abs_jac(0, x)
    # assert np.all(approx_grad.success)
    assert np.allclose(grad, approx_grad.df)

    def h(x):
        def inner(x):
            vf = lambda x: algo.KL(x, rho)[0]
            return np.apply_along_axis(vf, axis=0, arr=x)
        return np.array([inner(x)])

    x = rng.normal(size = 4) * 0.1
    approx_grad = jacobian(h, x)
    grad = algo.KL(x, rho)[1]
    assert np.all(approx_grad.success)
    assert np.allclose(grad, approx_grad.df)

    def j(x):
        def inner(x):
            vf = lambda x: algo.KLnormal(x, rho)[0]
            return np.apply_along_axis(vf, axis=0, arr=x)
        return np.array([inner(x)])

    x = rng.normal(size = 4) * 0.1
    approx_grad = jacobian(j, x)
    grad = algo.KLnormal(x, rho)[1]
    # assert np.all(approx_grad.success)
    assert np.allclose(grad, approx_grad.df)
