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

class KLHRSUBSINH(MCMCBase):
    def __init__(self, bsmodel,
                 theta = None,
                 seed = None,
                 N = 8,
                 K = 16,
                 J = 2,
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

        self.N = N
        self.K = K
        self.J = J
        self.l = l
        self._tol = tol
        self._grad_clip = grad_clip
        self._scale_clip = scale_clip
        self._gtol = gtol

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
        self._onlinepca = OnlinePCA(self.D, K = self.J, l = self.l)
        self._eigvecs = np.zeros((self.D, self.J + 1))
        self._eigvals = np.ones(self.J + 1)

        self._draw = 0
        self.acceptance_probability = 0
        self.grad_evals = 0

    def _unpack(self, eta):
        m = eta[0]
        c = self._scale_clip
        s = np.exp(np.clip(eta[1], -c, c)) + self._tol
        e = eta[2]
        return m, s, e

    def _cosh(self, x):
        c = self._scale_clip
        return np.cosh(np.clip(x, -c, c))

    def _sinh(self, x):
        c = self._scale_clip
        return np.sinh(np.clip(x, -c, c))

    def _tanh(self, x):
        c = self._scale_clip
        return np.tanh(np.clip(x, -c, c))

    def _T(self, x, eta):
        m, s, e = self._unpack(eta)
        return m + s * self._sinh(np.arcsinh(x) + e)

    def _grad_T(self, x, eta):
        m, s, e = self._unpack(eta)
        grad = np.ones(3)
        asinhpe = np.arcsinh(x) + e
        grad[1] = s * self._sinh(asinhpe)
        grad[2] = s * self._cosh(asinhpe)
        return grad

    def _T_inv(self, x, eta):
        m, s, e = self._unpack(eta)
        z = (x - m) / s
        return self._sinh(np.arcsinh(z) - e)

    def _CDF(self, x, eta):
        t_inv = self._T_inv(x, eta)
        return sp.ndtr(t_inv)

    def _CDF_inv(self, x, eta):
        phi_inv = sp.ndtri(x)
        return self._T(phi_inv, eta)

    def _log_abs_jac(self, x, eta):
        _, _, e = self._unpack(eta)
        out = -eta[1]
        asinhpe = np.arcsinh(x) + e
        out -= np.log(self._cosh(asinhpe))
        return out

    def _grad_log_abs_jac(self, x, eta):
        _, _, e = self._unpack(eta)
        grad = np.zeros(3)
        grad[1] = -1
        asinhpe = np.arcsinh(x) + e
        grad[2] = -self._tanh(asinhpe)
        return grad

    def _logp_grad(self, x):
        logp, grad = self.model.log_density_gradient(x)
        c = self._grad_clip
        return logp, np.clip(grad, -c, c)

    def KL(self, eta, rho):
        out = 0.0
        grad = np.zeros(3)
        for xn, wn in zip(self.x, self.w):
            t = self._T(xn, eta)
            xi = t * rho + self.theta
            logp, grad_logp = self._logp_grad(xi)
            log_abs_jac = self._log_abs_jac(xn, eta)
            out += wn * (log_abs_jac - logp)
            grad_log_abs_jac = self._grad_log_abs_jac(xn, eta)
            grad_T = self._grad_T(xn, eta)
            grad -= wn * grad_logp.dot(rho) * grad_T
            grad += wn * grad_log_abs_jac
        return out, grad

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
        h = o["hess_inv"][0,0]
        s = 0.0
        if h > 0 and np.isfinite(h):
            s = 0.5 * np.log(h)
        init = np.array([o.x[0], s, 0.0])
        o = minimize(self.KL,
                     init,
                     args = (rho,),
                     jac = True,
                     method = "BFGS",
                     options = {"gtol": self._gtol, "maxiter": 4})
        return o.x

    def _random_direction(self):
        evals = self._eigvals
        p = evals / np.sum(evals)
        j = self.rng.choice(np.size(p), p = p)
        m = self._eigvecs[:, j]
        S = np.diag(self._cov)
        rho = self.rng.multivariate_normal(m, S)
        return rho / np.linalg.norm(rho + self._tol)

    def _overrelaxed_proposal(self, eta):
        K = self.K
        u = self._CDF(np.zeros(1), eta)
        r = st.binom(K, u).rvs(random_state = self.rng)
        up = 0
        if r > K - r:
            v = st.beta(K - r + 1, 2 * r - K).rvs(random_state = self.rng)
            up = u * v
        elif r < K - r:
            v = st.beta(r + 1, K - 2 * r).rvs(random_state = self.rng)
            up = 1 - (1 - u) * v
        elif r == K - r:
            up = u
        return self._CDF_inv(up, eta)

    def _log_std_normal(self, x):
        return -0.5 * x * x

    def _log_q(self, x, eta):
        m, s, e = self._unpack(eta)
        ld = self._log_std_normal(self._T_inv(x, eta))
        z = (x - m) / s
        ld += np.log(self._cosh(np.arcsinh(z) - e))
        ld -= eta[1]
        ld -= 0.5 * np.log1p(z * z)
        return ld

    def _metropolis_step(self, eta, rho):
        xi = self._overrelaxed_proposal(eta)
        thetap = xi * rho + self.theta

        r = self.model.log_density(thetap)
        r -= self.model.log_density(self.theta)
        r += self._log_q(0.0, eta)
        r -= self._log_q(xi[0], eta)

        self.grad_evals += 2

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

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "earnings"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    algo = KLHRSUBSINH(bs_model)

    rng = np.random.default_rng()
    rho = rng.multivariate_normal(np.zeros(algo.D), np.eye(algo.D))
    rho /= np.linalg.norm(rho)

    def f(x):
        def inner(x):
            vf = lambda x: algo._T(0, x)
            return np.apply_along_axis(vf, axis=0, arr=x)
        return np.array([inner(x)])

    x = rng.normal(size = 3) * 0.1
    approx_grad = jacobian(f, x)
    grad = algo._grad_T(0, x)
    assert np.all(approx_grad.success)
    assert np.allclose(grad, approx_grad.df)

    def g(x):
        def inner(x):
            vf = lambda x: algo._log_abs_jac(0, x)
            return np.apply_along_axis(vf, axis=0, arr=x)
        return np.array([inner(x)])

    x = rng.normal(size = 3) * 0.1
    approx_grad = jacobian(g, x)
    grad = algo._grad_log_abs_jac(0, x)
    # assert np.all(approx_grad.success)
    assert np.allclose(grad, approx_grad.df)

    def h(x):
        def inner(x):
            vf = lambda x: algo.KL(x, rho)[0]
            return np.apply_along_axis(vf, axis=0, arr=x)
        return np.array([inner(x)])

    x = rng.normal(size = 3) * 0.1
    approx_grad = jacobian(h, x)
    grad = algo.KL(x, rho)[1]
    # assert np.all(approx_grad.success)
    assert np.allclose(grad, approx_grad.df)
