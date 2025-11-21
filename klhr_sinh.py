import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
import scipy.special as sp
import scipy.stats as st

from bsmodel import BSModel
from smoother import Smoother
from onlinemoments import OnlineMoments
from onlinepca import OnlinePCA
from mcmc import MCMCBase
from windowedadaptation import WindowedAdaptation

class KLHRSINH(MCMCBase):
    def __init__(self, bsmodel,
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
                 tol = 1e-10,
                 grad_clip = 1e15,
                 scale_clip = 300,
                 scale_dir_cov = False,
                 overrelaxed = True,
                 eigen_method_one = False,
                 max_init_tries = 100):
        super().__init__(bsmodel, -1, theta = theta, seed = seed)

        self.N = N
        self.K = K
        self.J = J
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
        self._eigvecs = np.zeros((self.D, self.J + 1))
        self._eigvals = np.ones(self.J + 1)
        # self._eigvecs = np.zeros((self.D, self.J))
        # self._eigvals = np.ones(self.J)
        self._smoothK = Smoother(self.K)
        self._prev_theta = np.zeros(self.D)
        self._msjd = 0.0

        self._draw = 0
        self.acceptance_probability = 0
        self.grad_evals = 0

        self._initialize()

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
        s = o["hess_inv"][0,0]
        s = (s > 0) * 0.5 * np.log(s)
        init = self.rng.normal(size = 3) * self._initscale
        init[0] = o.x[0]
        init[1] = s
        o = minimize(self.KL,
                     init,
                     args = (rho,),
                     jac = True,
                     method = "BFGS",
                     options = {"gtol": 1e-3})
        self.grad_evals += o["nfev"] * self.N
        return o.x

    def _random_direction(self):
        evals = self._eigvals
        p = evals / np.sum(evals)
        if self._eigen_method_one:
            j = self.rng.choice(np.size(p), p = p)
            m = self._eigvecs[:, j]
        else:
            m = np.sum(p * self._eigvecs, axis = 1)
        S = np.diag(self._cov)
        rho = self.rng.multivariate_normal(m, S)
        return rho / np.linalg.norm(rho + self._tol)

    def _overrelaxed_proposal(self, eta):
        K = self.K
        u = self._CDF(np.zeros(1), eta)
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
        if self._overrelaxed:
            zp = self._overrelaxed_proposal(eta)
        else:
            zp = self._T(self.rng.normal(size = 1), eta)
        thetap = zp * rho + self.theta

        r = self.model.log_density(thetap)
        r -= self.model.log_density(self.theta)
        r += self._log_q(0, eta)
        r -= self._log_q(zp, eta)

        a = np.log(self.rng.uniform()) < np.minimum(0, r)
        self._prev_theta = self.theta
        self.theta = a * thetap + (1 - a) * self.theta

        d = a - self.acceptance_probability
        self.acceptance_probability += d / self._draw
        return self.theta

    def draw(self):
        self._draw += 1
        if self._draw % 5_000 == 0:
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
            K = self._smoothK.optimum()
            # self.K = int(np.clip(K, 1, 50)) # TODO needs testing
            self._smoothK.reset()
        else:
            _, g = self._logp_grad(theta)
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

    algo = KLHRSINH(bs_model)

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
