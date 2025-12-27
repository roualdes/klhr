import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
import scipy.stats as st

from bsmodel import BSModel
from onlinemoments import OnlineMoments
from mcmc import MCMCBase
from windowedadaptation import WindowedAdaptation

class KLHRReflection(MCMCBase):
    def __init__(self, bsmodel,
                 theta = None,
                 seed = None,
                 steps = 5,
                 N = 6,
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

        self.steps = steps
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
        self._cov = np.ones(self.D)
        self._draw = 0
        self.acceptance_probability = 0
        self.grad_evals = 0

    def _unpack(self, eta):
        m = eta[0]
        c = self._scale_clip
        s = np.exp(np.clip(eta[1], -c, c)) + self._tol
        return m, s

    def _logp_grad(self, x):
        l, g = self.model.log_density_gradient(x)
        c = self._grad_clip
        return l, np.clip(g, -c, c)

    def KL(self, eta, rho, theta):
        m, s = self._unpack(eta)
        out = 0.0
        grad = np.zeros(2)
        for xn, wn in zip(self.x, self.w):
            y = s * xn + m
            xi = y * rho + theta
            logp, grad_logp = self.model.log_density_gradient(xi)
            out += wn * logp
            w_grad_rho = wn * grad_logp.dot(rho)
            grad[0] += w_grad_rho
            grad[1] += w_grad_rho * xn * s
        out += eta[1]
        grad[1] += 1
        return -out, -grad

    def logp_grad_rho(self, xi, rho, theta):
        l, g = self.model.log_density_gradient(xi * rho + theta)
        return -l, -g.dot(rho)

    def fit(self, rho, theta):
        o = minimize(self.logp_grad_rho,
                     self.rng.normal() * self._initscale,
                     args = (rho, theta,),
                     jac = True,
                     method = "BFGS")
        self.grad_evals += o["nfev"]
        h = o["hess_inv"][0,0]
        s = 0.0
        if h > 0 and np.isfinite(h):
            s = 0.5 * np.log(h)
        init = np.array([o.x[0], s])
        o = minimize(self.KL,
                     init,
                     args = (rho, theta),
                     jac = True,
                     method = "BFGS",
                     options = {"gtol": self._gtol})
        self.grad_evals += o["nfev"] * self.N
        return o.x

    def _random_direction(self):
        m = np.zeros(self.D)
        S = np.eye(self.D)
        rho = self.rng.multivariate_normal(m, S)
        return rho / np.linalg.norm(rho + self._tol)

    def _log_q(self, x, eta):
        m, s = self._unpack(eta)
        z = (x - m) / s
        return -np.log(s) - 0.5 * z * z

    def _reflect(self, rho, theta):
        self.grad_evals += 1
        _, g = self.model.log_density_gradient(theta)
        n = g / np.linalg.norm(g)
        return rho - 2 * n * np.dot(rho, n)

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

    def _walk(self, rho, theta):
        thetaw = theta
        rhow = rho
        r = 0.0
        for _ in range(self.steps):
            eta = self.fit(rhow, thetaw)
            r += self._log_q(0.0, eta)
            m, s = self._unpack(eta)
            xi = self._overrelaxed_proposal(eta)
            # xi = self.rng.normal(loc = m, scale = s)
            r -= self._log_q(xi[0], eta)
            thetaw = xi * rhow + thetaw
            rhow = self._reflect(rhow, thetaw)
        return rhow, thetaw, r

    def _metropolis_step(self, rho):
        rhow, thetaw, r = self._walk(rho, self.theta)
        r += self.model.log_density(thetaw)
        r += -0.5 * rhow.dot(rhow)
        r -= self.model.log_density(self.theta)
        r -= -0.5 * rho.dot(rho)
        self.grad_evals += 2

        a = np.log(self.rng.uniform()) < np.minimum(0, r)
        self.theta = a * thetaw + (1 - a) * self.theta

        d = a - self.acceptance_probability
        self.acceptance_probability += d / self._draw
        return self.theta

    def draw(self):
        self._draw += 1
        rho = self._random_direction()
        theta = self._metropolis_step(rho)

        if self._windowedadaptation.window_closed(self._draw):
            self._cov = self._onlinemoments.var()
            self._onlinemoments.reset()
        else:
            self._onlinemoments.update(theta)

        return theta

if __name__ == "__main__":
    # TODO fix tests
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
