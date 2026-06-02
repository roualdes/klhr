import warnings

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, minimize_scalar
import scipy.stats as st
from scipy.linalg import sqrtm

from bsmodel import BSModel
from onlinemoments import OnlineMoments
from onlinepca import OnlinePCA
from mcmc import MCMCBase
from windowedadaptation import WindowedAdaptation

class KLHRBAM(MCMCBase):
    def __init__(self,
                 bsmodel,
                 theta = None,
                 seed = None,
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

        self.K = K
        self.J = J
        self.l = l
        self._tol = tol
        self._grad_clip = grad_clip
        self._scale_clip = scale_clip
        self._gtol = gtol

        # TODO delete
        # self._gh = [self._gausshermite(n) for n in [4, 8]]

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

    # TODO delete
    # def _gausshermite(self, n):
    #     x, w = hermgauss(n)
    #     # normalize roots and weights
    #     x *= np.sqrt(2)
    #     w /= np.sqrt(np.pi)
    #     return x, w

    def _unpack(self, eta):
        m = eta[0]
        c = self._scale_clip
        s = np.exp(np.clip(eta[1], -c, c)) + self._tol
        return m, s

    def _logp_grad(self, x):
        l, g = self.model.log_density_gradient(x)
        c = self._grad_clip
        return l, np.clip(g, -c, c)

    # TODO delete
    # def KL(self, eta, rho, N):
    #     m, s = self._unpack(eta)
    #     out = 0.0
    #     grad = np.zeros(2)
    #     x, w = self._gh[N]
    #     for xn, wn in zip(x, w):
    #         y = s * xn + m
    #         xi = y * rho + self.theta
    #         logp, grad_logp = self._logp_grad(xi)
    #         out += wn * logp
    #         w_grad_rho = wn * grad_logp.dot(rho)
    #         grad[0] += w_grad_rho
    #         grad[1] += w_grad_rho * xn * s
    #     out += eta[1]
    #     grad[1] += 1
    #     return -out, -grad

    def logp_grad_rho(self, xi, rho):
        l, g = self.model.log_density_gradient(xi * rho + self.theta)
        return l, g.dot(rho)

    def _outer_map(self, A, B):
        return np.einsum('ni,nj->nij', A, B)

    def BaM(self, m, s, rho):
        T = 5
        B = 2
        I = np.eye(1)
        for t in range(T):
            lt = 2 * 1 / (t + 1) # B * D / (t + 1)
            N = st.norm(loc = m, scale = s)
            z = N.rvs(size = (B, 1))
            g = np.zeros(shape = (B, 1))
            for b in range(B):
                _, grad = self.logp_grad_rho(z[b], rho)
                g[b] = grad
            zbar = np.mean(z)
            gbar = np.mean(g, axis=0)
            C = np.mean(self._outer_map(z - zbar, z - zbar), axis=0)
            G = np.mean(self._outer_map(g - gbar, g - gbar), axis=0)
            U = lt * G + lt * np.outer(gbar, gbar) / (1 + lt)
            V = s + lt * C + lt * np.outer(m - zbar, m - zbar) / (1 + lt)
            s = 2 * np.linalg.solve(I + sqrtm(I + 4 * (U @ V)).T, V.T)
            m = 1 * m / (1 + lt) + lt * (s @ gbar + zbar) / (1 + lt)
        return np.array([m.flatten()[0], s.flatten()[0]])

    def nlogp_grad_rho(self, xi, rho):
        l, g = self.model.log_density_gradient(xi * rho + self.theta)
        return -l, -g.dot(rho)

    def fitBaM(self, rho):
        o = minimize(self.nlogp_grad_rho,
                     self.rng.normal() * self._initscale,
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        self.grad_evals += o["nfev"]
        h = o["hess_inv"][0, 0]
        s = 0.0
        if h > 0 and np.isfinite(h):
            s = 0.5 * np.log(h)
        eta = self.BaM(o.x[0], np.exp(s), rho)
        return eta

    def fitKL(self, rho):
        o = minimize(self.logp_grad_rho,
                     self.rng.normal() * self._initscale,
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        self.grad_evals += o["nfev"]
        h = o["hess_inv"][0, 0]
        s = 0.0
        if h > 0 and np.isfinite(h):
            s = 0.5 * np.log(h)
        init = np.array([o.x[0], s])

        for attempt in range(2):
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                try:
                    o = minimize(self.KL,
                                 init,
                                 args = (rho, attempt,),
                                 jac = True,
                                 method = "BFGS",
                                 options = {"gtol": self._gtol})
                    self.grad_evals += o["nfev"] * 4 * (attempt + 1)
                    return o.x
                except FloatingPointError as e:
                    pass
        return init

    def _random_direction(self):
        evals = self._eigvals
        p = evals / np.sum(evals)
        j = self.rng.choice(np.size(p), p = p)
        m = self._eigvecs[:, j]
        S = np.diag(self._cov)
        rho = self.rng.multivariate_normal(m, S)
        return rho / np.linalg.norm(rho + self._tol)

    def _log_q(self, x, eta):
        m, s = self._unpack(eta)
        z = (x - m) / s
        return -np.log(s) - 0.5 * z * z

    def _overrelaxed_proposal(self, eta):
        m, s = self._unpack(eta)
        K = self.K
        Normal = st.norm(m, s)
        u = Normal.cdf(np.zeros(1))
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
        return Normal.ppf(up)

    def _metropolis_step(self, eta, rho):
        xi = self._overrelaxed_proposal(eta)
        thetap = xi * rho + self.theta

        r = self.model.log_density(thetap)
        r -= self.model.log_density(self.theta)
        r += self._log_q(0, eta)
        r -= self._log_q(xi[0], eta)

        self.grad_evals += 2

        a = np.log(self.rng.uniform()) < np.minimum(0, r)
        self.theta = a * thetap + (1 - a) * self.theta

        d = a - self.acceptance_probability
        self.acceptance_probability += d / self._draw
        return self.theta

    def draw(self):
        self._draw += 1
        rho = self._random_direction()
        etakl = self.fitBaM(rho)
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
