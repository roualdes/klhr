import sys

import numpy as np
import scipy.stats as st

from bsmodel import BSModel
from onlinemoments import OnlineMoments
from onlinepca import OnlinePCA
from mcmc import MCMCBase
from windowedadaptation import WindowedAdaptation

class Slice(MCMCBase):
    """Copied from https://glizen.com/radfordneal/slice.software.html"""
    def __init__(self,
                 bsmodel,
                 theta = None,
                 seed = None,
                 w = 1,
                 m = np.inf,
                 lower = -np.inf,
                 upper = np.inf,
                 J = 2,
                 l = 4,
                 initscale = 0.1,
                 warmup = 1_000,
                 windowsize = 50,
                 windowscale = 2,
                 tol = 1e-12,
                 scale_dir_cov = False,
                 eigen_method_one = True,
                 max_init_tries = 100):
        super().__init__(bsmodel, -1, theta = theta, seed = seed)

        # TODO checks
        self.w = w
        self.m = m
        self.lower = lower
        self.upper = upper

        self.J = J
        self.l = l

        self._initscale = initscale
        self._windowedadaptation = \
            WindowedAdaptation(warmup,
                               windowsize = windowsize,
                               windowscale = windowscale)
        self._onlinemoments = OnlineMoments(self.D)
        self._mean = np.zeros(self.D)
        self._cov = np.ones(self.D)
        self._scale_dir_cov = scale_dir_cov
        self._eigen_method_one = eigen_method_one
        self._onlinemoments_density = OnlineMoments(self.D)
        self._onlinepca = OnlinePCA(self.D, K = self.J, l = self.l)
        self._eigvecs = np.zeros((self.D, self.J + 1))
        self._eigvals = np.ones(self.J + 1)

        self._draw = 0
        self.acceptance_probability = 0
        # TODO self.ld_evaluations = 0

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

    def _uni_slice(self):
        x0 = self.theta
        gx0 = self.model.log_density(x0)
        logy = gx0 - self.rng.exponential()

        u = self.rng.uniform(low = 0, high = self.w)
        L = x0 - u
        R = x0 + (self.w - u)

        if np.isinf(self.m):
            while True:
                if L <= self.lower: break
                gL = self.model.log_density(L)
                # TODO self.ld_evaluations += 1
                if gL <= logy: break
                L -= self.w
            while True:
                if R >= self.upper: break
                gR =  self.model.log_density(R)
                # TODO self.ld_evaluations += 1
                if gR <= logy: break
                R += self.w
        elif m > 1:
            J = np.floor(self.rng.uniform(low = 0, high = self.m))
            K = (self.m - 1) - J
            while J > 0:
                if L <= self.lower: break
                gL = self.model.log_density(L)
                # TODO self.ld_evaluations += 1
                if gL <= logy: break
                L -= self.w
                J -= 1
            while K > 0:
                if R >= self.upper: break
                gR = self.model.log_density(R)
                # TODO self.ld_evaluations += 1
                if gR <= logy: break
                R += w
                K -= 1


        # Shrink interval to lower and upper bounds.
        if L < self.lower: L = self.lower
        if R > self.upper: R = self.upper

        # Sample from the interval, shrinking it on each rejection.
        while True:
            x1 = self.rng.uniform(low = L, high = R)
            gx1 = self.model.log_density(x1)
            # TODO self.ld_evaluations += 1
            if gx1 >= logy: break
            if x1 > x0:
                R = x1
            else:
                L = x1

        # update the point sampled
        self.theta = x1

    def draw(self):
        self._draw += 1
        self._uni_slice()
        return self.theta

if __name__ == "__main__":

    import numpy as np
    from pathlib import Path

    import bridgestan as bs
    from bsmodel import BSModel

    bs.set_bridgestan_path(Path.home().expanduser() / "bridgestan")

    model = "one_normal"
    source_dir = Path(__file__).resolve().parent
    bs_model = BSModel(stan_file = source_dir / f"stan/{model}.stan",
                       data_file = source_dir / f"stan/{model}.json")

    algo = Slice(bs_model)
    draws = algo.sample(10_000)

    xx = np.linspace(-10, 2, 101)
    fx = np.zeros_like(xx)
    for n, xxn in enumerate(xx):
        fx[n] = np.exp(algo.model.log_density(np.array([xxn]),
                                              propto = False))

    print(f"mean = {np.mean(draws, axis=0)}")
    print(f"std = {np.std(draws, ddof = 1, axis=0)}")
