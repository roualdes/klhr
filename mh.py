import traceback

import numpy as np

import mcmc

class MH(mcmc.MCMCBase):
    def __init__(self, model, stepsize, seed = None, theta = None):
        super().__init__(model, stepsize, seed = seed, theta = theta)
        self._draw = 0
        self.acceptance_probability = 0

    def proposal_density(self, thetap, theta):
        stepsize = self.stepsize
        z = (thetap - theta) / stepsize
        return -0.5 * z.dot(z)

    def draw(self):
        self._draw += 1
        try:
            xi = self.rng.normal(size = self.D)
            thetap = self.theta + xi * self.stepsize

            r = self.log_density(thetap)
            r -= self.log_density(self.theta)
            r += self.proposal_density(self.theta, thetap)
            r -= self.proposal_density(thetap, self.theta)

            accept = np.log(self.rng.uniform()) < np.minimum(0.0, r)
            self.theta = a * thetap + (1 - a) * self.theta
            d = accept - self.acceptance_probability
            self.acceptance_probability += d / self._draw

        except Exception as e:
            # traceback.print_exc()
            pass
        return self.theta
