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

            a = self.log_density(thetap)
            a += self.proposal_density(self.theta, thetap)
            a -= self.log_density(self.theta)
            a -= self.proposal_density(thetap, self.theta)

            accept = np.log(self.rng.uniform()) < np.minimum(0.0, a)
            if accept:
                self.theta = thetap

            d = accept - self.acceptance_probability
            self.acceptance_probability += d / self._draw

        except Exception as e:
            # traceback.print_exc()
            pass
        return self.theta
