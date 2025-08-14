import numpy as np

class MCMCBase():
    def __init__(self, model, stepsize, theta = None, seed = None):
        self.model = model
        self.D = self.model.dim()
        self.stepsize = stepsize

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

        if theta is None:
            self.theta = self.rng.normal(scale = 0.1, size = self.D)
        else:
            self.theta = theta

    def __iter__(self):
        return self

    def __next__(self):
        return self.draw()

    def log_density(self, theta):
        try:
            return self.model.log_density(theta)
        except Exception as e:
            return np.NINF

    def sample(self, M):
        D = self.D
        thetas = np.empty((M, D), dtype=np.float64)
        thetas[0] = self.theta;
        for m in range(1, M):
            thetas[m] = self.draw()
        return thetas

    def sample_constrained(self, M):
        D = self.D
        thetas = np.empty((M, D), dtype=np.float64)
        thetas[0] = self.model.constrain(self.theta)
        for m in range(1, M):
            theta_m = self.draw()
            thetas[m].model.constrain(theta_m)
        return thetas
