import numpy as np

class OnlineMoments():
    def __init__(self, D):
        self.D = D
        self.N = 0
        self.m = np.zeros(self.D)
        self.v = np.zeros(self.D)

    def update(self, x):
        self.N += 1
        w = 1 / self.N
        d = x - self.m
        self.m += d * w
        self.v += -self.v * w + d * d * w * (1 - w)

    def mean(self):
        return self.m

    def var(self):
        if self.N > 2:
            return self.v * self.N / (self.N - 1)
        return np.ones(self.D)

    def reset(self):
        self.N = 0
        self.m = np.zeros(self.D)
        self.v = np.zeros(self.D)
