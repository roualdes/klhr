import numpy as np

class Smoother():
    def __init__(self, x0, kappa = -0.75):
        self._initial = x0
        self._x = x0
        self._kappa = kappa
        self._count = 0

    def update(self, d):
        self._count += 1
        k = self._count ** self._kappa
        self._x = k * (self._x + d) + (1 - k) * self._x

    def optimum(self):
        return self._x

    def reset(self):
        self._count = 0
        self._x = self._initial
