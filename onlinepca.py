import numpy as np

class OnlinePCA():
    """Adapted from https://www.cse.msu.edu/~weng/research/CCIPCApami.pdf"""
    def __init__(self, D, K = 1, l = 0, tol = 1e-10):
        self.D = D
        self.K = K
        self.l = l
        self.v = np.zeros((self.D, self.K))
        self.n = 0
        self.tol = tol

    def update(self, u):
        """Procedure 1"""
        self.n += 1
        for i in range(min(self.K, self.n)):
            if i == self.n - 1:
                self.v[:, i] = u
            else:
                w = (self.n - 1 - self.l) / self.n
                v = self.v[:, i]
                nv = np.linalg.norm(v)
                self.v[:, i] = w * v + (1 - w) * u * u.dot(v) / (nv + self.tol) # eq 10
                v = self.v[:, i]
                nv = np.linalg.norm(v)
                u = u - u.dot(v) * v / (nv * nv + self.tol) # eq 11

    def values(self):
        nv = np.linalg.norm(self.v, axis = 0)
        if np.any(np.isnan(nv) | np.isinf(nv)):
            nv = np.zeros_like(self.v)
        return nv + self.tol

    def vectors(self):
        return self.v / (self.values())

    def reset(self):
        self.n = 0
        self.v = np.zeros((self.D, self.K))

if __name__ == "__main__":
    rng = np.random.default_rng()

    # iid Normal data
    D = 2
    N = 10_000
    X = rng.normal(size = (N, D))

    opca = OnlinePCA(D, K = 2)
    for n in range(N):
        opca.update(X[n])

    eigvecs = opca.vectors()
    print(eigvecs)

    U, S, Vt = np.linalg.svd(np.cov(X.T), hermitian = True)
    print(Vt.T)

    # AR(1) data
    D = 100
    N = 1_000
    y = np.zeros((D, N))
    alpha = 0.9
    beta = np.sqrt(1 - alpha * alpha)
    y[0] = rng.normal(size = N)
    for d in range(1, D):
        y[d] = alpha * y[d-1] + rng.normal(size = N) * beta
    X = y.T

    K = 2
    opca = OnlinePCA(D, K = K)
    for n in range(N):
        opca.update(X[n])

    # print(opca.vectors())
    print(f"OPCA = {opca.values()}")

    U, S, Vt = np.linalg.svd(np.cov(y), hermitian = True)
    #print(Vt.T[:, 0:K])
    print(f"SVD = {S[:K]}")
