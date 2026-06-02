import numpy as np

def fft_nextgoodsize(N):
    if N <= 2:
        return 2
    while True:
        m = N
        while np.remainder(m, 2) == 0:
            m /= 2
        while np.remainder(m, 3) == 0:
            m /= 3
        while np.remainder(m, 5) == 0:
            m /= 5
        if m <= 1:
            return N
        N += 1


def autocovariance(x):
    N = np.size(x)
    Mt2 = 2 * fft_nextgoodsize(N)
    yc = np.zeros(Mt2)
    yc[:N] = x - np.mean(x)
    t = np.fft.fft(yc)
    ac = np.fft.fft(np.conj(t) * t)
    return np.real(ac)[:N] / (N * N * 2)


def isconstant(x):
    mn = np.min(x)
    mx = np.max(x)
    return np.isclose(mn, mx)


def ess(x):
    niterations, nchains = np.shape(x)
    if niterations < 3:
        return np.nan
    if np.any(np.isnan(x)):
        return np.nan
    if np.any(np.isinf(x)):
        return np.nan
    if isconstant(x):
        return np.nan

    acov = np.apply_along_axis(autocovariance, 0, x)
    chain_mean = np.mean(x, axis = 0)
    mean_var = np.mean(acov[0, :]) * niterations / (niterations - 1)
    var_plus = mean_var * (niterations - 1) / niterations
    if nchains > 1:
        var_plus += np.var(chain_mean, ddof = 1)

    rhohat = np.zeros(niterations)
    rhohat_even = 1
    rhohat[0] = rhohat_even
    rhohat_odd = 1 - (mean_var - np.mean(acov[1, :])) / var_plus
    rhohat[1] = rhohat_odd

    t = 1
    while t < niterations - 4 and rhohat_even + rhohat_odd > 0:
        rhohat_even = 1 - (mean_var - np.mean(acov[t + 1, :])) / var_plus
        rhohat_odd = 1 - (mean_var - np.mean(acov[t + 2, :])) / var_plus
        if rhohat_even + rhohat_odd >= 0:
            rhohat[t + 1] = rhohat_even
            rhohat[t + 2] = rhohat_odd
        t += 2

    max_t = t
    if rhohat_even > 0:
        rhohat[max_t + 1] = rhohat_even

    t = 1
    while t <= max_t - 3:
        if rhohat[t + 1] + rhohat[t + 2] > rhohat[t - 1] + rhohat[t]:
            rhohat[t + 1] = (rhohat[t - 1] + rhohat[t]) / 2
            rhohat[t + 2] = rhohat[t + 1]
        t += 2

    ess = nchains * niterations
    tau = -1 + 2 * np.sum(rhohat[0:np.maximum(1, max_t)]) + rhohat[max_t + 1]
    tau = np.maximum(tau, 1 / np.log10(ess))
    return ess / tau
