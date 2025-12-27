import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import numpy as np
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import minimize

from bsmodel import BSModel
import bridgestan as bs
from klhr import KLHR
from klhr_sub_sinh import KLHRSUBSINH
from klhr_sinh import KLHRSINH

bs.set_bridgestan_path(Path.home() / "bridgestan")

bs_model = BSModel(stan_file = "stan/ar1.stan",
                   data_file = "stan/ar1.json")
D = bs_model.dim()

def random_direction(rng, D):
    x = rng.normal(size = D)
    return x / np.linalg.norm(x)

def to_line(rho, x, o):
    return x * rho + o

def rotation_matrix(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c, -s], [s, c]])

cols = ["#0072B2", "#D55E00", "#009E73", "#F0E442"]

rng = np.random.default_rng()

xx = np.linspace(-4, 4, 301)

fig, axs = plt.subplots(4, 4, layout = "constrained", figsize = (14, 14))

for i, ax in enumerate(axs.flat):
    pt = rng.normal(size = D)
    algo = KLHRSUBSINH(bs_model, theta = pt)
    rho = random_direction(rng, D)
    line = np.array([np.array([xn]) * rho + pt for xn in xx])
    fx = np.array([np.exp(bs_model.log_density(ln)) for ln in line])
    fx /= np.max(fx)
    ax.plot(xx, fx)
    eta = algo.fit(rho)
    qx = np.exp(algo._log_q(xx, eta))
    qx /= np.max(qx)
    ax.plot(xx, qx, linestyle = "dashed")

fig.savefig("experiments/plot_ar1_general_approximations.png")
