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
from klhr_sinh import KLHRSINH

bs.set_bridgestan_path(Path.home() / "bridgestan")


bs_model = BSModel(stan_file = "stan/funnel.stan",
                   data_file = "stan/funnel.json")

x = np.linspace(-20, 20, 301)
x2 = np.linspace(-15, 15, 301)
y = np.linspace(-20, 20, 301)

z = np.zeros(2)
Z = np.zeros((301, 301))
for ix, xn in enumerate(x):
    for iy, yn in enumerate(y):
        z[0] = xn
        z[1] = yn
        Z[ix, iy] = bs_model.log_density(z)

def random_direction(rng):
    x = rng.normal(size = 2)
    return x / np.linalg.norm(x)

def to_line(rho, x, o):
    return x * rho + o

def rotation_matrix(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c, -s], [s, c]])

rng = np.random.default_rng(204)

cols = ["#0072B2", "#D55E00", "#009E73", "#F0E442"]
pt = np.zeros(2)
pt[0] = rng.normal() * 3
pt[1] = rng.normal() * np.exp(0.5 * pt[0])
rho = random_direction(rng)
r = rotation_matrix(30)


sinhklhr = KLHRSINH(bs_model, theta = pt)
klhr = KLHR(bs_model, theta = pt)


fig = plt.figure(layout = "constrained", figsize = (14, 6))
gs = GridSpec(2, 4, figure = fig)
ax1 = fig.add_subplot(gs[:, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[0, 3])
ax4 = fig.add_subplot(gs[1, 2])
ax5 = fig.add_subplot(gs[1, 3])

pos = ax1.contour(x, y, Z, levels = np.linspace(-5, 0, 10), colors = "black", linestyles = "solid", alpha = 0.1)
ax1.scatter(pt[0], pt[1], color = "black")

xx = np.linspace(-10, 10, 201)

for idx in range(4):
    line = np.array([to_line(rho, xn, pt) for xn in xx])
    fx = np.array([np.exp(bs_model.log_density(ln)) for ln in line])
    fx /= np.max(fx)
    eta = sinhklhr.fit(rho)
    qx = np.exp(sinhklhr._logq(xx, eta))
    qx /= np.max(qx)
    zeta = klhr.fit(rho)
    px = np.exp(klhr._logq(xx, zeta))
    px /= np.max(px)
    ax1.plot(line[:, 0], line[:, 1], color = cols[idx])
    rho = r @ rho

    if idx == 0:
        ax2.plot(xx, fx, color = cols[0], linewidth = 2, label = r"$p(x)$")
        ax2.plot(xx, qx, color = "black", linestyle = "dashed", label = "sinh")
        ax2.plot(xx, px, color = "black", linestyle = "dotted", label = "norm")
        ax2.set_xlabel(r"$\xi$")
        ax2.legend()

    if idx == 1:
        ax3.plot(xx, fx, linewidth = 2, color = cols[1])
        ax3.plot(xx, qx, color = "black", linestyle = "dashed")
        ax3.plot(xx, px, color = "black", linestyle = "dotted", label = "norm")

    if idx == 2:
        ax4.plot(xx, fx, linewidth = 2, color = cols[2])
        ax4.plot(xx, qx, color = "black", linestyle = "dashed")
        ax4.plot(xx, px, color = "black", linestyle = "dotted", label = "norm")

    if idx == 3:
        ax5.plot(xx, fx, linewidth = 2, color = cols[3])
        ax5.plot(xx, qx, color = "black", linestyle = "dashed")
        ax5.plot(xx, px, color = "black", linestyle = "dotted", label = "norm")

fig.savefig("plot_funnel_general_approximations.png")



# rng = np.random.default_rng(204)


# cols = ["#0072B2", "#D55E00", "#009E73", "#F0E442"]
# pt = np.zeros(2)
# pt[1] = 2
# rho = np.array([0, -1]) # random_direction(rng)


# sinhklhr = KLHRSINH(bs_model, theta = pt)
# klhr = KLHR(bs_model, theta = pt)


# fig = plt.figure(layout = "constrained", figsize = (14, 6))
# gs = GridSpec(2, 4, figure = fig)
# ax1 = fig.add_subplot(gs[:, :2])
# ax2 = fig.add_subplot(gs[0, 2])
# ax3 = fig.add_subplot(gs[0, 3])
# ax4 = fig.add_subplot(gs[1, 2])
# ax5 = fig.add_subplot(gs[1, 3])

# pos = ax1.contour(x, y, Z, levels = np.linspace(-5, 0, 10), colors = "black", linestyles = "solid", alpha = 0.1)
# ax1.scatter(pt[0], pt[1], color = "black")

# xx = np.linspace(-10, 10, 201)

# for idx in range(4):
#     line = np.array([to_line(rho, xn, pt) for xn in xx])
#     fx = np.array([np.exp(bs_model.log_density(ln)) for ln in line])
#     fx /= np.max(fx)
#     eta = sinhklhr.fit(rho)
#     qx = np.exp(sinhklhr._logq(xx, eta))
#     qx /= np.max(qx)
#     zeta = klhr.fit(rho)
#     px = np.exp(klhr._logq(xx, zeta))
#     px /= np.max(px)
#     ax1.plot(line[:, 0], line[:, 1], color = cols[idx])
#     pt[1] -= 2

#     if idx == 0:
#         ax2.plot(xx, fx, color = cols[0], linewidth = 2, label = r"$f(x)$")
#         ax2.plot(xx, qx, color = "black", linestyle = "dashed", label = "sinh")
#         ax2.plot(xx, px, color = "black", linestyle = "dotted", label = "norm")
#         ax2.set_xlabel(r"$\xi$")
#         ax2.legend()

#     if idx == 1:
#         ax3.plot(xx, fx, linewidth = 2, color = cols[1])
#         ax3.plot(xx, qx, color = "black", linestyle = "dashed")
#         ax3.plot(xx, px, color = "black", linestyle = "dotted", label = "norm")

#     if idx == 2:
#         ax4.plot(xx, fx, linewidth = 2, color = cols[2])
#         ax4.plot(xx, qx, color = "black", linestyle = "dashed")
#         ax4.plot(xx, px, color = "black", linestyle = "dotted", label = "norm")

#     if idx == 3:
#         ax5.plot(xx, fx, linewidth = 2, color = cols[3])
#         ax5.plot(xx, qx, color = "black", linestyle = "dashed")
#         ax5.plot(xx, px, color = "black", linestyle = "dotted", label = "norm")
