import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def trace_plot(draws, log_density, stop_idx, plot_name):
    plt.clf()
    M = np.shape(draws)[0]
    idx = np.arange(M)
    fig = plt.figure(figsize = (14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios = [1, 1, 0.85])

    axs = np.empty((2, 2), dtype = object)
    axs[0, 0] = fig.add_subplot(gs[0, 0])
    axs[0, 1] = fig.add_subplot(gs[0, 1])
    axs[1, 0] = fig.add_subplot(gs[1, 0])
    axs[1, 1] = fig.add_subplot(gs[1, 1])
    ax_log_density = fig.add_subplot(gs[2, :])

    axs[0, 0].plot(idx, draws[:, 0])
    axs[0, 0].axvline(x = stop_idx, linestyle = "--")
    axs[0, 0].set_ylabel(r"$\beta_0$")

    axs[0, 1].plot(idx, draws[:, 1])
    axs[0, 1].axvline(x = stop_idx, linestyle = "--")
    axs[0, 1].set_ylabel(r"$\beta_1$")

    axs[1, 0].plot(idx, draws[:, 2])
    axs[1, 0].axvline(x = stop_idx, linestyle = "--")
    axs[1, 0].set_ylabel(r"$\sigma$")

    axs[1, 1].plot(idx, draws[:, 3])
    axs[1, 1].axvline(x = stop_idx, linestyle = "--")
    axs[1, 1].set_ylabel(r"$s$")

    ax_log_density.plot(idx, log_density)
    ax_log_density.axvline(x = stop_idx, linestyle = "--")
    ax_log_density.set_ylabel("log density")
    ax_log_density.set_xlabel("iteration")

    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()

def acceptance_rate_plot(ar, plot_name):
    plt.clf()
    plt.plot(ar)
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()

def nfev_plot(nfev, plot_name):
    plt.clf()
    plt.plot(nfev)
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()

def scatter_plot(x, y, plot_name):
    plt.clf()
    plt.scatter(x, y)
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()

f = h5py.File("draws/earnings.h5", "r")

draws = np.asarray(f["earnings/draws"])
log_density = np.asarray(f["earnings/log_density"])
stop_idx = np.asarray(f["earnings/stop_transport_idx"])
print(f"Stopped transport phase at: {stop_idx}")
trace_plot(draws, log_density, stop_idx, "draws/earnings_trace_plots.png")

ar = np.asarray(f["earnings/acceptance_rate"])
acceptance_rate_plot(ar, "draws/earnings_acceptance_rate.png")

nfev = np.asarray(f["earnings/nfev"])
mdx = np.arange(np.size(nfev)) + 1
nfev = nfev.flatten() / mdx
nfev_plot(nfev, "draws/earnings_nfev.png")

scatter_plot(draws[:, 0], draws[:, 1], "draws/earnings_scatter_plot.png")
