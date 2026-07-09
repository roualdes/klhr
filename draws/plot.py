import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

def trace_plot(draws, log_density, plot_name):
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
    axs[0, 0].set_ylabel(r"$\beta_0$")

    axs[0, 1].plot(idx, draws[:, 1])
    axs[0, 1].set_ylabel(r"$\beta_1$")

    axs[1, 0].plot(idx, draws[:, 2])
    axs[1, 0].set_ylabel(r"$\sigma$")

    axs[1, 1].plot(idx, draws[:, 3])
    axs[1, 1].set_ylabel(r"$s$")

    ax_log_density.plot(idx, log_density)
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

def earnings(tbl, model):
    draws = np.asarray(tbl["draws"])
    log_density = np.asarray(tbl["log_density"])
    trace_plot(draws, log_density, f"draws/{model}_trace_plots.png")

    ar = np.asarray(tbl["acceptance_rate"])
    acceptance_rate_plot(ar, f"draws/{model}_acceptance_rate.png")

    nfev = np.asarray(tbl["nfev"])
    mdx = np.arange(np.size(nfev)) + 1
    nfev = nfev.flatten() / mdx
    nfev_plot(nfev, f"draws/{model}_nfev.png")

    scatter_plot(draws[:, 0], draws[:, 1], f"draws/{model}_scatter_plot.png")

def funnel(tbl, model):
    draws = np.asarray(tbl["draws"])
    idx = np.arange(np.shape(draws)[0]) + 1
    plt.clf()
    plt.plot(idx, draws[:, 0])
    plt.tight_layout()
    plt.savefig(f"draws/{model}_trace_plot.png")
    plt.close()
    scatter_plot(draws[:, 1], draws[:, 0], f"draws/{model}_scatter_plot.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="earnings")
    args = parser.parse_args()

    with h5py.File("draws/experiments.h5", "r") as f:
        tbl = f[args.model]
        model = args.model
        if model == "earnings":
            earnings(tbl, model)
        elif model == "funnel":
            funnel(tbl, model)
        else:
            print(f"don't know model {model}")

if __name__ == "__main__":
    main()
