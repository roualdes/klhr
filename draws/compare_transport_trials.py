import argparse
from pathlib import Path

import h5py
import numpy as np


def first_ridge_iteration(draws, beta0_threshold, beta1_threshold):
    idx = np.flatnonzero(
        (draws[:, 0] < beta0_threshold) & (draws[:, 1] > beta1_threshold)
    )
    if idx.size == 0:
        return None
    return int(idx[0])


def summarize(path, warmup, beta0_threshold, beta1_threshold):
    with h5py.File(path, "r") as h5:
        root = h5["earnings"]
        draws = np.asarray(root["draws"])
        log_density = np.asarray(root["log_density"]).reshape(-1)
        phase = np.asarray(root["diagnostics/phase"]).reshape(-1).astype(int)
        selected = np.asarray(root["diagnostics/selected_candidate"]).reshape(-1)
        diag_jump = np.asarray(root["diagnostics/diag_jump"]).reshape(-1)
        stop = int(np.asarray(root["stop_transport_idx"]))

    transport = phase == 0
    if np.any(transport):
        tdraws = draws[transport]
        tlogp = log_density[transport]
        tjump = diag_jump[transport]
        accepted_transport = int(np.sum(selected[transport] == 1))
        transport_endpoint = tdraws[-1]
        transport_logp_gain = tlogp[-1] - tlogp[0]
        max_abs_transport = np.max(np.abs(tdraws), axis=0)
        max_transport_jump = np.nanmax(tjump)
    else:
        accepted_transport = 0
        transport_endpoint = np.full(draws.shape[1], np.nan)
        transport_logp_gain = np.nan
        max_abs_transport = np.full(draws.shape[1], np.nan)
        max_transport_jump = np.nan

    warmup = min(warmup, draws.shape[0])
    posterior_mean = np.mean(draws[warmup:], axis=0)
    ridge = first_ridge_iteration(draws, beta0_threshold, beta1_threshold)

    return {
        "file": Path(path).name,
        "ridge": ridge,
        "stop": stop,
        "accepted_transport": accepted_transport,
        "transport_endpoint": transport_endpoint,
        "transport_logp_gain": transport_logp_gain,
        "max_abs_transport": max_abs_transport,
        "max_transport_jump": max_transport_jump,
        "posterior_mean": posterior_mean,
        "final_logp": log_density[-1],
    }


def fmt(x):
    if x is None:
        return "inf"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.4g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--warmup", type=int, default=15_000)
    parser.add_argument("--beta0-threshold", type=float, default=-10_000.0)
    parser.add_argument("--beta1-threshold", type=float, default=500.0)
    args = parser.parse_args()

    rows = [
        summarize(path, args.warmup, args.beta0_threshold, args.beta1_threshold)
        for path in args.files
    ]

    print(
        "variant                      ridge  accT  max|b0|T  max|b1|T  "
        "max sigmaT     max sT  max jumpT  mean b0  mean b1  mean sigma  mean s"
    )
    for row in rows:
        m = row["posterior_mean"]
        mt = row["max_abs_transport"]
        print(
            f"{row['file']:<28} "
            f"{fmt(row['ridge']):>6} "
            f"{row['accepted_transport']:>5d} "
            f"{fmt(mt[0]):>9} "
            f"{fmt(mt[1]):>9} "
            f"{fmt(mt[2]):>11} "
            f"{fmt(mt[3]):>9} "
            f"{fmt(row['max_transport_jump']):>10} "
            f"{fmt(m[0]):>8} "
            f"{fmt(m[1]):>8} "
            f"{fmt(m[2]):>11} "
            f"{fmt(m[3]):>7}"
        )

    print("\nTransport endpoint by variant:")
    for row in rows:
        x = row["transport_endpoint"]
        print(
            f"  {row['file']:<28} "
            f"beta0={fmt(x[0])} beta1={fmt(x[1])} "
            f"sigma={fmt(x[2])} s={fmt(x[3])} "
            f"logp_gain={fmt(row['transport_logp_gain'])}"
        )


if __name__ == "__main__":
    main()
