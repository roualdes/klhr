import argparse
from pathlib import Path

import h5py
import numpy as np


PHASES = {
    0: "transport",
    1: "warmup",
    2: "sampling",
}


def as_1d(dataset):
    return np.asarray(dataset).reshape(-1)


def finite(x):
    x = np.asarray(x)
    return x[np.isfinite(x)]


def stats_line(label, x):
    x = finite(x)
    if x.size == 0:
        print(f"  {label:<28} no finite values")
        return
    q = np.quantile(x, [0.05, 0.5, 0.95])
    print(
        f"  {label:<28} n={x.size:6d} "
        f"mean={np.mean(x): .4g} median={q[1]: .4g} "
        f"q05={q[0]: .4g} q95={q[2]: .4g}"
    )


def load_h5(path, group):
    with h5py.File(path, "r") as h5:
        root = h5[group]
        diag = root["diagnostics"]
        out = {
            "draws": np.asarray(root["draws"]),
            "log_density": as_1d(root["log_density"]),
            "stop_transport_idx": int(np.asarray(root["stop_transport_idx"])),
            "phase": as_1d(diag["phase"]).astype(int),
            "cos_grad_move": as_1d(diag["cos_grad_move"]),
            "beta_slope": as_1d(diag["beta_slope"]),
            "logp_gain": as_1d(diag["logp_gain"]),
            "diag_jump": as_1d(diag["diag_jump"]),
            "move_norm": as_1d(diag["move_norm"]),
            "move": np.asarray(diag["move_unconstrained"]),
        }
        return out


def print_run_header(data, path):
    draws = data["draws"]
    print(f"file: {path}")
    print(f"draws: {draws.shape[0]} iterations, {draws.shape[1]} parameters")
    print(f"transport stop index: {data['stop_transport_idx']}")
    print(
        "final draw: "
        f"beta0={draws[-1, 0]:.6g} beta1={draws[-1, 1]:.6g} "
        f"sigma={draws[-1, 2]:.6g} s={draws[-1, 3]:.6g} "
        f"logp={data['log_density'][-1]:.6g}"
    )


def orthogonality_checks(data):
    print("\n1. Gradient-Orthogonality Checks")
    moved = data["move_norm"] > 1e-12
    for phase_id, phase_name in PHASES.items():
        mask = (data["phase"] == phase_id) & moved
        stats_line(f"{phase_name} move cos", data["cos_grad_move"][mask])


def ridge_checks(data, beta0_threshold, beta1_threshold, window):
    draws = data["draws"]
    ridge_mask = (
        (draws[:, 0] < beta0_threshold) &
        (draws[:, 1] > beta1_threshold)
    )
    ridge_idx = np.flatnonzero(ridge_mask)

    print("\n2. Ridge Discovery Checks")
    if ridge_idx.size == 0:
        print(
            "  first ridge event           not found "
            f"(beta0 < {beta0_threshold:g}, beta1 > {beta1_threshold:g})"
        )
        return None

    first = int(ridge_idx[0])
    print(
        "  first ridge event           "
        f"iteration {first} "
        f"(beta0={draws[first, 0]:.6g}, beta1={draws[first, 1]:.6g})"
    )
    print(f"  after transport by          {first - data['stop_transport_idx']} iterations")

    lo = max(0, first - window)
    hi = min(draws.shape[0], first + window + 1)
    mask = np.zeros(draws.shape[0], dtype=bool)
    mask[lo:hi] = True
    mask &= data["move_norm"] > 1e-12
    stats_line("ridge-window move cos", data["cos_grad_move"][mask])
    stats_line("ridge-window beta slope", data["beta_slope"][mask])
    stats_line("ridge-window logp gain", data["logp_gain"][mask])
    return first


def transport_endpoint_checks(data):
    draws = data["draws"]
    stop = data["stop_transport_idx"]
    stop = min(stop, draws.shape[0] - 1)
    delta = draws[stop] - draws[0]

    print("\n3. Transport Endpoint Checks")
    print(
        "  endpoint delta              "
        f"beta0={delta[0]: .6g} beta1={delta[1]: .6g} "
        f"sigma={delta[2]: .6g} s={delta[3]: .6g}"
    )
    print(
        "  endpoint value              "
        f"beta0={draws[stop, 0]: .6g} beta1={draws[stop, 1]: .6g} "
        f"sigma={draws[stop, 2]: .6g} s={draws[stop, 3]: .6g}"
    )

    transport = data["phase"] == 0
    moved = transport & (data["move_norm"] > 1e-12)
    stats_line("transport beta slope", data["beta_slope"][moved])
    stats_line("transport logp gain", data["logp_gain"][moved])
    stats_line("transport diag jump", data["diag_jump"][moved])


def top_rows_by_abs(values, n):
    values = np.nan_to_num(values, nan=0.0)
    return np.argsort(-np.abs(values))[:n]


def transport_move_beta_checks(data, n):
    transport_idx = np.flatnonzero(
        (data["phase"] == 0) & (data["move_norm"] > 1e-12)
    )
    print("\n4. Transport Move Beta Checks")
    if transport_idx.size == 0:
        print("  no accepted transport moves")
        return

    moves = data["move"][transport_idx]
    stats_line("accepted |d_beta0|", np.abs(moves[:, 0]))
    stats_line("accepted |d_beta1|", np.abs(moves[:, 1]))
    stats_line("accepted diag jump", data["diag_jump"][transport_idx])

    print("\n  largest accepted transport beta0 moves")
    print(
        "  iter    d_beta0    d_beta1   beta_slope   "
        "cos_grad    logp_gain  diag_jump"
    )
    for row in top_rows_by_abs(moves[:, 0], n):
        i = transport_idx[row]
        print(
            f"  {i:5d} "
            f"{moves[row, 0]: 10.4g} "
            f"{moves[row, 1]: 10.4g} "
            f"{data['beta_slope'][i]: 11.4g} "
            f"{data['cos_grad_move'][i]: 10.4g} "
            f"{data['logp_gain'][i]: 11.4g} "
            f"{data['diag_jump'][i]: 9.4g}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Focused KLHR gradient/tangent transport diagnostics."
    )
    parser.add_argument("--file", default="draws/earnings.h5")
    parser.add_argument("--group", default="earnings")
    parser.add_argument("--beta0-threshold", type=float, default=-10_000.0)
    parser.add_argument("--beta1-threshold", type=float, default=500.0)
    parser.add_argument("--window", type=int, default=75)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    path = Path(args.file)
    data = load_h5(path, args.group)

    print_run_header(data, path)
    orthogonality_checks(data)
    ridge_checks(data, args.beta0_threshold, args.beta1_threshold, args.window)
    transport_endpoint_checks(data)
    transport_move_beta_checks(data, args.top)

    print("\nReading Guide")
    print("  Accepted transport move cosines should stay near zero.")
    print("  If ridge discovery is still late, inspect whether transport moves")
    print("  produce beta0/beta1 displacement without destabilizing other parameters.")


if __name__ == "__main__":
    main()
