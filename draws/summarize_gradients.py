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
            "selected_candidate": as_1d(diag["selected_candidate"]).astype(int),
        }
        cand = diag["transport_candidates"]
        out["candidates"] = {
            "probability": np.asarray(cand["probability"]),
            "logp_gain": np.asarray(cand["logp_gain"]),
            "cos_grad_move": np.asarray(cand["cos_grad_move"]),
            "beta_slope": np.asarray(cand["beta_slope"]),
            "delta_beta0": np.asarray(cand["delta_beta0"]),
            "delta_beta1": np.asarray(cand["delta_beta1"]),
            "diag_jump": np.asarray(cand["diag_jump"]),
            "move_norm": np.asarray(cand["move_norm"]),
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

    cand = data["candidates"]
    transport = data["phase"] == 0
    proposal_cols = np.arange(1, cand["cos_grad_move"].shape[1])
    all_candidate_cos = cand["cos_grad_move"][transport][:, proposal_cols].ravel()

    selected_rows = []
    selected_cols = []
    for i in np.flatnonzero(transport):
        col = int(data["selected_candidate"][i])
        if 0 <= col < cand["cos_grad_move"].shape[1]:
            selected_rows.append(i)
            selected_cols.append(col)
    selected_rows = np.asarray(selected_rows, dtype=int)
    selected_cols = np.asarray(selected_cols, dtype=int)
    selected_cos = cand["cos_grad_move"][selected_rows, selected_cols]

    stats_line("all transport candidate cos", all_candidate_cos)
    stats_line("selected candidate cos", selected_cos)
    if finite(all_candidate_cos).size and finite(selected_cos).size:
        tilt = np.mean(finite(selected_cos)) - np.mean(finite(all_candidate_cos))
        print(f"  selection cosine tilt        {tilt: .4g}")


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


def transport_candidate_beta_checks(data, n):
    cand = data["candidates"]
    transport_idx = np.flatnonzero(data["phase"] == 0)
    proposal_cols = np.arange(1, cand["delta_beta0"].shape[1])
    if transport_idx.size == 0 or proposal_cols.size == 0:
        print("\n4. Transport Candidate Beta Checks")
        print("  no transport candidate proposals")
        return

    transport_delta_beta0 = cand["delta_beta0"][transport_idx][:, proposal_cols]
    flat_order = top_rows_by_abs(transport_delta_beta0.ravel(), n)

    print("\n4. Transport Candidate Beta Checks")
    stats_line(
        "all candidate |d_beta0|",
        np.abs(cand["delta_beta0"][transport_idx][:, proposal_cols]).ravel(),
    )
    stats_line(
        "all candidate |d_beta1|",
        np.abs(cand["delta_beta1"][transport_idx][:, proposal_cols]).ravel(),
    )

    selected_rows = []
    selected_cols = []
    for i in transport_idx:
        col = int(data["selected_candidate"][i])
        if col > 0 and col < cand["delta_beta0"].shape[1]:
            selected_rows.append(i)
            selected_cols.append(col)
    selected_rows = np.asarray(selected_rows, dtype=int)
    selected_cols = np.asarray(selected_cols, dtype=int)
    if selected_rows.size > 0:
        stats_line(
            "selected |d_beta0|",
            np.abs(cand["delta_beta0"][selected_rows, selected_cols]),
        )
        stats_line(
            "selected |d_beta1|",
            np.abs(cand["delta_beta1"][selected_rows, selected_cols]),
        )

    print("\n  largest candidate beta0 moves")
    print(
        "  iter cand    d_beta0    d_beta1   beta_slope   "
        "cos_grad    logp_gain      prob"
    )
    n_cols = proposal_cols.size
    for flat in flat_order:
        row = flat // n_cols
        col = proposal_cols[flat % n_cols]
        i = transport_idx[row]
        print(
            f"  {i:5d} {col:4d} "
            f"{cand['delta_beta0'][i, col]: 10.4g} "
            f"{cand['delta_beta1'][i, col]: 10.4g} "
            f"{cand['beta_slope'][i, col]: 11.4g} "
            f"{cand['cos_grad_move'][i, col]: 10.4g} "
            f"{cand['logp_gain'][i, col]: 11.4g} "
            f"{cand['probability'][i, col]: 9.4g}"
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
    transport_candidate_beta_checks(data, args.top)

    print("\nReading Guide")
    print("  Candidate and selected cosines should stay near zero.")
    print("  If ridge discovery is still late, inspect whether transport candidates")
    print("  ever propose large beta0/beta1 moves and whether those moves get nontrivial probability.")


if __name__ == "__main__":
    main()
