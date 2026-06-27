import argparse
import csv
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np


DIRECTION_LAWS = ("kappa", "projected")


def first_ridge_iteration(draws, beta0_threshold, beta1_threshold):
    idx = np.flatnonzero(
        (draws[:, 0] < beta0_threshold) & (draws[:, 1] > beta1_threshold)
    )
    if idx.size == 0:
        return None
    return int(idx[0])


def summarize_h5(path, warmup, beta0_threshold, beta1_threshold):
    with h5py.File(path, "r") as h5:
        root = h5["earnings"]
        draws = np.asarray(root["draws"])
        log_density = np.asarray(root["log_density"]).reshape(-1)
        phase = np.asarray(root["diagnostics/phase"]).reshape(-1).astype(int)
        move_norm = np.asarray(root["diagnostics/move_norm"]).reshape(-1)
        diag_jump = np.asarray(root["diagnostics/diag_jump"]).reshape(-1)
        stop_transport_idx = int(np.asarray(root["stop_transport_idx"]))
        nfev = np.asarray(root["nfev"]).reshape(-1)

    transport = phase == 0
    transport_draws = draws[transport]
    posterior_draws = draws[min(warmup, draws.shape[0]) :]

    if transport_draws.size:
        max_abs_beta0_transport = float(np.nanmax(np.abs(transport_draws[:, 0])))
        max_abs_beta1_transport = float(np.nanmax(np.abs(transport_draws[:, 1])))
        max_sigma_transport = float(np.nanmax(transport_draws[:, 2]))
        max_s_transport = float(np.nanmax(transport_draws[:, 3]))
        accepted_transport = int(np.sum(move_norm[transport] > 1e-12))
        max_transport_jump = float(np.nanmax(diag_jump[transport]))
        transport_endpoint = transport_draws[-1]
        transport_logp_gain = float(log_density[transport][-1] - log_density[transport][0])
    else:
        max_abs_beta0_transport = np.nan
        max_abs_beta1_transport = np.nan
        max_sigma_transport = np.nan
        max_s_transport = np.nan
        accepted_transport = 0
        max_transport_jump = np.nan
        transport_endpoint = np.full(draws.shape[1], np.nan)
        transport_logp_gain = np.nan

    means = np.mean(posterior_draws, axis=0)
    ridge = first_ridge_iteration(draws, beta0_threshold, beta1_threshold)

    return {
        "ridge_iteration": "" if ridge is None else ridge,
        "ridge_found": ridge is not None,
        "stop_transport_idx": stop_transport_idx,
        "accepted_transport": accepted_transport,
        "transport_logp_gain": transport_logp_gain,
        "transport_endpoint_beta0": float(transport_endpoint[0]),
        "transport_endpoint_beta1": float(transport_endpoint[1]),
        "transport_endpoint_sigma": float(transport_endpoint[2]),
        "transport_endpoint_s": float(transport_endpoint[3]),
        "max_abs_beta0_transport": max_abs_beta0_transport,
        "max_abs_beta1_transport": max_abs_beta1_transport,
        "max_sigma_transport": max_sigma_transport,
        "max_s_transport": max_s_transport,
        "max_transport_jump": max_transport_jump,
        "mean_beta0": float(means[0]),
        "mean_beta1": float(means[1]),
        "mean_sigma": float(means[2]),
        "mean_s": float(means[3]),
        "final_logp": float(log_density[-1]),
        "nfev": int(nfev[-1]),
    }


def run_one(args, direction_law, replicate):
    seed = args.seed_base + replicate
    output = args.output_dir / f"{direction_law}_{replicate:02d}.h5"
    command = [
        str(args.executable),
        "--seed",
        str(seed),
        "--sampler",
        "sas",
        "--transport-direction",
        direction_law,
        "--transport-proposal",
        "random",
        "--transport-kappa",
        str(args.transport_kappa),
        "--transport-steps",
        str(args.transport_steps),
        "--gradient-history",
        str(args.gradient_history),
        "--projection-probability",
        str(args.projection_probability),
        "--n",
        str(args.n),
        "--warmup",
        str(args.warmup),
        "--output",
        str(output),
    ]

    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=args.cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime_seconds = time.perf_counter() - start

    row = summarize_h5(
        output,
        args.warmup,
        args.beta0_threshold,
        args.beta1_threshold,
    )
    row.update(
        {
            "direction_law": direction_law,
            "sampler": "sas",
            "transport_proposal": "random",
            "transport_kappa": args.transport_kappa,
            "transport_steps": args.transport_steps,
            "gradient_history": args.gradient_history,
            "projection_probability": args.projection_probability,
            "replicate": replicate,
            "seed": seed,
            "runtime_seconds": runtime_seconds,
            "output": str(output),
            "stdout": result.stdout.strip().replace("\n", " | "),
        }
    )
    return row


def write_rows(path, rows):
    fieldnames = [
        "direction_law",
        "sampler",
        "transport_proposal",
        "transport_kappa",
        "transport_steps",
        "gradient_history",
        "projection_probability",
        "replicate",
        "seed",
        "runtime_seconds",
        "ridge_iteration",
        "ridge_found",
        "stop_transport_idx",
        "accepted_transport",
        "transport_logp_gain",
        "transport_endpoint_beta0",
        "transport_endpoint_beta1",
        "transport_endpoint_sigma",
        "transport_endpoint_s",
        "max_abs_beta0_transport",
        "max_abs_beta1_transport",
        "max_sigma_transport",
        "max_s_transport",
        "max_transport_jump",
        "mean_beta0",
        "mean_beta1",
        "mean_sigma",
        "mean_s",
        "final_logp",
        "nfev",
        "output",
        "stdout",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_ridges(rows):
    return np.array(
        [float(row["ridge_iteration"]) for row in rows if row["ridge_found"]],
        dtype=float,
    )


def print_summary(rows):
    print(
        "direction  n found  runtime_mean  ridge_median  ridge_mean  "
        "ridge_q25  ridge_q75  accT_med  max|b0|T_med  max_sigmaT_med  "
        "max_sT_med  mean_beta0  mean_beta1"
    )
    for direction_law in DIRECTION_LAWS:
        subset = [row for row in rows if row["direction_law"] == direction_law]
        ridges = finite_ridges(subset)
        runtime = np.array([row["runtime_seconds"] for row in subset], dtype=float)
        acc_t = np.array([row["accepted_transport"] for row in subset], dtype=float)
        max_b0 = np.array([row["max_abs_beta0_transport"] for row in subset], dtype=float)
        max_sigma = np.array([row["max_sigma_transport"] for row in subset], dtype=float)
        max_s = np.array([row["max_s_transport"] for row in subset], dtype=float)
        mean_beta0 = np.array([row["mean_beta0"] for row in subset], dtype=float)
        mean_beta1 = np.array([row["mean_beta1"] for row in subset], dtype=float)

        if ridges.size:
            q25, median, q75 = np.quantile(ridges, [0.25, 0.5, 0.75])
            ridge_mean = np.mean(ridges)
        else:
            q25 = median = q75 = ridge_mean = np.nan

        print(
            f"{direction_law:<9} "
            f"{len(subset):2d} {ridges.size:5d} "
            f"{np.mean(runtime):12.3f} "
            f"{median:12.1f} "
            f"{ridge_mean:10.1f} "
            f"{q25:9.1f} "
            f"{q75:9.1f} "
            f"{np.median(acc_t):8.1f} "
            f"{np.median(max_b0):13.4g} "
            f"{np.median(max_sigma):15.4g} "
            f"{np.median(max_s):11.4g} "
            f"{np.mean(mean_beta0):11.4g} "
            f"{np.mean(mean_beta1):11.4g}"
        )

    print("\nPer-replicate ridge iterations:")
    print("replicate  kappa  projected")
    by_key = {
        (row["replicate"], row["direction_law"]): row for row in rows
    }
    for replicate in sorted({row["replicate"] for row in rows}):
        cells = []
        for direction_law in DIRECTION_LAWS:
            row = by_key[(replicate, direction_law)]
            cells.append(str(row["ridge_iteration"]) if row["ridge_found"] else "inf")
        print(f"{replicate:9d} {cells[0]:>6} {cells[1]:>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2026062900)
    parser.add_argument("--n", type=int, default=30_000)
    parser.add_argument("--warmup", type=int, default=15_000)
    parser.add_argument("--transport-steps", type=int, default=100)
    parser.add_argument("--transport-kappa", type=float, default=10.0)
    parser.add_argument("--gradient-history", type=int, default=3)
    parser.add_argument("--projection-probability", type=float, default=0.5)
    parser.add_argument("--beta0-threshold", type=float, default=-10_000.0)
    parser.add_argument("--beta1-threshold", type=float, default=500.0)
    parser.add_argument("--executable", type=Path, default=Path("build/examples/example"))
    parser.add_argument("--cwd", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("draws/transport_direction_trials"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("draws/transport_direction_trials/summary.csv"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = args.replicates * len(DIRECTION_LAWS)
    completed = 0
    for replicate in range(args.replicates):
        for direction_law in DIRECTION_LAWS:
            completed += 1
            print(f"[{completed:02d}/{total}] {direction_law} replicate {replicate}")
            row = run_one(args, direction_law, replicate)
            rows.append(row)
            write_rows(args.summary_csv, rows)
            ridge = row["ridge_iteration"] if row["ridge_found"] else "inf"
            print(
                f"  runtime={row['runtime_seconds']:.3f}s "
                f"ridge={ridge} "
                f"accepted_transport={row['accepted_transport']} "
                f"max|b0|T={row['max_abs_beta0_transport']:.4g} "
                f"max_sigmaT={row['max_sigma_transport']:.4g} "
                f"max_sT={row['max_s_transport']:.4g}",
                flush=True,
            )

    print()
    print_summary(rows)
    print(f"\nsummary csv: {args.summary_csv}")


if __name__ == "__main__":
    main()
