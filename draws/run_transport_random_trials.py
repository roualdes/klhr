import argparse
import csv
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np


VARIANTS = (
    ("sas_random", "sas"),
    ("normal_random", "normal"),
)


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
        phase = np.asarray(root["diagnostics/phase"]).reshape(-1).astype(int)
        selected = np.asarray(root["diagnostics/selected_candidate"]).reshape(-1)
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
        accepted_transport = int(np.sum(selected[transport] == 1))
    else:
        max_abs_beta0_transport = np.nan
        max_abs_beta1_transport = np.nan
        max_sigma_transport = np.nan
        max_s_transport = np.nan
        accepted_transport = 0

    means = np.mean(posterior_draws, axis=0)
    ridge = first_ridge_iteration(draws, beta0_threshold, beta1_threshold)

    return {
        "ridge_iteration": "" if ridge is None else ridge,
        "ridge_found": ridge is not None,
        "stop_transport_idx": stop_transport_idx,
        "accepted_transport": accepted_transport,
        "max_abs_beta0_transport": max_abs_beta0_transport,
        "max_abs_beta1_transport": max_abs_beta1_transport,
        "max_sigma_transport": max_sigma_transport,
        "max_s_transport": max_s_transport,
        "mean_beta0": float(means[0]),
        "mean_beta1": float(means[1]),
        "mean_sigma": float(means[2]),
        "mean_s": float(means[3]),
        "nfev": int(nfev[-1]),
    }


def run_one(args, variant_name, approximation, replicate):
    seed = args.seed_base + replicate
    output = args.output_dir / f"{variant_name}_{replicate:02d}.h5"
    command = [
        str(args.executable),
        "--seed",
        str(seed),
        "--transport-approx",
        approximation,
        "--transport-proposal",
        "random",
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
            "variant": variant_name,
            "approximation": approximation,
            "proposal": "random",
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
        "variant",
        "approximation",
        "proposal",
        "replicate",
        "seed",
        "runtime_seconds",
        "ridge_iteration",
        "ridge_found",
        "stop_transport_idx",
        "accepted_transport",
        "max_abs_beta0_transport",
        "max_abs_beta1_transport",
        "max_sigma_transport",
        "max_s_transport",
        "mean_beta0",
        "mean_beta1",
        "mean_sigma",
        "mean_s",
        "nfev",
        "output",
        "stdout",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print(
        "variant          n found  runtime_mean  ridge_median  ridge_mean  "
        "max|b0|T_med  max_sigmaT_med  max_sT_med  mean_beta0  mean_beta1"
    )
    for variant_name, _ in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant_name]
        finite_ridges = np.array(
            [float(row["ridge_iteration"]) for row in subset if row["ridge_found"]],
            dtype=float,
        )
        runtime = np.array([row["runtime_seconds"] for row in subset], dtype=float)
        max_b0 = np.array([row["max_abs_beta0_transport"] for row in subset], dtype=float)
        max_sigma = np.array([row["max_sigma_transport"] for row in subset], dtype=float)
        max_s = np.array([row["max_s_transport"] for row in subset], dtype=float)
        mean_beta0 = np.array([row["mean_beta0"] for row in subset], dtype=float)
        mean_beta1 = np.array([row["mean_beta1"] for row in subset], dtype=float)

        ridge_median = np.nan if finite_ridges.size == 0 else np.median(finite_ridges)
        ridge_mean = np.nan if finite_ridges.size == 0 else np.mean(finite_ridges)
        print(
            f"{variant_name:<15} "
            f"{len(subset):2d} {finite_ridges.size:5d} "
            f"{np.mean(runtime):12.3f} "
            f"{ridge_median:12.1f} "
            f"{ridge_mean:10.1f} "
            f"{np.median(max_b0):12.4g} "
            f"{np.median(max_sigma):14.4g} "
            f"{np.median(max_s):11.4g} "
            f"{np.mean(mean_beta0):11.4g} "
            f"{np.mean(mean_beta1):11.4g}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2026062600)
    parser.add_argument("--n", type=int, default=30_000)
    parser.add_argument("--warmup", type=int, default=15_000)
    parser.add_argument("--gradient-history", type=int, default=3)
    parser.add_argument("--projection-probability", type=float, default=0.5)
    parser.add_argument("--beta0-threshold", type=float, default=-10_000.0)
    parser.add_argument("--beta1-threshold", type=float, default=500.0)
    parser.add_argument("--executable", type=Path, default=Path("build/examples/example"))
    parser.add_argument("--cwd", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("draws/transport_random_trials"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("draws/transport_random_trials/summary.csv"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = args.replicates * len(VARIANTS)
    completed = 0
    for replicate in range(args.replicates):
        for variant_name, approximation in VARIANTS:
            completed += 1
            print(f"[{completed:02d}/{total}] {variant_name} replicate {replicate}")
            row = run_one(args, variant_name, approximation, replicate)
            rows.append(row)
            write_rows(args.summary_csv, rows)
            ridge = row["ridge_iteration"] if row["ridge_found"] else "inf"
            print(
                f"  runtime={row['runtime_seconds']:.3f}s "
                f"ridge={ridge} "
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
