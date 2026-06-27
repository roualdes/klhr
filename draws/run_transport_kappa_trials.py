import argparse
import csv
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np


DEFAULT_KAPPAS = (10.0, 12.0, 15.0, 20.0)


def first_ridge_iteration(draws, beta0_threshold, beta1_threshold):
    idx = np.flatnonzero(
        (draws[:, 0] < beta0_threshold) & (draws[:, 1] > beta1_threshold)
    )
    if idx.size == 0:
        return None
    return int(idx[0])


def finite_attempts(attempts):
    attempts = np.asarray(attempts, dtype=float)
    return attempts[(attempts > 0) & np.isfinite(attempts) & (attempts < 10_000)]


def summarize_h5(path, warmup, beta0_threshold, beta1_threshold):
    with h5py.File(path, "r") as h5:
        root = h5["earnings"]
        draws = np.asarray(root["draws"])
        log_density = np.asarray(root["log_density"]).reshape(-1)
        phase = np.asarray(root["diagnostics/phase"]).reshape(-1).astype(int)
        move_norm = np.asarray(root["diagnostics/move_norm"]).reshape(-1)
        diag_jump = np.asarray(root["diagnostics/diag_jump"]).reshape(-1)
        direction_attempts = np.asarray(
            root["diagnostics/transport_direction_attempts"]
        ).reshape(-1)
        stop_transport_idx = int(np.asarray(root["stop_transport_idx"]))
        nfev = np.asarray(root["nfev"]).reshape(-1)

    transport = phase == 0
    transport_draws = draws[transport]
    transport_attempts = direction_attempts[transport]
    successful_attempts = finite_attempts(transport_attempts)
    posterior_draws = draws[min(warmup, draws.shape[0]) :]

    if transport_draws.size:
        max_abs_beta0_transport = float(np.nanmax(np.abs(transport_draws[:, 0])))
        max_abs_beta1_transport = float(np.nanmax(np.abs(transport_draws[:, 1])))
        max_sigma_transport = float(np.nanmax(transport_draws[:, 2]))
        max_s_transport = float(np.nanmax(transport_draws[:, 3]))
        accepted_transport = int(np.sum(move_norm[transport] > 1e-12))
        max_transport_jump = float(np.nanmax(diag_jump[transport]))
    else:
        max_abs_beta0_transport = np.nan
        max_abs_beta1_transport = np.nan
        max_sigma_transport = np.nan
        max_s_transport = np.nan
        accepted_transport = 0
        max_transport_jump = np.nan

    means = np.mean(posterior_draws, axis=0)
    ridge = first_ridge_iteration(draws, beta0_threshold, beta1_threshold)

    if successful_attempts.size:
        rho_attempt_mean = float(np.mean(successful_attempts))
        rho_attempt_median = float(np.median(successful_attempts))
        rho_attempt_q90 = float(np.quantile(successful_attempts, 0.9))
        rho_attempt_max = float(np.max(successful_attempts))
    else:
        rho_attempt_mean = np.nan
        rho_attempt_median = np.nan
        rho_attempt_q90 = np.nan
        rho_attempt_max = np.nan

    return {
        "ridge_iteration": "" if ridge is None else ridge,
        "ridge_found": ridge is not None,
        "stop_transport_idx": stop_transport_idx,
        "accepted_transport": accepted_transport,
        "rho_attempt_count": int(successful_attempts.size),
        "rho_fallback_count": int(np.sum(transport_attempts >= 10_000)),
        "rho_attempt_mean": rho_attempt_mean,
        "rho_attempt_median": rho_attempt_median,
        "rho_attempt_q90": rho_attempt_q90,
        "rho_attempt_max": rho_attempt_max,
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


def format_kappa(kappa):
    return f"{kappa:g}".replace(".", "p")


def run_one(args, kappa, replicate):
    seed = args.seed_base + replicate
    output = args.output_dir / f"kappa_{format_kappa(kappa)}_{replicate:02d}.h5"
    command = [
        str(args.executable),
        "--seed",
        str(seed),
        "--sampler",
        "sas",
        "--transport-direction",
        "kappa",
        "--transport-proposal",
        "random",
        "--transport-kappa",
        str(kappa),
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
            "kappa": kappa,
            "sampler": "sas",
            "transport_direction": "kappa",
            "transport_proposal": "random",
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
        "kappa",
        "sampler",
        "transport_direction",
        "transport_proposal",
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
        "rho_attempt_count",
        "rho_fallback_count",
        "rho_attempt_mean",
        "rho_attempt_median",
        "rho_attempt_q90",
        "rho_attempt_max",
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


def read_transport_attempts(path):
    with h5py.File(path, "r") as h5:
        root = h5["earnings"]
        phase = np.asarray(root["diagnostics/phase"]).reshape(-1).astype(int)
        attempts = np.asarray(
            root["diagnostics/transport_direction_attempts"]
        ).reshape(-1)
    return finite_attempts(attempts[phase == 0])


def print_summary(rows):
    print(
        "kappa  n found  runtime_mean  ridge_median  ridge_mean  "
        "ridge_cens_med  rho_mean  rho_median  rho_q90  fallback  "
        "accT_med  mean_beta0  mean_beta1"
    )
    for kappa in sorted({float(row["kappa"]) for row in rows}):
        subset = [row for row in rows if float(row["kappa"]) == kappa]
        ridges = np.array(
            [float(row["ridge_iteration"]) for row in subset if row["ridge_found"]],
            dtype=float,
        )
        censored_ridges = np.array(
            [
                float(row["ridge_iteration"]) if row["ridge_found"] else 30_000.0
                for row in subset
            ],
            dtype=float,
        )
        attempts = np.concatenate(
            [read_transport_attempts(row["output"]) for row in subset]
        )
        runtime = np.array([row["runtime_seconds"] for row in subset], dtype=float)
        acc_t = np.array([row["accepted_transport"] for row in subset], dtype=float)
        mean_beta0 = np.array([row["mean_beta0"] for row in subset], dtype=float)
        mean_beta1 = np.array([row["mean_beta1"] for row in subset], dtype=float)
        fallback = np.sum([row["rho_fallback_count"] for row in subset])

        ridge_median = np.median(ridges) if ridges.size else np.nan
        ridge_mean = np.mean(ridges) if ridges.size else np.nan
        print(
            f"{kappa:5g} "
            f"{len(subset):2d} {ridges.size:5d} "
            f"{np.mean(runtime):12.3f} "
            f"{ridge_median:12.1f} "
            f"{ridge_mean:10.1f} "
            f"{np.median(censored_ridges):14.1f} "
            f"{np.mean(attempts):9.3f} "
            f"{np.median(attempts):11.1f} "
            f"{np.quantile(attempts, 0.9):7.1f} "
            f"{int(fallback):8d} "
            f"{np.median(acc_t):8.1f} "
            f"{np.mean(mean_beta0):11.4g} "
            f"{np.mean(mean_beta1):11.4g}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2026063000)
    parser.add_argument("--kappas", type=float, nargs="+", default=DEFAULT_KAPPAS)
    parser.add_argument("--n", type=int, default=30_000)
    parser.add_argument("--warmup", type=int, default=15_000)
    parser.add_argument("--transport-steps", type=int, default=100)
    parser.add_argument("--gradient-history", type=int, default=3)
    parser.add_argument("--projection-probability", type=float, default=0.5)
    parser.add_argument("--beta0-threshold", type=float, default=-10_000.0)
    parser.add_argument("--beta1-threshold", type=float, default=500.0)
    parser.add_argument("--executable", type=Path, default=Path("build/examples/example"))
    parser.add_argument("--cwd", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("draws/transport_kappa_trials"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("draws/transport_kappa_trials/summary.csv"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = args.replicates * len(args.kappas)
    completed = 0
    for replicate in range(args.replicates):
        for kappa in args.kappas:
            completed += 1
            print(f"[{completed:02d}/{total}] kappa={kappa:g} replicate {replicate}")
            row = run_one(args, kappa, replicate)
            rows.append(row)
            write_rows(args.summary_csv, rows)
            ridge = row["ridge_iteration"] if row["ridge_found"] else "inf"
            print(
                f"  runtime={row['runtime_seconds']:.3f}s "
                f"ridge={ridge} "
                f"rho_mean={row['rho_attempt_mean']:.3f} "
                f"rho_median={row['rho_attempt_median']:.1f} "
                f"accepted_transport={row['accepted_transport']} "
                f"max|b0|T={row['max_abs_beta0_transport']:.4g}",
                flush=True,
            )

    print()
    print_summary(rows)
    print(f"\nsummary csv: {args.summary_csv}")


if __name__ == "__main__":
    main()
