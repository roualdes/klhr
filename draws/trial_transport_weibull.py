#!/usr/bin/env python3

import argparse
import csv
import io
import json
import math
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SETTINGS = ((8, 4), (8, 8), (16, 4), (16, 8))
RIDGE_THRESHOLD = -10_000.0
REFERENCE_BETA = (-54_413.10, 1_158.90)


def run_one(executable, root, seed, nodes, maxiter, steps):
    start = time.perf_counter()
    result = subprocess.run(
        [executable, str(seed), str(nodes), str(maxiter), str(steps)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    rows = []
    for row in csv.DictReader(io.StringIO(result.stdout)):
        rows.append({
            "iteration": int(row["iteration"]),
            "beta0": float(row["beta0"]),
            "beta1": float(row["beta1"]),
            "sigma": float(row["sigma"]),
            "s": float(row["s"]),
            "nfev": int(row["nfev"]),
            "logp": float(row["logp"]),
        })
    if len(rows) != steps:
        raise RuntimeError(
            f"seed {seed}, N={nodes}, maxiter={maxiter}: "
            f"expected {steps} rows, got {len(rows)}"
        )
    return seed, nodes, maxiter, elapsed, rows


def summarize_run(seed, nodes, maxiter, elapsed, rows, mean_height, mean_earn):
    ridge_rows = [row for row in rows if row["beta0"] <= RIDGE_THRESHOLD]
    first_ridge = ridge_rows[0]["iteration"] if ridge_rows else None
    post_ridge = rows[first_ridge - 1:] if first_ridge is not None else []
    normalizer = math.hypot(mean_height, 1.0)

    def perpendicular(row):
        return abs(row["beta0"] + mean_height * row["beta1"] - mean_earn) / normalizer

    def along_from_reference(row):
        db0 = row["beta0"] - REFERENCE_BETA[0]
        db1 = row["beta1"] - REFERENCE_BETA[1]
        return abs((-mean_height * db0 + db1) / normalizer)

    def values(name, selected=rows):
        return [row[name] for row in selected]

    summary = {
        "seed": seed,
        "nodes": nodes,
        "maxiter": maxiter,
        "runtime_seconds": elapsed,
        "ridge_discovered": first_ridge is not None,
        "first_ridge_iteration": first_ridge if first_ridge is not None else len(rows) + 1,
        "nfev": rows[-1]["nfev"],
        "endpoint_beta0": rows[-1]["beta0"],
        "endpoint_beta1": rows[-1]["beta1"],
        "endpoint_sigma": rows[-1]["sigma"],
        "endpoint_s": rows[-1]["s"],
        "min_beta0": min(values("beta0")),
        "max_beta0": max(values("beta0")),
        "min_beta1": min(values("beta1")),
        "max_beta1": max(values("beta1")),
        "max_sigma": max(values("sigma")),
        "max_s": max(values("s")),
        "max_perpendicular_distance": max(map(perpendicular, rows)),
        "endpoint_perpendicular_distance": perpendicular(rows[-1]),
    }
    if post_ridge:
        summary.update({
            "post_ridge_min_beta0": min(values("beta0", post_ridge)),
            "post_ridge_max_abs_along_distance": max(map(along_from_reference, post_ridge)),
            "post_ridge_max_perpendicular_distance": max(map(perpendicular, post_ridge)),
            "post_ridge_max_sigma": max(values("sigma", post_ridge)),
            "post_ridge_max_s": max(values("s", post_ridge)),
        })
    else:
        summary.update({
            "post_ridge_min_beta0": math.nan,
            "post_ridge_max_abs_along_distance": math.nan,
            "post_ridge_max_perpendicular_distance": math.nan,
            "post_ridge_max_sigma": math.nan,
            "post_ridge_max_s": math.nan,
        })
    return summary


def finite(values):
    return [value for value in values if math.isfinite(value)]


def mean(values):
    values = finite(values)
    return statistics.fmean(values) if values else math.nan


def median(values):
    values = finite(values)
    return statistics.median(values) if values else math.nan


def percentile(values, probability):
    values = sorted(finite(values))
    if not values:
        return math.nan
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def print_summary(summaries, steps):
    print("\nWeibull reflected-transport fit trial")
    print(f"runs per setting: {len(summaries) // len(SETTINGS)}; transport steps: {steps}")
    print("ridge: beta0 <= -10000; failures are censored at steps + 1")
    print()
    header = (
        " N  iter  ridge  ridge iteration       nfev       sec  "
        "post-ridge |along|  post-ridge perp  post-ridge min b0  max sigma  max s"
    )
    print(header)
    for nodes, maxiter in SETTINGS:
        group = [row for row in summaries if row["nodes"] == nodes and row["maxiter"] == maxiter]
        successful = [row for row in group if row["ridge_discovered"]]
        ridge_iterations = [row["first_ridge_iteration"] for row in group]
        print(
            f"{nodes:2d} {maxiter:5d}  {len(successful):2d}/{len(group):<2d}  "
            f"{mean(ridge_iterations):6.1f}/{median(ridge_iterations):5.1f}  "
            f"{mean([r['nfev'] for r in group]):10.0f}  "
            f"{mean([r['runtime_seconds'] for r in group]):8.2f}  "
            f"{median([r['post_ridge_max_abs_along_distance'] for r in group]):18.1f}  "
            f"{median([r['post_ridge_max_perpendicular_distance'] for r in group]):15.1f}  "
            f"{median([r['post_ridge_min_beta0'] for r in group]):17.1f}  "
            f"{median([r['post_ridge_max_sigma'] for r in group]):9.1f}  "
            f"{median([r['post_ridge_max_s'] for r in group]):.1f}"
        )
        print(
            " " * 22
            + f"ridge q25/q75={percentile(ridge_iterations, 0.25):.1f}/{percentile(ridge_iterations, 0.75):.1f}; "
            + f"post-ridge min beta0 q05={percentile([r['post_ridge_min_beta0'] for r in group], 0.05):.1f}; "
            + f"max sigma q95={percentile([r['post_ridge_max_sigma'] for r in group], 0.95):.1f}; "
            + f"max s q95={percentile([r['post_ridge_max_s'] for r in group], 0.95):.1f}"
        )
        print(
            " " * 22
            + f"all-transport max sigma median/q95={median([r['max_sigma'] for r in group]):.1f}/"
            + f"{percentile([r['max_sigma'] for r in group], 0.95):.1f}; "
            + f"max s median/q95={median([r['max_s'] for r in group]):.1f}/"
            + f"{percentile([r['max_s'] for r in group], 0.95):.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--executable", default="./build/examples/transport_fit_trial"
    )
    parser.add_argument(
        "--output", default="draws/transport_weibull_fit_trial.csv"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with (root / "stan/earnings.json").open() as stream:
        data = json.load(stream)
    mean_height = statistics.fmean(data["height"])
    mean_earn = statistics.fmean(data["earn"])

    jobs = [
        (str(root / args.executable), str(root), seed, nodes, maxiter, args.steps)
        for seed in range(1, args.runs + 1)
        for nodes, maxiter in SETTINGS
    ]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, *job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            seed, nodes, maxiter, elapsed, _ = result
            print(
                f"[{completed:3d}/{len(futures)}] seed={seed:2d} "
                f"N={nodes:2d} maxiter={maxiter} {elapsed:.2f}s",
                flush=True,
            )

    summaries = [
        summarize_run(*result, mean_height, mean_earn) for result in results
    ]
    summaries.sort(key=lambda row: (row["nodes"], row["maxiter"], row["seed"]))
    output = root / args.output
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print_summary(summaries, args.steps)
    print(f"\nraw per-run summaries: {output}")


if __name__ == "__main__":
    main()
