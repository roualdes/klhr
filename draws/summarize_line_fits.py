import argparse

import h5py
import numpy as np


def finite_summary(name, values):
    values = np.asarray(values).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(f"  {name:29s} n=0")
        return

    q05, q25, median, q75, q95 = np.quantile(
        values, [0.05, 0.25, 0.5, 0.75, 0.95]
    )
    print(
        f"  {name:29s} n={values.size:7d}"
        f" mean={np.mean(values): .6g}"
        f" q05={q05: .6g}"
        f" q25={q25: .6g}"
        f" median={median: .6g}"
        f" q75={q75: .6g}"
        f" q95={q95: .6g}"
    )


def read(root, stage, name):
    key = f"line_fit_{stage}_{name}"
    if key not in root:
        raise RuntimeError(f"Missing line-fit diagnostic dataset: {key}")
    return np.asarray(root[key]).reshape(-1)


def read_optional(root, stage, name, default):
    key = f"line_fit_{stage}_{name}"
    if key not in root:
        return np.asarray(default).reshape(-1)
    return np.asarray(root[key]).reshape(-1)


def rate(name, values, mask):
    values = np.asarray(values, dtype=bool)
    count = int(np.sum(mask))
    hits = int(np.sum(values[mask])) if count else 0
    fraction = hits / count if count else np.nan
    print(f"  {name:29s} {hits:7d} / {count:7d} ({fraction:.4f})")


def summarize_stage(root, stage):
    attempted = read(root, stage, "attempted").astype(bool)
    mode_success = read(root, stage, "mode_success").astype(bool)
    hessian_usable = read(root, stage, "hessian_usable").astype(bool)
    hessian_identity = read(root, stage, "hessian_identity").astype(bool)
    kl_attempted = read_optional(
        root, stage, "kl_attempted", attempted
    ).astype(bool)
    kl_success = read(root, stage, "kl_success").astype(bool)
    saturated = read(root, stage, "scale_saturated").astype(bool)

    initial_scale = read(root, stage, "initial_scale")
    laplace_scale = read(root, stage, "laplace_scale")
    final_scale = read(root, stage, "final_scale")
    laplace_log_correction = read(
        root, stage, "laplace_log_scale_correction"
    )
    laplace_ratio = read(root, stage, "laplace_scale_ratio")
    log_correction = read(root, stage, "log_scale_correction")
    ratio = read(root, stage, "scale_ratio")
    bound_fraction = read(root, stage, "scale_bound_fraction")
    transform_derivative = read(root, stage, "scale_transform_derivative")
    inverse_hessian = read(root, stage, "inverse_hessian")
    location_correction = read(root, stage, "location_correction")
    final_skew = read(root, stage, "final_skew")
    objective_improvement = read(root, stage, "kl_objective_improvement")
    initial_gradient = read(root, stage, "kl_initial_gradient_norm")
    final_gradient = read(root, stage, "kl_final_gradient_norm")
    mode_nfev = read(root, stage, "mode_nfev")
    kl_nfev = read(root, stage, "kl_nfev")

    print(f"\n{stage.capitalize()} fit")
    print(f"  {'attempted':29s} {np.sum(attempted):7d} / {attempted.size:7d}")
    rate("mode converged", mode_success, attempted)
    rate("inverse Hessian usable", hessian_usable, attempted)
    rate("inverse Hessian identity", hessian_identity, attempted)
    rate("KL optimizer attempted", kl_attempted, attempted)
    rate("KL optimizer converged", kl_success, attempted & kl_attempted)
    rate("scale >= 90% of bound", saturated, attempted)

    print("\n  Scale and Laplace diagnostics")
    finite_summary("inverse Hessian", inverse_hessian[attempted])
    finite_summary("pure Laplace scale", laplace_scale[attempted])
    finite_summary("initial scale", initial_scale[attempted])
    finite_summary("final scale", final_scale[attempted])
    finite_summary(
        "Laplace log correction", laplace_log_correction[attempted]
    )
    finite_summary("Laplace scale ratio", laplace_ratio[attempted])
    finite_summary("log-scale correction", log_correction[attempted])
    finite_summary("scale ratio", ratio[attempted])
    finite_summary("scale bound fraction", bound_fraction[attempted])
    finite_summary("transform derivative", transform_derivative[attempted])
    finite_summary("location correction", location_correction[attempted])
    if np.any(np.isfinite(final_skew[attempted])):
        finite_summary("final skew", final_skew[attempted])

    print("\n  KL optimization diagnostics")
    finite_summary("objective improvement", objective_improvement[kl_attempted])
    finite_summary("initial gradient norm", initial_gradient[kl_attempted])
    finite_summary("final gradient norm", final_gradient[kl_attempted])
    finite_summary("mode gradient calls", mode_nfev[attempted])
    finite_summary("KL gradient calls", kl_nfev[kl_attempted])
    print(
        f"  {'total KL gradient calls':29s}"
        f" {int(np.nansum(kl_nfev[kl_attempted])):7d}"
    )

    correction_mask = (
        attempted
        & hessian_usable
        & np.isfinite(laplace_log_correction)
    )
    corrections = laplace_log_correction[correction_mask]
    if corrections.size:
        median_correction = np.median(corrections)
        residual = corrections - median_correction
        print("\n  Frozen median log-scale correction check")
        print(f"  median correction             {median_correction: .6g}")
        print(f"  multiplicative correction     {np.exp(median_correction): .6g}")
        finite_summary("residual log correction", residual)
        for factor in (1.1, 1.25, 2.0):
            covered = np.mean(np.abs(residual) <= np.log(factor))
            print(f"  residual within factor {factor:<4g} {covered:.4f}")

    improvement = objective_improvement[
        kl_attempted & np.isfinite(objective_improvement)
    ]
    if improvement.size:
        print("\n  Small KL-improvement frequencies")
        for threshold in (1e-3, 1e-2, 1e-1):
            fraction = np.mean(improvement <= threshold)
            print(f"  improvement <= {threshold:<7g} {fraction:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="draws/experiments.h5")
    parser.add_argument("--model", default="earnings")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as h5:
        if args.model not in h5:
            available = ", ".join(h5.keys())
            raise RuntimeError(
                f"Model {args.model!r} not found; available groups: {available}"
            )
        root = h5[args.model]
        print(f"file: {args.file}")
        print(f"model: {args.model}")
        summarize_stage(root, "forward")
        summarize_stage(root, "reverse")


if __name__ == "__main__":
    main()
