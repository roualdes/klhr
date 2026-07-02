import argparse

import h5py
import numpy as np


def finite_summary(name, x):
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"{name:20s} n=0")
        return

    qs = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    sd = np.std(x, ddof=1) if x.size > 1 else 0.0
    print(
        f"{name:20s} n={x.size:7d}"
        f" mean={np.mean(x): .6g}"
        f" sd={sd: .6g}"
        f" min={np.min(x): .6g}"
        f" q05={qs[0]: .6g}"
        f" q25={qs[1]: .6g}"
        f" median={qs[2]: .6g}"
        f" q75={qs[3]: .6g}"
        f" q95={qs[4]: .6g}"
        f" max={np.max(x): .6g}"
    )


def current_before(draws):
    out = np.full_like(draws, np.nan)
    if draws.shape[0] > 1:
        out[1:, :] = draws[:-1, :]
    return out


def safe_ratio(num, den, eps):
    out = np.full_like(num, np.nan, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > eps)
    out[ok] = num[ok] / den[ok]
    return out


def print_largest_abs(name, x, k, beta_d0=None, beta_d1=None, beta_slope=None):
    x = np.asarray(x).reshape(-1)
    finite = np.flatnonzero(np.isfinite(x))
    if finite.size == 0:
        return

    idx = finite[np.argsort(np.abs(x[finite]))[::-1][:k]]
    print(f"\nLargest |{name}|")
    if beta_d0 is None:
        print(f"{'iter':>8s} {name:>18s}")
        for i in idx:
            print(f"{i:8d} {x[i]:18.6g}")
    else:
        print(
            f"{'iter':>8s}"
            f" {name:>18s}"
            f" {'d_beta0':>13s}"
            f" {'d_beta1':>13s}"
            f" {'slope':>13s}"
        )
        for i in idx:
            print(
                f"{i:8d}"
                f" {x[i]:18.6g}"
                f" {beta_d0[i]:13.6g}"
                f" {beta_d1[i]:13.6g}"
                f" {beta_slope[i]:13.6g}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="draws/experiments.h5")
    parser.add_argument("--model", default="earnings")
    parser.add_argument("--beta0-col", type=int, default=0)
    parser.add_argument("--beta1-col", type=int, default=1)
    parser.add_argument("--slope-eps", type=float, default=1e-8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    with h5py.File(args.file, "r") as f:
        root = f[args.model]
        required = ["sas_m", "sas_sampled_xi", "sas_accepted_xi", "sas_accepted"]
        missing = [name for name in required if name not in root]
        if missing:
            names = ", ".join(missing)
            raise RuntimeError(f"Missing SAS diagnostic dataset(s): {names}")

        m = np.asarray(root["sas_m"]).reshape(-1)
        sampled_xi = np.asarray(root["sas_sampled_xi"]).reshape(-1)
        accepted_xi = np.asarray(root["sas_accepted_xi"]).reshape(-1)
        accepted = np.asarray(root["sas_accepted"]).reshape(-1).astype(bool)
        if "draws" in root and "proposal_draws" in root:
            draws = np.asarray(root["draws"])
            proposed = np.asarray(root["proposal_draws"])
        else:
            draws = None
            proposed = None

    beta_d0 = None
    beta_d1 = None
    beta_slope = None
    if draws is not None and proposed is not None:
        current = current_before(draws)
        beta_d0 = proposed[:, args.beta0_col] - current[:, args.beta0_col]
        beta_d1 = proposed[:, args.beta1_col] - current[:, args.beta1_col]
        beta_slope = safe_ratio(beta_d0, beta_d1, args.slope_eps)

    print(f"file: {args.file}")
    print(f"model: {args.model}")
    print(f"draws: {sampled_xi.size}")
    print(f"sas accepted: {np.sum(accepted)} / {accepted.size}"
          f" ({np.mean(accepted):.4f})")
    print()

    finite_summary("sas m", m)
    finite_summary("sampled xi", sampled_xi)
    finite_summary("accepted xi", accepted_xi)
    finite_summary("rejected xi", sampled_xi[~accepted])
    if beta_d0 is not None:
        finite_summary("beta d0", beta_d0)
        finite_summary("beta d1", beta_d1)
        finite_summary("beta slope", beta_slope)
        finite_summary("accepted slope", beta_slope[accepted])
        finite_summary("rejected slope", beta_slope[~accepted])

    print_largest_abs("sampled xi", sampled_xi, args.top_k,
                      beta_d0, beta_d1, beta_slope)
    print_largest_abs("accepted xi", accepted_xi, args.top_k,
                      beta_d0, beta_d1, beta_slope)


if __name__ == "__main__":
    main()
