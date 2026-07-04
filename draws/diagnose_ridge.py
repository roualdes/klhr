import argparse

import h5py
import numpy as np


def finite_summary(name, x):
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"{name:26s} n=0")
        return

    qs = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    sd = np.std(x, ddof=1) if x.size > 1 else 0.0
    print(
        f"{name:26s} n={x.size:7d}"
        f" mean={np.mean(x): .6g}"
        f" sd={sd: .6g}"
        f" q05={qs[0]: .6g}"
        f" q25={qs[1]: .6g}"
        f" median={qs[2]: .6g}"
        f" q75={qs[3]: .6g}"
        f" q95={qs[4]: .6g}"
    )


def first_crossing(x, threshold, direction):
    x = np.asarray(x).reshape(-1)
    if direction == "lower":
        hit = np.flatnonzero(x <= threshold)
    else:
        hit = np.flatnonzero(x >= threshold)
    return None if hit.size == 0 else int(hit[0])


def crossing_mask(x, threshold, direction):
    if direction == "lower":
        return x <= threshold
    return x >= threshold


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


def print_best_proposals(idx, proposed, current, draws, line_location,
                         line_distance, line_location_label,
                         line_distance_label, log_accept,
                         logp_gain, accepted, score_col, direction,
                         beta0_col, beta1_col, slope, top_k):
    if idx.size == 0:
        return

    score = proposed[idx, score_col]
    finite = np.isfinite(score)
    idx = idx[finite]
    if idx.size == 0:
        return

    order = np.argsort(proposed[idx, score_col])
    if direction == "upper":
        order = order[::-1]
    idx = idx[order[:top_k]]

    print("\nBest pre-ridge proposed beta-plane values")
    print(
        f"{'iter':>8s}"
        f" {'cur_b0':>13s}"
        f" {'prop_b0':>13s}"
        f" {'post_b0':>13s}"
        f" {'cur_b1':>13s}"
        f" {'prop_b1':>13s}"
        f" {'post_b1':>13s}"
        f" {'d_b0':>13s}"
        f" {'d_b1':>13s}"
        f" {'slope':>13s}"
        f" {line_location_label[:13]:>13s}"
        f" {line_distance_label[:13]:>13s}"
        f" {'logp_gain':>13s}"
        f" {'log_acc':>13s}"
        f" {'acc':>5s}"
    )
    for i in idx:
        cur_b0 = current[i, beta0_col]
        prop_b0 = proposed[i, beta0_col]
        post_b0 = draws[i, beta0_col]
        cur_b1 = current[i, beta1_col]
        prop_b1 = proposed[i, beta1_col]
        post_b1 = draws[i, beta1_col]
        d_b0 = prop_b0 - cur_b0 if np.isfinite(cur_b0) and np.isfinite(prop_b0) else np.nan
        d_b1 = prop_b1 - cur_b1 if np.isfinite(cur_b1) and np.isfinite(prop_b1) else np.nan
        print(
            f"{i:8d}"
            f" {cur_b0:13.6g}"
            f" {prop_b0:13.6g}"
            f" {post_b0:13.6g}"
            f" {cur_b1:13.6g}"
            f" {prop_b1:13.6g}"
            f" {post_b1:13.6g}"
            f" {d_b0:13.6g}"
            f" {d_b1:13.6g}"
            f" {slope[i]:13.6g}"
            f" {line_location[i]:13.6g}"
            f" {line_distance[i]:13.6g}"
            f" {logp_gain[i]:13.6g}"
            f" {log_accept[i]:13.6g}"
            f" {int(accepted[i]):5d}"
        )


def summarize_window(name, mask, line_location, line_distance,
                     line_location_label, line_distance_label,
                     log_accept, logp_gain, prop_delta,
                     beta_d0, beta_d1, beta_slope, beta_norm,
                     accepted, valid, extra=None):
    print(f"\n{name}")
    print(f"  steps: {np.sum(mask)}")
    print(f"  valid proposals: {np.sum(mask & valid)}")
    if np.sum(mask) > 0:
        print(f"  accepted: {np.sum(mask & accepted)} / {np.sum(mask)}"
              f" ({np.mean(accepted[mask]):.4f})")
    finite_summary(f"  {line_location_label}", line_location[mask])
    finite_summary(f"  {line_distance_label}", line_distance[mask])
    finite_summary("  proposal logp gain", logp_gain[mask])
    finite_summary("  log accept", log_accept[mask])
    finite_summary("  proposed delta", prop_delta[mask])
    finite_summary("  beta d0", beta_d0[mask])
    finite_summary("  beta d1", beta_d1[mask])
    finite_summary("  beta slope d0/d1", beta_slope[mask])
    finite_summary("  beta step norm", beta_norm[mask])
    finite_summary("  accepted beta slope", beta_slope[mask & accepted])
    finite_summary("  rejected beta slope", beta_slope[mask & ~accepted])
    if extra:
        for extra_name, extra_values in extra.items():
            finite_summary(f"  {extra_name}", extra_values[mask])
    finite_summary("  accepted prop delta", prop_delta[mask & accepted])
    finite_summary("  rejected prop delta", prop_delta[mask & ~accepted])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="draws/experiments.h5")
    parser.add_argument("--model", default="earnings")
    parser.add_argument("--ridge-col", type=int, default=0)
    parser.add_argument("--ridge-threshold", type=float, default=-10000.0)
    parser.add_argument("--ridge-direction", choices=["lower", "upper"],
                        default="lower")
    parser.add_argument("--beta0-col", type=int, default=0)
    parser.add_argument("--beta1-col", type=int, default=1)
    parser.add_argument("--slope-eps", type=float, default=1e-8)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    with h5py.File(args.file, "r") as f:
        root = f[args.model]
        required = [
            "draws",
            "log_density",
            "proposal_draws",
            "proposal_log_accept",
            "proposal_log_density",
            "proposal_valid",
            "proposal_accepted",
        ]
        missing = [name for name in required if name not in root]
        if missing:
            names = ", ".join(missing)
            raise RuntimeError(f"Missing diagnostic dataset(s): {names}")

        has_sas = "sas_m" in root and "sas_sampled_xi" in root
        has_leapfrog = "leapfrog_tau" in root
        if not has_sas and not has_leapfrog:
            raise RuntimeError("Missing line proposal diagnostics: expected SAS or leapfrog datasets")

        draws = np.asarray(root["draws"])
        log_density = np.asarray(root["log_density"]).reshape(-1)
        proposed = np.asarray(root["proposal_draws"])
        log_accept = np.asarray(root["proposal_log_accept"]).reshape(-1)
        proposal_log_density = np.asarray(root["proposal_log_density"]).reshape(-1)
        valid = np.asarray(root["proposal_valid"]).reshape(-1).astype(bool)
        accepted = np.asarray(root["proposal_accepted"]).reshape(-1).astype(bool)
        if has_sas:
            line_location = np.asarray(root["sas_m"]).reshape(-1)
            line_distance = np.asarray(root["sas_sampled_xi"]).reshape(-1)
            line_location_label = "sas m"
            line_distance_label = "sampled xi"
            sampler_name = "sas"
            extra = {}
        else:
            line_location = np.asarray(root["leapfrog_log_tau_scale"]).reshape(-1)
            line_distance = np.asarray(root["leapfrog_tau"]).reshape(-1)
            line_location_label = "log tau scale"
            line_distance_label = "tau"
            sampler_name = "leapfrog"
            extra = {}
            if "leapfrog_log_tau_shape" in root:
                extra["log tau shape"] = np.asarray(root["leapfrog_log_tau_shape"]).reshape(-1)
            if "leapfrog_direction_grad_cos" in root:
                extra["direction grad cos"] = np.asarray(root["leapfrog_direction_grad_cos"]).reshape(-1)
            if "leapfrog_momentum_norm" in root:
                extra["leapfrog momentum norm"] = np.asarray(root["leapfrog_momentum_norm"]).reshape(-1)
        for key, label in [
            ("transport_distance", "transport distance"),
            ("transport_reflections", "transport reflections"),
            ("transport_logp_gain", "transport logp gain"),
            ("transport_uturn", "transport uturn"),
            ("transport_moved", "transport moved"),
            ("transport_direction_norm", "transport direction norm"),
            ("transport_variance", "transport variance"),
        ]:
            if key in root:
                extra[label] = np.asarray(root[key])

    current = current_before(draws)
    current_log_density = np.full_like(log_density, np.nan)
    if log_density.size > 1:
        current_log_density[1:] = log_density[:-1]
    logp_gain = proposal_log_density - current_log_density
    n = draws.shape[0]
    idx = np.arange(n)

    first_draw = first_crossing(draws[:, args.ridge_col],
                                args.ridge_threshold,
                                args.ridge_direction)
    first_proposal = first_crossing(proposed[:, args.ridge_col],
                                    args.ridge_threshold,
                                    args.ridge_direction)

    if args.ridge_direction == "lower":
        prop_delta = current[:, args.ridge_col] - proposed[:, args.ridge_col]
    else:
        prop_delta = proposed[:, args.ridge_col] - current[:, args.ridge_col]
    beta_d0 = proposed[:, args.beta0_col] - current[:, args.beta0_col]
    beta_d1 = proposed[:, args.beta1_col] - current[:, args.beta1_col]
    beta_slope = safe_ratio(beta_d0, beta_d1, args.slope_eps)
    beta_norm = np.sqrt(beta_d0 * beta_d0 + beta_d1 * beta_d1)

    pre_end = n if first_draw is None else first_draw
    pre = idx < pre_end
    near_start = max(0, pre_end - args.window)
    near = (idx >= near_start) & (idx < pre_end)
    all_mask = np.ones(n, dtype=bool)

    proposed_cross = crossing_mask(proposed[:, args.ridge_col],
                                   args.ridge_threshold,
                                   args.ridge_direction)
    draw_cross = crossing_mask(draws[:, args.ridge_col],
                               args.ridge_threshold,
                               args.ridge_direction)
    pre_cross = pre & proposed_cross

    print(f"file: {args.file}")
    print(f"model: {args.model}")
    print(f"sampler diagnostics: {sampler_name}")
    print(f"draws: {n}")
    print(f"ridge column: {args.ridge_col}")
    print(f"ridge threshold: {args.ridge_threshold:g} ({args.ridge_direction})")
    print(f"beta-plane columns: beta0={args.beta0_col}, beta1={args.beta1_col}")
    print(f"first accepted ridge draw: {first_draw if first_draw is not None else 'inf'}")
    print(f"first proposed ridge crossing: "
          f"{first_proposal if first_proposal is not None else 'inf'}")
    if first_draw is not None and first_proposal is not None:
        print(f"proposal lead time: {first_draw - first_proposal}")
    print()

    print("Pre-Ridge Crossing Check")
    print(f"  pre-ridge proposed crossings: {np.sum(pre_cross)}")
    print(f"  pre-ridge accepted proposed crossings: {np.sum(pre_cross & accepted)}")
    print(f"  pre-ridge rejected proposed crossings: {np.sum(pre_cross & ~accepted)}")
    print(f"  accepted ridge draws: {np.sum(draw_cross)}")

    summarize_window("All Steps", all_mask, line_location, line_distance,
                     line_location_label, line_distance_label,
                     log_accept, logp_gain,
                     prop_delta, beta_d0, beta_d1, beta_slope, beta_norm,
                     accepted, valid, extra)
    summarize_window("Pre-Ridge Steps", pre, line_location, line_distance,
                     line_location_label, line_distance_label,
                     log_accept, logp_gain,
                     prop_delta, beta_d0, beta_d1, beta_slope, beta_norm,
                     accepted, valid, extra)
    summarize_window(f"Last {args.window} Pre-Ridge Steps", near,
                     line_location, line_distance,
                     line_location_label, line_distance_label,
                     log_accept, logp_gain, prop_delta,
                     beta_d0, beta_d1, beta_slope, beta_norm,
                     accepted, valid, extra)

    print_best_proposals(
        np.flatnonzero(pre & valid),
        proposed,
        current,
        draws,
        line_location,
        line_distance,
        line_location_label,
        line_distance_label,
        log_accept,
        logp_gain,
        accepted,
        args.ridge_col,
        args.ridge_direction,
        args.beta0_col,
        args.beta1_col,
        beta_slope,
        args.top_k,
    )


if __name__ == "__main__":
    main()
