"""Step 25: paired finite-time diagnostics for chaotic transverse dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from goldstone_lyapunov.lyapunov import compute_lyapunov_spectrum, integrate_trajectory
from goldstone_lyapunov.metrics import count_near_zero, kaplan_yorke_dimension, ks_entropy
from goldstone_lyapunov.models import ProductSymmetryChaos, RandomRateRNN, S1Attractor, T2Attractor
from goldstone_lyapunov.plotting import COLORS, clean_axis, ensure_dir, save_dict_rows_csv, save_figure, save_json, save_summary
from goldstone_lyapunov.symmetry import s1_tangent, t2_tangents


def _embedded_s1(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[:2] = s1_tangent(x[:2])
    return out


def _embedded_t2(x: np.ndarray, idx: int) -> np.ndarray:
    out = np.zeros_like(x)
    tangents = t2_tangents(x[:4])
    out[:4] = tangents[idx]
    return out


def _geometric_exponent(model, x0: np.ndarray, tangent_fn, dt: float, t_burn: float, t_total: float) -> float:
    _, burn = integrate_trajectory(model, x0, dt=dt, t_total=t_burn)
    _, states = integrate_trajectory(model, burn[-1], dt=dt, t_total=t_total)
    n0 = np.linalg.norm(tangent_fn(states[0]))
    n1 = np.linalg.norm(tangent_fn(states[-1]))
    return float((np.log(max(n1, 1e-300)) - np.log(max(n0, 1e-300))) / t_total)


def _row(model_name: str, seed: int, t_total: float, spectrum: np.ndarray, zero_tol: float) -> dict:
    return {
        "model": model_name,
        "seed": seed,
        "t_total": t_total,
        "lambda_max": float(np.max(spectrum)),
        "near_zero_count": count_near_zero(spectrum, zero_tol),
        "ks_entropy": ks_entropy(spectrum, tol=zero_tol),
        "kaplan_yorke_dimension": kaplan_yorke_dimension(spectrum, tol=zero_tol),
        "spectrum": spectrum,
    }


def run(
    output_dir: str | Path | None = None,
    quick: bool = False,
    seeds: list[int] | None = None,
    t_totals: list[float] | None = None,
    N: int | None = None,
    dt: float | None = None,
    t_burn: float | None = None,
    zero_tol: float | None = None,
) -> dict:
    out = ensure_dir(output_dir or Path("results") / "exp25_finite_time_chaos_diagnostics")
    seeds = list(seeds if seeds is not None else ([1, 3, 5] if quick else [1, 3, 5, 7, 9, 11]))
    t_totals = list(t_totals if t_totals is not None else ([60.0, 120.0, 240.0] if quick else [120.0, 240.0, 480.0, 720.0]))
    N = int(N if N is not None else (20 if quick else 48))
    dt = float(dt if dt is not None else (0.04 if quick else 0.025))
    t_burn = float(t_burn if t_burn is not None else (30.0 if quick else 80.0))
    zero_tol = float(zero_tol if zero_tol is not None else (1e-2 if quick else 4e-3))
    rows = []
    tangent_rows = []

    for seed in seeds:
        rng = np.random.default_rng(2500 + seed)
        x_rnn = rng.normal(0.0, 0.1, size=N)
        for t_total in t_totals:
            rnn = RandomRateRNN(N=N, g=3.0, seed=seed)
            rnn_spec = compute_lyapunov_spectrum(rnn, x_rnn, dt=dt, t_burn=t_burn, t_total=t_total, reorthonormalize_every=0.4, seed=2600 + seed)
            rows.append(_row("RNN", seed, t_total, rnn_spec, zero_tol))

            s1 = ProductSymmetryChaos(S1Attractor(), RandomRateRNN(N=N, g=3.0, seed=seed))
            x_s1 = np.concatenate([np.array([0.85, 0.2]), x_rnn])
            s1_spec = compute_lyapunov_spectrum(s1, x_s1, dt=dt, t_burn=t_burn, t_total=t_total, reorthonormalize_every=0.4, seed=2700 + seed)
            rows.append(_row("S1+RNN", seed, t_total, s1_spec, zero_tol))
            tangent_rows.append(
                {
                    "model": "S1+RNN",
                    "seed": seed,
                    "t_total": t_total,
                    "direct_protected_count": 1,
                    "lambda_sym_1": _geometric_exponent(s1, x_s1, _embedded_s1, dt, t_burn, t_total),
                    "lambda_sym_2": None,
                }
            )

            t2 = ProductSymmetryChaos(T2Attractor(alpha=(1.0, 0.8), R=(1.0, 1.1)), RandomRateRNN(N=N, g=3.0, seed=seed))
            x_t2 = np.concatenate([np.array([0.85, 0.2, -0.35, 0.8]), x_rnn])
            t2_spec = compute_lyapunov_spectrum(t2, x_t2, dt=dt, t_burn=t_burn, t_total=t_total, reorthonormalize_every=0.4, seed=2800 + seed)
            rows.append(_row("T2+RNN", seed, t_total, t2_spec, zero_tol))
            tangent_rows.append(
                {
                    "model": "T2+RNN",
                    "seed": seed,
                    "t_total": t_total,
                    "direct_protected_count": 2,
                    "lambda_sym_1": _geometric_exponent(t2, x_t2, lambda x: _embedded_t2(x, 0), dt, t_burn, t_total),
                    "lambda_sym_2": _geometric_exponent(t2, x_t2, lambda x: _embedded_t2(x, 1), dt, t_burn, t_total),
                }
            )

    save_dict_rows_csv(out / "finite_time_chaos_rows.csv", rows)
    save_dict_rows_csv(out / "finite_time_chaos_tangents.csv", tangent_rows)

    by_seed_time = {}
    for row in rows:
        by_seed_time.setdefault((row["seed"], row["t_total"]), {})[row["model"]] = row
    delta_rows = []
    for (seed, t_total), vals in by_seed_time.items():
        if all(k in vals for k in ["RNN", "S1+RNN", "T2+RNN"]):
            for model, q in [("S1+RNN", 1), ("T2+RNN", 2)]:
                delta_rows.append(
                    {
                        "model": model,
                        "q": q,
                        "seed": seed,
                        "t_total": t_total,
                        "zero_delta": vals[model]["near_zero_count"] - vals["RNN"]["near_zero_count"],
                        "ks_delta_abs": abs(vals[model]["ks_entropy"] - vals["RNN"]["ks_entropy"]),
                        "dky_delta": vals[model]["kaplan_yorke_dimension"] - vals["RNN"]["kaplan_yorke_dimension"],
                    }
                )
    save_dict_rows_csv(out / "finite_time_chaos_deltas.csv", delta_rows)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4), constrained_layout=True)
    rng_plot = np.random.default_rng(925)
    for model, q, color in [("S1+RNN", 1, COLORS["blue"]), ("T2+RNN", 2, COLORS["orange"])]:
        means = []
        lows = []
        highs = []
        for t in t_totals:
            vals = [r["zero_delta"] for r in delta_rows if r["model"] == model and r["t_total"] == t]
            jitter = rng_plot.uniform(-0.035, 0.035, size=len(vals)) * max(t_totals)
            axes[0, 0].scatter(np.asarray([t] * len(vals)) + jitter, vals, color=color, alpha=0.35, s=18)
            means.append(float(np.mean(vals)))
            lows.append(float(np.percentile(vals, 10)))
            highs.append(float(np.percentile(vals, 90)))
        axes[0, 0].plot(t_totals, means, marker="D", color=color, label=f"{model}, expected {q}")
        axes[0, 0].fill_between(t_totals, lows, highs, color=color, alpha=0.12, linewidth=0)
        axes[0, 0].axhline(q, color=color, linestyle="--", linewidth=0.9)
    axes[0, 0].set_xlabel("Lyapunov integration time")
    axes[0, 0].set_ylabel("finite-time zero-count delta")
    axes[0, 0].set_title("QR zero counts are diagnostics")
    axes[0, 0].legend(frameon=False)

    for model, q, color in [("S1+RNN", 1, COLORS["blue"]), ("T2+RNN", 2, COLORS["orange"])]:
        means = []
        lows = []
        highs = []
        for t in t_totals:
            vals = [r["dky_delta"] for r in delta_rows if r["model"] == model and r["t_total"] == t]
            jitter = rng_plot.uniform(-0.035, 0.035, size=len(vals)) * max(t_totals)
            axes[0, 1].scatter(np.asarray([t] * len(vals)) + jitter, vals, color=color, alpha=0.35, s=18)
            means.append(float(np.mean(vals)))
            lows.append(float(np.percentile(vals, 10)))
            highs.append(float(np.percentile(vals, 90)))
        axes[0, 1].plot(t_totals, means, marker="D", color=color, label=model)
        axes[0, 1].fill_between(t_totals, lows, highs, color=color, alpha=0.12, linewidth=0)
        axes[0, 1].axhline(q, color=color, linestyle="--", linewidth=0.9)
    axes[0, 1].set_xlabel("Lyapunov integration time")
    axes[0, 1].set_ylabel("D_KY delta")
    axes[0, 1].set_title("Dimension diagnostic approaches q")

    for model, color in [("S1+RNN", COLORS["blue"]), ("T2+RNN", COLORS["orange"])]:
        means = []
        lows = []
        highs = []
        for t in t_totals:
            vals = [r["ks_delta_abs"] for r in delta_rows if r["model"] == model and r["t_total"] == t]
            jitter = rng_plot.uniform(-0.035, 0.035, size=len(vals)) * max(t_totals)
            axes[1, 0].scatter(np.asarray([t] * len(vals)) + jitter, vals, color=color, alpha=0.35, s=18)
            means.append(float(np.mean(vals)))
            lows.append(float(np.percentile(vals, 10)))
            highs.append(float(np.percentile(vals, 90)))
        axes[1, 0].plot(t_totals, means, marker="D", color=color, label=model)
        axes[1, 0].fill_between(t_totals, lows, highs, color=color, alpha=0.12, linewidth=0)
    axes[1, 0].set_xlabel("Lyapunov integration time")
    axes[1, 0].set_ylabel("|KS entropy delta|")
    axes[1, 0].set_title("Entropy remains transverse")
    axes[1, 0].set_yscale("symlog", linthresh=1e-8)

    for model, color in [("S1+RNN", COLORS["blue"]), ("T2+RNN", COLORS["orange"])]:
        means = []
        for t in t_totals:
            vals = []
            for row in tangent_rows:
                if row["model"] != model or row["t_total"] != t:
                    continue
                vals.append(abs(row["lambda_sym_1"]))
                if row["lambda_sym_2"] is not None:
                    vals.append(abs(row["lambda_sym_2"]))
            plot_vals = [max(v, 1e-16) for v in vals]
            jitter = rng_plot.uniform(-0.035, 0.035, size=len(vals)) * max(t_totals)
            axes[1, 1].scatter(np.asarray([t] * len(plot_vals)) + jitter, plot_vals, color=color, alpha=0.45, s=18)
            means.append(float(max(np.mean(vals), 1e-16)) if vals else float("nan"))
        axes[1, 1].plot(t_totals, means, marker="D", color=color, label=model)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(5e-17, 1e-8)
    axes[1, 1].set_xlabel("Lyapunov integration time")
    axes[1, 1].set_ylabel("|direct group-tangent exponent|")
    axes[1, 1].set_title("Primary protected-mode evidence")
    axes[1, 1].legend(frameon=False)
    for ax in axes.ravel():
        clean_axis(ax)
    save_figure(fig, out / "finite_time_chaos_diagnostics.png")

    direct_max = max(
        max(abs(r["lambda_sym_1"]), abs(r["lambda_sym_2"]) if r["lambda_sym_2"] is not None else 0.0)
        for r in tangent_rows
    )
    longest = max(t_totals)
    summary_deltas = [r for r in delta_rows if r["t_total"] == longest]
    metrics = {
        "quick": quick,
        "N": N,
        "seeds": seeds,
        "t_totals": t_totals,
        "n_spectrum_rows": len(rows),
        "n_delta_rows": len(delta_rows),
        "n_tangent_rows": len(tangent_rows),
        "zero_tol": zero_tol,
        "max_abs_direct_group_tangent_exponent": float(direct_max),
        "mean_longest_s1_zero_delta": float(np.mean([r["zero_delta"] for r in summary_deltas if r["model"] == "S1+RNN"])),
        "mean_longest_t2_zero_delta": float(np.mean([r["zero_delta"] for r in summary_deltas if r["model"] == "T2+RNN"])),
        "mean_longest_abs_ks_delta": float(np.mean([r["ks_delta_abs"] for r in summary_deltas])),
        "rows": rows,
        "tangent_rows": tangent_rows,
        "delta_rows": delta_rows,
        "passed": bool(direct_max < zero_tol),
    }
    save_json(out / "metrics.json", metrics)
    save_summary(
        out / "summary.md",
        "Experiment 25: Finite-Time Chaos Diagnostics",
        [
            f"- Swept {len(seeds)} paired seeds and {len(t_totals)} integration times.",
            f"- Max absolute direct group-tangent exponent: {direct_max:.3e}.",
            f"- Mean longest-time absolute KS entropy delta: {metrics['mean_longest_abs_ks_delta']:.3e}.",
            "- Interpretation: finite-time QR zero counts are convergence diagnostics; direct group tangents are primary evidence.",
        ],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--t-totals", type=float, nargs="*", default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-burn", type=float, default=None)
    parser.add_argument("--zero-tol", type=float, default=None)
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        quick=args.quick,
        seeds=args.seeds,
        t_totals=args.t_totals,
        N=args.N,
        dt=args.dt,
        t_burn=args.t_burn,
        zero_tol=args.zero_tol,
    )


if __name__ == "__main__":
    main()
