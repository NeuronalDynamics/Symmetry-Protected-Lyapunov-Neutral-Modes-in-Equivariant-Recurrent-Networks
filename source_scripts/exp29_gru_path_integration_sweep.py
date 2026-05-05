"""Step 29: systematic GRU path-integration baseline sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from goldstone_lyapunov.experiments.exp10_trained_path_integrators import _eval_gru, _train_gru_baseline
from goldstone_lyapunov.models import PhaseIntegrator
from goldstone_lyapunov.plotting import COLORS, clean_axis, ensure_dir, save_dict_rows_csv, save_figure, save_json, save_summary
from goldstone_lyapunov.tasks import circular_rmse, generate_piecewise_velocity, simulate_phase_integrator


def _panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold", color=COLORS["dark"])


def _mean_sem(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _mean_row(rows: list[dict], **filters) -> float:
    vals = []
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            vals.append(row["rmse"])
    return float(np.mean(vals)) if vals else float("nan")


def _select_best_gru(rows: list[dict], train_horizons: list[float], hidden_sizes: list[int], speed_scale: float, horizon: float) -> tuple[float, int]:
    best = (float("inf"), train_horizons[0], hidden_sizes[0])
    for train_horizon in train_horizons:
        for hidden in hidden_sizes:
            val = _mean_row(rows, model="GRU", train_horizon=train_horizon, hidden_size=hidden, speed_scale=speed_scale, test_horizon=horizon)
            if np.isfinite(val) and val < best[0]:
                best = (val, train_horizon, hidden)
    return float(best[1]), int(best[2])


def run(output_dir: str | Path | None = None, quick: bool = False) -> dict:
    out = ensure_dir(output_dir or Path("results") / "exp29_gru_path_integration_sweep")
    dt = 0.05 if quick else 0.025
    train_seeds = list(range(2 if quick else 3))
    train_horizons = [4.0, 8.0, 16.0, 32.0] if quick else [8.0, 16.0, 32.0, 64.0]
    hidden_sizes = [16, 32, 64] if quick else [32, 64, 96]
    test_horizons = [8.0, 16.0, 32.0, 64.0, 96.0] if quick else [8.0, 16.0, 32.0, 64.0, 96.0, 128.0]
    speed_scales = [0.35, 0.55, 0.9, 1.2] if quick else [0.25, 0.35, 0.55, 0.9, 1.2]
    n_iters = 160 if quick else 260
    broken_epsilon = 0.02

    rows = []
    trained_count = 0
    for train_seed in train_seeds:
        for train_horizon in train_horizons:
            for hidden_size in hidden_sizes:
                gru = _train_gru_baseline(
                    quick=quick,
                    dt=dt,
                    seed=29000 + train_seed + 37 * hidden_size + int(train_horizon),
                    n_iters=n_iters,
                    hidden_size=hidden_size,
                    train_horizon=train_horizon,
                )
                if gru is None:
                    metrics = {
                        "quick": quick,
                        "torch_available": False,
                        "passed": False,
                        "inconclusive": True,
                        "note": "PyTorch is unavailable, so the GRU baseline sweep was skipped.",
                    }
                    save_json(out / "metrics.json", metrics)
                    save_summary(out / "summary.md", "Experiment 29: GRU Path-Integration Sweep", ["- PyTorch unavailable; skipped."])
                    return metrics
                trained_count += 1
                for speed_scale in speed_scales:
                    for test_horizon in test_horizons:
                        times, velocities = generate_piecewise_velocity(
                            test_horizon,
                            dt=dt,
                            segment_duration=0.6,
                            scale=speed_scale,
                            seed=29100 + train_seed + int(10 * test_horizon) + int(1000 * speed_scale),
                        )
                        del times
                        target_sim = simulate_phase_integrator(PhaseIntegrator(epsilon=0.0), velocities, dt=dt, theta0=0.0)
                        target = target_sim["theta_target"]
                        exact = target_sim
                        broken = simulate_phase_integrator(PhaseIntegrator(epsilon=broken_epsilon), velocities, dt=dt, theta0=0.0)
                        gru_pred = _eval_gru(gru, velocities, dt=dt)
                        shared = {
                            "train_seed": train_seed,
                            "train_horizon": train_horizon,
                            "hidden_size": hidden_size,
                            "speed_scale": speed_scale,
                            "test_horizon": test_horizon,
                        }
                        rows.append({**shared, "model": "exact", "rmse": circular_rmse(exact["theta_pred"], target)})
                        rows.append({**shared, "model": "broken", "rmse": circular_rmse(broken["theta_pred"], target)})
                        rows.append({**shared, "model": "GRU", "rmse": circular_rmse(gru_pred, target)})

    save_dict_rows_csv(out / "gru_path_integration_sweep.csv", rows)

    max_horizon = max(test_horizons)
    train_best, hidden_best = _select_best_gru(rows, train_horizons, hidden_sizes, speed_scale=0.55, horizon=max_horizon)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, speed_scale, label in [(axes_flat[0], 0.55, "in-distribution speed"), (axes_flat[1], max(speed_scales), "OOD speed")]:
        for model, color in [("exact", COLORS["blue"]), ("broken", COLORS["red"]), ("GRU", COLORS["green"])]:
            means, sems = [], []
            for horizon in test_horizons:
                if model == "GRU":
                    vals = [
                        r["rmse"]
                        for r in rows
                        if r["model"] == "GRU"
                        and r["speed_scale"] == speed_scale
                        and r["test_horizon"] == horizon
                        and r["train_horizon"] == train_best
                        and r["hidden_size"] == hidden_best
                    ]
                else:
                    vals = [
                        r["rmse"]
                        for r in rows
                        if r["model"] == model
                        and r["speed_scale"] == speed_scale
                        and r["test_horizon"] == horizon
                        and r["train_horizon"] == train_best
                        and r["hidden_size"] == hidden_best
                    ]
                mean, sem = _mean_sem(vals)
                means.append(mean)
                sems.append(sem)
            ax.errorbar(test_horizons, means, yerr=sems, marker="o", color=color, capsize=3, label=model)
        ax.set_yscale("log")
        ax.set_xlabel("test horizon")
        ax.set_ylabel("circular RMSE")
        ax.set_title(label)
        ax.legend(frameon=False, fontsize=8)
    _panel_label(axes_flat[0], "A")
    _panel_label(axes_flat[1], "B")

    for ax, speed_scale, title in [(axes_flat[2], 0.55, "GRU long-horizon RMSE"), (axes_flat[3], max(speed_scales), "GRU OOD long-horizon RMSE")]:
        heat = np.zeros((len(train_horizons), len(hidden_sizes)))
        for i, train_horizon in enumerate(train_horizons):
            for j, hidden_size in enumerate(hidden_sizes):
                heat[i, j] = _mean_row(rows, model="GRU", speed_scale=speed_scale, test_horizon=max_horizon, train_horizon=train_horizon, hidden_size=hidden_size)
        image = ax.imshow(heat, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=max(2.0, float(np.nanmax(heat))))
        ax.set_xticks(np.arange(len(hidden_sizes)))
        ax.set_xticklabels([str(h) for h in hidden_sizes])
        ax.set_yticks(np.arange(len(train_horizons)))
        ax.set_yticklabels([str(h) for h in train_horizons])
        ax.set_xlabel("hidden size")
        ax.set_ylabel("train horizon")
        ax.set_title(title)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="white" if heat[i, j] < 0.5 else COLORS["dark"], fontsize=8)
        cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("RMSE")
    _panel_label(axes_flat[2], "C")
    _panel_label(axes_flat[3], "D")

    ax = axes_flat[4]
    box_data = []
    labels = []
    colors = []
    for model, color in [("exact", COLORS["blue"]), ("broken", COLORS["red"]), ("GRU", COLORS["green"])]:
        vals = [
            r["rmse"]
            for r in rows
            if r["model"] == model
            and r["test_horizon"] == max_horizon
            and r["speed_scale"] == max(speed_scales)
            and r["train_horizon"] == train_best
            and r["hidden_size"] == hidden_best
        ]
        box_data.append(vals)
        labels.append(model)
        colors.append(color)
    box = ax.boxplot(box_data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    ax.set_yscale("log")
    ax.set_ylabel("long-horizon OOD RMSE")
    ax.set_title("Best-config distribution")
    _panel_label(ax, "E")

    ax = axes_flat[5]
    for hidden_size in hidden_sizes:
        gaps = []
        for train_horizon in train_horizons:
            long_rmse = _mean_row(rows, model="GRU", speed_scale=max(speed_scales), test_horizon=max_horizon, train_horizon=train_horizon, hidden_size=hidden_size)
            train_like_rmse = _mean_row(rows, model="GRU", speed_scale=0.55, test_horizon=min(test_horizons), train_horizon=train_horizon, hidden_size=hidden_size)
            gaps.append(long_rmse / max(train_like_rmse, 1e-12))
        ax.plot(train_horizons, gaps, marker="o", label=f"H={hidden_size}")
    ax.set_yscale("log")
    ax.set_xlabel("train horizon")
    ax.set_ylabel("OOD long / train-like short RMSE")
    ax.set_title("GRU extrapolation gap")
    ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "F")

    for ax in axes_flat:
        clean_axis(ax)
    save_figure(fig, out / "gru_path_integration_sweep.png")

    exact_long = [
        r["rmse"]
        for r in rows
        if r["model"] == "exact" and r["test_horizon"] == max_horizon and r["speed_scale"] == max(speed_scales) and r["train_horizon"] == train_best and r["hidden_size"] == hidden_best
    ]
    broken_long = [
        r["rmse"]
        for r in rows
        if r["model"] == "broken" and r["test_horizon"] == max_horizon and r["speed_scale"] == max(speed_scales) and r["train_horizon"] == train_best and r["hidden_size"] == hidden_best
    ]
    gru_long = [
        r["rmse"]
        for r in rows
        if r["model"] == "GRU" and r["test_horizon"] == max_horizon and r["speed_scale"] == max(speed_scales) and r["train_horizon"] == train_best and r["hidden_size"] == hidden_best
    ]
    metrics = {
        "quick": quick,
        "torch_available": True,
        "trained_models": trained_count,
        "train_seeds": train_seeds,
        "train_horizons": train_horizons,
        "hidden_sizes": hidden_sizes,
        "test_horizons": test_horizons,
        "speed_scales": speed_scales,
        "best_train_horizon": train_best,
        "best_hidden_size": hidden_best,
        "mean_exact_long_ood_rmse": float(np.mean(exact_long)),
        "mean_broken_long_ood_rmse": float(np.mean(broken_long)),
        "mean_gru_best_long_ood_rmse": float(np.mean(gru_long)),
        "median_gru_best_long_ood_rmse": float(np.median(gru_long)),
        "rows": len(rows),
        "passed": bool(np.mean(exact_long) < 1e-4 and np.mean(gru_long) > 10.0 * np.mean(exact_long)),
    }
    save_json(out / "metrics.json", metrics)
    save_summary(
        out / "summary.md",
        "Experiment 29: GRU Path-Integration Sweep",
        [
            f"- Trained {trained_count} GRU baselines across seeds, hidden sizes, and train horizons.",
            f"- Best config by longest in-distribution horizon: train horizon={train_best}, hidden size={hidden_best}.",
            f"- Exact long OOD RMSE: {metrics['mean_exact_long_ood_rmse']:.3e}.",
            f"- Best-GRU long OOD RMSE: {metrics['mean_gru_best_long_ood_rmse']:.3e}.",
            f"- Broken long OOD RMSE: {metrics['mean_broken_long_ood_rmse']:.3e}.",
        ],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(output_dir=args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
