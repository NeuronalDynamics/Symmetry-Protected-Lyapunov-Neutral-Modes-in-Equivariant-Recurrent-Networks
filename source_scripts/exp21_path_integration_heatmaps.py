"""Step 21: path-integration heatmaps over horizon, speed, and breaking."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from goldstone_lyapunov.models import PhaseIntegrator
from goldstone_lyapunov.plotting import COLORS, clean_axis, ensure_dir, save_dict_rows_csv, save_figure, save_json, save_summary
from goldstone_lyapunov.tasks import angle_wrap, circular_rmse, generate_piecewise_velocity, simulate_phase_integrator


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.11,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=COLORS["dark"],
    )


def _annotate_heatmap(ax, values: np.ndarray, fmt: str = ".2g") -> None:
    if values.size > 80:
        return
    finite = values[np.isfinite(values)]
    midpoint = float(np.nanmedian(finite)) if finite.size else 0.0
    for iy in range(values.shape[0]):
        for ix in range(values.shape[1]):
            val = values[iy, ix]
            if not np.isfinite(val):
                continue
            color = "white" if val < midpoint else COLORS["dark"]
            ax.text(ix, iy, format(val, fmt), ha="center", va="center", fontsize=6.5, color=color)


def run(output_dir: str | Path | None = None, quick: bool = False) -> dict:
    out = ensure_dir(output_dir or Path("results") / "exp21_path_integration_heatmaps")
    seeds = list(range(4 if quick else 12))
    horizons = [12.0, 30.0, 60.0] if quick else [12.0, 30.0, 60.0, 100.0, 150.0]
    epsilons = [0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2] if quick else [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 6e-2]
    speed_scales = [0.3, 0.6, 1.0] if quick else [0.25, 0.45, 0.65, 0.9, 1.2]
    dt = 0.05 if quick else 0.025
    rows = []

    for seed in seeds:
        for scale in speed_scales:
            for horizon in horizons:
                _, velocities = generate_piecewise_velocity(horizon, dt=dt, segment_duration=0.75, scale=scale, seed=2100 + seed + int(10 * horizon) + int(100 * scale))
                target = simulate_phase_integrator(PhaseIntegrator(epsilon=0.0), velocities, dt=dt, theta0=0.0)["theta_target"]
                for eps in epsilons:
                    sim = simulate_phase_integrator(PhaseIntegrator(epsilon=eps), velocities, dt=dt, theta0=0.0)
                    err = np.abs(((sim["theta_pred"] - target + np.pi) % (2.0 * np.pi)) - np.pi)
                    rows.append(
                        {
                            "seed": seed,
                            "speed_scale": scale,
                            "horizon": horizon,
                            "epsilon": eps,
                            "rmse": circular_rmse(sim["theta_pred"], target),
                            "final_abs_error": float(err[-1]),
                            "survived_0p25rad": bool(np.max(err) < 0.25),
                        }
                    )

    save_dict_rows_csv(out / "path_integration_heatmaps.csv", rows)

    def mean_grid(x_values, y_values, fixed_key: str, fixed_value: float, x_key: str, y_key: str, value_key: str = "rmse") -> np.ndarray:
        grid = np.zeros((len(y_values), len(x_values)))
        for i, y in enumerate(y_values):
            for j, x in enumerate(x_values):
                vals = [r[value_key] for r in rows if r[fixed_key] == fixed_value and r[x_key] == x and r[y_key] == y]
                grid[i, j] = np.mean(vals)
        return grid

    middle_speed = speed_scales[len(speed_scales) // 2]
    max_horizon = max(horizons)
    heat_horizon_eps = mean_grid(epsilons, horizons, "speed_scale", middle_speed, "epsilon", "horizon")
    heat_speed_eps = mean_grid(epsilons, speed_scales, "horizon", max_horizon, "epsilon", "speed_scale")
    survival_speed_eps = mean_grid(epsilons, speed_scales, "horizon", max_horizon, "epsilon", "speed_scale", value_key="survived_0p25rad")

    _, velocities = generate_piecewise_velocity(max_horizon, dt=dt, segment_duration=0.75, scale=middle_speed, seed=999)
    exact = simulate_phase_integrator(PhaseIntegrator(epsilon=0.0), velocities, dt=dt, theta0=0.0)
    broken = simulate_phase_integrator(PhaseIntegrator(epsilon=max(epsilons)), velocities, dt=dt, theta0=0.0)
    times = exact["theta_target"].size
    time_axis = np.arange(times) * dt
    err_exact = np.abs(angle_wrap(exact["theta_pred"] - exact["theta_target"]))
    err_broken = np.abs(angle_wrap(broken["theta_pred"] - exact["theta_target"]))

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.2), constrained_layout=True)
    image = axes[0, 0].imshow(np.log10(heat_horizon_eps + 1e-10), origin="lower", aspect="auto", cmap="magma")
    axes[0, 0].set_xticks(np.arange(len(epsilons)))
    axes[0, 0].set_xticklabels([f"{e:g}" for e in epsilons], rotation=35, ha="right")
    axes[0, 0].set_yticks(np.arange(len(horizons)))
    axes[0, 0].set_yticklabels([f"{h:g}" for h in horizons])
    axes[0, 0].set_xlabel("breaking epsilon")
    axes[0, 0].set_ylabel("horizon")
    axes[0, 0].set_title(f"log10 RMSE, speed={middle_speed:g}")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04)
    _annotate_heatmap(axes[0, 0], np.log10(heat_horizon_eps + 1e-10), ".1f")
    _panel_label(axes[0, 0], "A")

    image = axes[0, 1].imshow(np.log10(heat_speed_eps + 1e-10), origin="lower", aspect="auto", cmap="magma")
    axes[0, 1].set_xticks(np.arange(len(epsilons)))
    axes[0, 1].set_xticklabels([f"{e:g}" for e in epsilons], rotation=35, ha="right")
    axes[0, 1].set_yticks(np.arange(len(speed_scales)))
    axes[0, 1].set_yticklabels([f"{s:g}" for s in speed_scales])
    axes[0, 1].set_xlabel("breaking epsilon")
    axes[0, 1].set_ylabel("velocity scale")
    axes[0, 1].set_title(f"log10 RMSE, horizon={max_horizon:g}")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04)
    _annotate_heatmap(axes[0, 1], np.log10(heat_speed_eps + 1e-10), ".1f")
    _panel_label(axes[0, 1], "B")

    image = axes[0, 2].imshow(survival_speed_eps, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 2].set_xticks(np.arange(len(epsilons)))
    axes[0, 2].set_xticklabels([f"{e:g}" for e in epsilons], rotation=35, ha="right")
    axes[0, 2].set_yticks(np.arange(len(speed_scales)))
    axes[0, 2].set_yticklabels([f"{s:g}" for s in speed_scales])
    axes[0, 2].set_xlabel("breaking epsilon")
    axes[0, 2].set_ylabel("velocity scale")
    axes[0, 2].set_title("P(max error < 0.25 rad)")
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046, pad=0.04)
    _annotate_heatmap(axes[0, 2], survival_speed_eps, ".2f")
    _panel_label(axes[0, 2], "C")

    axes[1, 0].plot(time_axis, exact["theta_target_unwrapped"], color=COLORS["dark"], label="target", linewidth=1.3)
    axes[1, 0].plot(time_axis, exact["theta_pred_unwrapped"], color=COLORS["blue"], linestyle="--", label="exact")
    axes[1, 0].plot(time_axis, broken["theta_pred_unwrapped"], color=COLORS["red"], label=f"broken eps={max(epsilons):g}")
    inset = axes[1, 0].inset_axes([0.05, 0.08, 0.38, 0.26])
    inset.step(time_axis[:-1], velocities, where="post", color=COLORS["gray"], linewidth=0.8)
    inset.set_title("velocity", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(False)
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("unwrapped phase")
    axes[1, 0].set_title("Long trajectory example")
    axes[1, 0].legend(frameon=False)
    _panel_label(axes[1, 0], "D")

    axes[1, 1].plot(time_axis, err_exact, color=COLORS["blue"], label="exact")
    axes[1, 1].plot(time_axis, err_broken, color=COLORS["red"], label="broken")
    axes[1, 1].set_yscale("symlog", linthresh=1e-5)
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("|circular error|")
    axes[1, 1].set_title("Error remains neutral or pins")
    axes[1, 1].legend(frameon=False)
    _panel_label(axes[1, 1], "E")

    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    axes[1, 2].plot(np.cos(theta), np.sin(theta), color=COLORS["light_gray"], linewidth=1.4)
    step = max(1, exact["states"].shape[0] // 700)
    axes[1, 2].plot(exact["states"][::step, 0], exact["states"][::step, 1], color=COLORS["blue"], label="exact")
    axes[1, 2].plot(broken["states"][::step, 0], broken["states"][::step, 1], color=COLORS["red"], label="broken")
    axes[1, 2].scatter([exact["states"][0, 0]], [exact["states"][0, 1]], color=COLORS["dark"], s=22, label="start", zorder=4)
    axes[1, 2].scatter([broken["states"][-1, 0]], [broken["states"][-1, 1]], color=COLORS["red"], s=32, marker="x", label="broken final", zorder=5)
    axes[1, 2].set_aspect("equal", adjustable="box")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("y")
    axes[1, 2].set_title("State-space pinning")
    axes[1, 2].legend(frameon=False, fontsize=8, loc="upper right")
    _panel_label(axes[1, 2], "F")
    for ax in axes[0, :]:
        ax.grid(False)
    for ax in axes[1, :]:
        clean_axis(ax)
    save_figure(fig, out / "path_integration_heatmaps.png")

    exact_long = [r["rmse"] for r in rows if r["horizon"] == max_horizon and r["epsilon"] == 0.0]
    broken_long = [r["rmse"] for r in rows if r["horizon"] == max_horizon and r["epsilon"] == max(epsilons)]
    metrics = {
        "quick": quick,
        "seeds": seeds,
        "horizons": horizons,
        "epsilons": epsilons,
        "speed_scales": speed_scales,
        "mean_exact_long_rmse": float(np.mean(exact_long)),
        "mean_strongly_broken_long_rmse": float(np.mean(broken_long)),
        "passed": bool(np.mean(exact_long) < 1e-4 and np.mean(broken_long) > 100.0 * np.mean(exact_long)),
        "rows": rows,
    }
    save_json(out / "metrics.json", metrics)
    save_summary(
        out / "summary.md",
        "Experiment 21: Path-Integration Heatmaps",
        [
            f"- Swept {len(horizons)} horizons, {len(epsilons)} breaking strengths, {len(speed_scales)} speed scales, and {len(seeds)} seeds.",
            f"- Mean exact RMSE at longest horizon: {metrics['mean_exact_long_rmse']:.3e}.",
            f"- Mean strongly-broken RMSE at longest horizon: {metrics['mean_strongly_broken_long_rmse']:.3e}.",
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
