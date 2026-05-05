"""Experiment 32: stronger path-integration baseline check.

This ICLR-readiness check tests whether the learned path-integration baseline
gap is mainly a small-budget artifact.  It keeps the exp31 task distribution,
initial phase cue, loss, optimizer family, and exact orthogonal-RNN constraint,
but compares the original 120-step/hidden-16 budget with a targeted stronger
500-step/hidden-32 budget for GRU, LSTM, and orthogonal-RNN baselines.

The output is a conservative diagnostic for paper wording.  It is task-level
evidence only; it is not theorem evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from goldstone_lyapunov.equivariant_s1_cell import EquivariantS1Cell
from goldstone_lyapunov.experiments.exp31_learned_equivariant_path_integration import (
    circular_metrics,
    ensure_dir,
    evaluate_one,
    ffloat,
    read_rows,
    sample_velocity_batch,
    write_json,
    write_rows,
)
from goldstone_lyapunov.path_integration_baselines import (
    OrthogonalRNNPathIntegrator,
    count_parameters,
    make_baseline,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "iclr_final_checks" / "stronger_baselines"
DEFAULT_FIG = ROOT / "figures_clean" / "iclr_final_checks"
LEARNED_OUT = ROOT / "results" / "learned_equivariant_pi"

MODELS = ["gru", "lstm", "orthogonal_rnn"]
MODEL_LABELS = {"gru": "GRU", "lstm": "LSTM", "orthogonal_rnn": "orthogonal RNN", "equivariant": "equivariant"}
COLORS = {"gru": "#4b72b0", "lstm": "#b35c00", "orthogonal_rnn": "#7a4da3", "equivariant": "#1f8a70"}
CONFIGS_FULL = [
    {"config_label": "original_budget", "hidden_size": 16, "train_steps": 120, "learning_rate": 2e-3},
    {"config_label": "stronger_budget", "hidden_size": 32, "train_steps": 500, "learning_rate": 2e-3},
]
CONFIGS_QUICK = [
    {"config_label": "smoke_original", "hidden_size": 16, "train_steps": 12, "learning_rate": 2e-3},
    {"config_label": "smoke_stronger", "hidden_size": 20, "train_steps": 18, "learning_rate": 2e-3},
]


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def mean_sem(values: list[float]) -> tuple[float, float]:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(vals.size))


def train_baseline_one(
    model_name: str,
    seed: int,
    phase_mode: str,
    config: dict[str, Any],
    out: Path,
    device: str,
    train_horizon: int = 64,
    speed_scale: float = 0.8,
    batch: int = 64,
) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dt = 0.1
    hidden_size = int(config["hidden_size"])
    train_steps = int(config["train_steps"])
    lr = float(config["learning_rate"])
    config_label = str(config["config_label"])
    grad_clip = 1.0
    model = make_baseline(model_name, hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    gen = torch.Generator(device=device)
    seed_offset = 70000 + 97 * seed + 1009 * MODELS.index(model_name) + (17 if phase_mode == "restricted" else 0)
    gen.manual_seed(seed_offset + hidden_size + train_steps)
    run_id = f"{model_name}_seed{seed}_T{train_horizon}_{phase_mode}_{config_label}"
    curves: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state = None
    log_every = max(10, train_steps // 25)
    start = time.perf_counter()
    status = "completed"
    warning = ""
    try:
        for step in range(1, train_steps + 1):
            model.train()
            velocities, phi0, target, _phi = sample_velocity_batch(
                batch, train_horizon, dt, speed_scale, phase_mode, device, gen
            )
            pred, _states = model(velocities, phi0)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if step == 1 or step % log_every == 0 or step == train_steps:
                model.eval()
                with torch.no_grad():
                    v_val, p0_val, target_val, _ = sample_velocity_batch(
                        batch, train_horizon, dt, speed_scale, "full", device, gen
                    )
                    pred_val, _ = model(v_val, p0_val)
                    val_loss = torch.mean((pred_val - target_val) ** 2)
                val_float = float(val_loss.detach().cpu())
                if val_float < best_val:
                    best_val = val_float
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                curves.append(
                    {
                        "run_id": run_id,
                        "model": model_name,
                        "seed": seed,
                        "train_horizon": train_horizon,
                        "hidden_size": hidden_size,
                        "speed_scale": speed_scale,
                        "phase_generalization": phase_mode,
                        "config_label": config_label,
                        "train_steps": train_steps,
                        "learning_rate": lr,
                        "step": step,
                        "train_loss": float(loss.detach().cpu()),
                        "validation_loss": val_float,
                    }
                )
    except Exception as exc:  # pragma: no cover - failure path is recorded for runs.
        status = "failed"
        warning = repr(exc)
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start
    ckpt_dir = ensure_dir(out / "checkpoints")
    ckpt_path = ckpt_dir / f"{run_id}.pt"
    if status == "completed":
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "hidden_size": hidden_size,
                "dt": dt,
                "seed": seed,
                "train_horizon": train_horizon,
                "phase_generalization": phase_mode,
                "config_label": config_label,
            },
            ckpt_path,
        )
    orth_err = float("nan")
    if isinstance(model, OrthogonalRNNPathIntegrator):
        orth_err = model.orthogonality_error()
    registry = {
        "run_id": run_id,
        "model": model_name,
        "seed": seed,
        "train_horizon": train_horizon,
        "hidden_size": hidden_size,
        "speed_scale": speed_scale,
        "phase_generalization": phase_mode,
        "config_label": config_label,
        "train_steps": train_steps,
        "batch_size": batch,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "gradient_clip": grad_clip,
        "loss": "MSE on normalized (cos phi, sin phi)",
        "runtime_seconds": elapsed,
        "parameter_count": count_parameters(model),
        "best_validation_loss": best_val,
        "checkpoint": str(ckpt_path.relative_to(ROOT)) if status == "completed" else "",
        "orthogonality_error": orth_err,
        "status": status,
        "warning": warning,
        "preliminary_less_than_3_seeds": False,
    }
    write_json(ensure_dir(out / "run_configs") / f"{run_id}.json", registry)
    return model, registry, curves


def write_parameter_counts(out: Path, device: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eq_ref = EquivariantS1Cell(hidden_size=16, dt=0.1).to(device)
    eq_params = count_parameters(eq_ref)
    rows.append({"model": "equivariant", "hidden_size": 16, "parameter_count": eq_params, "orthogonality_error": ""})
    for model_name in MODELS:
        for hidden in [8, 12, 16, 24, 32, 48, 64]:
            model = make_baseline(model_name, hidden).to(device)
            orth = model.orthogonality_error() if isinstance(model, OrthogonalRNNPathIntegrator) else float("nan")
            rows.append(
                {
                    "model": model_name,
                    "hidden_size": hidden,
                    "parameter_count": count_parameters(model),
                    "orthogonality_error": orth,
                    "abs_param_diff_to_equivariant_h16": abs(count_parameters(model) - eq_params),
                }
            )
    write_rows(out / "parameter_counts.csv", rows)
    return rows


def load_equivariant_reference() -> tuple[list[dict[str, str]], int]:
    eval_rows = read_rows(LEARNED_OUT / "evaluation_metrics.csv")
    reg_rows = read_rows(LEARNED_OUT / "model_registry.csv")
    eq_params = 1237
    for row in reg_rows:
        if row.get("model") == "equivariant" and int(ffloat(row.get("hidden_size"))) == 16:
            eq_params = int(ffloat(row.get("parameter_count")))
            break
    filtered = [
        row
        for row in eval_rows
        if row.get("model") == "equivariant"
        and int(ffloat(row.get("train_horizon"))) == 64
        and int(ffloat(row.get("hidden_size"))) == 16
    ]
    return filtered, eq_params


def aggregate_eval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = [
        "model",
        "config_label",
        "phase_generalization",
        "hidden_size",
        "train_steps",
        "learning_rate",
        "test_horizon",
        "test_speed_scale",
    ]
    for row in rows:
        grouped[tuple(row.get(k, "") for k in keys)].append(row)
    summary: list[dict[str, Any]] = []
    for key, vals in sorted(grouped.items()):
        rmse = [ffloat(v.get("circular_rmse")) for v in vals]
        final = [ffloat(v.get("final_circular_rmse")) for v in vals]
        success = [ffloat(v.get("success_probability")) for v in vals]
        mean_rmse, sem_rmse = mean_sem(rmse)
        mean_final, sem_final = mean_sem(final)
        mean_success, sem_success = mean_sem(success)
        row = dict(zip(keys, key))
        row.update(
            {
                "n_seeds": len({int(ffloat(v.get("seed"))) for v in vals}),
                "mean_circular_rmse": mean_rmse,
                "sem_circular_rmse": sem_rmse,
                "mean_final_circular_rmse": mean_final,
                "sem_final_circular_rmse": sem_final,
                "mean_success_probability": mean_success,
                "sem_success_probability": sem_success,
            }
        )
        summary.append(row)
    return summary


def make_statistical_summary(out: Path, eval_rows: list[dict[str, Any]], registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = aggregate_eval(eval_rows)
    write_rows(out / "baseline_sweep_summary.csv", summary)
    eq_rows, _eq_params = load_equivariant_reference()
    target_rows = [
        r
        for r in summary
        if int(ffloat(r["test_horizon"])) == 256 and abs(ffloat(r["test_speed_scale"]) - 1.8) < 1e-9
    ]
    eq_target = [
        ffloat(r["circular_rmse"])
        for r in eq_rows
        if r.get("phase_generalization") == "full"
        and int(ffloat(r.get("test_horizon"))) == 256
        and abs(ffloat(r.get("test_speed_scale")) - 1.8) < 1e-9
    ]
    by_key = {(r["model"], r["config_label"], r["phase_generalization"]): r for r in target_rows}
    lines = ["# Stronger Baseline Statistical Summary", ""]
    lines.append("This check is task-level evidence and is not theorem evidence.")
    lines.append("")
    lines.append("## Runs")
    status_counts = defaultdict(int)
    for r in registry:
        status_counts[(r["model"], r["config_label"], r["status"])] += 1
    for key, count in sorted(status_counts.items()):
        lines.append(f"- {key[0]} / {key[1]} / {key[2]}: {count} runs.")
    lines.append("")
    lines.append("## Long-Horizon Speed-OOD RMSE at Horizon 256, Speed 1.8")
    if eq_target:
        m_eq, s_eq = mean_sem(eq_target)
        lines.append(f"- Equivariant reference from exp31 full-phase six-seed run: {m_eq:.4g} +/- {s_eq:.2g} (n={len(eq_target)}).")
    for r in target_rows:
        lines.append(
            f"- {MODEL_LABELS.get(r['model'], r['model'])} {r['config_label']} {r['phase_generalization']}: "
            f"{ffloat(r['mean_circular_rmse']):.4g} +/- {ffloat(r['sem_circular_rmse']):.2g} "
            f"(n={r['n_seeds']})."
        )
    lines.append("")
    lines.append("## Original vs Stronger")
    for model_name in MODELS:
        for phase in ["full", "restricted"]:
            original = by_key.get((model_name, "original_budget", phase))
            stronger = by_key.get((model_name, "stronger_budget", phase))
            if original and stronger:
                o = ffloat(original["mean_circular_rmse"])
                s = ffloat(stronger["mean_circular_rmse"])
                delta = (s - o) / o if o > 0 else float("nan")
                direction = "improved" if delta < 0 else "worsened"
                lines.append(f"- {MODEL_LABELS[model_name]} {phase}: stronger budget {direction} by {-delta:.1%} relative to original.")
    lines.append("")
    lines.append("## Interpretation")
    strongest = [
        ffloat(r["mean_circular_rmse"])
        for r in target_rows
        if r.get("phase_generalization") == "full" and r.get("config_label") == "stronger_budget"
    ]
    if strongest and eq_target:
        best = min(strongest)
        eq_mean, _eq_sem = mean_sem(eq_target)
        lines.append(
            f"The stronger GRU/LSTM baselines improve materially under the extra budget, but the best tested stronger full-phase baseline remains {best / eq_mean:.1f}x higher RMSE than the equivariant reference on this long-horizon speed-OOD slice."
        )
    lines.append(
        "If a stronger configuration improves materially, the manuscript should describe the original comparison as a small-budget matched protocol and use this check to report whether the performance gap persisted under the tested extra budget."
    )
    lines.append("No universal statement about GRUs, LSTMs, or orthogonal RNNs is supported by this table.")
    (out / "statistical_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def grouped_values(rows: list[dict[str, Any]], filt: dict[str, Any], ykey: str = "circular_rmse") -> dict[Any, list[float]]:
    grouped: dict[Any, list[float]] = defaultdict(list)
    for r in rows:
        ok = True
        for k, v in filt.items():
            rv = r.get(k)
            if isinstance(v, float):
                ok = ok and abs(ffloat(rv) - v) < 1e-9
            else:
                ok = ok and rv == v
        if ok:
            grouped[r.get("test_horizon")].append(ffloat(r.get(ykey)))
    return grouped


def horizon_values(grouped: dict[Any, list[float]], horizon: int) -> list[float]:
    vals: list[float] = []
    for key, key_vals in grouped.items():
        if int(ffloat(key)) == int(horizon):
            vals.extend(key_vals)
    return vals


def plot_training_curves(curves: list[dict[str, Any]], fig_dir: Path) -> None:
    ensure_dir(fig_dir)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.75), constrained_layout=True, sharey=True)
    for ax, model_name in zip(axes, MODELS):
        config_labels = sorted({row.get("config_label", "") for row in curves if row.get("model") == model_name})
        for config_label in config_labels:
            for metric in ["train_loss", "validation_loss"]:
                grouped: dict[int, list[float]] = defaultdict(list)
                for row in curves:
                    if (
                        row.get("model") == model_name
                        and row.get("config_label") == config_label
                        and row.get("phase_generalization") == "full"
                    ):
                        grouped[int(ffloat(row["step"]))].append(ffloat(row[metric]))
                xs = sorted(grouped)
                if not xs:
                    continue
                ys = np.asarray([np.nanmean(grouped[x]) for x in xs], dtype=float)
                sem = np.asarray(
                    [
                        np.nanstd(grouped[x], ddof=1) / math.sqrt(len(grouped[x])) if len(grouped[x]) > 1 else 0.0
                        for x in xs
                    ],
                    dtype=float,
                )
                is_stronger = "stronger" in config_label
                linestyle = "-" if metric == "validation_loss" else "--"
                lw = 1.4 if is_stronger else 0.8
                label = f"{config_label.replace('_', ' ')} {'val' if metric == 'validation_loss' else 'train'}"
                ax.plot(xs, ys, linestyle=linestyle, color=COLORS[model_name], lw=lw, label=label)
                ax.fill_between(xs, np.maximum(ys - sem, 1e-12), ys + sem, color=COLORS[model_name], alpha=0.10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("training step")
        ax.set_title(MODEL_LABELS[model_name])
    axes[0].set_ylabel("MSE loss")
    axes[0].legend(frameon=False, fontsize=6)
    fig.suptitle("Stronger baseline convergence check (full phase, mean +/- SEM)", fontsize=10)
    fig.savefig(fig_dir / "fig_stronger_baseline_training_curves.pdf")
    fig.savefig(fig_dir / "fig_stronger_baseline_training_curves.png", dpi=300)
    plt.close(fig)


def plot_eval(eval_rows: list[dict[str, Any]], fig_dir: Path) -> None:
    eq_rows, _eq_params = load_equivariant_reference()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.85), constrained_layout=True)
    # A: long horizon at in-distribution speed.
    for model_name in MODELS:
        for config_label, ls in [("original_budget", "--"), ("stronger_budget", "-")]:
            grouped = grouped_values(
                eval_rows,
                {
                    "model": model_name,
                    "config_label": config_label,
                    "phase_generalization": "full",
                    "test_speed_scale": 0.8,
                },
            )
            xs = sorted(int(ffloat(x)) for x in grouped)
            if xs:
                ys = [mean_sem(horizon_values(grouped, x))[0] for x in xs]
                es = [mean_sem(horizon_values(grouped, x))[1] for x in xs]
                axes[0].errorbar(xs, ys, yerr=es, marker="o", ms=3, ls=ls, color=COLORS[model_name], capsize=2)
    eq_group = grouped_values(eq_rows, {"model": "equivariant", "phase_generalization": "full", "test_speed_scale": 0.8})
    xs = sorted(int(ffloat(x)) for x in eq_group)
    if xs:
        axes[0].errorbar(
            xs,
            [mean_sem(horizon_values(eq_group, x))[0] for x in xs],
            yerr=[mean_sem(horizon_values(eq_group, x))[1] for x in xs],
            marker="s",
            color=COLORS["equivariant"],
            capsize=2,
            label="equivariant ref.",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("test horizon")
    axes[0].set_ylabel("circular RMSE")
    axes[0].set_title("A  Horizon")
    axes[0].grid(True, alpha=0.25)

    # B: speed-OOD at horizon 256, full phase.
    labels: list[str] = []
    means: list[float] = []
    sems: list[float] = []
    colors: list[str] = []
    for model_name in MODELS:
        for config_label in ["original_budget", "stronger_budget"]:
            vals = [
                ffloat(r["circular_rmse"])
                for r in eval_rows
                if r["model"] == model_name
                and r.get("config_label") == config_label
                and r.get("phase_generalization") == "full"
                and int(ffloat(r["test_horizon"])) == 256
                and abs(ffloat(r["test_speed_scale"]) - 1.8) < 1e-9
            ]
            m, s = mean_sem(vals)
            labels.append(f"{MODEL_LABELS[model_name]}\n{config_label.split('_')[0]}")
            means.append(m)
            sems.append(s)
            colors.append(COLORS[model_name])
    eq_vals = [
        ffloat(r["circular_rmse"])
        for r in eq_rows
        if r["phase_generalization"] == "full"
        and int(ffloat(r["test_horizon"])) == 256
        and abs(ffloat(r["test_speed_scale"]) - 1.8) < 1e-9
    ]
    m, s = mean_sem(eq_vals)
    labels.append("equivariant\nref.")
    means.append(m)
    sems.append(s)
    colors.append(COLORS["equivariant"])
    x = np.arange(len(labels))
    axes[1].bar(x, means, yerr=sems, color=colors, alpha=0.88, capsize=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
    axes[1].set_ylabel("circular RMSE")
    axes[1].set_title("B  Speed OOD")
    axes[1].grid(True, axis="y", alpha=0.25)

    # C: restricted phase training, full-circle eval.
    labels = []
    means = []
    sems = []
    colors = []
    for model_name in MODELS:
        for config_label in ["original_budget", "stronger_budget"]:
            vals = [
                ffloat(r["circular_rmse"])
                for r in eval_rows
                if r["model"] == model_name
                and r.get("config_label") == config_label
                and r.get("phase_generalization") == "restricted"
                and int(ffloat(r["test_horizon"])) == 256
                and abs(ffloat(r["test_speed_scale"]) - 0.8) < 1e-9
            ]
            m, s = mean_sem(vals)
            labels.append(f"{MODEL_LABELS[model_name]}\n{config_label.split('_')[0]}")
            means.append(m)
            sems.append(s)
            colors.append(COLORS[model_name])
    eq_vals = [
        ffloat(r["circular_rmse"])
        for r in eq_rows
        if r["phase_generalization"] == "restricted"
        and int(ffloat(r["test_horizon"])) == 256
        and abs(ffloat(r["test_speed_scale"]) - 0.8) < 1e-9
    ]
    m, s = mean_sem(eq_vals)
    labels.append("equivariant\nref.")
    means.append(m)
    sems.append(s)
    colors.append(COLORS["equivariant"])
    x = np.arange(len(labels))
    axes[2].bar(x, means, yerr=sems, color=colors, alpha=0.88, capsize=2)
    axes[2].set_yscale("log")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
    axes[2].set_ylabel("circular RMSE")
    axes[2].set_title("C  Restricted phase")
    axes[2].grid(True, axis="y", alpha=0.25)
    fig.savefig(fig_dir / "fig_stronger_baseline_eval.pdf")
    fig.savefig(fig_dir / "fig_stronger_baseline_eval.png", dpi=300)
    plt.close(fig)


def plot_parameter_matched(
    eval_rows: list[dict[str, Any]], param_rows: list[dict[str, Any]], fig_dir: Path, out: Path
) -> None:
    eq_rows, eq_params = load_equivariant_reference()
    trained_params = {
        (r["model"], int(ffloat(r["hidden_size"]))): int(ffloat(r["parameter_count"]))
        for r in param_rows
        if r.get("model") in MODELS
    }
    candidates = []
    for model_name in MODELS:
        best = None
        for config_label in ["original_budget", "stronger_budget"]:
            vals = [
                ffloat(r["circular_rmse"])
                for r in eval_rows
                if r["model"] == model_name
                and r.get("config_label") == config_label
                and r.get("phase_generalization") == "full"
                and int(ffloat(r["test_horizon"])) == 256
                and abs(ffloat(r["test_speed_scale"]) - 1.8) < 1e-9
            ]
            if not vals:
                continue
            hidden = 16 if config_label == "original_budget" else 32
            params = trained_params.get((model_name, hidden), 0)
            item = {
                "model": model_name,
                "config_label": config_label,
                "hidden_size": hidden,
                "parameter_count": params,
                "rmse_mean": mean_sem(vals)[0],
                "rmse_sem": mean_sem(vals)[1],
                "param_diff": abs(params - eq_params),
            }
            if best is None or item["param_diff"] < best["param_diff"]:
                best = item
        if best is not None:
            candidates.append(best)
    eq_vals = [
        ffloat(r["circular_rmse"])
        for r in eq_rows
        if r["phase_generalization"] == "full"
        and int(ffloat(r["test_horizon"])) == 256
        and abs(ffloat(r["test_speed_scale"]) - 1.8) < 1e-9
    ]
    labels = ["equivariant\nref."] + [f"{MODEL_LABELS[c['model']]}\n{c['config_label'].split('_')[0]}" for c in candidates]
    means = [mean_sem(eq_vals)[0]] + [c["rmse_mean"] for c in candidates]
    sems = [mean_sem(eq_vals)[1]] + [c["rmse_sem"] for c in candidates]
    colors = [COLORS["equivariant"]] + [COLORS[c["model"]] for c in candidates]
    params = [eq_params] + [c["parameter_count"] for c in candidates]
    fig, ax = plt.subplots(figsize=(5.8, 3.0), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sems, color=colors, capsize=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("circular RMSE")
    ax.set_title("Parameter-nearest baseline view")
    ax.grid(True, axis="y", alpha=0.25)
    for xi, p in zip(x, params):
        ax.text(xi, means[xi] * 1.2 if math.isfinite(means[xi]) and means[xi] > 0 else 1.0, f"{p} params", ha="center", va="bottom", fontsize=6, rotation=90)
    fig.savefig(fig_dir / "fig_parameter_matched_baselines.pdf")
    fig.savefig(fig_dir / "fig_parameter_matched_baselines.png", dpi=300)
    plt.close(fig)
    write_rows(out / "parameter_nearest_summary.csv", candidates)


def write_manifest(
    out: Path,
    args: argparse.Namespace,
    registry: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    param_rows: list[dict[str, Any]],
    started: float,
    completed: float,
) -> None:
    manifest = {
        "experiment": "exp32_stronger_path_integration_baselines",
        "purpose": "ICLR final stronger-baseline and convergence check",
        "task_level_evidence_only": True,
        "models": MODELS,
        "configs": CONFIGS_QUICK if args.quick else CONFIGS_FULL,
        "seeds": args.seeds,
        "phase_generalization_modes": args.phase_modes,
        "train_horizon": args.train_horizon,
        "speed_scale": args.speed_scale,
        "evaluation_horizons": args.test_horizons,
        "evaluation_speeds": args.test_speeds,
        "optimizer": "AdamW",
        "batch_size": args.batch_size,
        "loss": "MSE on normalized (cos phi, sin phi)",
        "orthogonal_rnn_constraint": "W=matrix_exp(A-A^T)",
        "runs": len(registry),
        "completed_runs": sum(1 for r in registry if r.get("status") == "completed"),
        "evaluation_rows": len(eval_rows),
        "parameter_count_rows": len(param_rows),
        "runtime_seconds": completed - started,
        "output_files": [
            "model_registry.csv",
            "training_curves.csv",
            "evaluation_metrics.csv",
            "parameter_counts.csv",
            "baseline_sweep_summary.csv",
            "statistical_summary.md",
            "reproducibility_manifest.md",
        ],
    }
    write_json(out / "reproducibility_manifest.json", manifest)
    lines = ["# Reproducibility Manifest: Stronger Baselines", ""]
    for key, value in manifest.items():
        lines.append(f"- {key}: {json.dumps(jsonable(value))}")
    (out / "reproducibility_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def enrich_eval_rows(rows: list[dict[str, Any]], registry: dict[str, Any]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        new = dict(row)
        new["config_label"] = registry.get("config_label", "")
        new["train_steps"] = registry.get("train_steps", "")
        new["learning_rate"] = registry.get("learning_rate", "")
        new["parameter_count"] = registry.get("parameter_count", "")
        enriched.append(new)
    return enriched


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    out = ensure_dir(Path(args.output_dir).resolve())
    fig_dir = ensure_dir(Path(args.figure_dir).resolve())
    device = args.device
    configs = CONFIGS_QUICK if args.quick else CONFIGS_FULL
    write_parameter_counts(out, device)
    param_rows = read_rows(out / "parameter_counts.csv")
    registry_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for model_name in args.models:
        for phase_mode in args.phase_modes:
            for seed in args.seeds:
                for config in configs:
                    model, registry, curves = train_baseline_one(
                        model_name=model_name,
                        seed=int(seed),
                        phase_mode=phase_mode,
                        config=config,
                        out=out,
                        device=device,
                        train_horizon=args.train_horizon,
                        speed_scale=args.speed_scale,
                        batch=args.batch_size,
                    )
                    registry_rows.append(registry)
                    curve_rows.extend(curves)
                    if registry["status"] == "completed":
                        eval_rows.extend(
                            enrich_eval_rows(
                                evaluate_one(
                                    model,
                                    registry,
                                    test_horizons=args.test_horizons,
                                    test_speeds=args.test_speeds,
                                    device=device,
                                    n_batches=args.eval_batches,
                                    batch=args.eval_batch_size,
                                ),
                                registry,
                            )
                        )
                    print(f"{registry['run_id']}: {registry['status']} best_val={registry['best_validation_loss']:.4g}")
    write_rows(out / "model_registry.csv", registry_rows)
    write_rows(out / "training_curves.csv", curve_rows)
    write_rows(out / "evaluation_metrics.csv", eval_rows)
    summary = make_statistical_summary(out, eval_rows, registry_rows)
    plot_training_curves(curve_rows, fig_dir)
    plot_eval(eval_rows, fig_dir)
    plot_parameter_matched(eval_rows, param_rows, fig_dir, out)
    completed = time.perf_counter()
    write_manifest(out, args, registry_rows, eval_rows, param_rows, started, completed)
    print(f"Wrote {len(registry_rows)} registry rows, {len(curve_rows)} curve rows, {len(eval_rows)} eval rows.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a smoke version.")
    parser.add_argument("--full", action="store_true", help="Run the targeted ICLR check.")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--phase-modes", nargs="+", default=["full", "restricted"])
    parser.add_argument("--train-horizon", type=int, default=64)
    parser.add_argument("--speed-scale", type=float, default=0.8)
    parser.add_argument("--test-horizons", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--test-speeds", nargs="+", type=float, default=[0.8, 1.8])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIG))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
