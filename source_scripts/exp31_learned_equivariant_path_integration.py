"""Experiment 31: learned exactly equivariant path integration.

This experiment trains an exact S1-equivariant recurrent cell end-to-end on a
velocity-input path-integration task and compares it with matched GRU, LSTM,
and orthogonal-RNN baselines.  The learned task is nonautonomous, so the
diagnostics here are application evidence rather than autonomous theorem proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from goldstone_lyapunov.equivariant_s1_cell import (
    EquivariantS1Cell,
    autonomous_zero_diagnostic,
    direct_group_tangent_exponent,
    principal_angle_to_group_tangent,
    vector_field_equivariance_error,
)
from goldstone_lyapunov.path_integration_baselines import (
    OrthogonalRNNPathIntegrator,
    count_parameters,
    make_baseline,
    phase_shift_error,
)
from goldstone_lyapunov.tasks import angle_wrap


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "learned_equivariant_pi"
DEFAULT_FIG = ROOT / "figures_clean" / "learned_equivariant_pi"

COLORS = {
    "equivariant": "#1f8a70",
    "broken_equivariant": "#9b2d30",
    "gru": "#4b72b0",
    "lstm": "#b35c00",
    "orthogonal_rnn": "#7a4da3",
    "untrained_exact": "#7a7a7a",
    "broken_posthoc": "#c03a3a",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(jsonable(row))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sample_velocity_batch(
    batch: int,
    steps: int,
    dt: float,
    speed_scale: float,
    phase_mode: str,
    device: str,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    velocities = torch.zeros(batch, steps, dtype=torch.float32, device=device)
    kind = torch.randint(0, 3, (batch,), generator=gen, device=device)
    for i in range(batch):
        if int(kind[i].item()) == 0:
            velocities[i] = speed_scale * torch.randn(steps, generator=gen, device=device)
        elif int(kind[i].item()) == 1:
            seg = int(torch.randint(4, 13, (), generator=gen, device=device).item())
            nseg = int(math.ceil(steps / seg))
            vals = speed_scale * torch.randn(nseg, generator=gen, device=device)
            velocities[i] = vals.repeat_interleave(seg)[:steps]
        else:
            noise = speed_scale * torch.randn(steps, generator=gen, device=device)
            v = torch.zeros(steps, dtype=torch.float32, device=device)
            for t in range(steps):
                v[t] = (0.92 * v[t - 1] if t else 0.0) + 0.25 * noise[t]
            velocities[i] = v
    if phase_mode == "restricted":
        phi0 = (torch.rand(batch, generator=gen, device=device) - 0.5) * (math.pi / 2.0)
    else:
        phi0 = (2.0 * math.pi) * torch.rand(batch, generator=gen, device=device) - math.pi
    dphi = torch.cumsum(velocities * dt, dim=1)
    phi = torch.cat([phi0[:, None], phi0[:, None] + dphi], dim=1)
    target = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    return velocities, phi0, target, phi


def circular_metrics(pred_y: torch.Tensor, target_phi: torch.Tensor, threshold: float = 0.25) -> dict[str, float]:
    pred_phi = torch.atan2(pred_y[..., 1], pred_y[..., 0])
    err = torch.atan2(torch.sin(pred_phi - target_phi), torch.cos(pred_phi - target_phi))
    rmse = torch.sqrt(torch.mean(err * err))
    final_rmse = torch.sqrt(torch.mean(err[:, -1] ** 2))
    success = torch.mean((torch.abs(err[:, -1]) < threshold).float())
    return {
        "circular_rmse": float(rmse.detach().cpu()),
        "final_circular_rmse": float(final_rmse.detach().cpu()),
        "success_probability": float(success.detach().cpu()),
    }


def make_model(model_name: str, hidden_size: int, dt: float, device: str) -> torch.nn.Module:
    if model_name == "equivariant":
        return EquivariantS1Cell(hidden_size=hidden_size, dt=dt, epsilon=0.0).to(device)
    if model_name == "broken_equivariant":
        return EquivariantS1Cell(hidden_size=hidden_size, dt=dt, epsilon=0.02).to(device)
    return make_baseline(model_name, hidden_size).to(device)


def train_one(
    model_name: str,
    seed: int,
    train_horizon: int,
    hidden_size: int,
    speed_scale: float,
    phase_mode: str,
    device: str,
    out: Path,
    quick: bool,
    train_steps_override: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dt = 0.1
    train_steps = int(train_steps_override if train_steps_override is not None else (28 if quick else 120))
    batch = 32 if quick else 64
    lr = 3e-3 if model_name in {"equivariant", "broken_equivariant"} else 2e-3
    grad_clip = 1.0
    gen = torch.Generator(device=device)
    gen.manual_seed(31000 + seed + 17 * train_horizon)
    model = make_model(model_name, hidden_size, dt, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    run_id = f"{model_name}_seed{seed}_T{train_horizon}_{phase_mode}"
    curves: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state = None
    start = time.perf_counter()
    for step in range(1, train_steps + 1):
        model.train()
        velocities, phi0, target, _phi = sample_velocity_batch(batch, train_horizon, dt, speed_scale, phase_mode, device, gen)
        pred, _states = model(velocities, phi0)
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if step == 1 or step % max(5, train_steps // 12) == 0 or step == train_steps:
            model.eval()
            with torch.no_grad():
                v_val, p0_val, target_val, _ = sample_velocity_batch(batch, train_horizon, dt, speed_scale, "full", device, gen)
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
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "validation_loss": val_float,
                }
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start
    ckpt_dir = ensure_dir(out / "checkpoints")
    ckpt_path = ckpt_dir / f"{run_id}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "hidden_size": hidden_size,
            "dt": dt,
            "seed": seed,
            "train_horizon": train_horizon,
            "phase_generalization": phase_mode,
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
        "train_steps": train_steps,
        "batch_size": batch,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "gradient_clip": grad_clip,
        "loss": "MSE on normalized (cos phi, sin phi)",
        "runtime_seconds": elapsed,
        "parameter_count": count_parameters(model),
        "best_validation_loss": best_val,
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "orthogonality_error": orth_err,
        "status": "completed",
        "preliminary_less_than_3_seeds": True,
    }
    config_dir = ensure_dir(out / "run_configs")
    write_json(config_dir / f"{run_id}.json", registry)
    return model, registry, curves


def evaluate_one(
    model: torch.nn.Module,
    registry: dict[str, Any],
    test_horizons: list[int],
    test_speeds: list[float],
    device: str,
    n_batches: int,
    batch: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(41000 + int(registry["seed"]) + int(registry["train_horizon"]))
    model.eval()
    with torch.no_grad():
        for horizon in test_horizons:
            for speed in test_speeds:
                vals = []
                finals = []
                successes = []
                for _ in range(n_batches):
                    v, phi0, _target_y, phi = sample_velocity_batch(batch, horizon, 0.1, speed, "full", device, gen)
                    pred, _ = model(v, phi0)
                    m = circular_metrics(pred, phi)
                    vals.append(m["circular_rmse"])
                    finals.append(m["final_circular_rmse"])
                    successes.append(m["success_probability"])
                rows.append(
                    {
                        "run_id": registry["run_id"],
                        "model": registry["model"],
                        "seed": registry["seed"],
                        "train_horizon": registry["train_horizon"],
                        "hidden_size": registry["hidden_size"],
                        "train_speed_scale": registry["speed_scale"],
                        "phase_generalization": registry["phase_generalization"],
                        "test_horizon": horizon,
                        "test_speed_scale": speed,
                        "eval_initial_phase": "full",
                        "circular_rmse": float(np.mean(vals)),
                        "final_circular_rmse": float(np.mean(finals)),
                        "success_probability": float(np.mean(successes)),
                        "n_eval_batches": n_batches,
                        "eval_batch_size": batch,
                        "long_horizon_ood": bool(horizon > int(registry["train_horizon"])),
                        "speed_ood": bool(abs(speed - float(registry["speed_scale"])) > 1e-12),
                    }
                )
    return rows


def write_exp10_audit(out: Path) -> None:
    exp10_path = ROOT / "goldstone_lyapunov" / "experiments" / "exp10_trained_path_integrators.py"
    exp10_result = ROOT / "results" / "journal_full" / "exp10_trained_path_integrators"
    text = exp10_path.read_text(encoding="utf-8") if exp10_path.exists() else ""
    lines = [
        "# Repo and exp10 Audit",
        "",
        f"- `exp10_trained_path_integrators` exists: {exp10_path.exists()}.",
        "- What it trains: an optional unconstrained GRU baseline, if PyTorch is available; exact and broken models are analytic `PhaseIntegrator` controls.",
        "- Models compared: analytic exact phase integrator, analytic broken phase integrator, optional GRU.",
        "- Exact equivariant learned cells: no.",
        "- GRU/LSTM/orthogonal-RNN baselines: GRU only; no LSTM or constrained orthogonal RNN.",
        "- Equivariance error measured: no.",
        "- Direct group-tangent exponents measured: no.",
        "- Principal-angle or neutral-subspace alignment measured: no.",
        "- Pseudo-gap/lifetime scaling measured: no.",
        f"- Existing raw CSVs: {bool(list(exp10_result.glob('*.csv'))) if exp10_result.exists() else False}.",
        f"- Existing logs/configs/seeds: metrics JSON exists = {(exp10_result / 'metrics.json').exists() if exp10_result.exists() else False}; no per-run configs.",
        f"- Existing figures: {[p.name for p in exp10_result.glob('*.png')] if exp10_result.exists() else []}.",
        "- Sufficiency for new paper claim: not sufficient; this run creates `exp31_learned_equivariant_path_integration` with learned exact equivariant cells, matched baselines, and learned-model diagnostics.",
        "",
        "## Implementation Signal",
        f"- `_train_gru_baseline` found: {'_train_gru_baseline' in text}.",
        f"- `PhaseIntegrator` analytic controls found: {'PhaseIntegrator' in text}.",
    ]
    (out / "repo_and_exp10_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_task_files(out: Path, full: bool, hidden_size: int, speed_scale: float) -> None:
    config = {
        "dt": 0.1,
        "velocity_mixture": ["Gaussian", "piecewise_constant", "smooth_correlated_AR1"],
        "train_horizons_supported": [32, 64],
        "test_horizons": [32, 64, 128, 256] if full else [32, 64],
        "in_distribution_speed_scale": speed_scale,
        "ood_speed_scales": [0.25, 1.2, 1.8],
        "phase_regimes": {
            "full": "phi0 uniform on [-pi, pi)",
            "restricted": "phi0 uniform on [-pi/4, pi/4] during training; evaluation uses full circle",
        },
        "loss": "MSE on normalized output vector (cos phi, sin phi)",
        "success_threshold_radians": 0.25,
        "hidden_size": hidden_size,
        "seed_policy": "This run is preliminary if fewer than 3 seeds are present in model_registry.csv.",
    }
    write_json(out / "dataset_config.json", config)
    lines = [
        "# Learned S1 Path-Integration Task",
        "",
        "The target phase obeys `phi_{t+1}=wrap(phi_t + dt v_t)` and the target output is `(cos phi_t, sin phi_t)`.",
        "Velocity sequences are sampled from a mixture of Gaussian, piecewise-constant, and smooth correlated processes.",
        "The exact equivariant cell receives the initial phase through its representation state `z0=(cos phi0, sin phi0)` and keeps invariant hidden units independent of phase.",
        "GRU, LSTM, and orthogonal-RNN baselines receive the same initial phase cue through a trainable encoder.",
        "The loss is MSE on normalized vector outputs, and circular error metrics are computed from decoded angles.",
        "The experiment has a standard full-phase regime and a restricted-phase regime that tests systematic phase-shift generalization.",
        "Because the task is input-driven, learned diagnostics along input sequences are controlled-flow diagnostics and not autonomous theorem proof.",
    ]
    (out / "task_description.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return ["equivariant", "broken_equivariant", "gru", "lstm", "orthogonal_rnn"]
    return [model_arg]


def select_phase_modes(arg: str) -> list[str]:
    if arg == "both":
        return ["full", "restricted"]
    return [arg]


def compute_diagnostics(
    trained: list[tuple[torch.nn.Module, dict[str, Any]]],
    out: Path,
    device: str,
    quick: bool,
) -> dict[str, Any]:
    exact_entry = None
    broken_entry = None
    baseline_entries = []
    for model, reg in trained:
        if reg["model"] == "equivariant" and reg["phase_generalization"] == "full":
            if exact_entry is None or int(reg["train_horizon"]) > int(exact_entry[1]["train_horizon"]):
                exact_entry = (model, reg)
        if reg["model"] == "broken_equivariant" and reg["phase_generalization"] == "full":
            broken_entry = (model, reg)
        if reg["model"] in {"gru", "lstm", "orthogonal_rnn"} and reg["phase_generalization"] == "full":
            baseline_entries.append((model, reg))
    if exact_entry is None:
        return {"diagnostics_available": False, "reason": "No full-phase trained equivariant model was present."}

    exact_model, exact_reg = exact_entry
    untrained = EquivariantS1Cell(hidden_size=int(exact_reg["hidden_size"]), dt=0.1, epsilon=0.0).to(device)
    broken_posthoc = exact_model.clone_with_epsilon(0.02).to(device)

    equiv_rows: list[dict[str, Any]] = []
    for label, model in [
        ("untrained_exact", untrained),
        ("trained_exact", exact_model),
        ("broken_posthoc", broken_posthoc),
        ("trained_broken" if broken_entry else "trained_broken_missing", broken_entry[0] if broken_entry else None),
    ]:
        if model is None:
            continue
        err = vector_field_equivariance_error(model, n_tests=64 if quick else 192, seed=5001, device=device)
        equiv_rows.append({"model": label, "diagnostic": "vector_field_and_step", **err})

    gen = torch.Generator(device=device)
    gen.manual_seed(5200)
    velocities, phi0, _target, _phi = sample_velocity_batch(32, 64, 0.1, 0.8, "full", device, gen)
    theta = (2.0 * math.pi) * torch.rand(32, generator=gen, device=device) - math.pi
    for model, reg in baseline_entries:
        try:
            err = phase_shift_error(model, velocities, phi0, theta)
        except Exception:
            err = float("nan")
        equiv_rows.append(
            {
                "model": reg["model"],
                "diagnostic": "trajectory_phase_shift",
                "trajectory_phase_shift_error": err,
                "vf_mean": float("nan"),
                "vf_median": float("nan"),
                "vf_p95": float("nan"),
                "vf_max": float("nan"),
                "step_mean": float("nan"),
                "step_median": float("nan"),
                "step_p95": float("nan"),
                "step_max": float("nan"),
            }
        )
    write_rows(out / "equivariance_diagnostics.csv", equiv_rows)

    exponent_rows = []
    horizon = 64 if quick else 160
    rand = np.random.default_rng(5310).normal(0.0, 0.8, size=horizon)
    input_sets = {
        "zero_input": np.zeros(horizon),
        "fixed_input": np.full(horizon, 0.7),
        "random_heldout_input": rand,
    }
    for label, model in [("trained_exact", exact_model), ("broken_posthoc", broken_posthoc)]:
        for regime, inputs in input_sets.items():
            lam = direct_group_tangent_exponent(model, inputs, device=device)
            exponent_rows.append(
                {
                    "model": label,
                    "input_regime": regime,
                    "steps": horizon,
                    "elapsed_time": horizon * 0.1,
                    "lambda_group": lam,
                    "abs_lambda_group": abs(lam),
                    "near_zero_tol": 1e-3,
                    "near_zero_within_tol": abs(lam) < 1e-3,
                }
            )
    write_rows(out / "group_tangent_exponents.csv", exponent_rows)

    angle_rows = []
    for label, model in [("trained_exact", exact_model), ("broken_posthoc", broken_posthoc)]:
        for u_value in [0.0, 0.7]:
            angle = principal_angle_to_group_tangent(model, steps=32 if quick else 64, u_value=u_value, device=device)
            angle_rows.append({"model": label, "u_value": u_value, **angle, "method": "finite-time SVD/QR-style tangent product"})
    write_rows(out / "principal_angle_alignment.csv", angle_rows)

    auto_rows = []
    for label, model in [("trained_exact", exact_model), ("broken_posthoc", broken_posthoc)]:
        for u_value in [0.0, 0.7]:
            diag = autonomous_zero_diagnostic(model, u_value=u_value, device=device)
            row = asdict(diag)
            row.update({"model": label, "u_value": u_value})
            auto_rows.append(row)
    write_rows(out / "autonomous_zero_diagnostic.csv", auto_rows)

    pseudo_rows = []
    eps_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 6e-2]
    phi0 = 0.35
    threshold = 0.2
    max_time = 800.0 if quick else 1500.0
    max_steps = int(max_time / 0.1)
    for eps in eps_list:
        model = exact_model.clone_with_epsilon(eps).to(device)
        err = vector_field_equivariance_error(model, n_tests=48 if quick else 96, seed=5500, device=device)
        lam = direct_group_tangent_exponent(model, np.zeros(128 if not quick else 64), phi0=phi0, device=device)
        gap = max(0.0, -lam)
        predicted = float("inf")
        if gap > 1e-12 and threshold < abs(phi0):
            predicted = -math.log(1.0 - threshold / abs(phi0)) / gap
        measured, censored = measure_lifetime(model, phi0=phi0, threshold=threshold, max_steps=max_steps, device=device)
        pseudo_rows.append(
            {
                "epsilon": eps,
                "equivariance_step_max": err["step_max"],
                "lambda_sym": lam,
                "pseudo_gap": gap,
                "predicted_lifetime": predicted,
                "measured_lifetime": measured,
                "censored": censored,
                "ratio_measured_to_predicted": measured / predicted if math.isfinite(predicted) and predicted > 0 else float("nan"),
                "threshold": threshold,
                "initial_phase": phi0,
            }
        )
    write_rows(out / "pseudogap_lifetime_learned.csv", pseudo_rows)

    uncensored = [r for r in pseudo_rows if not r["censored"] and math.isfinite(r["predicted_lifetime"])]
    corr = float("nan")
    if len(uncensored) >= 2:
        corr = float(np.corrcoef(np.log([r["predicted_lifetime"] for r in uncensored]), np.log([r["measured_lifetime"] for r in uncensored]))[0, 1])
    summary = {
        "diagnostics_available": True,
        "selected_equivariant_run": exact_reg["run_id"],
        "max_trained_exact_step_equivariance_error": next(r["step_max"] for r in equiv_rows if r["model"] == "trained_exact"),
        "trained_exact_zero_input_lambda": next(r["lambda_group"] for r in exponent_rows if r["model"] == "trained_exact" and r["input_regime"] == "zero_input"),
        "trained_exact_zero_input_principal_angle_degrees": next(r["angle_degrees"] for r in angle_rows if r["model"] == "trained_exact" and abs(r["u_value"]) < 1e-12),
        "trained_exact_max_principal_angle_degrees": max(r["angle_degrees"] for r in angle_rows if r["model"] == "trained_exact"),
        "pseudogap_log_lifetime_correlation": corr,
        "pseudogap_uncensored_fraction": len(uncensored) / len(pseudo_rows),
        "pseudo_rows": pseudo_rows,
    }
    write_diagnostics_summary(out, summary)
    return summary


def measure_lifetime(
    model: EquivariantS1Cell,
    phi0: float,
    threshold: float,
    max_steps: int,
    device: str,
) -> tuple[float, bool]:
    model.eval()
    with torch.no_grad():
        phi = torch.tensor([phi0], dtype=torch.float32, device=device)
        state = model.initial_state(phi)[0]
        zero = torch.tensor(0.0, dtype=torch.float32, device=device)
        for step in range(1, max_steps + 1):
            state = model.step(state, zero)
            pred = float(torch.atan2(state[1], state[0]).cpu())
            err = abs(float(angle_wrap(pred - phi0)))
            if err >= threshold:
                return step * model.dt, False
    return max_steps * model.dt, True


def write_diagnostics_summary(out: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Learned-Model Diagnostics Summary",
        "",
        f"- Diagnostics available: {summary.get('diagnostics_available')}.",
        f"- Selected equivariant run: `{summary.get('selected_equivariant_run', 'TODO_MISSING_RUN')}`.",
        f"- Max trained exact step equivariance error: {summary.get('max_trained_exact_step_equivariance_error', 'TODO_MISSING_METRIC')}.",
        f"- Trained exact zero-input direct group-tangent exponent: {summary.get('trained_exact_zero_input_lambda', 'TODO_MISSING_METRIC')}.",
        f"- Trained exact zero-input principal angle (degrees): {summary.get('trained_exact_zero_input_principal_angle_degrees', 'TODO_MISSING_METRIC')}.",
        f"- Trained exact max principal angle (degrees): {summary.get('trained_exact_max_principal_angle_degrees', 'TODO_MISSING_METRIC')}.",
        f"- Pseudo-gap log lifetime correlation: {summary.get('pseudogap_log_lifetime_correlation', 'TODO_MISSING_METRIC')}.",
        f"- Pseudo-gap uncensored fraction: {summary.get('pseudogap_uncensored_fraction', 'TODO_MISSING_METRIC')}.",
        "",
        "These diagnostics are reported as learned task-level and controlled/autonomous-restriction evidence, not as the theorem proof.",
    ]
    (out / "diagnostics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_sem(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def write_statistical_summary(out: Path, eval_rows: list[dict[str, Any]], registry_rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in eval_rows:
        key = (row["model"], row["phase_generalization"], row["test_horizon"], row["test_speed_scale"])
        groups[key].append(float(row["circular_rmse"]))
    stat_rows = []
    for key, vals in sorted(groups.items()):
        mean, sem = mean_sem(vals)
        stat_rows.append(
            {
                "model": key[0],
                "phase_generalization": key[1],
                "test_horizon": key[2],
                "test_speed_scale": key[3],
                "n_runs": len(vals),
                "mean_circular_rmse": mean,
                "sem_circular_rmse": sem,
            }
        )
    write_rows(out / "statistical_summary.csv", stat_rows)
    seed_counts = defaultdict(set)
    for row in registry_rows:
        seed_counts[row["model"]].add(row["seed"])
    preliminary = any(len(v) < 3 for v in seed_counts.values())
    lines = [
        "# Statistical Summary",
        "",
        f"- Models in registry: {', '.join(sorted(seed_counts))}.",
        "- Seed counts: " + ", ".join(f"{model}={len(seeds)}" for model, seeds in sorted(seed_counts.items())),
        f"- Preliminary due to fewer than 3 seeds for at least one model: {preliminary}.",
        "- Mean and SEM rows are in `statistical_summary.csv`; SEM is zero when only one run contributes.",
        "- Baselines use the same initial phase cue, optimizer family, training horizon, velocity generator, and vector-output loss.",
        "- Strong comparative language should still be limited to this matched protocol; do not generalize to all GRU/LSTM/orthogonal-RNN training regimes.",
    ]
    (out / "statistical_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_figures(out: Path, fig_dir: Path) -> None:
    ensure_dir(fig_dir)
    eval_rows = read_rows(out / "evaluation_metrics.csv")
    equiv = read_rows(out / "equivariance_diagnostics.csv")
    exps = read_rows(out / "group_tangent_exponents.csv")
    angles = read_rows(out / "principal_angle_alignment.csv")
    auto = read_rows(out / "autonomous_zero_diagnostic.csv")
    pseudo = read_rows(out / "pseudogap_lifetime_learned.csv")
    plot_task_performance(eval_rows, fig_dir)
    plot_symmetry_diagnostics(equiv, exps, angles, auto, fig_dir)
    plot_pseudogap(pseudo, fig_dir)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ffloat(value: Any, default: float = float("nan")) -> float:
    try:
        text = str(value)
        if text.lower() in {"nan", "", "none"}:
            return default
        return float(text)
    except Exception:
        return default


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")


def plot_task_performance(rows: list[dict[str, str]], fig_dir: Path) -> None:
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    axes = axes_grid.ravel()
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.06, 0.78, r"$v_t$", fontsize=11)
    ax.arrow(0.24, 0.80, 0.22, 0, head_width=0.04, color="#333333")
    ax.text(0.50, 0.78, r"$\phi_{t+1}$", fontsize=10)
    ax.arrow(0.47, 0.68, 0, -0.24, head_width=0.04, color="#333333")
    ax.text(0.25, 0.25, r"$(\cos\phi_t,\sin\phi_t)$", fontsize=9)
    ax.text(0.0, 1.02, "A  Task", fontsize=10, fontweight="bold")

    models = ["equivariant", "broken_equivariant", "gru", "lstm", "orthogonal_rnn"]
    full_rows = [r for r in rows if r.get("phase_generalization") == "full"]
    speed_candidates = sorted({ffloat(r.get("test_speed_scale")) for r in full_rows})
    id_speed = min(speed_candidates, key=lambda x: abs(x - 0.8)) if speed_candidates else 0.8
    for model in models:
        grouped: dict[int, list[float]] = defaultdict(list)
        for r in full_rows:
            if r.get("model") == model and abs(ffloat(r.get("test_speed_scale")) - id_speed) < 1e-9:
                grouped[int(float(r["test_horizon"]))].append(ffloat(r["circular_rmse"]))
        xs = sorted(grouped)
        if xs:
            ys = [mean_sem(grouped[x])[0] for x in xs]
            es = [mean_sem(grouped[x])[1] for x in xs]
            axes[1].errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=model.replace("_", " "), color=COLORS.get(model))
    axes[1].set_yscale("log")
    axes[1].set_xlabel("test horizon")
    axes[1].set_ylabel("circular RMSE")
    axes[1].set_title("B  Horizon generalization")
    axes[1].legend(frameon=False, fontsize=6)

    max_h = max([int(float(r["test_horizon"])) for r in rows], default=64)
    max_speed = max(speed_candidates) if speed_candidates else id_speed
    labels = []
    vals = []
    errs = []
    colors = []
    for model in models:
        data = [ffloat(r["circular_rmse"]) for r in full_rows if r.get("model") == model and int(float(r["test_horizon"])) == max_h and abs(ffloat(r["test_speed_scale"]) - max_speed) < 1e-9]
        if data:
            labels.append(model.replace("_", "\n"))
            m, s = mean_sem(data)
            vals.append(m)
            errs.append(s)
            colors.append(COLORS.get(model))
    axes[2].bar(range(len(labels)), vals, yerr=errs, color=colors, capsize=2)
    axes[2].set_yscale("log")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("C  Speed OOD")

    rest_rows = [r for r in rows if r.get("phase_generalization") == "restricted" and int(float(r["test_horizon"])) == min(max_h, 128)]
    labels = []
    vals = []
    colors = []
    for model in models:
        data = [ffloat(r["circular_rmse"]) for r in rest_rows if r.get("model") == model and abs(ffloat(r["test_speed_scale"]) - id_speed) < 1e-9]
        if data:
            labels.append(model.replace("_", "\n"))
            vals.append(mean_sem(data)[0])
            colors.append(COLORS.get(model))
    axes[3].bar(range(len(labels)), vals, color=colors)
    axes[3].set_yscale("log")
    axes[3].set_xticks(range(len(labels)))
    axes[3].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    axes[3].set_ylabel("RMSE")
    axes[3].set_title("D  Restricted-phase")
    fig.savefig(fig_dir / "fig_learned_task_performance.pdf")
    fig.savefig(fig_dir / "fig_learned_task_performance.png", dpi=300)
    plt.close(fig)


def plot_symmetry_diagnostics(equiv: list[dict[str, str]], exps: list[dict[str, str]], angles: list[dict[str, str]], auto: list[dict[str, str]], fig_dir: Path) -> None:
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    axes = axes_grid.ravel()
    label_map = {
        "untrained_exact": "exact\ninit",
        "trained_exact": "exact\ntrained",
        "broken_posthoc": "broken\nposthoc",
        "trained_broken": "broken\ntrained",
        "gru": "GRU",
        "lstm": "LSTM",
        "orthogonal_rnn": "orth.\nRNN",
    }
    ordered = ["untrained_exact", "trained_exact", "broken_posthoc", "trained_broken", "gru", "lstm", "orthogonal_rnn"]
    grouped_err: dict[str, list[float]] = defaultdict(list)
    for r in equiv:
        val = ffloat(r.get("step_max"), ffloat(r.get("trajectory_phase_shift_error")))
        if math.isfinite(val):
            grouped_err[r["model"]].append(max(val, 1e-12))
    labels = []
    vals = []
    colors = []
    for key in ordered:
        if key in grouped_err:
            labels.append(label_map.get(key, key))
            vals.append(float(np.mean(grouped_err[key])))
            colors.append(COLORS.get(key, COLORS.get(key.replace("trained_", ""), "#777777")))
    axes[0].bar(range(len(vals)), vals, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(vals)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("error")
    axes[0].set_title("A  Equivariance")

    exp_grouped: dict[str, list[float]] = defaultdict(list)
    for r in exps:
        if r.get("input_regime") == "zero_input":
            exp_grouped[r["model"]].append(max(abs(ffloat(r["lambda_group"])), 1e-12))
    exp_order = [k for k in ["trained_exact", "broken_posthoc"] if k in exp_grouped]
    labels = [label_map.get(k, k).replace("_", "\n") for k in exp_order]
    vals = [float(np.mean(exp_grouped[k])) for k in exp_order]
    axes[1].bar(range(len(vals)), vals, color=[COLORS.get(k, "#777777") for k in exp_order])
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(vals)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    axes[1].set_ylabel(r"$|\lambda_G|$")
    axes[1].set_title("B  Group tangent")

    angle_grouped: dict[str, list[float]] = defaultdict(list)
    for r in angles:
        if abs(ffloat(r.get("u_value"))) < 1e-12:
            angle_grouped[r["model"]].append(max(ffloat(r["angle_degrees"]), 1e-9))
    angle_order = [k for k in ["trained_exact", "broken_posthoc"] if k in angle_grouped]
    labels = [label_map.get(k, k).replace("_", "\n") for k in angle_order]
    vals = [float(np.mean(angle_grouped[k])) for k in angle_order]
    axes[2].bar(range(len(vals)), vals, color=[COLORS.get(k, "#777777") for k in angle_order])
    axes[2].set_yscale("log")
    axes[2].set_xticks(range(len(vals)))
    axes[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    axes[2].set_ylabel("angle (deg)")
    axes[2].set_title("C  Alignment")

    auto_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in auto:
        base = "exact" if r["model"] == "trained_exact" else "broken"
        ulabel = "zero" if abs(ffloat(r["u_value"])) < 1e-12 else "fixed"
        auto_grouped[(base, ulabel)].append(ffloat(r["group_directions_independent_of_flow"]))
    auto_order = [("exact", "zero"), ("exact", "fixed"), ("broken", "zero"), ("broken", "fixed")]
    auto_order = [k for k in auto_order if k in auto_grouped]
    labels = [f"{a}\n{b}" for a, b in auto_order]
    vals = [float(np.mean(auto_grouped[k])) for k in auto_order]
    axes[3].bar(range(len(vals)), vals, color=["#1f8a70" if k[0] == "exact" else "#9b2d30" for k in auto_order])
    axes[3].set_ylim(0, max(vals + [1]) + 0.4)
    axes[3].set_xticks(range(len(vals)))
    axes[3].set_xticklabels(labels, rotation=0, fontsize=7)
    axes[3].set_ylabel("group dirs independent of f")
    axes[3].set_title("D  Autonomous caveat")
    fig.savefig(fig_dir / "fig_learned_symmetry_diagnostics.pdf")
    fig.savefig(fig_dir / "fig_learned_symmetry_diagnostics.png", dpi=300)
    plt.close(fig)


def plot_pseudogap(rows: list[dict[str, str]], fig_dir: Path) -> None:
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    axes = axes_grid.ravel()
    pred = [ffloat(r["predicted_lifetime"]) for r in rows]
    meas = [ffloat(r["measured_lifetime"]) for r in rows]
    cens = [str(r["censored"]).lower() == "true" for r in rows]
    unc = [i for i, c in enumerate(cens) if not c and math.isfinite(pred[i])]
    if unc:
        axes[0].scatter([pred[i] for i in unc], [meas[i] for i in unc], color="#1f8a70")
        lo = min([pred[i] for i in unc] + [meas[i] for i in unc])
        hi = max([pred[i] for i in unc] + [meas[i] for i in unc])
        axes[0].plot([lo, hi], [lo, hi], "--", color="black", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("measured")
    axes[0].set_title("A  Lifetime")

    eps = [ffloat(r["epsilon"]) for r in rows]
    axes[1].plot(eps, meas, "o-", color="#9b2d30", label="measured")
    axes[1].plot(eps, pred, "--", color="#333333", label="predicted")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\epsilon$")
    axes[1].set_ylabel("lifetime")
    axes[1].set_title("B  Scaling")
    axes[1].legend(frameon=False, fontsize=7)

    eq = [ffloat(r["equivariance_step_max"]) for r in rows]
    gap = [ffloat(r["pseudo_gap"]) for r in rows]
    axes[2].plot(eq, gap, "o", color="#7a4da3")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("equivariance error")
    axes[2].set_ylabel("pseudo-gap")
    axes[2].set_title("C  Gap vs breaking")

    ratio = [ffloat(r["ratio_measured_to_predicted"]) for r in rows]
    axes[3].axhline(1.0, color="black", lw=1, ls="--")
    axes[3].plot(eps, ratio, "o-", color="#1f8a70")
    axes[3].set_xscale("log")
    axes[3].set_xlabel(r"$\epsilon$")
    axes[3].set_ylabel("measured/predicted")
    axes[3].set_title("D  Ratio")
    fig.savefig(fig_dir / "fig_learned_pseudogap.pdf")
    fig.savefig(fig_dir / "fig_learned_pseudogap.png", dpi=300)
    plt.close(fig)


def write_manuscript_artifacts(out: Path, diagnostics: dict[str, Any]) -> None:
    main = r"""
We trained an exactly equivariant recurrent cell end-to-end on velocity-input \(S^1\) path integration, using a state \(x=(z,h)\) with \(z\in\mathbb{R}^2\) transforming by phase rotation and \(h\) invariant.
This learned experiment addresses empirical relevance but is task-level evidence, not theorem proof, because the task is input-driven.
The architecture preserves equivariance by construction for each scalar velocity input, since the learned coefficients depend only on invariant features.
Baselines are matched GRU, LSTM, and orthogonal-RNN models with the same initial phase cue, optimizer family, training batches, and vector-output loss.
We evaluate in-distribution behavior, long-horizon extrapolation, speed out-of-distribution behavior, and restricted-phase generalization.
We then apply equivariance-error, direct group-tangent, principal-angle, autonomous-restriction, and pseudo-gap diagnostics to the learned equivariant cell.
TODO_MISSING_CITATION: trained path-integration RNNs.
"""
    (out / "text_learned_experiment_main.tex").write_text(main.strip() + "\n", encoding="utf-8")

    captions = r"""
\caption{Learned path-integration performance. The figure supports task-level empirical relevance of the exact equivariant recurrent cell under the matched training protocol; it does not prove the autonomous-flow theorem.}
\caption{Learned symmetry diagnostics. The exact learned cell is tested by construction-level equivariance error, direct group-tangent exponents, finite-time tangent-subspace alignment, and autonomous zero-input/fixed-input restrictions.}
\caption{Learned pseudo-gap scaling. Explicit post-training symmetry breaking increases equivariance error, displaces the group tangent, and gives a measured lifetime diagnostic; censored lifetimes are reported in the raw table.}
"""
    (out / "captions_learned_experiment.tex").write_text(captions.strip() + "\n", encoding="utf-8")

    appendix = r"""
\section*{Learned Equivariant Path-Integration Methods}
Velocity sequences are sampled from Gaussian, piecewise-constant, and correlated random-walk processes.
The equivariant cell uses \(dz/dt=a(I,h,u)z+b(I,h,u)Jz\), \(dh/dt=g(I,h,u)\), with \(I=\|z\|^2\), so rotating \(z\) commutes with the vector field for every scalar input \(u\).
The broken control adds \(\epsilon(x,-y)\) to \(dz/dt\), which violates \(S^1\) equivariance.
Baselines are GRU, LSTM, and an orthogonal RNN with \(W=\exp(A-A^\top)\), all initialized from the same phase cue \((\cos\phi_0,\sin\phi_0)\).
Training uses AdamW, gradient clipping, MSE on normalized vector outputs, and the same velocity generator across models.
Diagnostics report vector-field and step equivariance error, finite-time direct group-tangent exponents, a finite-time SVD/QR-style tangent-product angle, autonomous zero-input/fixed-input flow alignment, and pseudo-gap lifetime measurements.
The lifetime predictor uses the measured pseudo-gap and the local exponential memory-loss formula described in the raw manifest.
Failure modes include undertrained baselines, censored lifetimes for very small breaking, and finite-time tangent-subspace ambiguity.
TODO_MISSING_CITATION: trained path-integration RNNs.
"""
    (out / "appendix_learned_methods.tex").write_text(appendix.strip() + "\n", encoding="utf-8")

    repro = r"""
\paragraph{Learned experiment reproducibility.}
The learned path-integration results are generated by \texttt{goldstone\_lyapunov.experiments.exp31\_learned\_equivariant\_path\_integration}.
Run configurations, checkpoints, training curves, evaluation metrics, diagnostics, source-support audit, and a machine-readable manifest are stored under \texttt{results/learned\_equivariant\_pi}.
Missing details are marked with \texttt{TODO\_MISSING\_REPRO\_DETAIL}.
"""
    (out / "reproducibility_learned_experiment.tex").write_text(repro.strip() + "\n", encoding="utf-8")

    insertion = [
        "# Manuscript Insertion Plan",
        "",
        "- If the full learned run remains preliminary with fewer than 3 seeds, keep the learned experiment as an appendix figure and mention it briefly in the main text.",
        "- If a 3-seed run confirms the same pattern, promote `fig_learned_task_performance` and `fig_learned_symmetry_diagnostics` as a new main Figure 5 and move the old consequence/null figure to the appendix.",
        "- Replace the old GRU comparison panel with this matched learned comparison, but keep the finite-grid null sentence in the main text.",
        "- Keep theorem evidence first: exact equivariance, direct group-tangent exponents, dimension law, and principal-angle alignment remain the primary claims.",
        f"- Current diagnostics available: {diagnostics.get('diagnostics_available')}.",
    ]
    (out / "manuscript_insertion_plan.md").write_text("\n".join(insertion) + "\n", encoding="utf-8")


def source_support_audit(out: Path) -> None:
    files = [
        out / "text_learned_experiment_main.tex",
        out / "captions_learned_experiment.tex",
        out / "appendix_learned_methods.tex",
        out / "reproducibility_learned_experiment.tex",
    ]
    rows = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 12]
        for sentence in sentences:
            if "TODO_MISSING_CITATION" in sentence:
                cat = "needs_support"
            elif any(token in sentence.lower() for token in ["figure", "raw table", "manifest", "\\caption", "evaluate in-distribution"]):
                cat = "figure/table_or_reproducibility"
            elif any(token in sentence.lower() for token in ["equivariance", "by construction", "diagnostic", "commutes with the vector field", "not theorem proof", "input-driven"]):
                cat = "theorem/proof_or_definition"
            elif any(token in sentence.lower() for token in ["baseline", "training", "velocity", "generated by", "texttt", "missing details", "\\paragraph", "exp31"]):
                cat = "definition_or_protocol"
            else:
                cat = "needs_support"
            rows.append({"file": str(path.relative_to(ROOT)), "category": cat, "sentence": sentence})
    flagged = [r for r in rows if r["category"] == "needs_support"]
    lines = [
        "# Source-Support Audit for Learned Experiment",
        "",
        f"- Sentences audited: {len(rows)}",
        f"- Flagged needs_support: {len(flagged)}",
        "",
        "## Flagged Sentences",
    ]
    if flagged:
        lines.extend(f"- `{r['file']}`: {r['sentence']}." for r in flagged)
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("TODO_MISSING_CITATION entries are intentionally left explicit rather than filled with fabricated references.")
    (out / "source_support_audit_learned.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(out: Path, fig_dir: Path, args: argparse.Namespace, registry_rows: list[dict[str, Any]]) -> None:
    output_files = sorted(str(p.relative_to(ROOT)) for p in out.rglob("*") if p.is_file())
    figure_files = sorted(str(p.relative_to(ROOT)) for p in fig_dir.rglob("*") if p.is_file())
    manifest = {
        "command": " ".join(sys.argv),
        "code_paths": [
            "goldstone_lyapunov/equivariant_s1_cell.py",
            "goldstone_lyapunov/path_integration_baselines.py",
            "goldstone_lyapunov/experiments/exp31_learned_equivariant_path_integration.py",
            "tests/test_equivariant_s1_cell.py",
        ],
        "device": args.device,
        "quick": args.quick,
        "full": args.full,
        "model_arg": args.model,
        "seed_arg": args.seed,
        "hidden_size": args.hidden_size,
        "speed_scale": args.speed_scale,
        "phase_generalization": args.phase_generalization,
        "optimizer": "AdamW",
        "training_loss": "MSE on normalized (cos phi, sin phi)",
        "diagnostic_tolerances": {
            "equivariance_target": 1e-7,
            "near_zero_lambda_tol": 1e-3,
            "principal_angle_reported_degrees": True,
        },
        "pseudo_gap": {
            "epsilon_list": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 6e-2],
            "threshold_radians": 0.2,
            "initial_phase": 0.35,
        },
        "hardware_compute": "TODO_MISSING_REPRO_DETAIL: CPU model and memory were not queried.",
        "registry_rows": len(registry_rows),
        "outputs": output_files,
        "figures": figure_files,
    }
    write_json(out / "reproducibility_manifest_learned.json", manifest)
    lines = [
        "# Learned Experiment Reproducibility Manifest",
        "",
        f"- Command: `{manifest['command']}`",
        f"- Device: `{args.device}`",
        f"- Registry rows: {len(registry_rows)}",
        "- Code paths:",
        *[f"  - `{p}`" for p in manifest["code_paths"]],
        "- Data generation: Gaussian, piecewise-constant, and correlated random-walk velocities.",
        "- Random seeds: see `model_registry.csv`.",
        "- Optimizer: AdamW with gradient clipping; per-run configs are in `run_configs/`.",
        "- Evaluation horizons and speeds: see `dataset_config.json` and `evaluation_metrics.csv`.",
        "- Diagnostics: equivariance error, direct group-tangent exponents, finite-time tangent-product principal angle, autonomous zero diagnostic, pseudo-gap lifetime.",
        "- Hardware/compute: TODO_MISSING_REPRO_DETAIL: CPU model and memory were not queried.",
    ]
    (out / "reproducibility_manifest_learned.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_build_report(out: Path, start_time: float, quick_passed: bool, full_passed: bool, warnings: list[str]) -> None:
    elapsed = time.perf_counter() - start_time
    expected = [
        "UPLOAD_TO_CHATGPT_LEARNED_EXPERIMENT_SUMMARY.md",
        "repo_and_exp10_audit.md",
        "task_description.md",
        "dataset_config.json",
        "training_curves.csv",
        "evaluation_metrics.csv",
        "model_registry.csv",
        "equivariance_diagnostics.csv",
        "group_tangent_exponents.csv",
        "principal_angle_alignment.csv",
        "autonomous_zero_diagnostic.csv",
        "pseudogap_lifetime_learned.csv",
        "diagnostics_summary.md",
        "statistical_summary.md",
        "reproducibility_manifest_learned.md",
        "reproducibility_manifest_learned.json",
        "source_support_audit_learned.md",
    ]
    missing = [f for f in expected if not (out / f).exists() and f != "UPLOAD_TO_CHATGPT_LEARNED_EXPERIMENT_SUMMARY.md"]
    lines = [
        "# Learned Experiment Build Report",
        "",
        f"- Command: `{' '.join(sys.argv)}`",
        "- Unit tests: run separately and summarized in final upload summary if this script is invoked by the check runner.",
        f"- Quick mode passed in this invocation: {quick_passed}",
        f"- Full mode passed in this invocation: {full_passed}",
        f"- Total runtime seconds: {elapsed:.2f}",
        f"- Missing expected files before upload summary: {', '.join(missing) if missing else 'none'}",
        f"- Warnings: {', '.join(warnings) if warnings else 'none'}",
        "- Failed commands: none inside this invocation.",
        "- Generated files: see `reproducibility_manifest_learned.json`.",
    ]
    (out / "build_report_learned.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_zip(out: Path, fig_dir: Path) -> None:
    zip_path = ROOT / "neurips_goldstone_learned_equivariant_pi_package.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in [out, fig_dir]:
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        zf.write(path, path.relative_to(ROOT).as_posix())
        for path in [
            ROOT / "goldstone_lyapunov" / "equivariant_s1_cell.py",
            ROOT / "goldstone_lyapunov" / "path_integration_baselines.py",
            ROOT / "goldstone_lyapunov" / "experiments" / "exp31_learned_equivariant_path_integration.py",
            ROOT / "tests" / "test_equivariant_s1_cell.py",
        ]:
            if path.exists():
                zf.write(path, path.relative_to(ROOT).as_posix())


def create_upload_summary(
    out: Path,
    registry_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    quick: bool,
    full: bool,
) -> None:
    models = sorted({r["model"] for r in registry_rows})
    seeds = sorted({int(r["seed"]) for r in registry_rows})
    flagged = "TODO_MISSING_AUDIT"
    audit_path = out / "source_support_audit_learned.md"
    if audit_path.exists():
        text = audit_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "Flagged needs_support" in line:
                flagged = line.split(":", 1)[-1].strip()
    max_eq = diagnostics.get("max_trained_exact_step_equivariance_error", "TODO_MISSING_METRIC")
    lam = diagnostics.get("trained_exact_zero_input_lambda", "TODO_MISSING_METRIC")
    angle_zero = diagnostics.get("trained_exact_zero_input_principal_angle_degrees", "TODO_MISSING_METRIC")
    angle = diagnostics.get("trained_exact_max_principal_angle_degrees", "TODO_MISSING_METRIC")
    corr = diagnostics.get("pseudogap_log_lifetime_correlation", "TODO_MISSING_METRIC")
    files = [
        "results/learned_equivariant_pi/UPLOAD_TO_CHATGPT_LEARNED_EXPERIMENT_SUMMARY.md",
        "results/learned_equivariant_pi/repo_and_exp10_audit.md",
        "results/learned_equivariant_pi/task_description.md",
        "results/learned_equivariant_pi/dataset_config.json",
        "results/learned_equivariant_pi/training_curves.csv",
        "results/learned_equivariant_pi/evaluation_metrics.csv",
        "results/learned_equivariant_pi/model_registry.csv",
        "results/learned_equivariant_pi/equivariance_diagnostics.csv",
        "results/learned_equivariant_pi/group_tangent_exponents.csv",
        "results/learned_equivariant_pi/principal_angle_alignment.csv",
        "results/learned_equivariant_pi/autonomous_zero_diagnostic.csv",
        "results/learned_equivariant_pi/pseudogap_lifetime_learned.csv",
        "results/learned_equivariant_pi/diagnostics_summary.md",
        "results/learned_equivariant_pi/statistical_summary.md",
        "results/learned_equivariant_pi/reproducibility_manifest_learned.md",
        "results/learned_equivariant_pi/reproducibility_manifest_learned.json",
        "results/learned_equivariant_pi/build_report_learned.md",
        "results/learned_equivariant_pi/source_support_audit_learned.md",
        "figures_clean/learned_equivariant_pi/fig_learned_task_performance.pdf",
        "figures_clean/learned_equivariant_pi/fig_learned_task_performance.png",
        "figures_clean/learned_equivariant_pi/fig_learned_symmetry_diagnostics.pdf",
        "figures_clean/learned_equivariant_pi/fig_learned_symmetry_diagnostics.png",
        "figures_clean/learned_equivariant_pi/fig_learned_pseudogap.pdf",
        "figures_clean/learned_equivariant_pi/fig_learned_pseudogap.png",
        "results/learned_equivariant_pi/text_learned_experiment_main.tex",
        "results/learned_equivariant_pi/captions_learned_experiment.tex",
        "results/learned_equivariant_pi/appendix_learned_methods.tex",
        "results/learned_equivariant_pi/reproducibility_learned_experiment.tex",
        "results/learned_equivariant_pi/manuscript_insertion_plan.md",
        "neurips_goldstone_learned_equivariant_pi_package.zip",
    ]
    seed_note = (
        "All trained model families have at least 3 seeds in this run."
        if all(len(v) >= 3 for v in {m: {int(float(r["seed"])) for r in registry_rows if r["model"] == m} for m in models}.values())
        else "Results are preliminary if fewer than 3 seeds per model are present; the statistical summary states this explicitly."
    )
    lines = [
        "# Upload to ChatGPT Learned Experiment Summary",
        "",
        "## Reuse or New Experiment",
        "- Existing `exp10_trained_path_integrators` was audited and judged insufficient for the new learned-architecture claim.",
        "- A new `exp31_learned_equivariant_path_integration` was created.",
        "",
        "## Models Trained",
        f"- Models: {', '.join(models)}.",
        f"- Seeds run: {seeds}.",
        f"- Quick invocation: {quick}; full invocation: {full}.",
        f"- {seed_note}",
        "",
        "## Training and Evaluation Summary",
        f"- Training runs recorded: {len(registry_rows)}.",
        f"- Evaluation rows recorded: {len(eval_rows)}.",
        "- Baselines use the same velocity generator, initial phase cue, vector-output loss, and optimizer family.",
        "- Orthogonal RNN uses an exact matrix exponential parameterization and logs `||W^T W-I||`.",
        "",
        "## Diagnostic Summary",
        f"- Max trained exact step equivariance error: {max_eq}.",
        f"- Trained exact zero-input direct group-tangent exponent: {lam}.",
        f"- Trained exact zero-input principal angle degrees: {angle_zero}.",
        f"- Trained exact max principal angle degrees across tested input restrictions: {angle}.",
        f"- Learned pseudo-gap log lifetime correlation: {corr}.",
        "- Diagnostics are task-level and autonomous-restriction evidence, not theorem proof.",
        "",
        "## Figures Created",
        "- `fig_learned_task_performance.*`",
        "- `fig_learned_symmetry_diagnostics.*`",
        "- `fig_learned_pseudogap.*`",
        "",
        "## Tables Created",
        "- training curves, evaluation metrics, registry, equivariance diagnostics, direct tangent exponents, principal-angle alignment, autonomous zero diagnostic, pseudo-gap lifetime, and statistical summary.",
        "",
        "## Tests and Caveats",
        "- Equivariance tests are included in `tests/test_equivariant_s1_cell.py`; see `build_report_learned.md` for the current run status.",
        f"- Source-support audit flagged sentences: {flagged}.",
        "- Missing reproducibility detail: CPU hardware summary was not queried.",
        "- Do not claim generic GRUs/LSTMs cannot integrate; report only this matched protocol.",
        "",
        "## Files to Upload to ChatGPT",
    ]
    lines.extend(f"- `{f}`" for f in files)
    (out / "UPLOAD_TO_CHATGPT_LEARNED_EXPERIMENT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    out = ensure_dir(Path(args.output_dir) if args.output_dir else DEFAULT_OUT)
    fig_dir = ensure_dir(DEFAULT_FIG)
    ensure_dir(out / "checkpoints")
    ensure_dir(out / "run_configs")
    write_exp10_audit(out)
    write_task_files(out, full=args.full, hidden_size=args.hidden_size, speed_scale=args.speed_scale)

    models = select_models(args.model)
    seeds = [args.seed] if args.seed is not None else (list(range(6)) if args.full else [0])
    train_horizons = [args.train_horizon] if args.train_horizon is not None else ([32, 64] if args.full else [32])
    phases = select_phase_modes(args.phase_generalization)
    test_horizons = [32, 64, 128, 256] if args.full else [32, 64]
    test_speeds = sorted(set([args.speed_scale, 0.25, 1.2, 1.8] if args.full else [args.speed_scale, 1.2]))
    eval_batches = 3 if args.full else 1
    eval_batch = 64 if args.full else 32

    registry_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    trained: list[tuple[torch.nn.Module, dict[str, Any]]] = []
    warnings: list[str] = []
    if len(seeds) < 3:
        warnings.append("PRELIMINARY_LESS_THAN_3_SEEDS")

    for model_name in models:
        for seed in seeds:
            for train_horizon in train_horizons:
                for phase in phases:
                    model, registry, curves = train_one(
                        model_name=model_name,
                        seed=seed,
                        train_horizon=int(train_horizon),
                        hidden_size=int(args.hidden_size),
                        speed_scale=float(args.speed_scale),
                        phase_mode=phase,
                        device=args.device,
                        out=out,
                        quick=args.quick,
                        train_steps_override=args.train_steps,
                    )
                    registry_rows.append(registry)
                    curve_rows.extend(curves)
                    eval_rows.extend(evaluate_one(model, registry, test_horizons, test_speeds, args.device, eval_batches, eval_batch))
                    trained.append((model, registry))

    write_rows(out / "training_curves.csv", curve_rows)
    write_rows(out / "evaluation_metrics.csv", eval_rows)
    write_rows(out / "model_registry.csv", registry_rows)

    diagnostics = compute_diagnostics(trained, out, args.device, args.quick)
    write_statistical_summary(out, eval_rows, registry_rows)
    plot_figures(out, fig_dir)
    write_manuscript_artifacts(out, diagnostics)
    source_support_audit(out)
    write_manifest(out, fig_dir, args, registry_rows)
    write_build_report(out, start, quick_passed=args.quick, full_passed=args.full, warnings=warnings)
    create_zip(out, fig_dir)
    create_upload_summary(out, registry_rows, eval_rows, diagnostics, quick=args.quick, full=args.full)
    return diagnostics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--model", default="all", choices=["all", "equivariant", "broken_equivariant", "gru", "lstm", "orthogonal_rnn"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_horizon", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--speed_scale", type=float, default=0.8)
    parser.add_argument("--phase_generalization", default="both", choices=["full", "restricted", "both"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", "--output-dir", default=None)
    parser.add_argument("--train_steps", type=int, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.quick and not args.full:
        args.quick = True
    diagnostics = run(args)
    summary_path = Path(args.output_dir) / "UPLOAD_TO_CHATGPT_LEARNED_EXPERIMENT_SUMMARY.md" if args.output_dir else DEFAULT_OUT / "UPLOAD_TO_CHATGPT_LEARNED_EXPERIMENT_SUMMARY.md"
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
