"""Generate the figures and tables used by the current Goldstone-Lyapunov draft.

This folder is intentionally a thin, runnable asset pipeline.  It assumes the
folder sits inside the repository root and reads the frozen raw result tables in
``../results``.  Heavy numerical experiments are not rerun by default; their
archived full-run figures are copied with a manifest entry and the corresponding
rerun scripts are included in ``source_scripts``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS = REPO_ROOT / "results"
JOURNAL = RESULTS / "journal_full"
LEARNED = RESULTS / "learned_equivariant_pi"
ICLR = RESULTS / "iclr_final_checks"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MAIN_FIGURES = [
    "fig1_dimension_law_clean",
    "fig2_neutral_geometry_clean_v2",
    "fig3_equivariant_rnn_clean",
    "fig4_pseudogap_breaking_clean",
    "figure5_learned_path_integration",
]

APPENDIX_FIGURES = [
    "figure_A1_existing_training_curves",
    "figure_A2_stronger_baseline_training_curves",
    "figure_A3_stronger_baseline_eval",
    "figure_A4_parameter_matched_baselines",
    "figure_A5_task_schematic",
    "figure_A6_learned_symmetry_diagnostics",
    "figure_A7_learned_pseudogap",
    "figure_A8_consequence_null",
    "figure_A9_finite_time_chaos_diagnostics",
    "figure_A10_large_chaotic_spectra",
    "figure_A11_path_integration_heatmaps",
    "figure_A12_gru_path_integration_sweep",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ffloat(value: Any, default: float = float("nan")) -> float:
    try:
        text = str(value)
        if text.lower() in {"", "nan", "none"}:
            return default
        return float(text)
    except Exception:
        return default


def mean_sem(values: list[float]) -> tuple[float, float]:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(vals.size))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def copy_pair(src_stem: Path, dst_stem: Path) -> list[Path]:
    outputs: list[Path] = []
    for suffix in [".pdf", ".png"]:
        src = src_stem.with_suffix(suffix)
        if not src.exists() and suffix == ".pdf":
            continue
        if not src.exists():
            raise FileNotFoundError(src)
        dst = dst_stem.with_suffix(suffix)
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


def write_latex_table(path: Path, headers: list[str], rows: list[list[Any]], caption: str, label: str) -> None:
    ensure_dir(path.parent)
    colspec = "l" * len(headers)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        escaped = [str(x).replace("_", "\\_") for x in row]
        lines.append(" & ".join(escaped) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_main_theorem_figures(fig_dir: Path, support_dir: Path, manifest: list[dict[str, str]]) -> None:
    import paper_runs.conference_fix as cf
    import paper_runs.make_fig2_neutral_geometry_v2 as f2

    ensure_dir(fig_dir)
    ensure_dir(support_dir)
    cf.FIGS = fig_dir
    cf.FIX = support_dir
    cf.setup_plot_style()
    auto_rows = cf.generate_autonomous_zero_diagnostic()
    cf.plot_fig1_dimension_law()
    f2.FIG_DIR = fig_dir
    f2.make_fig2()
    cf.plot_fig3_equivariant_rnn()
    cf.plot_fig4_pseudogap()
    cf.plot_fig5_consequence_null()
    for stem in ["fig1_dimension_law_clean", "fig2_neutral_geometry_clean_v2", "fig3_equivariant_rnn_clean", "fig4_pseudogap_breaking_clean"]:
        manifest.append(
            {
                "asset": stem,
                "kind": "main_figure",
                "generator": "paper_runs/conference_fix.py and paper_runs/make_fig2_neutral_geometry_v2.py",
                "data": "results/journal_full raw CSV tables",
                "output": rel(fig_dir / f"{stem}.pdf"),
            }
        )
    manifest.append(
        {
            "asset": "figure_A8_consequence_null",
            "kind": "appendix_figure",
            "generator": "paper_runs/conference_fix.py::plot_fig5_consequence_null",
            "data": "results/journal_full/exp15_path_integration_benchmark and exp17_finite_grid_null",
            "output": rel(fig_dir / "figure_A8_consequence_null.pdf"),
        }
    )
    copy_pair(fig_dir / "fig5_consequence_null_clean", fig_dir / "figure_A8_consequence_null")
    if not auto_rows:
        raise RuntimeError("Autonomous zero diagnostic produced no rows.")


def generate_learned_figures(fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    from goldstone_lyapunov.experiments import exp31_learned_equivariant_path_integration as exp31

    learned_fig_dir = ensure_dir(fig_dir / "learned_equivariant_pi")
    exp31.plot_figures(LEARNED, learned_fig_dir)
    copy_pair(learned_fig_dir / "fig_learned_task_performance", fig_dir / "figure5_learned_path_integration")
    copy_pair(learned_fig_dir / "fig_learned_symmetry_diagnostics", fig_dir / "figure_A6_learned_symmetry_diagnostics")
    copy_pair(learned_fig_dir / "fig_learned_pseudogap", fig_dir / "figure_A7_learned_pseudogap")
    for asset, source in [
        ("figure5_learned_path_integration", "fig_learned_task_performance"),
        ("figure_A6_learned_symmetry_diagnostics", "fig_learned_symmetry_diagnostics"),
        ("figure_A7_learned_pseudogap", "fig_learned_pseudogap"),
    ]:
        manifest.append(
            {
                "asset": asset,
                "kind": "main_or_appendix_figure",
                "generator": "goldstone_lyapunov/experiments/exp31_learned_equivariant_path_integration.py plotting functions",
                "data": "results/learned_equivariant_pi CSV tables",
                "output": rel(fig_dir / f"{asset}.pdf"),
                "source_figure": source,
            }
        )
    generate_task_schematic(fig_dir / "figure_A5_task_schematic", manifest)


def generate_task_schematic(stem: Path, manifest: list[dict[str, str]]) -> None:
    ensure_dir(stem.parent)
    t = np.linspace(0.0, 1.0, 120)
    velocity = 0.55 * np.sin(2 * np.pi * 2.2 * t) + 0.2 * np.sign(np.sin(2 * np.pi * 5 * t))
    phase = np.cumsum(velocity) * (t[1] - t[0])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), constrained_layout=True)
    axes[0].plot(t, velocity, color="#4b72b0")
    axes[0].set_title("velocity input")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("v(t)")
    axes[1].plot(t, phase, color="#1f8a70")
    axes[1].set_title("integrated phase")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("phi(t)")
    theta = np.linspace(0, 2 * np.pi, 200)
    axes[2].plot(np.cos(theta), np.sin(theta), color="#dddddd")
    axes[2].plot(np.cos(phase), np.sin(phase), color="#b35c00")
    axes[2].scatter([np.cos(phase[0])], [np.sin(phase[0])], color="black", s=18, label="start")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_title("target output")
    axes[2].set_xlabel("cos phi")
    axes[2].set_ylabel("sin phi")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    manifest.append(
        {
            "asset": stem.name,
            "kind": "appendix_figure",
            "generator": "draft_asset_generation_code/run_all.py::generate_task_schematic",
            "data": "deterministic illustrative velocity trace",
            "output": rel(stem.with_suffix(".pdf")),
        }
    )


def generate_iclr_figures(fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    from goldstone_lyapunov.experiments import exp32_stronger_path_integration_baselines as exp32

    iclr_fig_dir = ensure_dir(fig_dir / "iclr_final_checks")
    curves = exp32.read_rows(ICLR / "stronger_baselines" / "training_curves.csv")
    eval_rows = exp32.read_rows(ICLR / "stronger_baselines" / "evaluation_metrics.csv")
    param_rows = exp32.read_rows(ICLR / "stronger_baselines" / "parameter_counts.csv")
    existing_curves = read_rows(LEARNED / "training_curves.csv")
    plot_existing_training_curves(existing_curves, iclr_fig_dir / "fig_training_curves_existing")
    exp32.plot_training_curves(curves, iclr_fig_dir)
    exp32.plot_eval(eval_rows, iclr_fig_dir)
    exp32.plot_parameter_matched(eval_rows, param_rows, iclr_fig_dir, ICLR / "stronger_baselines")
    copy_pair(iclr_fig_dir / "fig_training_curves_existing", fig_dir / "figure_A1_existing_training_curves")
    copy_pair(iclr_fig_dir / "fig_stronger_baseline_training_curves", fig_dir / "figure_A2_stronger_baseline_training_curves")
    copy_pair(iclr_fig_dir / "fig_stronger_baseline_eval", fig_dir / "figure_A3_stronger_baseline_eval")
    copy_pair(iclr_fig_dir / "fig_parameter_matched_baselines", fig_dir / "figure_A4_parameter_matched_baselines")
    for asset in APPENDIX_FIGURES[:4]:
        manifest.append(
            {
                "asset": asset,
                "kind": "appendix_figure",
                "generator": "draft_asset_generation_code/run_all.py plus exp32 plotting functions",
                "data": "results/iclr_final_checks and results/learned_equivariant_pi",
                "output": rel(fig_dir / f"{asset}.pdf"),
            }
        )


def plot_existing_training_curves(rows: list[dict[str, str]], stem: Path) -> None:
    labels = {
        "equivariant": "equivariant",
        "broken_equivariant": "broken eq.",
        "gru": "GRU",
        "lstm": "LSTM",
        "orthogonal_rnn": "orthogonal RNN",
    }
    colors = {
        "equivariant": "#1f8a70",
        "broken_equivariant": "#9b2d30",
        "gru": "#4b72b0",
        "lstm": "#b35c00",
        "orthogonal_rnn": "#7a4da3",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    for ax, metric, title in [(axes[0], "train_loss", "Train loss"), (axes[1], "validation_loss", "Validation loss")]:
        for model in labels:
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                if row.get("model") == model:
                    grouped[int(ffloat(row.get("step")))].append(ffloat(row.get(metric)))
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
            ax.plot(xs, ys, label=labels[model], color=colors[model], lw=1.5)
            ax.fill_between(xs, np.maximum(ys - sem, 1e-12), ys + sem, color=colors[model], alpha=0.14)
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def copy_heavy_appendix_figures(fig_dir: Path, manifest: list[dict[str, str]]) -> None:
    heavy = [
        (
            "figure_A9_finite_time_chaos_diagnostics",
            JOURNAL / "exp25_finite_time_chaos_diagnostics" / "finite_time_chaos_diagnostics.png",
            "goldstone_lyapunov/experiments/exp25_finite_time_chaos_diagnostics.py",
        ),
        (
            "figure_A10_large_chaotic_spectra",
            JOURNAL / "exp26_large_chaotic_spectra" / "large_chaotic_spectra.png",
            "goldstone_lyapunov/experiments/exp26_large_chaotic_spectra.py",
        ),
        (
            "figure_A11_path_integration_heatmaps",
            JOURNAL / "exp21_path_integration_heatmaps" / "path_integration_heatmaps.png",
            "goldstone_lyapunov/experiments/exp21_path_integration_heatmaps.py",
        ),
        (
            "figure_A12_gru_path_integration_sweep",
            JOURNAL / "exp29_gru_path_integration_sweep" / "gru_path_integration_sweep.png",
            "goldstone_lyapunov/experiments/exp29_gru_path_integration_sweep.py",
        ),
    ]
    for asset, src, generator in heavy:
        if not src.exists():
            raise FileNotFoundError(src)
        dst = fig_dir / f"{asset}.png"
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        manifest.append(
            {
                "asset": asset,
                "kind": "appendix_figure_archived_full_run",
                "generator": generator,
                "data": rel(src.parent),
                "output": rel(dst),
                "note": "Copied archived full-run PNG by default; rerun command is listed in rerun_commands.ps1.",
            }
        )


def generate_tables(table_dir: Path, manifest: list[dict[str, str]]) -> None:
    ensure_dir(table_dir)
    generate_table1(table_dir, manifest)
    generate_table_a1(table_dir, manifest)
    generate_table_a2(table_dir, manifest)
    generate_table_a3(table_dir, manifest)
    generate_table_a4(table_dir, manifest)
    generate_table_a5(table_dir, manifest)


def generate_table1(table_dir: Path, manifest: list[dict[str, str]]) -> None:
    rows = read_rows(JOURNAL / "exp28_nontrivial_equivariant_rnn" / "nontrivial_equivariant_rnn_metrics.csv")
    exact = [r for r in rows if r.get("variant") == "hidden_irrep_rnn"]
    broken = [r for r in rows if r.get("variant") == "broken_hidden_irrep"]
    table = [
        {
            "diagnostic": "max equivariance error",
            "exact_branch": f"{max(ffloat(r.get('equivariance_error')) for r in exact):.3e}",
            "broken_control": f"{max(ffloat(r.get('equivariance_error')) for r in broken):.3e}",
        },
        {
            "diagnostic": "max |direct group-tangent exponent|",
            "exact_branch": f"{max(abs(ffloat(r.get('direct_tangent_exponent'))) for r in exact):.3e}",
            "broken_control": f"{np.mean([abs(ffloat(r.get('direct_tangent_exponent'))) for r in broken]):.3e}",
        },
        {
            "diagnostic": "max tangent covariance angle (deg)",
            "exact_branch": f"{max(ffloat(r.get('tangent_covariance_angle_degrees')) for r in exact):.3e}",
            "broken_control": "not a protected subspace",
        },
        {
            "diagnostic": "protected near-zero count match",
            "exact_branch": str(all(int(ffloat(r.get("near_zero_count"))) >= int(ffloat(r.get("expected_neutral"))) for r in exact)),
            "broken_control": "zero mode removed",
        },
    ]
    write_rows(table_dir / "table1_nontrivial_equivariant_rnn_metrics.csv", table)
    write_latex_table(
        table_dir / "table1_nontrivial_equivariant_rnn_metrics.tex",
        ["diagnostic", "exact branch", "broken control"],
        [[r["diagnostic"], r["exact_branch"], r["broken_control"]] for r in table],
        "Nontrivial equivariant RNN-style metrics.",
        "tab:nontrivial-equivariant-rnn",
    )
    manifest.append({"asset": "Table 1", "kind": "table", "generator": "generate_table1", "data": "exp28_nontrivial_equivariant_rnn", "output": rel(table_dir / "table1_nontrivial_equivariant_rnn_metrics.tex")})


def generate_table_a1(table_dir: Path, manifest: list[dict[str, str]]) -> None:
    table = [
        {"claim": "Theorem-level neutral modes", "evidence": "Theorem 1; Figures 1-3", "role": "primary"},
        {"claim": "Assumption necessity", "evidence": "Figures 2 and 4; Table A2", "role": "controls"},
        {"claim": "Explicit breaking opens pseudo-gaps", "evidence": "Figure 4; Figure A7", "role": "consequence"},
        {"claim": "Learned path-integration relevance", "evidence": "Figure 5; Tables A3-A5", "role": "task-level"},
        {"claim": "Finite-grid null", "evidence": "Figure A8", "role": "caveat"},
        {"claim": "Chaotic coexistence", "evidence": "Figures A9-A10", "role": "robustness diagnostic"},
    ]
    write_rows(table_dir / "table_A1_claim_to_evidence_hierarchy.csv", table)
    write_latex_table(table_dir / "table_A1_claim_to_evidence_hierarchy.tex", ["claim", "evidence", "role"], [[r["claim"], r["evidence"], r["role"]] for r in table], "Claim-to-evidence hierarchy.", "tab:claim-hierarchy")
    manifest.append({"asset": "Table A1", "kind": "table", "generator": "generate_table_a1", "data": "manuscript claim hierarchy", "output": rel(table_dir / "table_A1_claim_to_evidence_hierarchy.tex")})


def generate_table_a2(table_dir: Path, manifest: list[dict[str, str]]) -> None:
    rows = read_rows(RESULTS / "conference_fix" / "autonomous_zero_diagnostic.csv")
    table = [
        {
            "model": r.get("model"),
            "q": r.get("orbit_dim_q"),
            "rank_EG": r.get("rank_EG"),
            "rank_flow_plus_EG": r.get("rank_flow_plus_EG"),
            "angle_flow_to_EG_deg": r.get("angle_flow_to_EG_degrees"),
            "independent_group_dirs": r.get("group_directions_independent_of_flow"),
            "theorem_level": r.get("theorem_level"),
        }
        for r in rows
    ]
    write_rows(table_dir / "table_A2_autonomous_zero_diagnostic.csv", table)
    write_latex_table(table_dir / "table_A2_autonomous_zero_diagnostic.tex", list(table[0].keys()), [[r[k] for k in table[0].keys()] for r in table], "Autonomous-flow zero-exponent diagnostic.", "tab:auto-zero")
    manifest.append({"asset": "Table A2", "kind": "table", "generator": "generate_table_a2", "data": "results/conference_fix/autonomous_zero_diagnostic.csv", "output": rel(table_dir / "table_A2_autonomous_zero_diagnostic.tex")})


def generate_table_a3(table_dir: Path, manifest: list[dict[str, str]]) -> None:
    equiv = read_rows(LEARNED / "equivariance_diagnostics.csv")
    exps = read_rows(LEARNED / "group_tangent_exponents.csv")
    angles = read_rows(LEARNED / "principal_angle_alignment.csv")
    pseudo = read_rows(LEARNED / "pseudogap_lifetime_learned.csv")
    stats = read_rows(LEARNED / "statistical_summary.csv")
    max_step = max(ffloat(r.get("step_max")) for r in equiv if r.get("model") == "trained_exact")
    zero_exp = [ffloat(r.get("lambda_group")) for r in exps if r.get("model") == "trained_exact" and r.get("input_regime") == "zero_input"]
    zero_angle = [ffloat(r.get("angle_degrees")) for r in angles if r.get("model") == "trained_exact" and abs(ffloat(r.get("u_value"))) < 1e-12]
    max_angle = max(ffloat(r.get("angle_degrees")) for r in angles if r.get("model") == "trained_exact")
    unc = [(ffloat(r.get("predicted_lifetime")), ffloat(r.get("measured_lifetime"))) for r in pseudo if str(r.get("censored")).lower() != "true"]
    corr = float("nan")
    if len(unc) >= 2:
        pred = np.log([u[0] for u in unc])
        meas = np.log([u[1] for u in unc])
        corr = float(np.corrcoef(pred, meas)[0, 1])
    long_rows = [r for r in stats if int(ffloat(r.get("test_horizon"))) == 256 and abs(ffloat(r.get("test_speed_scale")) - 1.8) < 1e-9 and r.get("phase_generalization") == "full"]
    table = [
        {"metric": "max trained exact step equivariance error", "value": f"{max_step:.3e}"},
        {"metric": "mean zero-input direct group-tangent exponent", "value": f"{np.mean(zero_exp):.3e}"},
        {"metric": "mean zero-input principal angle (deg)", "value": f"{np.mean(zero_angle):.3e}"},
        {"metric": "max tested principal angle (deg)", "value": f"{max_angle:.3e}"},
        {"metric": "pseudo-gap log lifetime correlation", "value": f"{corr:.4f}"},
    ]
    for row in long_rows:
        table.append({"metric": f"{row.get('model')} long OOD RMSE", "value": f"{ffloat(row.get('mean_circular_rmse')):.4g} +/- {ffloat(row.get('sem_circular_rmse')):.2g}"})
    write_rows(table_dir / "table_A3_learned_diagnostics.csv", table)
    write_latex_table(table_dir / "table_A3_learned_diagnostics.tex", ["metric", "value"], [[r["metric"], r["value"]] for r in table], "Learned equivariant path-integration diagnostics.", "tab:learned-diagnostics")
    manifest.append({"asset": "Table A3", "kind": "table", "generator": "generate_table_a3", "data": "results/learned_equivariant_pi", "output": rel(table_dir / "table_A3_learned_diagnostics.tex")})


def generate_table_a4(table_dir: Path, manifest: list[dict[str, str]]) -> None:
    rows = read_rows(LEARNED / "model_registry.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("model", "")].append(row)
    table = []
    for model, vals in sorted(grouped.items()):
        table.append(
            {
                "model": model,
                "runs": len(vals),
                "seeds": len({v.get("seed") for v in vals}),
                "hidden_size": ",".join(sorted({v.get("hidden_size") for v in vals})),
                "train_steps": ",".join(sorted({v.get("train_steps") for v in vals})),
                "learning_rate": ",".join(sorted({v.get("learning_rate") for v in vals})),
                "parameter_count": ",".join(sorted({v.get("parameter_count") for v in vals})),
            }
        )
    write_rows(table_dir / "table_A4_training_configuration.csv", table)
    write_latex_table(table_dir / "table_A4_training_configuration.tex", list(table[0].keys()), [[r[k] for k in table[0].keys()] for r in table], "Learned path-integration training configuration.", "tab:training-config")
    manifest.append({"asset": "Table A4", "kind": "table", "generator": "generate_table_a4", "data": "results/learned_equivariant_pi/model_registry.csv", "output": rel(table_dir / "table_A4_training_configuration.tex")})


def generate_table_a5(table_dir: Path, manifest: list[dict[str, str]]) -> None:
    rows = read_rows(ICLR / "stronger_baselines" / "baseline_sweep_summary.csv")
    table = [
        {
            "model": r.get("model"),
            "config": r.get("config_label"),
            "phase": r.get("phase_generalization"),
            "horizon": r.get("test_horizon"),
            "speed": r.get("test_speed_scale"),
            "mean_rmse": f"{ffloat(r.get('mean_circular_rmse')):.4g}",
            "sem": f"{ffloat(r.get('sem_circular_rmse')):.2g}",
            "n_seeds": r.get("n_seeds"),
        }
        for r in rows
        if int(ffloat(r.get("test_horizon"))) == 256 and abs(ffloat(r.get("test_speed_scale")) - 1.8) < 1e-9
    ]
    write_rows(table_dir / "table_A5_stronger_baseline_check.csv", table)
    write_latex_table(table_dir / "table_A5_stronger_baseline_check.tex", list(table[0].keys()), [[r[k] for k in table[0].keys()] for r in table], "Stronger-baseline check for learned path integration.", "tab:stronger-baseline")
    manifest.append({"asset": "Table A5", "kind": "table", "generator": "generate_table_a5", "data": "results/iclr_final_checks/stronger_baselines/baseline_sweep_summary.csv", "output": rel(table_dir / "table_A5_stronger_baseline_check.tex")})


def write_rerun_commands(path: Path) -> None:
    lines = [
        "# Heavy/Full Rerun Commands",
        "",
        "# Main clean figures from frozen journal tables:",
        "python draft_asset_generation_code/run_all.py",
        "",
        "# Learned experiment full rerun:",
        "python -m goldstone_lyapunov.experiments.exp31_learned_equivariant_path_integration --full",
        "",
        "# Stronger baseline check full rerun:",
        "python -m goldstone_lyapunov.experiments.exp32_stronger_path_integration_baselines --full",
        "",
        "# Heavy appendix figures, if raw full figures need to be regenerated:",
        "python -m goldstone_lyapunov.experiments.exp25_finite_time_chaos_diagnostics",
        "python -m goldstone_lyapunov.experiments.exp26_large_chaotic_spectra",
        "python -m goldstone_lyapunov.experiments.exp21_path_integration_heatmaps",
        "python -m goldstone_lyapunov.experiments.exp29_gru_path_integration_sweep",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_outputs(out: Path, manifest: list[dict[str, str]]) -> list[str]:
    failures: list[str] = []
    for row in manifest:
        output = row.get("output", "")
        if not output:
            continue
        path = REPO_ROOT / output
        if not path.exists() or path.stat().st_size == 0:
            failures.append(output)
    for stem in MAIN_FIGURES:
        pdf = out / "figures" / f"{stem}.pdf"
        png = out / "figures" / f"{stem}.png"
        if not pdf.exists() or pdf.stat().st_size == 0:
            failures.append(rel(pdf))
        if stem != "figure5_learned_path_integration" and (not png.exists() or png.stat().st_size == 0):
            failures.append(rel(png))
    for table in ["table1_nontrivial_equivariant_rnn_metrics", "table_A1_claim_to_evidence_hierarchy", "table_A2_autonomous_zero_diagnostic", "table_A3_learned_diagnostics", "table_A4_training_configuration", "table_A5_stronger_baseline_check"]:
        tex = out / "tables" / f"{table}.tex"
        csv_path = out / "tables" / f"{table}.csv"
        if not tex.exists() or tex.stat().st_size == 0:
            failures.append(rel(tex))
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            failures.append(rel(csv_path))
    return failures


def write_readme_snapshot(out: Path, manifest: list[dict[str, str]], failures: list[str]) -> None:
    lines = [
        "# Generated Draft Assets",
        "",
        "This directory was produced by `draft_asset_generation_code/run_all.py`.",
        "",
        f"- Figure/table manifest rows: {len(manifest)}",
        f"- Verification failures: {len(failures)}",
        "",
        "Main figure stems:",
    ]
    lines.extend(f"- `{stem}`" for stem in MAIN_FIGURES)
    lines.append("")
    lines.append("Appendix figure stems:")
    lines.extend(f"- `{stem}`" for stem in APPENDIX_FIGURES)
    lines.append("")
    lines.append("Tables:")
    lines.extend(
        [
            "- `table1_nontrivial_equivariant_rnn_metrics`",
            "- `table_A1_claim_to_evidence_hierarchy`",
            "- `table_A2_autonomous_zero_diagnostic`",
            "- `table_A3_learned_diagnostics`",
            "- `table_A4_training_configuration`",
            "- `table_A5_stronger_baseline_check`",
        ]
    )
    if failures:
        lines.append("")
        lines.append("Failures:")
        lines.extend(f"- `{failure}`" for failure in failures)
    (out / "README_GENERATED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_source_scripts() -> None:
    dst = ensure_dir(HERE / "source_scripts")
    sources = [
        REPO_ROOT / "paper_runs" / "conference_fix.py",
        REPO_ROOT / "paper_runs" / "make_fig2_neutral_geometry_v2.py",
        REPO_ROOT / "goldstone_lyapunov" / "experiments" / "exp31_learned_equivariant_path_integration.py",
        REPO_ROOT / "goldstone_lyapunov" / "experiments" / "exp32_stronger_path_integration_baselines.py",
        REPO_ROOT / "goldstone_lyapunov" / "experiments" / "exp21_path_integration_heatmaps.py",
        REPO_ROOT / "goldstone_lyapunov" / "experiments" / "exp25_finite_time_chaos_diagnostics.py",
        REPO_ROOT / "goldstone_lyapunov" / "experiments" / "exp26_large_chaotic_spectra.py",
        REPO_ROOT / "goldstone_lyapunov" / "experiments" / "exp29_gru_path_integration_sweep.py",
    ]
    for src in sources:
        if src.exists():
            shutil.copy2(src, dst / src.name)


def required_inputs() -> list[Path]:
    return [
        JOURNAL / "exp23_dimension_law_torus_sweep" / "dimension_law_torus_sweep.csv",
        JOURNAL / "exp18_clv_principal_angle_sweep" / "principal_angle_sweep.csv",
        JOURNAL / "exp28_nontrivial_equivariant_rnn" / "nontrivial_equivariant_rnn_metrics.csv",
        JOURNAL / "exp19_pseudogap_lifetime" / "pseudogap_lifetime.csv",
        JOURNAL / "exp24_random_breaking_ensemble" / "random_breaking_ensemble.csv",
        LEARNED / "evaluation_metrics.csv",
        LEARNED / "training_curves.csv",
        ICLR / "stronger_baselines" / "baseline_sweep_summary.csv",
    ]


def run(output_dir: Path, check_only: bool = False) -> int:
    missing = [path for path in required_inputs() if not path.exists()]
    if missing:
        for path in missing:
            print(f"MISSING: {path}", file=sys.stderr)
        return 2
    if check_only:
        print("Input check passed.")
        return 0
    out = ensure_dir(output_dir)
    fig_dir = ensure_dir(out / "figures")
    table_dir = ensure_dir(out / "tables")
    support_dir = ensure_dir(out / "support")
    manifest: list[dict[str, str]] = []
    generate_main_theorem_figures(fig_dir, support_dir, manifest)
    generate_learned_figures(fig_dir, manifest)
    generate_iclr_figures(fig_dir, manifest)
    copy_heavy_appendix_figures(fig_dir, manifest)
    generate_tables(table_dir, manifest)
    write_rows(out / "asset_manifest.csv", manifest)
    write_json(out / "asset_manifest.json", manifest)
    write_rerun_commands(HERE / "rerun_commands.ps1")
    copy_source_scripts()
    failures = verify_outputs(out, manifest)
    write_readme_snapshot(out, manifest, failures)
    print(
        json.dumps(
            {
                "output_dir": rel(out),
                "n_manifest_rows": len(manifest),
                "n_failures": len(failures),
                "failures": failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(HERE / "generated_assets"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(Path(args.output_dir), check_only=args.check_only))


if __name__ == "__main__":
    main()
