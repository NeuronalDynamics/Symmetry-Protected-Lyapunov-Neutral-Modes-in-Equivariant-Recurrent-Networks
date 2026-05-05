"""Generate the conference-fix manuscript package and clean figures.

This script is intentionally conservative: it reads existing raw outputs from
``results/journal_full`` and writes a reviewer-facing bundle under
``results/conference_fix`` plus clean main figures under ``figures_clean``.
It does not fabricate missing measurements; absent inputs are reported as
TODO_MISSING_* in the generated manifests.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from goldstone_lyapunov.models import (
    CollapseCounterexample,
    PhaseIntegrator,
    S1Attractor,
    S1CoupledIrrepAttractor,
    SONSphereAttractor,
    T2Attractor,
    UMSphereAttractor,
)
from goldstone_lyapunov.symmetry import (
    s1_tangent,
    son_tangent_basis,
    t2_tangents,
    um_tangent_basis,
    weighted_s1_tangent,
)


RESULTS = ROOT / "results"
JOURNAL = RESULTS / "journal_full"
FIX = RESULTS / "conference_fix"
FIGS = ROOT / "figures_clean"
DOCS = ROOT / "docs"


def ensure_dirs() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def ffloat(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return default
        return float(text)
    except Exception:
        return default


def iint(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def parse_spectrum(text: str) -> list[float]:
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return [float(x) for x in parsed]
    except Exception:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        return [float(x) for x in nums]


def group_values(rows: list[dict[str, str]], key: str, value: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(ffloat(row.get(value)))
    return grouped


def mean_or_nan(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def std_or_nan(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.std(vals)) if vals else float("nan")


def sem_or_nan(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.std(vals) / math.sqrt(len(vals))) if vals else float("nan")


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGS / f"{stem}.pdf")
    fig.savefig(FIGS / f"{stem}.png")
    plt.close(fig)


def matrix_rank(cols: np.ndarray, tol: float = 1e-10) -> int:
    if cols.size == 0:
        return 0
    return int(np.linalg.matrix_rank(cols, tol=tol))


def orth_basis(cols: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if cols.size == 0:
        return np.zeros((cols.shape[0], 0))
    u, s, _vh = np.linalg.svd(cols, full_matrices=False)
    return u[:, s > tol]


def autonomous_row(
    model_name: str,
    group: str,
    stabilizer: str,
    orbit_dim: int,
    state: np.ndarray,
    flow: np.ndarray,
    tangent_cols: np.ndarray,
    notes: str,
    theorem_level: bool = True,
) -> dict[str, Any]:
    flow = np.asarray(flow, dtype=float).reshape(-1)
    tangent_cols = np.asarray(tangent_cols, dtype=float)
    if tangent_cols.ndim == 1:
        tangent_cols = tangent_cols[:, None]
    flow_norm = float(np.linalg.norm(flow))
    group_rank = matrix_rank(tangent_cols)
    flow_rank = 1 if flow_norm > 1e-10 else 0
    combined = np.column_stack([flow, tangent_cols])
    combined_rank = matrix_rank(combined)
    q_independent_of_flow = combined_rank - flow_rank
    basis = orth_basis(tangent_cols)
    if flow_norm > 1e-10 and basis.shape[1] > 0:
        projection = basis @ (basis.T @ flow)
        projection_ratio = float(np.linalg.norm(projection) / flow_norm)
        projection_ratio = min(1.0, max(0.0, projection_ratio))
        angle = float(np.degrees(np.arccos(projection_ratio)))
        cosines = []
        for j in range(tangent_cols.shape[1]):
            tj = tangent_cols[:, j]
            denom = flow_norm * np.linalg.norm(tj)
            cosines.append(float(np.dot(flow, tj) / denom) if denom > 1e-12 else float("nan"))
    else:
        projection_ratio = float("nan")
        angle = float("nan")
        cosines = []
    return {
        "model": model_name,
        "group": group,
        "stabilizer": stabilizer,
        "orbit_dim_q": orbit_dim,
        "state_dim": int(state.size),
        "flow_norm": flow_norm,
        "flow_defined": bool(flow_rank),
        "rank_EG": group_rank,
        "rank_flow_plus_EG": combined_rank,
        "projection_norm_over_flow_norm": projection_ratio,
        "angle_flow_to_EG_degrees": angle,
        "cosine_flow_each_generator": json.dumps(cosines),
        "group_directions_independent_of_flow": int(q_independent_of_flow),
        "theorem_level": theorem_level,
        "notes": notes,
    }


def generate_autonomous_zero_diagnostic() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    s1 = S1Attractor(R=1.0)
    x = np.array([1.0, 0.0])
    rows.append(
        autonomous_row(
            "S1Attractor",
            "S1",
            "trivial",
            1,
            x,
            s1.f(x),
            s1_tangent(x),
            "Attracting circle consists of equilibria, so f(x)=0 while the group tangent is nonzero.",
        )
    )

    t2 = T2Attractor(R=(1.0, 1.0))
    x = np.array([1.0, 0.0, 0.0, 1.0])
    rows.append(
        autonomous_row(
            "T2Attractor",
            "T2",
            "trivial",
            2,
            x,
            t2.f(x),
            np.column_stack(t2_tangents(x)),
            "Attracting torus consists of equilibria with two independent group tangents.",
        )
    )

    coupled = S1CoupledIrrepAttractor(hidden_size=10, hidden_seed=0)
    x = coupled.template_state(theta=0.37)
    rows.append(
        autonomous_row(
            "S1CoupledIrrepAttractor",
            "weighted S1",
            "trivial",
            1,
            x,
            coupled.f(x),
            weighted_s1_tangent(x, weights=(1, 2)),
            "Non-radial equivariant RNN-style orbit with a slaved second harmonic and hidden invariant rates.",
        )
    )

    so3 = SONSphereAttractor(n=3)
    x = np.array([1.0, 0.2, -0.3])
    x = x / np.linalg.norm(x)
    rows.append(
        autonomous_row(
            "SONSphereAttractor_n3",
            "SO(3)",
            "SO(2)",
            2,
            x,
            so3.f(x),
            son_tangent_basis(x),
            "SO(3)/SO(2) sphere example with nontrivial stabilizer.",
        )
    )

    so5 = SONSphereAttractor(n=5)
    x = np.array([1.0, 0.3, -0.2, 0.4, 0.1])
    x = x / np.linalg.norm(x)
    rows.append(
        autonomous_row(
            "SONSphereAttractor_n5",
            "SO(5)",
            "SO(4)",
            4,
            x,
            so5.f(x),
            son_tangent_basis(x),
            "Higher-dimensional SO(n)/SO(n-1) sphere example.",
        )
    )

    um3 = UMSphereAttractor(m=3)
    x = np.array([1.0, 0.1, 0.3, -0.4, 0.2, 0.5])
    x = x / np.linalg.norm(x)
    rows.append(
        autonomous_row(
            "UMSphereAttractor_m3",
            "U(3)",
            "U(2)",
            5,
            x,
            um3.f(x),
            um_tangent_basis(x),
            "U(m)/U(m-1) complex-sphere example represented in real coordinates.",
        )
    )

    rel = PhaseIntegrator(R=1.0, v_default=0.7)
    x = np.array([1.0, 0.0])
    rows.append(
        autonomous_row(
            "PhaseIntegrator_constant_velocity",
            "S1",
            "trivial",
            1,
            x,
            rel.f(x),
            s1_tangent(x),
            "Relative-equilibrium control: the autonomous flow direction coincides with the group tangent.",
            theorem_level=False,
        )
    )

    collapse = CollapseCounterexample()
    x = np.array([1.0, 0.0])
    rows.append(
        autonomous_row(
            "CollapseCounterexample",
            "SO(2)",
            "violated asymptotically",
            0,
            x,
            collapse.f(x),
            s1_tangent(x),
            "Exact equivariance control where the persistent nondegenerate orbit assumption fails.",
            theorem_level=False,
        )
    )

    write_csv(FIX / "autonomous_zero_diagnostic.csv", rows)

    md_lines = [
        "# Autonomous-Flow Zero-Exponent Diagnostic",
        "",
        "Continuous-time autonomous flows can carry a zero Lyapunov exponent in the flow direction f(x).",
        "This diagnostic separates that direction from the analytical group-tangent bundle E^G_x.",
        "",
        "| model | q | rank(E^G) | rank([f,E^G]) | flow defined | angle flow-to-E^G (deg) | independent group directions | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        angle = row["angle_flow_to_EG_degrees"]
        angle_text = "undefined (f=0)" if not math.isfinite(float(angle)) else f"{float(angle):.3g}"
        md_lines.append(
            f"| {row['model']} | {row['orbit_dim_q']} | {row['rank_EG']} | {row['rank_flow_plus_EG']} | "
            f"{row['flow_defined']} | {angle_text} | {row['group_directions_independent_of_flow']} | {row['notes']} |"
        )
    md_lines.extend(
        [
            "",
            "Interpretation: fixed-point continuous attractors have f(x)=0 on the orbit, so their group tangents are not inferred from an ordinary flow direction.",
            "The constant-velocity phase-integrator row is a relative-equilibrium control where the flow is tangent to the S1 orbit, illustrating the caveat that one group direction may coincide with time translation.",
        ]
    )
    (FIX / "autonomous_zero_diagnostic.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    text = r"""
Autonomous flows can have a zero exponent in the time-translation direction \(f(x)\).
The protected modes studied here are generated by infinitesimal group actions \(\xi_M(x)\), not by reading off an undifferentiated near-zero exponent.
In some relative-equilibrium examples, one group direction may coincide with the flow.
Product-group and \(q>1\) examples therefore test multiplicity beyond the time-translation direction.
Accordingly, the paper reports direct group-tangent exponents and tangent-subspace alignment rather than relying on a single zero exponent.
"""
    (FIX / "text_autonomous_zero_caveat.tex").write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")

    plot_autonomous_zero_figure(rows, "fig_autonomous_zero_diagnostic")
    return rows


def plot_autonomous_zero_figure(rows: list[dict[str, Any]], stem: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    labels = [r["model"].replace("Attractor", "").replace("Sphere", " sphere").replace("_", " ") for r in rows]
    labels = [s.replace("S1CoupledIrrep", "coupled irrep").replace("PhaseIntegrator constant velocity", "relative eq.") for s in labels]
    x = np.arange(len(rows))

    axes[0].bar(x - 0.18, [r["rank_EG"] for r in rows], width=0.36, label=r"rank $E^G$")
    axes[0].bar(x + 0.18, [r["rank_flow_plus_EG"] for r in rows], width=0.36, label=r"rank $[f,E^G]$")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("rank")
    axes[0].set_title("A  Subspace ranks")
    axes[0].legend(frameon=False)

    angles = []
    colors = []
    for row in rows:
        angle = ffloat(row["angle_flow_to_EG_degrees"])
        if math.isfinite(angle):
            angles.append(angle)
            colors.append("#2f6fbb")
        else:
            angles.append(95.0)
            colors.append("#888888")
    axes[1].scatter(x, angles, s=36, c=colors, zorder=3)
    for idx, row in enumerate(rows):
        if not row["flow_defined"]:
            axes[1].text(idx, 92.0, r"$f=0$", ha="center", va="bottom", fontsize=7, rotation=90)
    axes[1].set_ylim(-3, 102)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("angle to group subspace (deg)")
    axes[1].set_title("B  Flow direction caveat")

    theorem_rows = [r for r in rows if r["theorem_level"]]
    q = [r["orbit_dim_q"] for r in theorem_rows]
    independent = [r["group_directions_independent_of_flow"] for r in theorem_rows]
    axes[2].plot(q, independent, "o-", color="#1f8a70", lw=1.8)
    max_q = max(q)
    axes[2].plot([0, max_q], [0, max_q], "--", color="black", lw=1.0, alpha=0.6, label="identity")
    axes[2].set_xlabel(r"orbit dimension $q$")
    axes[2].set_ylabel("independent of flow")
    axes[2].set_title("C  Product/multiplicity check")
    axes[2].set_xlim(0, max_q + 0.5)
    axes[2].set_ylim(0, max_q + 0.5)
    axes[2].legend(frameon=False)

    savefig(fig, stem)


def generate_asset_map() -> None:
    candidate_mains = [
        ROOT / "neurips_goldstone_project_final" / "main.tex",
        ROOT / "neurips_goldstone_project" / "main.tex",
        ROOT / "paper" / "main.tex",
        ROOT / "main.tex",
    ]
    all_tex = sorted(p for p in ROOT.rglob("*.tex") if "pytest-cache-files" not in str(p))
    all_bib = sorted(p for p in ROOT.rglob("*.bib") if "pytest-cache-files" not in str(p))
    searches = [
        "exp18_clv_principal_angle_sweep",
        "exp19_pseudogap_lifetime",
        "exp23_dimension_law_torus_sweep",
        "exp24_random_breaking_ensemble",
        "exp27_more_symmetry_families",
        "exp28_nontrivial_equivariant_rnn",
        "exp29_gru_path_integration_sweep",
        "exp17_finite_grid_null",
    ]
    lines = [
        "# Repository Asset Map",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Draft and Bibliography",
    ]
    found_main = [p for p in candidate_mains if p.exists()]
    if found_main:
        lines.extend(f"- current LaTeX draft candidate: `{rel(p)}`" for p in found_main)
    else:
        lines.append("- TODO_MISSING_DRAFT: no main LaTeX file found at the expected draft paths.")
    if all_tex:
        lines.append("- TeX files found:")
        lines.extend(f"  - `{rel(p)}`" for p in all_tex)
    else:
        lines.append("- No existing TeX files found before this conference-fix generation.")
    if all_bib:
        lines.append("- Bibliography files found:")
        lines.extend(f"  - `{rel(p)}`" for p in all_bib)
    else:
        lines.append("- TODO_MISSING_BIB: no bibliography file found before this generation.")

    lines.extend(
        [
            "",
            "## Figure and Result Directories",
            f"- clean figure output: `{rel(FIGS)}`",
            f"- raw full results: `{rel(JOURNAL)}`" if JOURNAL.exists() else "- TODO_MISSING_RESULTS: `results/journal_full` not found.",
            f"- experiment logs: `{rel(JOURNAL / 'logs')}`" if (JOURNAL / "logs").exists() else "- TODO_MISSING_LOGS: `results/journal_full/logs` not found.",
            f"- old full atlas: `{rel(JOURNAL / 'results_atlas_full')}`" if (JOURNAL / "results_atlas_full").exists() else "- Old full atlas not found.",
            "",
            "## Figure Scripts",
        ]
    )
    figure_scripts = sorted((ROOT / "paper_runs").rglob("*.py"))
    lines.extend(f"- `{rel(p)}`" for p in figure_scripts)

    lines.extend(["", "## Data Files by Experiment"])
    for name in searches:
        paths = sorted((JOURNAL / name).glob("*")) if (JOURNAL / name).exists() else []
        lines.append(f"### {name}")
        if paths:
            for p in paths:
                lines.append(f"- `{rel(p)}`")
        else:
            lines.append(f"- TODO_MISSING_EXPERIMENT_OUTPUT: `{rel(JOURNAL / name)}`")

    status_files = sorted(JOURNAL.glob("*.json")) + sorted((JOURNAL / "logs").glob("*")) if (JOURNAL / "logs").exists() else sorted(JOURNAL.glob("*.json"))
    lines.extend(["", "## Audit/Status/Log Files"])
    if status_files:
        lines.extend(f"- `{rel(p)}`" for p in status_files)
    else:
        lines.append("- TODO_MISSING_STATUS: no JSON/log status files found.")

    (FIX / "repo_asset_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_fig1_dimension_law() -> None:
    torus = read_csv(JOURNAL / "exp23_dimension_law_torus_sweep" / "dimension_law_torus_sweep.csv")
    so = read_csv(JOURNAL / "exp27_more_symmetry_families" / "so_n_sphere_family.csv")
    um = read_csv(JOURNAL / "exp27_more_symmetry_families" / "unitary_sphere_family.csv")
    product = read_csv(JOURNAL / "exp27_more_symmetry_families" / "product_symmetry_family.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.55), constrained_layout=True)

    q_groups = sorted(group_values(torus, "q", "near_zero_count").items(), key=lambda kv: float(kv[0]))
    qx = [float(k) for k, _ in q_groups]
    qy = [mean_or_nan(v) for _, v in q_groups]
    qerr = [std_or_nan(v) for _, v in q_groups]
    axes[0].errorbar(qx, qy, yerr=qerr, fmt="o", color="#275d9f", capsize=2, label="product tori")
    if qx:
        axes[0].plot([min(qx), max(qx)], [min(qx), max(qx)], "--", color="black", lw=1.0, label="prediction")
    axes[0].set_xlabel(r"expected $\dim(G/H)$")
    axes[0].set_ylabel("observed near-zero count")
    axes[0].set_title("A  Dimension law")
    axes[0].legend(frameon=False, loc="upper left")

    def family_points(rows: list[dict[str, str]], marker: str, label: str, color: str) -> None:
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            grouped[iint(row.get("expected_neutral"))].append(ffloat(row.get("near_zero_count")))
        xs = sorted(grouped)
        ys = [mean_or_nan(grouped[x]) for x in xs]
        err = [std_or_nan(grouped[x]) for x in xs]
        axes[1].errorbar(xs, ys, yerr=err, fmt=marker, color=color, capsize=2, label=label)

    family_points(so, "o", "SO(n)/SO(n-1)", "#1f8a70")
    family_points(um, "s", "U(m)/U(m-1)", "#b35c00")
    family_points(product, "^", "Tq x sphere", "#7a4da3")
    expected = [iint(r.get("expected_neutral")) for r in so + um + product]
    if expected:
        lo, hi = min(expected), max(expected)
        axes[1].plot([lo, hi], [lo, hi], "--", color="black", lw=1.0)
    axes[1].set_xlabel("expected neutral count")
    axes[1].set_ylabel("observed count")
    axes[1].set_title("B  Other continuous symmetries")
    axes[1].legend(frameon=False, loc="upper left")

    for qtarget, color in [(1, "#275d9f"), (2, "#1f8a70"), (4, "#b35c00"), (8, "#7a4da3")]:
        candidates = [r for r in torus if iint(r.get("q")) == qtarget and iint(r.get("seed")) == 0]
        if candidates:
            spec = parse_spectrum(candidates[0].get("spectrum", ""))
            axes[2].plot(np.arange(1, len(spec) + 1), spec, "o-", ms=3, lw=1.0, color=color, label=f"q={qtarget}")
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_xlabel("spectrum index")
    axes[2].set_ylabel(r"$\lambda_i$")
    axes[2].set_title("C  Spectrum families")
    axes[2].legend(frameon=False, ncol=2)

    savefig(fig, "fig1_dimension_law_clean")


def plot_fig2_neutral_geometry(auto_rows: list[dict[str, Any]]) -> None:
    angle_rows = read_csv(JOURNAL / "exp18_clv_principal_angle_sweep" / "principal_angle_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.75), constrained_layout=True)

    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in angle_rows:
        grouped[(row.get("model", "model"), ffloat(row.get("window")))].append(ffloat(row.get("max_angle_degrees")))
    models = sorted(set(k[0] for k in grouped))
    colors = ["#275d9f", "#1f8a70", "#b35c00", "#7a4da3", "#9b2d30"]
    for idx, model in enumerate(models):
        xs = sorted(k[1] for k in grouped if k[0] == model)
        ys = [max(1e-12, mean_or_nan(grouped[(model, x)])) for x in xs]
        axes[0].plot(xs, ys, "o-", lw=1.2, ms=3.5, color=colors[idx % len(colors)], label=model)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("alignment window")
    axes[0].set_ylabel("max principal angle (deg)")
    axes[0].set_title("A  Neutral subspace alignment")
    axes[0].legend(frameon=False, ncol=2)

    plot_rows = [r for r in auto_rows if r["model"] not in {"CollapseCounterexample"}]
    x = np.arange(len(plot_rows))
    label_map = {
        "S1Attractor": "S1",
        "T2Attractor": "T2",
        "S1CoupledIrrepAttractor": "irrep RNN",
        "SONSphereAttractor_n3": "SO3",
        "SONSphereAttractor_n5": "SO5",
        "UMSphereAttractor_m3": "U3",
        "PhaseIntegrator_constant_velocity": "relative eq.",
    }
    labels = [label_map.get(str(r["model"]), str(r["model"])) for r in plot_rows]
    y = []
    colors_b = []
    for row in plot_rows:
        angle = ffloat(row["angle_flow_to_EG_degrees"])
        if math.isfinite(angle):
            y.append(max(1e-8, angle))
            colors_b.append("#275d9f")
        else:
            y.append(100.0)
            colors_b.append("#888888")
    axes[1].scatter(x, y, s=32, c=colors_b)
    for idx, row in enumerate(plot_rows):
        if not row["flow_defined"]:
            axes[1].text(idx, 72, "f=0", ha="center", va="bottom", fontsize=7, rotation=90)
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("flow-to-group angle (deg)")
    axes[1].set_title("B  Flow-zero diagnostic")

    theorem_rows = [r for r in auto_rows if r["theorem_level"]]
    q = [r["orbit_dim_q"] for r in theorem_rows]
    indep = [r["group_directions_independent_of_flow"] for r in theorem_rows]
    axes[2].plot(q, indep, "o", color="#1f8a70", label="diagnostic rows")
    max_q = max(q) if q else 1
    axes[2].plot([0, max_q], [0, max_q], "--", color="black", lw=1.0, label="q directions")
    axes[2].set_xlabel(r"orbit dimension $q$")
    axes[2].set_ylabel("group directions independent of f")
    axes[2].set_title("C  Multiplicity beyond flow")
    axes[2].set_xlim(0, max_q + 0.5)
    axes[2].set_ylim(0, max_q + 0.5)
    axes[2].legend(frameon=False)

    savefig(fig, "fig2_neutral_geometry_clean")


def plot_architecture_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    blocks = [
        (0.07, 0.62, 0.18, 0.2, "z\nq=1", "#d8ecff"),
        (0.42, 0.62, 0.18, 0.2, "w\nq=2", "#dff4df"),
        (0.77, 0.62, 0.18, 0.2, "h\ninv.", "#f5e7c6"),
        (0.42, 0.18, 0.18, 0.17, "broken\ncontrol", "#f2d6d6"),
    ]
    for x0, y0, w, h, text, color in blocks:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=color, edgecolor="#333333", lw=1.0))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=7.5)
    arrows = [
        ((0.25, 0.72), (0.42, 0.72), ""),
        ((0.60, 0.72), (0.77, 0.72), ""),
        ((0.51, 0.62), (0.51, 0.35), "pins phase"),
    ]
    for start, end, label in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, lw=1.0, color="#333333"))
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        yoff = 0.055 if abs(start[1] - end[1]) < 1e-6 else 0.02
        if label:
            ax.text(mx, my + yoff, label, ha="center", va="center", fontsize=6.5)
    ax.text(0.03, 0.94, "A  Equivariant RNN-style branch", weight="bold", fontsize=9)


def plot_fig3_equivariant_rnn() -> None:
    rows = read_csv(JOURNAL / "exp28_nontrivial_equivariant_rnn" / "nontrivial_equivariant_rnn_metrics.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.75), constrained_layout=True, gridspec_kw={"width_ratios": [1.55, 0.9, 0.9, 0.82]})
    plot_architecture_panel(axes[0])

    exact = [r for r in rows if r.get("variant") == "hidden_irrep_rnn"]
    broken = [r for r in rows if r.get("variant") == "broken_hidden_irrep"]
    labels = ["exact", "broken"]
    eq_vals = [max(1e-18, max(ffloat(r.get("equivariance_error")) for r in exact)), max(1e-18, min(ffloat(r.get("equivariance_error")) for r in broken))]
    axes[1].bar(labels, eq_vals, color=["#1f8a70", "#9b2d30"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("equivariance error")
    axes[1].set_title("B  Equivariance")

    direct_vals = [
        max(1e-18, max(abs(ffloat(r.get("direct_tangent_exponent"))) for r in exact)),
        max(1e-18, statistics.mean(abs(ffloat(r.get("direct_tangent_exponent"))) for r in broken)),
    ]
    axes[2].bar(labels, direct_vals, color=["#1f8a70", "#9b2d30"])
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$|\lambda_{\mathrm{group}}|$")
    axes[2].set_title("C  Group exponent")

    angle_vals = [
        max(1e-12, max(ffloat(r.get("tangent_covariance_angle_degrees")) for r in exact)),
        float("nan"),
    ]
    axes[3].bar([labels[0]], [angle_vals[0]], color="#1f8a70")
    axes[3].text(1, angle_vals[0] * 1.35, "no protected\nneutral\nsubspace", ha="center", va="bottom", fontsize=6.5)
    axes[3].set_yscale("log")
    axes[3].set_xticks([0, 1])
    axes[3].set_xticklabels(labels)
    axes[3].set_ylabel("angle (deg)")
    axes[3].set_title("D  Alignment")

    savefig(fig, "fig3_equivariant_rnn_clean")


def plot_fig4_pseudogap() -> None:
    life = read_csv(JOURNAL / "exp19_pseudogap_lifetime" / "pseudogap_lifetime.csv")
    breaking = read_csv(JOURNAL / "exp24_random_breaking_ensemble" / "random_breaking_ensemble.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.55), constrained_layout=True)

    pred = [ffloat(r.get("predicted_lifetime")) for r in life]
    meas = [ffloat(r.get("measured_lifetime")) for r in life]
    matrix = [r.get("matrix", "") for r in life]
    colors = {"weak_axis": "#275d9f", "unit_axis": "#1f8a70", "rotated_strong": "#b35c00"}
    for m in sorted(set(matrix)):
        xs = [p for p, mm in zip(pred, matrix) if mm == m]
        ys = [v for v, mm in zip(meas, matrix) if mm == m]
        axes[0].scatter(xs, ys, s=24, label=m, color=colors.get(m, "#555555"))
    lo = min(pred + meas)
    hi = max(pred + meas)
    axes[0].plot([lo, hi], [lo, hi], "--", color="black", lw=1.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("predicted lifetime")
    axes[0].set_ylabel("measured lifetime")
    axes[0].set_title("A  Lifetime prediction")
    axes[0].legend(frameon=False)

    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for r in life:
        grouped[r.get("matrix", "")].append((ffloat(r.get("epsilon")), ffloat(r.get("measured_lifetime")), ffloat(r.get("predicted_lifetime"))))
    for m, vals in grouped.items():
        vals = sorted(vals)
        axes[1].plot([v[0] for v in vals], [v[1] for v in vals], "o-", ms=3, lw=1.0, color=colors.get(m, "#555555"), label=m)
        axes[1].plot([v[0] for v in vals], [v[2] for v in vals], "--", lw=0.9, color=colors.get(m, "#555555"), alpha=0.65)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"breaking $\epsilon$")
    axes[1].set_ylabel("lifetime")
    axes[1].set_title("B  Pseudo-gap controls memory")

    expected = [ffloat(r.get("lambda_expected")) for r in breaking]
    measured = [ffloat(r.get("lambda_measured")) for r in breaking]
    eqerr = [ffloat(r.get("equivariance_error")) for r in breaking]
    sc = axes[2].scatter(expected, measured, c=eqerr, s=18, cmap="viridis")
    lo = min(expected + measured)
    hi = max(expected + measured)
    axes[2].plot([lo, hi], [lo, hi], "--", color="black", lw=1.0)
    axes[2].set_xlabel(r"predicted $\lambda_{\mathrm{sym}}$")
    axes[2].set_ylabel(r"measured $\lambda_{\mathrm{sym}}$")
    axes[2].set_title("C  Random breaking")
    cbar = fig.colorbar(sc, ax=axes[2], fraction=0.05, pad=0.02)
    cbar.set_label("equivariance error")

    savefig(fig, "fig4_pseudogap_breaking_clean")


def plot_fig5_consequence_null() -> None:
    path_rows = read_csv(JOURNAL / "exp15_path_integration_benchmark" / "path_integration_benchmark.csv")
    grid_disc = read_csv(JOURNAL / "exp17_finite_grid_null" / "discrete_cyclic_equivariance.csv")
    grid_cont = read_csv(JOURNAL / "exp17_finite_grid_null" / "continuous_shift_equivariance.csv")
    gru_metrics = read_json(JOURNAL / "exp29_gru_path_integration_sweep" / "metrics.json", {})
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.55), constrained_layout=True)

    for label, filt, color in [
        ("exact", lambda r: r.get("model") == "exact", "#1f8a70"),
        ("broken eps=0.06", lambda r: r.get("model") == "broken" and abs(ffloat(r.get("epsilon")) - 0.06) < 1e-12, "#9b2d30"),
        ("GRU tested", lambda r: r.get("model") == "GRU", "#7a4da3"),
    ]:
        grouped: dict[float, list[float]] = defaultdict(list)
        for r in path_rows:
            if filt(r):
                grouped[ffloat(r.get("horizon"))].append(ffloat(r.get("rmse")))
        xs = sorted(grouped)
        ys = [mean_or_nan(grouped[x]) for x in xs]
        sem = [sem_or_nan(grouped[x]) for x in xs]
        axes[0].errorbar(xs, ys, yerr=sem, fmt="o-", ms=3, lw=1.1, capsize=2, label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("test horizon")
    axes[0].set_ylabel("circular RMSE")
    axes[0].set_title("A  Path integration consequence")
    axes[0].legend(frameon=False)

    disc_x = [ffloat(r.get("element")) for r in grid_disc]
    disc_y = [max(1e-18, ffloat(r.get("error"))) for r in grid_disc]
    cont_x = [ffloat(r.get("element")) for r in grid_cont]
    cont_y = [max(1e-18, ffloat(r.get("error"))) for r in grid_cont]
    axes[1].plot(disc_x, disc_y, "o", ms=3, color="#1f8a70", label=r"integer roll $C_N$")
    axes[1].plot(cont_x, cont_y, "-", lw=1.2, color="#9b2d30", label="generic shift")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("shift element")
    axes[1].set_ylabel("equivariance error")
    axes[1].set_title("B  Finite-grid null")
    axes[1].legend(frameon=False)

    vals = [
        ffloat(gru_metrics.get("mean_exact_long_ood_rmse")),
        ffloat(gru_metrics.get("mean_broken_long_ood_rmse")),
        ffloat(gru_metrics.get("mean_gru_best_long_ood_rmse")),
    ]
    axes[2].bar(["exact", "broken", "best GRU\nsweep"], vals, color=["#1f8a70", "#9b2d30", "#7a4da3"])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("long OOD RMSE")
    axes[2].set_title("C  Task-level comparison")

    savefig(fig, "fig5_consequence_null_clean")


def write_text_fragments() -> None:
    novelty = r"""
The equivariant-flow identity underlying our result is classical in equivariant dynamical systems and relative-equilibrium theory \cite{golubitsky1988,krupa1990}.
Our contribution is not to rediscover equivariance, but to specialize this identity into a Lyapunov-mode statement for finite-dimensional recurrent neural dynamics, identify the orbit and stabilizer assumptions needed for neutral memory coordinates, provide diagnostics that distinguish protected neutral modes from generic finite-time near-zero QR exponents, and show how explicit symmetry breaking opens a pseudo-gap that controls finite memory lifetime and path-integration drift.
The numerical protocol combines Oseledets/Benettin-style Lyapunov computations \cite{oseledets1968,benettin1980Theory,benettin1980Numerical,wolf1985}, covariant or principal-angle alignment diagnostics \cite{ginelli2007}, continuous-attractor and path-integration tests \cite{seung1996,seung1998,burakFiete2009}, and secondary chaotic-RNN robustness checks motivated by random-rate-network theory \cite{sompolinsky1988}.
Gradient Flossing is adjacent motivation for controlling Jacobian structure during learning, but the present claims concern exact symmetry-protected tangent directions in the vector field \cite{engelken2023gradientflossing}.
"""
    (FIX / "text_novelty_framing.tex").write_text(textwrap.dedent(novelty).strip() + "\n", encoding="utf-8")

    gru = r"""
The GRU sweep is a controlled task-level comparison, not theorem evidence.
It does not prove that generic GRUs cannot integrate phase.
It shows that the tested unconstrained GRU sweep did not recover the tested long-horizon out-of-distribution behavior supplied by exact symmetry in this construction.
Most GRU training details and heatmaps are therefore kept in the appendix, while the main text uses the sweep only as functional context for the exact and explicitly broken integrators.
"""
    (FIX / "text_gru_caveat.tex").write_text(textwrap.dedent(gru).strip() + "\n", encoding="utf-8")

    theorem_summary = """
# Theorem Revision Summary

- The revised theorem is stated for a finite-dimensional smooth state space M, a complete C1 flow phi_t generated by dot{x}=f(x), and a smooth Lie-group action of G on M.
- Exact equivariance is stated as phi_t(g x)=g phi_t(x).
- The invariant set K is compact and supports an invariant measure for which Lyapunov exponents exist.
- The stabilizer type H is constant along K and q=dim(G/H).
- The group-tangent bundle E^G_x=T_x(G dot x) is assumed uniformly nondegenerate on K.
- The proof differentiates phi_t(exp(s xi)x)=exp(s xi)phi_t(x) to obtain D phi_t(x) xi_M(x)=xi_M(phi_t(x)).
- Compactness and nondegeneracy give uniform upper/lower norm bounds, so the restricted exponential growth rate is zero.
- The conclusion is that the full spectrum contains at least q symmetry-protected zero exponents, counted with multiplicity.
- A separate paragraph now explains the relation to ordinary autonomous-flow zero exponents and points to the autonomous-flow diagnostic.
"""
    (FIX / "theorem_revision_summary.md").write_text(textwrap.dedent(theorem_summary).strip() + "\n", encoding="utf-8")


def latex_main() -> str:
    return r"""
\documentclass[10pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{hyperref}
\newtheorem{theorem}{Theorem}
\newcommand{\R}{\mathbb{R}}
\newcommand{\dd}{\mathrm{d}}
\title{Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks}
\author{Conference-fix draft generated from the reproducible Goldstone-Lyapunov codebase}
\date{}
\begin{document}
\maketitle

\begin{abstract}
Exact continuous Lie-group equivariance in a finite-dimensional recurrent vector field gives geometrically identifiable neutral Lyapunov directions when trajectories remain on a persistent nondegenerate group orbit.
We call these directions symmetry-protected Lyapunov neutral modes, or Goldstone-like modes in a dynamical systems sense, and not literal quantum Goldstone bosons \cite{goldstone1961,goldstoneSalamWeinberg1962}.
The main evidence is a theorem, exact equivariance checks, direct group-tangent exponent measurements, tangent-subspace alignment diagnostics, dimension-law scaling, stabilizer controls, and symmetry-breaking pseudo-gap scaling.
Chaotic RNN spectra, path integration, and GRU comparisons are consequences and robustness diagnostics rather than proof of the theorem.
\end{abstract}

\section{Framing and Contribution}
The equivariant-flow identity underlying our result is classical in equivariant dynamical systems and relative-equilibrium theory \cite{golubitsky1988,krupa1990}.
Our contribution is not to rediscover equivariance, but to specialize this identity into a Lyapunov-mode statement for finite-dimensional recurrent neural dynamics, identify the orbit and stabilizer assumptions needed for neutral memory coordinates, provide diagnostics that distinguish protected neutral modes from generic finite-time near-zero QR exponents, and show how explicit symmetry breaking opens a pseudo-gap that controls finite memory lifetime and path-integration drift.
The numerical protocol combines Oseledets and Benettin-style Lyapunov computations \cite{oseledets1968,benettin1980Theory,benettin1980Numerical,wolf1985}, covariant or principal-angle alignment diagnostics \cite{ginelli2007}, continuous-attractor and path-integration tests \cite{seung1996,seung1998,burakFiete2009}, and secondary chaotic-RNN robustness checks motivated by random-rate-network theory \cite{sompolinsky1988}.
Gradient Flossing is adjacent motivation for controlling Jacobian structure during learning, but the present claims concern exact symmetry-protected tangent directions in the vector field \cite{engelken2023gradientflossing}.

\section{Theorem}
Let \(M\) be a finite-dimensional smooth state space, let \(\dot{x}=f(x)\) generate a complete \(C^1\) flow \(\phi_t\), and let a Lie group \(G\) act smoothly on \(M\).
The vector field is exactly \(G\)-equivariant when \(\phi_t(g\cdot x)=g\cdot \phi_t(x)\) for all \(g\in G\), \(x\in M\), and \(t\in\R\).
Let \(K\subset M\) be compact, invariant, and support an invariant measure for which Lyapunov exponents exist \cite{oseledets1968}.
Assume that the stabilizer type is constant on \(K\), equal to \(H\), and write \(q=\dim(G/H)\).
Assume also that the group-tangent bundle \(E^G_x=T_x(G\cdot x)=\{\xi_M(x):\xi\in\mathfrak{g}\}\) has rank \(q\) and is uniformly nondegenerate on \(K\).

\begin{theorem}[Symmetry-protected neutral Lyapunov modes]
Under the assumptions above, \(E^G\) is invariant under \(D\phi_t\), all Lyapunov exponents of \(D\phi_t|_{E^G}\) are zero, and the full Lyapunov spectrum contains at least \(q=\dim(G/H)\) zero exponents counted with multiplicity.
\end{theorem}

\begin{proof}
Exact equivariance gives \(\phi_t(g\cdot x)=g\cdot\phi_t(x)\).
For \(g(s)=\exp(s\xi)\), differentiating the identity at \(s=0\) gives
\[
D\phi_t(x)\,\xi_M(x)=\xi_M(\phi_t(x)).
\]
Thus \(E^G_x\) is mapped into \(E^G_{\phi_t(x)}\).
The constant-rank and stabilizer assumptions make the map onto \(E^G_{\phi_t(x)}\), so \(E^G\) is an invariant tangent subbundle.
Uniform nondegeneracy on compact \(K\) gives constants \(0<c<C<\infty\) such that nonzero group tangents have norms bounded between \(c\|\xi\|\) and \(C\|\xi\|\) after choosing local generators.
The identity above therefore bounds \(\|D\phi_t(x)v\|\) above and below by constants independent of \(t\) for every nonzero \(v\in E^G_x\).
Taking \(t^{-1}\log\|D\phi_t(x)v\|\) and passing to Lyapunov limits gives zero.
Since \(\operatorname{rank}E^G=q\), the full spectrum contains at least \(q\) symmetry-protected zero exponents.
\end{proof}

\paragraph{Relation to ordinary autonomous-flow zeros.}
Autonomous flows can have a zero exponent in the time-translation direction \(f(x)\).
The protected modes here are generated by infinitesimal group actions \(\xi_M(x)\).
In some relative-equilibrium examples, one group direction may coincide with the flow.
Product-group and \(q>1\) examples test multiplicity beyond the time-translation direction.
The paper reports direct group-tangent exponents and tangent-subspace alignment rather than relying on a single zero exponent, as summarized in Figure~\ref{fig:geometry}.

\section{Verification Protocol and Main Evidence}
Figure~\ref{fig:dimension} tests the dimension law across exact continuous product tori, real sphere orbits \(SO(n)/SO(n-1)\), complex sphere orbits \(U(m)/U(m-1)\), and additive product-group examples.
The finite-dimensional theorem models use exact Lie-group representations, not finite circulant grids.
The observed near-zero counts match the predicted orbit dimension within the stated numerical tolerance in the full run summarized by the reproducibility manifest.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures_clean/fig1_dimension_law_clean.pdf}
\caption{Dimension-law evidence for exact continuous symmetry models. The panels support the theorem-level claim that exact equivariance plus a persistent nondegenerate orbit gives at least \(\dim(G/H)\) near-zero exponents in numerical spectra, while the inset spectra illustrate rather than prove the theorem.}
\label{fig:dimension}
\end{figure}

Figure~\ref{fig:geometry} checks that the numerical neutral subspaces align with analytical group tangents and separates those tangents from the ordinary autonomous-flow direction.
These diagnostics are the primary empirical checks because finite-time QR spectra in chaotic systems can contain noisy near-zero values.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures_clean/fig2_neutral_geometry_clean.pdf}
\caption{Neutral-subspace geometry and autonomous-flow caveat. Principal-angle diagnostics identify the group-tangent subspace directly, and the flow diagnostic reports whether the analytical protected directions are independent of \(f(x)\).}
\label{fig:geometry}
\end{figure}

Figure~\ref{fig:rnn} replaces the previous nontrivial-equvariant-RNN atlas image with a compact diagnostic built from raw experiment metrics.
The exact hidden-rate irreducible architecture has machine-precision equivariance error, near-zero direct group-tangent exponent, and sub-microdegree tangent alignment, while the explicitly broken control loses the protected zero mode.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures_clean/fig3_equivariant_rnn_clean.pdf}
\caption{A non-radial equivariant RNN-style construction and its broken control. This figure supports the claim that the theorem is not limited to uncoupled radial normal forms; it does not claim that the broken control is a trained baseline.}
\label{fig:rnn}
\end{figure}

Figure~\ref{fig:pseudogap} shows that explicit symmetry breaking opens a pseudo-gap and that the measured finite memory lifetime follows the predicted gap-controlled scale.
The random anisotropic breaking ensemble further checks that the displaced symmetry exponent follows perturbative predictions rather than an index-by-index spectral coincidence.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures_clean/fig4_pseudogap_breaking_clean.pdf}
\caption{Symmetry breaking opens pseudo-gaps that predict finite memory lifetime. The figure supports the breaking-to-lifetime consequence of the theorem, not a claim about quantum Goldstone particles.}
\label{fig:pseudogap}
\end{figure}

Figure~\ref{fig:consequence} reports task-level consequences and a finite-grid null control.
The GRU sweep is a controlled comparison, not theorem evidence, and it does not prove that generic GRUs cannot integrate.
The finite-grid control shows exact integer-roll \(C_N\) symmetry and failure under generic continuous shifts, so finite circulant grids are not used as exact continuous \(S^1\) theorem models.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures_clean/fig5_consequence_null_clean.pdf}
\caption{Task consequence and null control. Exact symmetry stabilizes long-horizon path integration in this construction, while finite grids serve as visualization/null controls rather than exact continuous-symmetry evidence.}
\label{fig:consequence}
\end{figure}

\section{Scope of the Claims}
The main theorem is a Lyapunov-mode specialization of classical equivariant-flow identities for finite-dimensional recurrent dynamics.
The chaotic RNN experiments are secondary robustness diagnostics because finite-time Lyapunov spectra in chaotic systems are noisy and can include the ordinary autonomous-flow zero exponent.
The path-integration experiments show functional relevance of protected and pseudo-gapped phase modes, but they are not a proof of the theorem.
The GRU sweep shows only that the tested unconstrained models did not recover the tested long-horizon out-of-distribution behavior supplied by exact symmetry in this construction.

\section{Reproducibility}
All plotted measurements in the main figures are regenerated from raw CSV or JSON files under \texttt{results/journal\_full}.
The reviewer-facing reproducibility manifest lists vector fields, group actions, solver tolerances, integration settings, random seeds, raw table paths, scripts, test status, and missing details marked as TODO\_MISSING\_*.

\bibliographystyle{plain}
\bibliography{references}
\end{document}
"""


def latex_appendix() -> str:
    return r"""
\section*{Appendix: Additional Diagnostics}
This appendix is intended to accompany \texttt{main\_neurips\_revised.tex}.
The full atlas, finite-time chaotic spectra, path-integration heatmaps, GRU sweep heatmaps, claim hierarchy, and claim audit tables are appendix-only materials.
Chaotic finite-time zero counts are treated as convergence diagnostics rather than theorem evidence.
The finite-grid null is included to emphasize that integer shifts on a discretized ring provide exact \(C_N\) symmetry, not exact continuous \(S^1\) symmetry.
Most GRU details are kept outside the main text because the GRU sweep is a task-level comparison and not a universal statement about gated recurrent networks.
"""


def latex_repro_appendix() -> str:
    return r"""
\section*{Appendix: Reproducibility Manifest}
The machine-readable manifest is stored at \texttt{results/conference\_fix/reproducibility\_manifest.json}.
The human-readable manifest is stored at \texttt{results/conference\_fix/reproducibility\_manifest.md}.
The experiment registry is stored at \texttt{results/conference\_fix/experiment\_registry.csv}.
Each main figure is regenerated by \texttt{paper\_runs/conference\_fix.py} from raw data under \texttt{results/journal\_full}.
Missing details are marked with \texttt{TODO\_MISSING\_*} rather than inferred.
"""


def write_latex_files() -> None:
    (ROOT / "main_neurips_revised.tex").write_text(textwrap.dedent(latex_main()).strip() + "\n", encoding="utf-8")
    (ROOT / "appendix_neurips_revised.tex").write_text(textwrap.dedent(latex_appendix()).strip() + "\n", encoding="utf-8")
    (ROOT / "appendix_reproducibility.tex").write_text(textwrap.dedent(latex_repro_appendix()).strip() + "\n", encoding="utf-8")


def write_bibliography() -> None:
    bib_src = DOCS / "references.bib"
    text = bib_src.read_text(encoding="utf-8") if bib_src.exists() else ""
    additions = {
        "oseledets1968": r"""
@article{oseledets1968,
  author = {Oseledets, V. I.},
  title = {A Multiplicative Ergodic Theorem. Lyapunov Characteristic Numbers for Dynamical Systems},
  journal = {Transactions of the Moscow Mathematical Society},
  volume = {19},
  pages = {197--231},
  year = {1968}
}
""",
        "engelken2023gradientflossing": r"""
@inproceedings{engelken2023gradientflossing,
  author = {Engelken, Rainer},
  title = {Gradient Flossing: Improving Gradient Descent through Dynamic Control of Jacobians},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023}
}
""",
    }
    report = ["# Bibliography Cleanup Report", ""]
    for key, entry in additions.items():
        if f"{{{key}," not in text:
            text += "\n" + textwrap.dedent(entry).strip() + "\n"
            report.append(f"- Added missing entry `{key}`.")
        else:
            report.append(f"- Entry `{key}` already present.")
    keys = re.findall(r"@\w+\{([^,\s]+),", text)
    duplicates = sorted(k for k in set(keys) if keys.count(k) > 1)
    if duplicates:
        report.append(f"- TODO_MISSING_BIB_CLEANUP: duplicate keys remain: {', '.join(duplicates)}")
    else:
        report.append("- No duplicate BibTeX keys detected.")
    required = [
        "oseledets1968",
        "benettin1980Theory",
        "benettin1980Numerical",
        "wolf1985",
        "ginelli2007",
        "golubitsky1988",
        "krupa1990",
        "goldstone1961",
        "goldstoneSalamWeinberg1962",
        "seung1996",
        "seung1998",
        "burakFiete2009",
        "sompolinsky1988",
        "engelken2023gradientflossing",
    ]
    missing = [key for key in required if f"{{{key}," not in text]
    if missing:
        report.append(f"- TODO_MISSING_BIB_ENTRY: {', '.join(missing)}")
    else:
        report.append("- All requested bibliography keys are present.")
    (ROOT / "references.bib").write_text(text.strip() + "\n", encoding="utf-8")
    (FIX / "bib_cleanup_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def write_repro_manifest() -> None:
    status = read_json(JOURNAL / "journal_full_status.json", {})
    experiments = status.get("experiments", {}) if isinstance(status, dict) else {}
    registry_rows = []
    for name, info in sorted(experiments.items()):
        out_dir = JOURNAL / name
        metrics = out_dir / "metrics.json"
        registry_rows.append(
            {
                "experiment": name,
                "passed": info.get("passed"),
                "inconclusive": info.get("inconclusive"),
                "quick": info.get("quick"),
                "status": info.get("status"),
                "output_dir": rel(out_dir),
                "metrics_json": rel(metrics) if metrics.exists() else "TODO_MISSING_METRICS_JSON",
            }
        )
    write_csv(FIX / "experiment_registry.csv", registry_rows)

    raw_tables = {
        "dimension_law": rel(JOURNAL / "exp23_dimension_law_torus_sweep" / "dimension_law_torus_sweep.csv"),
        "principal_angle": rel(JOURNAL / "exp18_clv_principal_angle_sweep" / "principal_angle_sweep.csv"),
        "nontrivial_equivariant_rnn": rel(JOURNAL / "exp28_nontrivial_equivariant_rnn" / "nontrivial_equivariant_rnn_metrics.csv"),
        "pseudogap_lifetime": rel(JOURNAL / "exp19_pseudogap_lifetime" / "pseudogap_lifetime.csv"),
        "random_breaking": rel(JOURNAL / "exp24_random_breaking_ensemble" / "random_breaking_ensemble.csv"),
        "path_integration": rel(JOURNAL / "exp15_path_integration_benchmark" / "path_integration_benchmark.csv"),
        "gru_sweep": rel(JOURNAL / "exp29_gru_path_integration_sweep" / "gru_path_integration_sweep.csv"),
        "finite_grid_discrete": rel(JOURNAL / "exp17_finite_grid_null" / "discrete_cyclic_equivariance.csv"),
        "finite_grid_continuous": rel(JOURNAL / "exp17_finite_grid_null" / "continuous_shift_equivariance.csv"),
    }
    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "project": "Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks",
        "raw_results_root": rel(JOURNAL),
        "figure_output_dir": rel(FIGS),
        "main_figure_script": rel(ROOT / "paper_runs" / "conference_fix.py"),
        "model_definitions": {
            "S1Attractor": "dot z = alpha (R^2-|z|^2) z, S1 action by complex phase rotation.",
            "T2Attractor": "two independent S1 blocks, T2 action by independent phase rotations.",
            "SONSphereAttractor": "dot x = alpha (R^2-||x||^2) x, SO(n) action on R^n, stabilizer SO(n-1).",
            "UMSphereAttractor": "same radial sphere in real coordinates for C^m, U(m) orbit with stabilizer U(m-1).",
            "S1CoupledIrrepAttractor": "weighted S1 representation with charge-1 z, charge-2 w, invariant hidden rates, and optional breaking.",
            "PhaseIntegrator": "controlled S1 phase integrator, exact when epsilon=0 and finite scalar input is treated equivariantly.",
        },
        "group_actions": {
            "S1": "2D rotation matrix or complex multiplication by exp(i theta).",
            "T2": "independent rotations on two complex amplitudes.",
            "SO(n)": "orthogonal determinant-one rotations of a real vector.",
            "U(m)": "unitary rotations of complex vector represented as interleaved real coordinates.",
            "finite_grid_control": "integer rolls are exact C_N actions; generic continuous shifts are not exact continuous S1 actions.",
        },
        "solver": {
            "lyapunov_method": "Benettin QR reorthonormalization with RK4 augmented state/tangent integration.",
            "near_zero_threshold_full": "0.0001 to 0.0003 depending on experiment metrics; see each metrics.json.",
            "qr_reorthonormalization_interval": "TODO_MISSING_REPRO_DETAIL: per-experiment values are not centralized in a single config file.",
            "integration_times": "TODO_MISSING_REPRO_DETAIL: see individual experiment scripts and metrics where recorded.",
            "random_seeds": "Recorded in experiment metrics and raw CSV tables.",
        },
        "raw_tables": raw_tables,
        "tests": {
            "pytest_status": "Filled in build_report.md after checks run.",
            "pytest_log": rel(FIX / "pytest_output.txt"),
        },
        "hardware_compute": "TODO_MISSING_REPRO_DETAIL: hardware summary was not recorded in the existing run outputs.",
        "experiment_registry": rel(FIX / "experiment_registry.csv"),
    }
    write_json(FIX / "reproducibility_manifest.json", manifest)

    md = [
        "# Reproducibility Manifest",
        "",
        f"- Generated: {manifest['generated']}",
        f"- Raw full results root: `{manifest['raw_results_root']}`",
        f"- Clean figure directory: `{manifest['figure_output_dir']}`",
        f"- Main generation script: `{manifest['main_figure_script']}`",
        "",
        "## Models and Group Actions",
    ]
    for name, desc in manifest["model_definitions"].items():
        md.append(f"- `{name}`: {desc}")
    md.append("")
    md.append("## Solver and Numerical Settings")
    for key, val in manifest["solver"].items():
        md.append(f"- `{key}`: {val}")
    md.append("")
    md.append("## Raw Result Tables")
    for key, val in raw_tables.items():
        md.append(f"- `{key}`: `{val}`")
    md.append("")
    md.append("## Missing Details")
    md.append(f"- {manifest['hardware_compute']}")
    md.append("- TODO_MISSING_REPRO_DETAIL items above should be filled from experiment-specific configs before final camera-ready archival.")
    (FIX / "reproducibility_manifest.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def sentence_split_latex(text: str) -> list[str]:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " MATH ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def classify_sentence(sentence: str, raw: str) -> str:
    lower = sentence.lower()
    if "\\cite" in raw:
        return "external_literature"
    if "figure~\\ref" in raw or "table~\\ref" in raw or "Figure" in sentence:
        return "figure_or_table"
    if any(
        word in lower
        for word in [
            "theorem",
            "proof",
            "assume",
            "assumption",
            "equivariance",
            "differentiating",
            "nondegeneracy",
            "lyapunov limits",
            "full spectrum",
            "protected modes",
            "infinitesimal group actions",
            "relative-equilibrium",
            "product-group",
            "identity above",
            "autonomous flows",
            "rank",
        ]
    ):
        return "theorem_or_proof"
    if any(
        word in lower
        for word in [
            "diagnostics",
            "machine-precision",
            "hidden-rate",
            "random anisotropic",
            "gru sweep",
            "chaotic rnn experiments",
            "finite-time qr",
        ]
    ):
        return "figure_or_table"
    if any(word in lower for word in ["reproducibility", "manifest", "raw csv", "results/journal"]):
        return "reproducibility_manifest"
    if any(word in lower for word in ["let ", "write ", "denote", "definition", "called", "not literal", "the vector field"]):
        return "definition_or_notational"
    return "claim_needs_support"


def source_support_audit() -> None:
    tex_path = ROOT / "main_neurips_revised.tex"
    raw_text = tex_path.read_text(encoding="utf-8")
    raw_sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", raw_text))
    clean_sentences = sentence_split_latex(raw_text)
    rows = []
    for idx, sentence in enumerate(clean_sentences, start=1):
        raw_match = raw_sentences[min(idx - 1, len(raw_sentences) - 1)] if raw_sentences else sentence
        category = classify_sentence(sentence, raw_match)
        proposed = ""
        if category == "claim_needs_support":
            proposed = "Add a citation, theorem reference, figure reference, or remove/soften this claim."
        rows.append({"sentence_id": idx, "category": category, "sentence": sentence, "proposed_fix": proposed})
    write_csv(FIX / "source_support_audit.csv", rows, fieldnames=["sentence_id", "category", "sentence", "proposed_fix"])
    counts = defaultdict(int)
    for row in rows:
        counts[row["category"]] += 1
    flagged = [row for row in rows if row["category"] == "claim_needs_support"]
    md = [
        "# Source-Support Audit",
        "",
        f"- Total sentences audited: {len(rows)}",
        f"- external_literature: {counts['external_literature']}",
        f"- theorem_or_proof: {counts['theorem_or_proof']}",
        f"- figure_or_table: {counts['figure_or_table']}",
        f"- reproducibility_manifest: {counts['reproducibility_manifest']}",
        f"- definition_or_notational: {counts['definition_or_notational']}",
        f"- flagged claim_needs_support: {counts['claim_needs_support']}",
        "",
        "## Flagged Sentences",
    ]
    if flagged:
        for row in flagged:
            md.append(f"- Sentence {row['sentence_id']}: {row['sentence']} Proposed fix: {row['proposed_fix']}")
    else:
        md.append("- None flagged by the practical audit.")
    md.append("")
    md.append("Note: this is a practical automated audit, not a substitute for human citation review.")
    (FIX / "source_support_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def run_command(command: list[str], cwd: Path, timeout: int = 600) -> dict[str, Any]:
    started = datetime.now().isoformat(timespec="seconds")
    try:
        proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
        return {
            "command": " ".join(command),
            "started": started,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "started": started,
            "exit_code": 124,
            "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def run_checks_and_build_report() -> None:
    commands: list[dict[str, Any]] = []
    pytest_result = run_command([sys.executable, "-m", "pytest", "-q"], ROOT, timeout=900)
    commands.append(pytest_result)
    (FIX / "pytest_output.txt").write_text(pytest_result["stdout"] + "\n" + pytest_result["stderr"], encoding="utf-8")

    latex_results: list[dict[str, Any]] = []
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if pdflatex:
        latex_results.append(run_command([pdflatex, "-interaction=nonstopmode", "main_neurips_revised.tex"], ROOT, timeout=300))
        if bibtex:
            latex_results.append(run_command([bibtex, "main_neurips_revised"], ROOT, timeout=120))
            latex_results.append(run_command([pdflatex, "-interaction=nonstopmode", "main_neurips_revised.tex"], ROOT, timeout=300))
            latex_results.append(run_command([pdflatex, "-interaction=nonstopmode", "main_neurips_revised.tex"], ROOT, timeout=300))
        commands.extend(latex_results)
    else:
        commands.append(
            {
                "command": "pdflatex -interaction=nonstopmode main_neurips_revised.tex",
                "started": datetime.now().isoformat(timespec="seconds"),
                "exit_code": "not_run",
                "stdout": "",
                "stderr": "pdflatex not found on PATH.",
                "timed_out": False,
            }
        )

    pdf_path = ROOT / "main_neurips_revised.pdf"
    page_count = "TODO_MISSING_BUILD_DETAIL: PDF was not compiled."
    if pdf_path.exists():
        page_count = "TODO_MISSING_BUILD_DETAIL: PDF page count tool not available in this script."
        try:
            data = pdf_path.read_bytes()
            page_count = str(len(re.findall(rb"/Type\s*/Page\b", data)))
        except Exception:
            pass

    required = [
        path
        for path in expected_files()
        if path
        not in {
            "results/conference_fix/build_report.md",
            "results/conference_fix/UPLOAD_TO_CHATGPT_SUMMARY.md",
        }
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    report = [
        "# Build Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Commands Run",
    ]
    for result in commands:
        report.append(f"- `{result['command']}`")
        report.append(f"  - exit code: {result['exit_code']}")
        if result.get("timed_out"):
            report.append("  - warning: command timed out")
        if result["stderr"]:
            first = result["stderr"].strip().splitlines()[:5]
            report.append("  - stderr excerpt: " + " | ".join(first))
    report.extend(
        [
            "",
            "## Generated Files",
            f"- Missing expected files: {', '.join(missing) if missing else 'none'}",
            f"- NeurIPS PDF compiled: {pdf_path.exists()}",
            f"- PDF page count: {page_count}",
            "- NeurIPS style file not present; compiled with available local article style." if pdf_path.exists() else "- Could not compile a PDF without a local pdflatex executable or full LaTeX toolchain.",
            "- Main-text page-count compliance cannot be asserted without the official NeurIPS style file.",
        ]
    )
    (FIX / "build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(FIX / "build_report_commands.json", commands)


def expected_files() -> list[str]:
    return [
        "main_neurips_revised.tex",
        "appendix_neurips_revised.tex",
        "appendix_reproducibility.tex",
        "references.bib",
        "figures_clean/fig1_dimension_law_clean.pdf",
        "figures_clean/fig1_dimension_law_clean.png",
        "figures_clean/fig2_neutral_geometry_clean.pdf",
        "figures_clean/fig2_neutral_geometry_clean.png",
        "figures_clean/fig3_equivariant_rnn_clean.pdf",
        "figures_clean/fig3_equivariant_rnn_clean.png",
        "figures_clean/fig4_pseudogap_breaking_clean.pdf",
        "figures_clean/fig4_pseudogap_breaking_clean.png",
        "figures_clean/fig5_consequence_null_clean.pdf",
        "figures_clean/fig5_consequence_null_clean.png",
        "figures_clean/fig_autonomous_zero_diagnostic.pdf",
        "figures_clean/fig_autonomous_zero_diagnostic.png",
        "results/conference_fix/autonomous_zero_diagnostic.csv",
        "results/conference_fix/autonomous_zero_diagnostic.md",
        "results/conference_fix/reproducibility_manifest.md",
        "results/conference_fix/reproducibility_manifest.json",
        "results/conference_fix/experiment_registry.csv",
        "results/conference_fix/source_support_audit.csv",
        "results/conference_fix/source_support_audit.md",
        "results/conference_fix/build_report.md",
        "results/conference_fix/bib_cleanup_report.md",
        "results/conference_fix/theorem_revision_summary.md",
        "results/conference_fix/UPLOAD_TO_CHATGPT_SUMMARY.md",
    ]


def create_upload_summary() -> None:
    exp28_metrics = read_json(JOURNAL / "exp28_nontrivial_equivariant_rnn" / "metrics.json", {})
    exp29_metrics = read_json(JOURNAL / "exp29_gru_path_integration_sweep" / "metrics.json", {})
    exp19_metrics = read_json(JOURNAL / "exp19_pseudogap_lifetime" / "metrics.json", {})
    audit = (FIX / "source_support_audit.md").read_text(encoding="utf-8") if (FIX / "source_support_audit.md").exists() else ""
    flagged_match = re.search(r"flagged claim_needs_support: (\d+)", audit)
    flagged = flagged_match.group(1) if flagged_match else "TODO_MISSING_AUDIT_COUNT"
    build = (FIX / "build_report.md").read_text(encoding="utf-8") if (FIX / "build_report.md").exists() else ""
    missing_line = next((line for line in build.splitlines() if "Missing expected files" in line), "- Missing expected files: TODO_MISSING_BUILD_REPORT")
    pytest_line = "pytest status: see build_report.md"
    if "pytest -q`" in build or "-m pytest -q" in build:
        pytest_line = "pytest was run; see build_report.md and pytest_output.txt for exit code."
    upload_files = [
        "main_neurips_revised.tex",
        "appendix_neurips_revised.tex",
        "appendix_reproducibility.tex",
        "references.bib",
        "figures_clean/fig1_dimension_law_clean.pdf",
        "figures_clean/fig1_dimension_law_clean.png",
        "figures_clean/fig2_neutral_geometry_clean.pdf",
        "figures_clean/fig2_neutral_geometry_clean.png",
        "figures_clean/fig3_equivariant_rnn_clean.pdf",
        "figures_clean/fig3_equivariant_rnn_clean.png",
        "figures_clean/fig4_pseudogap_breaking_clean.pdf",
        "figures_clean/fig4_pseudogap_breaking_clean.png",
        "figures_clean/fig5_consequence_null_clean.pdf",
        "figures_clean/fig5_consequence_null_clean.png",
        "figures_clean/fig_autonomous_zero_diagnostic.pdf",
        "figures_clean/fig_autonomous_zero_diagnostic.png",
        "results/conference_fix/repo_asset_map.md",
        "results/conference_fix/autonomous_zero_diagnostic.csv",
        "results/conference_fix/autonomous_zero_diagnostic.md",
        "results/conference_fix/reproducibility_manifest.md",
        "results/conference_fix/reproducibility_manifest.json",
        "results/conference_fix/experiment_registry.csv",
        "results/conference_fix/source_support_audit.csv",
        "results/conference_fix/source_support_audit.md",
        "results/conference_fix/build_report.md",
        "results/conference_fix/bib_cleanup_report.md",
        "results/conference_fix/theorem_revision_summary.md",
        "results/conference_fix/text_autonomous_zero_caveat.tex",
        "results/conference_fix/text_novelty_framing.tex",
        "results/conference_fix/text_gru_caveat.tex",
    ]
    summary = [
        "# Upload to ChatGPT Summary",
        "",
        "## What Changed",
        "- Created a clean conference-fix package around the safer thesis: exact continuous equivariance plus a persistent nondegenerate group orbit gives symmetry-protected Lyapunov neutral modes.",
        "- Added an autonomous-flow zero-exponent diagnostic to separate group tangents from ordinary time-translation directions.",
        "- Rewrote novelty, theorem, proof, autonomous-flow caveat, and GRU caveat text fragments.",
        "- Generated a clean five-figure NeurIPS-style set from raw CSV/JSON outputs rather than using atlas panels.",
        "- Created a reviewer-facing reproducibility manifest, experiment registry, bibliography cleanup report, source-support audit, and build report.",
        "",
        "## New Diagnostics Added",
        "- `results/conference_fix/autonomous_zero_diagnostic.csv`",
        "- `results/conference_fix/autonomous_zero_diagnostic.md`",
        "- `figures_clean/fig_autonomous_zero_diagnostic.pdf` and `.png`",
        "",
        "## New Clean Figures",
        "- Figure 1: dimension law across product tori, SO(n), U(m), and product-group families.",
        "- Figure 2: principal-angle neutral geometry plus autonomous-flow caveat.",
        "- Figure 3: clean nontrivial equivariant RNN architecture and exact/broken diagnostics.",
        "- Figure 4: pseudo-gap lifetime and random breaking ensemble.",
        "- Figure 5: path-integration consequence plus finite-grid null and downweighted GRU summary.",
        "",
        "## Removed or Downweighted from Main Text",
        "- The old atlas-style overview is appendix-only material.",
        "- `results/journal_full/exp28_nontrivial_equivariant_rnn/nontrivial_equivariant_rnn.png` is not used in the revised main text.",
        "- Chaotic QR zero counts are described as finite-time diagnostics, not theorem proof.",
        "- The GRU sweep is framed as a controlled task comparison, not a universal failure of GRUs.",
        "",
        "## Validated Quantitative Anchors",
        f"- Nontrivial equivariant RNN max unbroken equivariance error: {exp28_metrics.get('max_unbroken_equivariance_error', 'TODO_MISSING_METRIC')}",
        f"- Nontrivial equivariant RNN max direct group-tangent exponent magnitude: {exp28_metrics.get('max_abs_unbroken_direct_tangent_exponent', 'TODO_MISSING_METRIC')}",
        f"- Nontrivial equivariant RNN max tangent covariance angle (deg): {exp28_metrics.get('max_unbroken_tangent_covariance_angle_degrees', 'TODO_MISSING_METRIC')}",
        f"- Broken hidden-irrep equivariance error floor: {exp28_metrics.get('min_broken_equivariance_error', 'TODO_MISSING_METRIC')}",
        f"- GRU sweep trained models: {exp29_metrics.get('trained_models', 'TODO_MISSING_METRIC')}; rows: {exp29_metrics.get('rows', 'TODO_MISSING_METRIC')}",
        f"- Exact long OOD RMSE: {exp29_metrics.get('mean_exact_long_ood_rmse', 'TODO_MISSING_METRIC')}",
        f"- Broken long OOD RMSE: {exp29_metrics.get('mean_broken_long_ood_rmse', 'TODO_MISSING_METRIC')}",
        f"- Best GRU long OOD RMSE: {exp29_metrics.get('mean_gru_best_long_ood_rmse', 'TODO_MISSING_METRIC')}",
        f"- Pseudo-gap log lifetime correlation: {exp19_metrics.get('log_lifetime_correlation', 'TODO_MISSING_METRIC')}",
        f"- Pseudo-gap uncensored fraction: {exp19_metrics.get('uncensored_fraction', 'TODO_MISSING_METRIC')}",
        "",
        "## Audit and Build Status",
        f"- Source-support audit flagged sentences: {flagged}",
        f"- {missing_line}",
        f"- {pytest_line}",
        "- LaTeX compile status and page count are recorded in `results/conference_fix/build_report.md`.",
        "",
        "## Remaining Missing Reproducibility Details",
        "- Hardware/compute summary was not present in the existing run outputs.",
        "- Some solver settings are experiment-specific and not centralized; the manifest points reviewers to experiment scripts and metrics.",
        "",
        "## Files to Upload to ChatGPT 5.5 Pro",
    ]
    summary.extend(f"- `{path}`" for path in upload_files)
    if (ROOT / "neurips_goldstone_conference_fix_package.zip").exists():
        summary.append("- `neurips_goldstone_conference_fix_package.zip`")
    (FIX / "UPLOAD_TO_CHATGPT_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def create_zip() -> None:
    zip_path = ROOT / "neurips_goldstone_conference_fix_package.zip"
    include_roots = [
        ROOT / "main_neurips_revised.tex",
        ROOT / "appendix_neurips_revised.tex",
        ROOT / "appendix_reproducibility.tex",
        ROOT / "references.bib",
        FIGS,
        FIX,
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in include_roots:
            if item.is_file():
                zf.write(item, rel(item))
            elif item.is_dir():
                for path in item.rglob("*"):
                    if path.is_file():
                        zf.write(path, rel(path))


def main() -> None:
    ensure_dirs()
    setup_plot_style()
    generate_asset_map()
    auto_rows = generate_autonomous_zero_diagnostic()
    plot_fig1_dimension_law()
    plot_fig2_neutral_geometry(auto_rows)
    plot_fig3_equivariant_rnn()
    plot_fig4_pseudogap()
    plot_fig5_consequence_null()
    write_text_fragments()
    write_bibliography()
    write_latex_files()
    write_repro_manifest()
    source_support_audit()
    run_checks_and_build_report()
    create_zip()
    create_upload_summary()
    print((FIX / "UPLOAD_TO_CHATGPT_SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
