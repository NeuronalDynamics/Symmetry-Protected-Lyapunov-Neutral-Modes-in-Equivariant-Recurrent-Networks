"""Regenerate the neutral-geometry figure with an unambiguous angle axis."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_clean"
ANGLE_CSV = ROOT / "results" / "journal_full" / "exp18_clv_principal_angle_sweep" / "principal_angle_sweep.csv"
AUTO_CSV = ROOT / "results" / "conference_fix" / "autonomous_zero_diagnostic.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ffloat(value: object, default: float = float("nan")) -> float:
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


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def make_fig2() -> None:
    angle_rows = read_csv(ANGLE_CSV)
    auto_rows = read_csv(AUTO_CSV)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.85), constrained_layout=True)

    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in angle_rows:
        grouped[(row.get("model", "model"), ffloat(row.get("window")))].append(ffloat(row.get("max_angle_degrees")))
    models = ["S1", "T2", "SO3/SO2"]
    colors = {"S1": "#275d9f", "T2": "#1f8a70", "SO3/SO2": "#b35c00"}
    for model in models:
        xs = sorted(k[1] for k in grouped if k[0] == model)
        ys = []
        es = []
        for x in xs:
            mean, sem = mean_sem(grouped[(model, x)])
            ys.append(mean)
            es.append(sem)
        if xs:
            axes[0].errorbar(xs, ys, yerr=es, marker="o", ms=3.5, lw=1.3, capsize=2, color=colors[model], label=model)
    axes[0].set_xlabel("alignment window")
    axes[0].set_ylabel("max principal angle (deg)")
    axes[0].set_title("A  Neutral subspace alignment")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-6, -6), useMathText=True)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=1)

    label_map = {
        "S1Attractor": "S1",
        "T2Attractor": "T2",
        "S1CoupledIrrepAttractor": "irrep\nRNN",
        "SONSphereAttractor_n3": "SO3",
        "SONSphereAttractor_n5": "SO5",
        "UMSphereAttractor_m3": "U3",
        "PhaseIntegrator_constant_velocity": "rel.\neq.",
        "CollapseCounterexample": "collapse\nctrl",
    }
    plot_rows = [
        row
        for row in auto_rows
        if row.get("model")
        in {
            "S1Attractor",
            "T2Attractor",
            "S1CoupledIrrepAttractor",
            "SONSphereAttractor_n3",
            "SONSphereAttractor_n5",
            "UMSphereAttractor_m3",
            "PhaseIntegrator_constant_velocity",
            "CollapseCounterexample",
        }
    ]
    ypos = np.arange(len(plot_rows))[::-1]
    xvals = []
    colors_b = []
    for row in plot_rows:
        angle = ffloat(row.get("angle_flow_to_EG_degrees"))
        if math.isfinite(angle):
            xvals.append(angle)
            colors_b.append("#275d9f" if as_bool(row.get("theorem_level")) else "#9b2d30")
        else:
            xvals.append(100.0)
            colors_b.append("#8a8a8a")
    axes[1].scatter(xvals, ypos, s=28, c=colors_b, zorder=3)
    for idx, row in enumerate(plot_rows):
        if not as_bool(row.get("flow_defined")):
            axes[1].text(96, ypos[idx] + 0.23, "f=0", ha="right", va="bottom", fontsize=6.5)
        if row.get("model") == "CollapseCounterexample":
            axes[1].text(xvals[idx] + 2.5, ypos[idx], "assumption fails", ha="left", va="center", fontsize=6.1)
    axes[1].set_xlim(-4, 106)
    axes[1].set_ylim(-0.6, len(plot_rows) - 0.4)
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels([label_map.get(row["model"], row["model"]) for row in plot_rows])
    axes[1].set_xlabel("flow-to-group angle (deg)")
    axes[1].set_title("B  Flow-zero caveat")
    axes[1].grid(True, axis="x", alpha=0.25)

    theorem_rows = [row for row in auto_rows if as_bool(row.get("theorem_level"))]
    q = [ffloat(row.get("orbit_dim_q")) for row in theorem_rows]
    indep = [ffloat(row.get("group_directions_independent_of_flow")) for row in theorem_rows]
    axes[2].scatter(q, indep, s=34, color="#1f8a70", label="theorem rows", zorder=3)
    max_q = max(q) if q else 1.0
    axes[2].plot([0, max_q + 0.25], [0, max_q + 0.25], "--", color="black", lw=1.0, label="all q independent")
    axes[2].scatter([1], [0], marker="x", color="#9b2d30", s=40, label="relative eq.")
    axes[2].set_xlabel(r"orbit dimension $q$")
    axes[2].set_ylabel("group directions independent of f")
    axes[2].set_title("C  Multiplicity beyond flow")
    axes[2].set_xlim(0, max_q + 0.55)
    axes[2].set_ylim(-0.2, max_q + 0.55)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, loc="upper left")
    savefig(fig, "fig2_neutral_geometry_clean_v2")


if __name__ == "__main__":
    make_fig2()
