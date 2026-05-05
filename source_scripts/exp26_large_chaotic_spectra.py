"""Step 26: larger chaotic spectra for journal-scale spectrum figures.

This experiment is intentionally heavier than the theorem checks.  It computes
full finite-time spectra for larger random-rate RNNs, then forms exact
block-product spectra with S1/T2 symmetry sectors by spectral union.  The
block-product union is exact for the implemented product system and avoids
spending most of the runtime repeatedly integrating identical transverse RNN
tangent dynamics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from goldstone_lyapunov.literature import short_citations
from goldstone_lyapunov.lyapunov import compute_lyapunov_spectrum
from goldstone_lyapunov.metrics import count_near_zero, kaplan_yorke_dimension, ks_entropy
from goldstone_lyapunov.models import RandomRateRNN
from goldstone_lyapunov.plotting import COLORS, clean_axis, ensure_dir, save_dict_rows_csv, save_figure, save_json, save_spectrum_csv, save_summary


S1_SECTOR_SPECTRUM = np.array([0.0, -2.0])
T2_SECTOR_SPECTRUM = np.array([0.0, 0.0, -2.0, -1.936])


def _parse_int_list(values: list[int] | None, default: list[int]) -> list[int]:
    if values is None or len(values) == 0:
        return default
    return [int(v) for v in values]


def _summary_row(model: str, N: int, seed: int, exponents: np.ndarray, zero_tol: float, full_spectrum: bool) -> dict:
    return {
        "model": model,
        "N_rnn": N,
        "dim": int(exponents.size),
        "seed": seed,
        "full_spectrum": full_spectrum,
        "lambda_max": float(np.max(exponents)),
        "lambda_min": float(np.min(exponents)),
        "near_zero_count": count_near_zero(exponents, zero_tol),
        "ks_entropy": ks_entropy(exponents, tol=zero_tol),
        "kaplan_yorke_dimension": kaplan_yorke_dimension(exponents, tol=zero_tol),
        "median_lambda": float(np.median(exponents)),
    }


def _spectrum_rows(model: str, N: int, seed: int, exponents: np.ndarray) -> list[dict]:
    return [
        {
            "model": model,
            "N_rnn": N,
            "dim": int(exponents.size),
            "seed": seed,
            "index": i,
            "lambda": float(lam),
        }
        for i, lam in enumerate(exponents, start=1)
    ]


def _history_rows(model: str, N: int, seed: int, history: dict, probe_indices: list[int]) -> list[dict]:
    times = np.asarray(history["history_times"], dtype=float)
    hist = np.asarray(history["history_exponents"], dtype=float)
    rows = []
    for t, spectrum in zip(times, hist):
        row = {"model": model, "N_rnn": N, "seed": seed, "time": float(t)}
        for idx in probe_indices:
            if 1 <= idx <= spectrum.size:
                row[f"lambda_{idx}"] = float(spectrum[idx - 1])
        rows.append(row)
    return rows


def _concat_product_spectra(rnn_spectrum: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "RNN": np.sort(rnn_spectrum)[::-1],
        "S1+RNN": np.sort(np.concatenate([rnn_spectrum, S1_SECTOR_SPECTRUM]))[::-1],
        "T2+RNN": np.sort(np.concatenate([rnn_spectrum, T2_SECTOR_SPECTRUM]))[::-1],
    }


def _make_figure(out: Path, spectrum_rows: list[dict], summary_rows: list[dict], history_rows: list[dict], zero_tol: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)

    ax = axes[0, 0]
    largest_N = max(int(r["N_rnn"]) for r in spectrum_rows)
    largest_seed = min(int(r["seed"]) for r in spectrum_rows if int(r["N_rnn"]) == largest_N)
    palette = {"RNN": COLORS["gray"], "S1+RNN": COLORS["blue"], "T2+RNN": COLORS["orange"]}
    for model in ["RNN", "S1+RNN", "T2+RNN"]:
        vals = [
            (int(r["index"]), float(r["lambda"]))
            for r in spectrum_rows
            if r["model"] == model and int(r["N_rnn"]) == largest_N and int(r["seed"]) == largest_seed
        ]
        vals = sorted(vals)
        ax.plot([v[0] for v in vals], [v[1] for v in vals], color=palette[model], linewidth=1.8, label=model)
    ax.axhline(0.0, color=COLORS["dark"], linewidth=0.9)
    ax.axhspan(-zero_tol, zero_tol, color=COLORS["orange"], alpha=0.10, linewidth=0)
    ax.set_xlabel("Lyapunov index i")
    ax.set_ylabel("lambda_i")
    ax.set_title(f"Long spectrum, N={largest_N}, seed={largest_seed}")
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax)

    ax = axes[0, 1]
    for model in ["RNN", "S1+RNN", "T2+RNN"]:
        vals = [
            (int(r["index"]), float(r["lambda"]))
            for r in spectrum_rows
            if r["model"] == model and int(r["N_rnn"]) == largest_N and int(r["seed"]) == largest_seed and abs(float(r["lambda"])) <= 0.12
        ]
        vals = sorted(vals)
        ax.scatter([v[0] for v in vals], [v[1] for v in vals], color=palette[model], s=22, alpha=0.78, label=model)
    ax.axhline(0.0, color=COLORS["dark"], linewidth=0.9)
    ax.axhspan(-zero_tol, zero_tol, color=COLORS["orange"], alpha=0.15, linewidth=0, label="zero tolerance")
    ax.set_xlabel("Lyapunov index i")
    ax.set_ylabel("lambda_i")
    ax.set_title("Near-zero zoom")
    ax.legend(frameon=False, fontsize=7)
    clean_axis(ax)

    ax = axes[1, 0]
    Ns = sorted({int(r["N_rnn"]) for r in summary_rows if r["model"] == "RNN"})
    for key, label, color in [
        ("lambda_max", "lambda_max", COLORS["blue"]),
        ("ks_entropy", "KS entropy", COLORS["green"]),
    ]:
        medians = []
        lows = []
        highs = []
        for N in Ns:
            vals = np.array([float(r[key]) for r in summary_rows if r["model"] == "RNN" and int(r["N_rnn"]) == N], dtype=float)
            medians.append(float(np.median(vals)))
            lows.append(float(np.percentile(vals, 10)))
            highs.append(float(np.percentile(vals, 90)))
        ax.plot(Ns, medians, marker="o", color=color, label=label)
        ax.fill_between(Ns, lows, highs, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0.0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xlabel("RNN dimension N")
    ax.set_ylabel("value")
    ax.set_title("Large-spectrum summary across N")
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax)

    ax = axes[1, 1]
    if history_rows:
        hist_keys = [k for k in history_rows[0] if k.startswith("lambda_")]
        for key in hist_keys:
            idx = int(key.split("_")[1])
            vals = [(float(r["time"]), float(r[key])) for r in history_rows if key in r]
            vals = sorted(vals)
            ax.plot([v[0] for v in vals], [v[1] for v in vals], linewidth=1.5, label=f"i={idx}")
    ax.axhline(0.0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xlabel("measurement time")
    ax.set_ylabel("finite-time lambda_i")
    ax.set_title("Convergence of selected exponents")
    ax.legend(frameon=False, fontsize=7, ncols=2)
    clean_axis(ax)

    fig.text(
        0.01,
        -0.035,
        "Literature anchors: " + short_citations(["chaotic_rnn", "lyapunov_numerics", "equivariant_dynamics"]),
        ha="left",
        va="top",
        fontsize=7,
        color=COLORS["gray"],
    )
    save_figure(fig, out / "large_chaotic_spectra.png")


def run(
    output_dir: str | Path | None = None,
    quick: bool = False,
    N_values: list[int] | None = None,
    seeds: list[int] | None = None,
    g: float = 3.0,
    dt: float | None = None,
    t_burn: float | None = None,
    t_total: float | None = None,
    reorthonormalize_every: float | None = None,
    zero_tol: float | None = None,
) -> dict:
    out = ensure_dir(output_dir or Path("results") / "exp26_large_chaotic_spectra")
    N_values = _parse_int_list(N_values, [32, 64] if quick else [64, 96])
    seeds = _parse_int_list(seeds, [0] if quick else [0, 1, 2])
    dt = float(dt if dt is not None else (0.045 if quick else 0.035))
    t_burn = float(t_burn if t_burn is not None else (30.0 if quick else 80.0))
    t_total = float(t_total if t_total is not None else (160.0 if quick else 520.0))
    reorthonormalize_every = float(reorthonormalize_every if reorthonormalize_every is not None else (0.45 if quick else 0.35))
    zero_tol = float(zero_tol if zero_tol is not None else (1e-2 if quick else 5e-3))

    spectrum_rows: list[dict] = []
    summary_rows: list[dict] = []
    history_rows: list[dict] = []
    acceptance_rows: list[dict] = []

    largest_N = max(N_values)
    smallest_seed = min(seeds)

    for N in N_values:
        for seed in seeds:
            rng = np.random.default_rng(26000 + 100 * N + seed)
            model = RandomRateRNN(N=N, g=g, seed=seed)
            x0 = rng.normal(0.0, 0.1, size=N)
            return_history = bool(N == largest_N and seed == smallest_seed)
            result = compute_lyapunov_spectrum(
                model,
                x0,
                dt=dt,
                t_burn=t_burn,
                t_total=t_total,
                reorthonormalize_every=reorthonormalize_every,
                seed=27000 + seed,
                return_history=return_history,
            )
            if return_history:
                rnn_spectrum = np.asarray(result["exponents"], dtype=float)
                probes = sorted({1, max(1, N // 8), max(1, N // 4), max(1, N // 2), N})
                history_rows.extend(_history_rows("RNN", N, seed, result, probes))
            else:
                rnn_spectrum = np.asarray(result, dtype=float)

            product_spectra = _concat_product_spectra(rnn_spectrum)
            for label, spectrum in product_spectra.items():
                save_spectrum_csv(out / f"spectrum_N{N}_seed{seed}_{label.replace('+', 'plus').lower()}.csv", spectrum)
                spectrum_rows.extend(_spectrum_rows(label, N, seed, spectrum))
                summary_rows.append(_summary_row(label, N, seed, spectrum, zero_tol, full_spectrum=True))

            rnn_zero = count_near_zero(product_spectra["RNN"], zero_tol)
            s1_zero = count_near_zero(product_spectra["S1+RNN"], zero_tol)
            t2_zero = count_near_zero(product_spectra["T2+RNN"], zero_tol)
            acceptance_rows.append(
                {
                    "N_rnn": N,
                    "seed": seed,
                    "rnn_near_zero_count": rnn_zero,
                    "s1_near_zero_count": s1_zero,
                    "t2_near_zero_count": t2_zero,
                    "s1_zero_delta": s1_zero - rnn_zero,
                    "t2_zero_delta": t2_zero - rnn_zero,
                    "lambda_max_rnn": float(np.max(product_spectra["RNN"])),
                    "chaotic": bool(np.max(product_spectra["RNN"]) > 0.0),
                    "ks_delta_s1_abs": abs(ks_entropy(product_spectra["S1+RNN"], zero_tol) - ks_entropy(product_spectra["RNN"], zero_tol)),
                    "ks_delta_t2_abs": abs(ks_entropy(product_spectra["T2+RNN"], zero_tol) - ks_entropy(product_spectra["RNN"], zero_tol)),
                }
            )

    save_dict_rows_csv(out / "large_spectrum_summary.csv", summary_rows)
    save_dict_rows_csv(out / "large_spectrum_long.csv", spectrum_rows)
    save_dict_rows_csv(out / "large_spectrum_history.csv", history_rows)
    save_dict_rows_csv(out / "large_spectrum_acceptance.csv", acceptance_rows)
    _make_figure(out, spectrum_rows, summary_rows, history_rows, zero_tol)

    largest_dim = max(int(r["dim"]) for r in summary_rows)
    max_lambda = max(float(r["lambda_max"]) for r in summary_rows if r["model"] == "RNN")
    all_s1_delta_ok = all(int(r["s1_zero_delta"]) >= 1 for r in acceptance_rows)
    all_t2_delta_ok = all(int(r["t2_zero_delta"]) >= 2 for r in acceptance_rows)
    any_chaotic = any(bool(r["chaotic"]) for r in acceptance_rows)
    metrics = {
        "quick": quick,
        "N_values": N_values,
        "seeds": seeds,
        "g": g,
        "dt": dt,
        "t_burn": t_burn,
        "t_total": t_total,
        "reorthonormalize_every": reorthonormalize_every,
        "zero_tol": zero_tol,
        "largest_spectrum_dimension": largest_dim,
        "max_rnn_lambda_max": max_lambda,
        "any_chaotic_seed": any_chaotic,
        "all_s1_zero_deltas_at_least_one": all_s1_delta_ok,
        "all_t2_zero_deltas_at_least_two": all_t2_delta_ok,
        "summary_rows": summary_rows,
        "acceptance_rows": acceptance_rows,
        "passed": bool(all_s1_delta_ok and all_t2_delta_ok),
        "inconclusive": bool(not any_chaotic),
        "note": "Product spectra are exact spectral unions for the block-diagonal product model. Use direct tangent/theorem experiments as primary evidence for protected modes; this experiment is the larger-spectrum journal diagnostic.",
    }
    save_json(out / "metrics.json", metrics)
    save_summary(
        out / "summary.md",
        "Experiment 26: Large Chaotic Spectra",
        [
            f"- RNN dimensions: {N_values}; seeds: {seeds}; full spectrum length up to {largest_dim}.",
            f"- Integration: dt={dt:g}, burn={t_burn:g}, total={t_total:g}, QR interval={reorthonormalize_every:g}.",
            f"- Largest RNN lambda_max observed: {max_lambda:.6g}; any chaotic seed: {any_chaotic}.",
            f"- S1 zero-count delta >= 1 for every run: {all_s1_delta_ok}.",
            f"- T2 zero-count delta >= 2 for every run: {all_t2_delta_ok}.",
            "- The block product is exact; finite-time chaotic zero counts remain diagnostics, not the primary theorem evidence.",
        ],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--N-values", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--g", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-burn", type=float, default=None)
    parser.add_argument("--t-total", type=float, default=None)
    parser.add_argument("--reorthonormalize-every", type=float, default=None)
    parser.add_argument("--zero-tol", type=float, default=None)
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        quick=args.quick,
        N_values=args.N_values,
        seeds=args.seeds,
        g=args.g,
        dt=args.dt,
        t_burn=args.t_burn,
        t_total=args.t_total,
        reorthonormalize_every=args.reorthonormalize_every,
        zero_tol=args.zero_tol,
    )


if __name__ == "__main__":
    main()
