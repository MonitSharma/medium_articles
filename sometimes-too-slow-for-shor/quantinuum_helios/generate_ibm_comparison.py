from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "comparison"
IBM_SUMMARY = PROJECT_ROOT / "data" / "summary" / "results_summary.csv"
HELIOS_SUMMARY = PROJECT_ROOT / "quantinuum_helios" / "data" / "summary" / "results_summary_helios_full_20260619.csv"
HELIOS_FULL_RAW = PROJECT_ROOT / "quantinuum_helios" / "data" / "raw" / "results_helios_hardware_full_20260619.jsonl"
HELIOS_N21_RAW = PROJECT_ROOT / "quantinuum_helios" / "data" / "raw" / "results_helios_hardware_N21_20260619_max6500.jsonl"


BLUE = "#2F6F9F"
GREEN = "#3A7D44"
RED = "#B23A48"
ORANGE = "#D17A22"
GRAY = "#8A8F98"
LIGHT_GRAY = "#D8DEE5"
TEXT = "#20242A"
PANEL_BG = "#FBFCFE"


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 360,
        "savefig.facecolor": "white",
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": "#30343B",
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.edgecolor": LIGHT_GRAY,
        "legend.fontsize": 10,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _maybe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_by_n(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            out[int(row["N"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(True, axis="y", alpha=0.28, color="#B8C1CC", linewidth=0.7)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = OUTPUT_DIR / filename
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.2f}", pad: float = 0.02) -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + ymax * pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=TEXT,
        )


def plot_n15_signal(ibm: dict[str, Any], helios: dict[str, Any]) -> None:
    metrics = [
        ("Strict yield", "factor_yield_mass", 100.0, "%"),
        ("Mass near peaks", "mass_near_peaks", 100.0, "%"),
        ("Yield / baseline", "yield_vs_baseline_ratio", 1.0, "x"),
        ("Peak enrichment", "peak_enrichment", 1.0, "x"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.1))
    axes = axes.ravel()

    for ax, (title, key, scale, suffix) in zip(axes, metrics):
        ibm_val = (_maybe_float(ibm.get(key)) or 0.0) * scale
        helios_val = (_maybe_float(helios.get(key)) or 0.0) * scale
        bars = ax.bar(
            ["IBM Torino", "Helios-1"],
            [ibm_val, helios_val],
            color=[BLUE, GREEN],
            width=0.58,
            edgecolor="white",
            linewidth=1.2,
        )
        _annotate_bars(ax, bars, fmt="{:.2f}" + suffix)
        ax.set_title(title, fontweight="bold", pad=9)
        _style_axis(ax)
        if title in {"Strict yield", "Mass near peaks"}:
            ax.set_ylabel("Percent of shots")
            ax.set_ylim(0, max(80, max(ibm_val, helios_val) * 1.22))
        else:
            ax.set_ylabel("Multiple")
            ax.set_ylim(0, max(18, max(ibm_val, helios_val) * 1.22))

    fig.suptitle("N=15 Completed Hardware Comparison: Signal Quality", fontsize=17, fontweight="bold", y=1.02)
    _save(fig, "fig1_n15_signal_quality.png")


def plot_n15_resources(ibm: dict[str, Any], helios_raw: dict[str, Any]) -> None:
    depth = [_maybe_float(ibm.get("depth")) or 0.0, _maybe_float(helios_raw.get("depth")) or 0.0]
    two_q = [_maybe_float(ibm.get("two_qubit_gates")) or 0.0, _maybe_float(helios_raw.get("two_qubit_gates")) or 0.0]
    runtime = [_maybe_float(ibm.get("runtime_sec")) or 0.0, _maybe_float(helios_raw.get("runtime_sec")) or 0.0]

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.15))
    labels = ["IBM Torino", "Helios-1"]

    for ax, values, title, ylabel, log_scale in [
        (axes[0], depth, "Circuit depth", "Depth", False),
        (axes[1], two_q, "Two-qubit gates", "2Q gate count", False),
        (axes[2], runtime, "End-to-end runtime", "Seconds", True),
    ]:
        bars = ax.bar(labels, values, color=[BLUE, GREEN], width=0.58, edgecolor="white", linewidth=1.2)
        _annotate_bars(ax, bars, fmt="{:.0f}")
        ax.set_title(title, fontweight="bold", pad=9)
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        _style_axis(ax)

    cost = _maybe_float(helios_raw.get("quantinuum_actual_cost_hqc"))
    if cost is not None:
        axes[2].text(
            1,
            runtime[1] * 0.35,
            f"{cost:.2f} HQC",
            ha="center",
            va="center",
            fontsize=10.5,
            color=GREEN,
            fontweight="bold",
        )

    fig.suptitle("N=15 Completed Hardware Comparison: Resources and Runtime", fontsize=17, fontweight="bold", y=1.03)
    _save(fig, "fig2_n15_resources_runtime.png")


def plot_cross_n_status(ibm_rows: list[dict[str, Any]], helios_rows: list[dict[str, Any]], helios_n21: dict[str, Any]) -> None:
    ns = [15, 21, 35]
    ibm_by_n = _row_by_n(ibm_rows)
    helios_by_n = _row_by_n(helios_rows)

    x = np.arange(len(ns))
    width = 0.34
    ibm_yield = [(_maybe_float(ibm_by_n[n].get("factor_yield_mass")) or 0.0) * 100 for n in ns]
    helios_yield = [
        ((_maybe_float(helios_by_n[n].get("factor_yield_mass")) or 0.0) * 100)
        if helios_by_n.get(n, {}).get("run_status") == "completed"
        else 0.0
        for n in ns
    ]

    fig, ax = plt.subplots(figsize=(13.6, 7.1))
    bars_ibm = ax.bar(
        x - width / 2,
        ibm_yield,
        width,
        label="IBM completed",
        color=BLUE,
        edgecolor="white",
        linewidth=1.1,
    )
    bars_helios = ax.bar(
        x + width / 2,
        helios_yield,
        width,
        label="Helios-1 completed",
        color=GREEN,
        edgecolor="white",
        linewidth=1.1,
    )
    _annotate_bars(ax, bars_ibm, fmt="{:.1f}%")
    _annotate_bars(ax, [bar for bar, value in zip(bars_helios, helios_yield) if value > 0], fmt="{:.1f}%")

    for idx, n_value in enumerate(ns):
        h_row = helios_by_n.get(n_value, {})
        status = h_row.get("run_status")
        if status != "completed":
            label = "no completed\nHelios shots"
            if n_value == 21:
                label = "rejected /\nterminated"
            ax.bar(
                x[idx] + width / 2,
                2.5,
                width,
                color="white",
                edgecolor=RED,
                linewidth=1.3,
                hatch="///",
            )
            ax.text(
                x[idx] + width / 2,
                5.2,
                label,
                ha="center",
                va="bottom",
                fontsize=9.2,
                color=RED,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("Strict factor yield (% of shots)")
    ax.set_ylim(0, 78)
    ax.set_title("Completed Factor Yield Across Tested Semiprimes", fontsize=16, fontweight="bold", pad=14)
    ax.legend(loc="upper right")
    _style_axis(ax)
    _save(fig, "fig3_factor_yield_by_n.png")


def plot_helios_attempts(helios_full_raw: list[dict[str, Any]], helios_n21_raw: dict[str, Any]) -> None:
    attempts = [
        ("N=15 full\ncompleted", next(r for r in helios_full_raw if int(r["N"]) == 15)),
        ("N=21 full\na=19 rejected", next(r for r in helios_full_raw if int(r["N"]) == 21)),
        ("N=21 rerun\na=4 terminated", helios_n21_raw),
        ("N=35 full\ncost failed", next(r for r in helios_full_raw if int(r["N"]) == 35)),
    ]

    estimates = [_maybe_float(row.get("quantinuum_cost_hqc")) or np.nan for _, row in attempts]
    max_caps = [_maybe_float(row.get("quantinuum_effective_max_cost_hqc")) or np.nan for _, row in attempts]
    actuals = [_maybe_float(row.get("quantinuum_actual_cost_hqc")) or np.nan for _, row in attempts]
    labels = [label for label, _ in attempts]
    statuses = [str(row.get("run_status")) for _, row in attempts]

    fig, ax = plt.subplots(figsize=(14.4, 7.4))
    x = np.arange(len(labels))
    width = 0.24
    bars_est = ax.bar(x - width, estimates, width, label="Nexus estimate", color=GRAY, edgecolor="white", linewidth=1.1)
    bars_cap = ax.bar(x, max_caps, width, label="Max-cost guard", color=ORANGE, edgecolor="white", linewidth=1.1)
    bars_actual = ax.bar(x + width, actuals, width, label="Actual charged", color=GREEN, edgecolor="white", linewidth=1.1)

    for bars in [bars_est, bars_cap, bars_actual]:
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height * 1.08,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
                rotation=0,
            )

    for idx, status in enumerate(statuses):
        color = GREEN if status == "completed" else RED
        ax.text(
            idx,
            1.55,
            status.replace("_", "\n"),
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(1.2, 60000)
    ax.set_ylabel("HQC, log scale")
    ax.set_title("Helios-1 Attempt Outcomes and HQC Guards", fontsize=16, fontweight="bold", pad=14)
    ax.legend(loc="upper left")
    _style_axis(ax)
    _save(fig, "fig4_helios_cost_attempts.png")


def write_comparison_table(
    ibm_rows: list[dict[str, Any]],
    helios_rows: list[dict[str, Any]],
    helios_full_raw: list[dict[str, Any]],
    helios_n21_raw: dict[str, Any],
) -> None:
    ibm_by_n = _row_by_n(ibm_rows)
    helios_by_n = _row_by_n(helios_rows)
    raw_by_n = _row_by_n(helios_full_raw)

    rows: list[dict[str, Any]] = []
    for n_value in [15, 21, 35]:
        ibm = ibm_by_n.get(n_value, {})
        helios = helios_by_n.get(n_value, {})
        raw = raw_by_n.get(n_value, {})
        rows.append(
            {
                "N": n_value,
                "ibm_backend": ibm.get("backend"),
                "ibm_status": ibm.get("run_status"),
                "ibm_factor_yield": ibm.get("factor_yield_mass"),
                "ibm_peak_enrichment": ibm.get("peak_enrichment"),
                "ibm_dominant_outcome_success": ibm.get("dominant_outcome_success"),
                "helios_backend": helios.get("backend"),
                "helios_status": helios.get("run_status"),
                "helios_factor_yield": helios.get("factor_yield_mass"),
                "helios_peak_enrichment": helios.get("peak_enrichment"),
                "helios_dominant_outcome_success": helios.get("dominant_outcome_success"),
                "helios_actual_cost_hqc": raw.get("quantinuum_actual_cost_hqc"),
                "helios_job_id": raw.get("job_id"),
            }
        )

    rows.append(
        {
            "N": "21-rerun-a4",
            "ibm_backend": "",
            "ibm_status": "",
            "ibm_factor_yield": "",
            "ibm_peak_enrichment": "",
            "ibm_dominant_outcome_success": "",
            "helios_backend": helios_n21_raw.get("backend"),
            "helios_status": helios_n21_raw.get("run_status"),
            "helios_factor_yield": "",
            "helios_peak_enrichment": "",
            "helios_dominant_outcome_success": "",
            "helios_actual_cost_hqc": helios_n21_raw.get("quantinuum_actual_cost_hqc"),
            "helios_job_id": helios_n21_raw.get("job_id"),
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "comparison_table.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ibm_rows = _load_csv(IBM_SUMMARY)
    helios_rows = _load_csv(HELIOS_SUMMARY)
    helios_full_raw = _load_jsonl(HELIOS_FULL_RAW)
    helios_n21_rows = _load_jsonl(HELIOS_N21_RAW)
    if not helios_n21_rows:
        raise SystemExit(f"Missing N=21 rerun raw file: {HELIOS_N21_RAW}")

    ibm_by_n = _row_by_n(ibm_rows)
    helios_by_n = _row_by_n(helios_rows)
    raw_by_n = _row_by_n(helios_full_raw)

    plot_n15_signal(ibm=ibm_by_n[15], helios=helios_by_n[15])
    plot_n15_resources(ibm=ibm_by_n[15], helios_raw=raw_by_n[15])
    plot_cross_n_status(ibm_rows=ibm_rows, helios_rows=helios_rows, helios_n21=helios_n21_rows[0])
    plot_helios_attempts(helios_full_raw=helios_full_raw, helios_n21_raw=helios_n21_rows[0])
    write_comparison_table(
        ibm_rows=ibm_rows,
        helios_rows=helios_rows,
        helios_full_raw=helios_full_raw,
        helios_n21_raw=helios_n21_rows[0],
    )
    print(f"Wrote comparison plots and table to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
