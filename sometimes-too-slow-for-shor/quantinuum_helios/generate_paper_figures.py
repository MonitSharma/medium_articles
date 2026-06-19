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
OUT_DIR = Path(__file__).resolve().parent / "paper_figures"
IBM_SUMMARY = PROJECT_ROOT / "data" / "summary" / "results_summary.csv"
IBM_RAW = PROJECT_ROOT / "data" / "raw" / "results_hardware_20260228_183452.jsonl"
HELIOS_SUMMARY = PROJECT_ROOT / "quantinuum_helios" / "data" / "summary" / "results_summary_helios_full_20260619.csv"
HELIOS_FULL_RAW = PROJECT_ROOT / "quantinuum_helios" / "data" / "raw" / "results_helios_hardware_full_20260619.jsonl"
HELIOS_N21_RAW = PROJECT_ROOT / "quantinuum_helios" / "data" / "raw" / "results_helios_hardware_N21_20260619_max6500.jsonl"
HELIOS_COST_DIR = PROJECT_ROOT / "quantinuum_helios" / "data" / "cost_estimates"


IBM = "#0072B2"
HELIOS = "#009E73"
ACCENT = "#CC79A7"
ORANGE = "#E69F00"
GRAY = "#6E7681"
LIGHT = "#E6E8EC"
DARK = "#1F2328"
RED = "#B5413C"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 11,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "legend.fontsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_latest_json(pattern: str) -> dict[str, Any]:
    matches = sorted(HELIOS_COST_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No cost-estimate file matched {pattern}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _by_n(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            out[int(row["N"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _f(value: Any) -> float:
    if value in {None, ""}:
        return float("nan")
    return float(value)


def _finish_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4B5563")
    ax.spines["bottom"].set_color("#4B5563")
    ax.grid(True, axis=grid_axis, color=LIGHT, linewidth=0.65)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(OUT_DIR / f"{name}{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _label_barh(ax: plt.Axes, value: float, y: float, text: str, color: str, x_pad: float) -> None:
    ax.text(value + x_pad, y, text, ha="left", va="center", fontsize=9, color=color, fontweight="bold")


def make_figure_1(ibm_n15: dict[str, Any], helios_n15: dict[str, Any], helios_n15_raw: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)

    labels = ["Strict factor yield", "Mass near ideal QPE peaks"]
    ibm_pct = [_f(ibm_n15["factor_yield_mass"]) * 100, _f(ibm_n15["mass_near_peaks"]) * 100]
    helios_pct = [_f(helios_n15["factor_yield_mass"]) * 100, _f(helios_n15["mass_near_peaks"]) * 100]
    y = np.arange(len(labels))[::-1]
    height = 0.32

    ax.barh(y + height / 2, ibm_pct, height, color=IBM, label="IBM Torino")
    ax.barh(y - height / 2, helios_pct, height, color=HELIOS, label="Helios-1")
    for yy, value in zip(y + height / 2, ibm_pct):
        _label_barh(ax, value, yy, f"IBM {value:.1f}%", IBM, 1.2)
    for yy, value in zip(y - height / 2, helios_pct):
        _label_barh(ax, value, yy, f"Helios {value:.1f}%", HELIOS, 1.2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 88)
    ax.set_xlabel("Fraction of measured shots (%)")
    ax.set_title("N=15: Helios-1 produced a much clearer measured signal", pad=12, fontweight="bold")
    _finish_axis(ax, "x")

    fig.text(
        0.01,
        -0.02,
        "Higher is better. Strict factor yield counts shots that recover non-trivial factors; QPE-peak mass measures concentration near the ideal period peaks.",
        ha="left",
        va="top",
        fontsize=8.2,
        color=GRAY,
    )
    _save(fig, "fig1_paper_n15_hardware_comparison")


def make_figure_2(
    ibm_rows: list[dict[str, Any]],
    helios_rows: list[dict[str, Any]],
    helios_full_raw: list[dict[str, Any]],
    helios_n21_raw: dict[str, Any],
) -> None:
    ibm_by_n = _by_n(ibm_rows)
    helios_by_n = _by_n(helios_rows)
    raw_by_n = _by_n(helios_full_raw)

    fig, ax = plt.subplots(figsize=(7.4, 3.9), constrained_layout=True)

    n15_actual = _f(raw_by_n[15].get("quantinuum_actual_cost_hqc"))
    labels = ["N=15 actual charge", "N=21 provider estimate", "N=35 extrapolated estimate"]
    values = [n15_actual, 27198, 134000]
    colors = [HELIOS, ORANGE, ACCENT]
    y = np.arange(len(labels))[::-1]

    ax.barh(y, values, color=colors, height=0.46)
    value_labels = [f"{n15_actual:.2f} HQC", "27.2k HQC", "134k HQC"]
    for yy, value, label, color in zip(y, values, value_labels, colors):
        _label_barh(ax, value, yy, label, color, 3600)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 190000)
    ax.set_xlabel("HQC for a 1024-shot Helios-1 run")
    ax.set_title("Helios-1 HQC cost rises quickly beyond the completed N=15 run", pad=12, fontweight="bold")
    _finish_axis(ax, "x")

    fig.text(
        0.01,
        -0.02,
        "N=15 is the actual completed charge. N=21 is a Nexus provider estimate. N=35 is extrapolated from circuit volume because Nexus did not return a completed estimate.",
        ha="left",
        va="top",
        fontsize=8.2,
        color=GRAY,
    )
    _save(fig, "fig2_paper_scaling_budget")


def _label_bar(ax: plt.Axes, x: float, height: float, text: str, color: str) -> None:
    ax.text(x, height * 1.08, text, ha="center", va="bottom", fontsize=7.4, color=color, fontweight="bold")


def make_figure_3(
    ibm_raw: list[dict[str, Any]],
    helios_full_raw: list[dict[str, Any]],
    helios_n21_cost: dict[str, Any],
    helios_n35_cost: dict[str, Any],
) -> None:
    ibm_by_n = _by_n(ibm_raw)
    helios_by_n = _by_n(helios_full_raw)
    ns = [15, 21, 35]
    labels = [f"N={n}" for n in ns]

    ibm_twoq = [_f(ibm_by_n[n]["two_qubit_gates"]) for n in ns]
    helios_twoq = [
        _f(helios_by_n[15]["two_qubit_gates"]),
        _f(helios_n21_cost["two_qubit_gates"]),
        _f(helios_n35_cost["two_qubit_gates"]),
    ]
    ibm_qubits = [_f(ibm_by_n[n]["num_qubits"]) for n in ns]
    helios_qubits = [
        _f(helios_by_n[15]["num_qubits"]),
        _f(helios_by_n[21]["num_qubits"]),
        _f(helios_by_n[35]["num_qubits"]),
    ]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.6, 3.75), constrained_layout=True)
    x = np.arange(len(ns))
    width = 0.36

    ax_a.bar(x - width / 2, ibm_twoq, width, color=IBM, label="IBM hardware compile")
    ax_a.bar(x + width / 2, helios_twoq, width, color=HELIOS, label="Helios QIR path")
    ax_a.set_yscale("log")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("Two-qubit gates (log scale)")
    ax_a.set_title("Entangling-gate scale", fontweight="bold")
    for xpos, value in zip(x - width / 2, ibm_twoq):
        _label_bar(ax_a, xpos, value, f"{value/1000:.1f}k" if value >= 1000 else f"{value:.0f}", IBM)
    for xpos, value in zip(x + width / 2, helios_twoq):
        _label_bar(ax_a, xpos, value, f"{value/1000:.1f}k" if value >= 1000 else f"{value:.0f}", HELIOS)
    ax_a.legend(frameon=False, loc="upper left")
    _finish_axis(ax_a, "y")

    ax_b.bar(x - width / 2, ibm_qubits, width, color=IBM)
    ax_b.bar(x + width / 2, helios_qubits, width, color=HELIOS)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Physical/logical qubits reported")
    ax_b.set_title("Qubit footprint", fontweight="bold")
    ax_b.set_ylim(0, 175)
    for xpos, value in zip(x - width / 2, ibm_qubits):
        _label_bar(ax_b, xpos, value, f"{value:.0f}", IBM)
    for xpos, value in zip(x + width / 2, helios_qubits):
        _label_bar(ax_b, xpos, value, f"{value:.0f}", HELIOS)
    _finish_axis(ax_b, "y")

    fig.suptitle("Circuit resources behind the HQC scaling", fontsize=12, fontweight="bold")
    fig.text(
        0.01,
        -0.03,
        "IBM bars use completed hardware-job metadata. Helios N=21/N=35 bars use QIR syntax-checker/cost-estimate artifacts; N=35 cost estimation failed but circuit volume was recorded.",
        ha="left",
        va="top",
        fontsize=8.0,
        color=GRAY,
    )
    _save(fig, "fig3_paper_resource_scale")


def main() -> None:
    ibm_rows = _load_csv(IBM_SUMMARY)
    ibm_raw = _load_jsonl(IBM_RAW)
    helios_rows = _load_csv(HELIOS_SUMMARY)
    helios_full_raw = _load_jsonl(HELIOS_FULL_RAW)
    helios_n21_raw = _load_jsonl(HELIOS_N21_RAW)[0]
    helios_n21_cost = _load_latest_json("helios_qir_cost_N21_1024shots_*.json")
    helios_n35_cost = _load_latest_json("helios_qir_cost_N35_1024shots_*.json")

    ibm_by_n = _by_n(ibm_rows)
    helios_by_n = _by_n(helios_rows)
    raw_by_n = _by_n(helios_full_raw)

    make_figure_1(ibm_by_n[15], helios_by_n[15], raw_by_n[15])
    make_figure_2(ibm_rows, helios_rows, helios_full_raw, helios_n21_raw)
    make_figure_3(ibm_raw, helios_full_raw, helios_n21_cost, helios_n35_cost)
    print(f"Wrote paper figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
