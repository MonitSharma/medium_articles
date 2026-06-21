"""Cross-hardware comparison for the N=15/21/35 Shor/QPE sweep.

Consolidates the completed IBM Quantum and Quantinuum Helios runs (including the
new N=21, a=2 Helios-1E job e03104e4) into:

  - comparison/full_hardware_comparison.csv   (one row per completed run)
  - comparison/hero_hardware_comparison.png/.pdf   (4-panel overview)
  - comparison/n21_base_comparison.png/.pdf        (N=21 base/period deep dive)

Run from sometimes-too-slow-for-shor/:
  python quantinuum_helios/generate_full_comparison.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
HELIOS = ROOT / "quantinuum_helios"
OUT = HELIOS / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

# ---- styling -------------------------------------------------------------
IBM = "#1f6feb"        # IBM blue
HEL = "#00b39b"        # Quantinuum teal
HEL_BAD = "#f0883e"    # Helios run that did not yield a clean signal
GRID = "#d0d7de"
INK = "#1b1f24"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#8b949e",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.dpi": 140,
})


def _f(row, key, default=None):
    v = row.get(key, "")
    if v in ("", None):
        return default
    try:
        return float(v)
    except ValueError:
        return v


def _row(path: Path, where=lambda r: True) -> dict:
    for r in csv.DictReader(open(path)):
        if where(r):
            return r
    raise SystemExit(f"no matching row in {path}")


# ---- load completed runs --------------------------------------------------
ibm_csv = ROOT / "data" / "summary" / "results_summary.csv"
ibm = {int(float(r["N"])): r for r in csv.DictReader(open(ibm_csv))}

hel_n15 = _row(HELIOS / "data/summary/results_summary_helios_full_20260619.csv",
               lambda r: int(float(r["N"])) == 15 and r["run_status"] == "completed")
hel_n21_a4 = _row(HELIOS / "data/summary/results_summary_helios_N21_Helios-1E_20260620_retry.csv")
hel_n21_a2 = _row(HELIOS / "data/summary/results_summary_helios_N21_a2_Helios-1E_20260620.csv")

# HQC charges (from job records / cost estimates)
HQC = {("Helios", 15, 7): 896.08, ("Helios", 21, 4): 27197.72, ("Helios", 21, 2): 27197.52}

# Public list prices used only for order-of-magnitude USD scale.
IBM_USD_PER_MIN = 96.0            # Pay-As-You-Go
HQC_USD = 12.50                   # Azure H2 Standard-bundle implied $/HQC


def usd_ibm(runtime_s):
    return runtime_s / 60.0 * IBM_USD_PER_MIN


def usd_hel(hqc):
    return hqc * HQC_USD


# Build a uniform run table.
runs = []


def add(provider, system, color, N, a, src, hqc=None):
    runtime = _f(src, "runtime_sec", 0.0)
    runs.append({
        "provider": provider, "system": system, "color": color,
        "N": N, "a": a,
        "r_true": _f(src, "r_true"),
        "depth": _f(src, "depth"),
        "two_qubit_gates": _f(src, "two_qubit_gates"),
        "strict_success": str(src.get("strict_success")).strip().lower() == "true",
        "factor_found": str(src.get("factor_found_in_histogram")).strip().lower() == "true",
        "yield": _f(src, "factor_yield_mass", 0.0),
        "baseline": _f(src, "strict_baseline_fp_rate", 0.0),
        "yield_ratio": _f(src, "yield_vs_baseline_ratio", 0.0),
        "mass_near_peaks": _f(src, "mass_near_peaks", 0.0),
        "uniform_peak_mass": _f(src, "uniform_peak_mass", 0.0),
        "peak_enrichment": _f(src, "peak_enrichment", 0.0),
        "runtime_sec": runtime,
        "hqc": hqc,
        "usd": usd_hel(hqc) if hqc is not None else usd_ibm(runtime),
        "cost_basis": "HQC" if hqc is not None else "runtime-min",
    })


add("IBM", ibm[15]["backend"], IBM, 15, 7, ibm[15])
add("IBM", ibm[21]["backend"], IBM, 21, 19, ibm[21])
add("IBM", ibm[35]["backend"], IBM, 35, 4, ibm[35])
add("Helios", "Helios-1", HEL, 15, 7, hel_n15, hqc=HQC[("Helios", 15, 7)])
add("Helios", "Helios-1E", HEL_BAD, 21, 4, hel_n21_a4, hqc=HQC[("Helios", 21, 4)])
add("Helios", "Helios-1E", HEL, 21, 2, hel_n21_a2, hqc=HQC[("Helios", 21, 2)])


# ---- write consolidated CSV ----------------------------------------------
csv_path = OUT / "full_hardware_comparison.csv"
cols = ["provider", "system", "N", "a", "r_true", "depth", "two_qubit_gates",
        "strict_success", "factor_found", "yield", "baseline", "yield_ratio",
        "mass_near_peaks", "uniform_peak_mass", "peak_enrichment",
        "runtime_sec", "hqc", "cost_basis", "usd"]
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in runs:
        w.writerow(r)
print("wrote", csv_path)


def label(r):
    return f"{r['provider']}\nN={r['N']}, a={r['a']}"


# ======================================================================
# HERO FIGURE: 4 panels
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
fig.suptitle("Shor / QPE order-finding on real hardware: IBM Quantum vs Quantinuum Helios",
             fontsize=16, fontweight="bold", y=0.98)

x = np.arange(len(runs))
labels = [label(r) for r in runs]
colors = [r["color"] for r in runs]

# Panel A: peak enrichment (signal quality) -- log scale
axA = axes[0, 0]
enr = [max(r["peak_enrichment"], 0.01) for r in runs]
bars = axA.bar(x, enr, color=colors, edgecolor="white", linewidth=1.2)
axA.axhline(1.0, color="#cf222e", ls="--", lw=1.3, zorder=0)
axA.text(len(runs) - 0.5, 1.05, "no enrichment (random output)", color="#cf222e",
         ha="right", va="bottom", fontsize=9)
axA.set_yscale("log")
axA.set_ylim(0.4, 40)
axA.set_title("A  Signal quality — mass near ideal QPE peaks\nvs a uniform-random baseline",
              fontweight="bold", loc="left")
axA.set_ylabel("peak enrichment  (×)")
for xi, v in zip(x, enr):
    axA.text(xi, v * 1.08, f"{v:.2f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")
axA.set_xticks(x)
axA.set_xticklabels(labels, fontsize=8.5)

# Panel B: factor yield vs strict null baseline (paired)
axB = axes[0, 1]
w = 0.38
yld = [r["yield"] for r in runs]
bl = [r["baseline"] for r in runs]
axB.bar(x - w / 2, yld, w, color=colors, edgecolor="white", label="measured factor yield")
axB.bar(x + w / 2, bl, w, color="#9aa0a6", edgecolor="white", label="strict null baseline")
axB.set_title("B  Per-shot factor yield vs the strict\nfalse-positive baseline", fontweight="bold", loc="left")
axB.set_ylabel("fraction of shots")
for xi, r in zip(x, runs):
    tag = "✓ factors" if r["strict_success"] else "✗ no factors"
    tc = "#1a7f37" if r["strict_success"] else "#cf222e"
    axB.text(xi, max(r["yield"], r["baseline"]) + 0.03, tag, ha="center", fontsize=8, color=tc, fontweight="bold")
axB.set_ylim(0, 0.8)
axB.set_xticks(x)
axB.set_xticklabels(labels, fontsize=8.5)
axB.legend(loc="upper right", fontsize=9, framealpha=0.95)

# Panel C: two-qubit gate count vs N (the oracle wall)
axC = axes[1, 0]
for prov, col, marker in [("IBM", IBM, "o"), ("Helios", HEL, "s")]:
    pts = sorted({(r["N"], r["two_qubit_gates"]) for r in runs if r["provider"] == prov})
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    axC.plot(xs, ys, marker=marker, color=col, lw=2.2, ms=9, label=prov)
    for xx, yy in zip(xs, ys):
        axC.annotate(f"{int(yy):,}", (xx, yy), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8.5, color=col, fontweight="bold")
axC.set_yscale("log")
axC.set_title("C  Compiled two-qubit gates per run\n(submitted circuit, not logical qubits)",
              fontweight="bold", loc="left")
axC.set_xlabel("N (number factored)")
axC.set_ylabel("two-qubit gates  (log)")
axC.set_xticks([15, 21, 35])
axC.legend(loc="upper left", fontsize=10)

# Panel D: cost to run -- USD scale (log), annotated with native basis
axD = axes[1, 1]
usd = [r["usd"] for r in runs]
bars = axD.bar(x, usd, color=colors, edgecolor="white", linewidth=1.2)
axD.set_yscale("log")
axD.set_title("D  Order-of-magnitude cost per completed run\n(IBM $96/min PAYG · Helios $12.50/HQC bundle)",
              fontweight="bold", loc="left")
axD.set_ylabel("estimated USD  (log)")
for xi, r, v in zip(x, runs, usd):
    native = f"{r['hqc']:,.0f} HQC" if r["hqc"] is not None else f"{r['runtime_sec']:.0f}s"
    axD.text(xi, v * 1.12, f"${v:,.0f}\n{native}", ha="center", va="bottom", fontsize=8, fontweight="bold")
axD.set_ylim(10, usd_hel(HQC[("Helios", 21, 4)]) * 4)
axD.set_xticks(x)
axD.set_xticklabels(labels, fontsize=8.5)

for ax in axes.flat:
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

legend = [Patch(facecolor=IBM, label="IBM Quantum (superconducting)"),
          Patch(facecolor=HEL, label="Quantinuum Helios (trapped-ion)"),
          Patch(facecolor=HEL_BAD, label="Helios run with unfavourable base (a=4, odd order)")]
fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.002))
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(OUT / "hero_hardware_comparison.png", bbox_inches="tight")
fig.savefig(OUT / "hero_hardware_comparison.pdf", bbox_inches="tight")
print("wrote", OUT / "hero_hardware_comparison.png")
plt.close(fig)


# ======================================================================
# N=21 BASE DEEP DIVE: spectra + period structure
# ======================================================================
def load_counts(path: Path) -> dict:
    row = json.loads(Path(path).read_text().splitlines()[0])
    return {k: int(v) for k, v in row["counts"].items()}, row


def spectrum(counts, t, bit_order):
    tot = sum(counts.values())
    spec = np.zeros(2 ** t)
    for bits, n in counts.items():
        b = bits[::-1] if bit_order == "reversed" else bits
        spec[int(b, 2)] += n
    return spec / tot


a4_counts, a4_row = load_counts(HELIOS / "data/raw/results_helios_hardware_N21_Helios-1E_20260620_retry.jsonl")
a2_counts, a2_row = load_counts(HELIOS / "data/raw/results_helios_hardware_N21_a2_Helios-1E_20260620.jsonl")
t21 = int(a2_row["t"])

fig2, axes2 = plt.subplots(2, 1, figsize=(13, 8.6))
fig2.suptitle("N=21 on Quantinuum Helios-1E: the base choice decides whether Shor can even work",
              fontsize=15, fontweight="bold", y=0.98)

for ax, counts, a, r, col, note in [
    (axes2[0], a4_counts, 4, 3, HEL_BAD, "order 3 is ODD → Shor cannot factor, regardless of hardware"),
    (axes2[1], a2_counts, 2, 6, HEL, "order 6 is even and yields factors in principle"),
]:
    spec = spectrum(counts, t21, "reversed")
    xs = np.arange(2 ** t21)
    ax.bar(xs, spec, width=1.0, color=col, alpha=0.85)
    peaks = sorted({round(k * (2 ** t21) / r) % (2 ** t21) for k in range(r)})
    for pk in peaks:
        ax.axvline(pk, color="#cf222e", ls="--", lw=1.0, alpha=0.7, zorder=5)
    ax.set_xlim(0, 2 ** t21)
    ax.set_ylabel("probability")
    enr = next(rr["peak_enrichment"] for rr in runs if rr["N"] == 21 and rr["a"] == a)
    yr = next(rr["yield_ratio"] for rr in runs if rr["N"] == 21 and rr["a"] == a)
    ax.set_title(f"a={a}  ·  true order r={r}  ·  {note}\n"
                 f"peak enrichment {enr:.2f}×   ·   factor yield / null {yr:.2f}×   ·   "
                 f"red lines = ideal QPE peaks at k·{2**t21}/{r}",
                 loc="left", fontsize=10.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes2[1].set_xlabel(f"measured counting-register value y  (0 … {2**t21 - 1})")
# IBM N=21 reference annotation
fig2.text(0.5, 0.005,
          f"Reference: IBM ibm_torino N=21, a=19 (order 6) recovered factors by histogram scan "
          f"(yield {ibm[21]['factor_yield_mass']}, enrichment {ibm[21]['peak_enrichment']}×) "
          f"using {int(float(ibm[21]['two_qubit_gates'])):,} two-qubit gates — "
          f"vs {int(a2_row['two_qubit_gates']):,} on Helios.",
          ha="center", fontsize=9, color="#57606a")
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
fig2.savefig(OUT / "n21_base_comparison.png", bbox_inches="tight")
fig2.savefig(OUT / "n21_base_comparison.pdf", bbox_inches="tight")
print("wrote", OUT / "n21_base_comparison.png")
plt.close(fig2)
