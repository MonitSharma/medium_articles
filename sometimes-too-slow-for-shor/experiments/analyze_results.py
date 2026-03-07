"""Analyze Shor sweep results and generate enriched CSV/plots."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shor.postprocess import (
    strict_null_baseline_fp_rate,
    exploratory_postprocess_y,
    per_shot_factor_yield,
    compute_ideal_peaks,
    histogram_vs_ideal_overlap,
)


SUMMARY_COLUMNS = [
    "N", "bits", "a", "t", "n_work",
    "backend", "depth", "two_qubit_gates", "shots",
    "run_status",
    # --- Strict layer (headline) ---
    "top1_success",
    "factor_yield_mass",
    "strict_baseline_fp_rate",
    "yield_vs_baseline_ratio",
    # --- Exploratory layer (transparency) ---
    "exploratory_yield_mass",
    "exploratory_baseline_fp_rate",
    "exploratory_yield_ratio",
    # --- Labels ---
    "factor_found_in_histogram",
    "dominant_outcome_success",
    # --- Ideal peak comparison ---
    "near_peak_bins_count",
    # weak structural diagnostics (secondary metrics)
    "r_true", "peaks_total", "peaks_hit",
    "peak_hit_fraction", "mass_near_peaks",
    "best_bit_order",
    "uniform_peak_mass", "peak_enrichment",
    # --- Legacy ---
    "strict_success", "p", "q", "r_min",
    "runtime_sec",
]

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)

def _as_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)

def _is_completed(row: dict[str, Any]) -> bool:
    return str(row.get("run_status") or "").strip().lower() == "completed"

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _run_exploratory_null_baseline(
    trials: int,
    t: int,
    a: int,
    N: int,
    seed: int,
    postprocess_fn,
) -> float:
    """
    Estimate the exploratory false-positive rate by testing random t-bit strings.
    """
    rng = random.Random(seed)
    false_positives = 0
    for _ in range(trials):
        y = rng.randint(0, (1 << t) - 1)
        if postprocess_fn(y=y, t=t, a=a, N=N) is not None:
            false_positives += 1
    return false_positives / trials if trials > 0 else 0.0


def _compute_exploratory_yield(
    counts: dict[str, int],
    t: int,
    a: int,
    N: int,
    try_reversed_bitorder: bool = True,
) -> float:
    """Per-shot yield under the exploratory postprocessing path."""
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    factorable_shots = 0
    for raw_bits, count in counts.items():
        bits = str(raw_bits).replace(" ", "")
        if not bits:
            continue

        candidates = [int(bits, 2)]
        if try_reversed_bitorder and bits[::-1] != bits:
            candidates.append(int(bits[::-1], 2))

        for y_val in candidates:
            if exploratory_postprocess_y(y=y_val, t=t, a=a, N=N) is not None:
                factorable_shots += count
                break

    return factorable_shots / total_shots

def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add strict/exploratory metrics and peak-overlap diagnostics to one row."""
    enriched = dict(row)
    counts = row.get("counts", {})
    N = row.get("N")
    a = row.get("a")
    t = row.get("t")

    null = {
        "top1_success": None, "factor_yield_mass": None,
        "strict_baseline_fp_rate": None, "yield_vs_baseline_ratio": None,
        "exploratory_yield_mass": None, "exploratory_baseline_fp_rate": None,
        "exploratory_yield_ratio": None,
        "factor_found_in_histogram": _as_bool(row.get("strict_success")),
        "dominant_outcome_success": None,
        "near_peak_bins_count": None,
        "r_true": None, "peaks_total": None, "peaks_hit": None,
        "peak_hit_fraction": None, "mass_near_peaks": None,
        "best_bit_order": None,
        "uniform_peak_mass": None, "peak_enrichment": None,
    }

    if not counts or not _is_completed(row) or N is None or a is None or t is None:
        enriched.update(null)
        return enriched

    # Strict layer (main metric)
    yield_info = per_shot_factor_yield(counts=counts, t=t, a=a, N=N)
    top1_success = yield_info["top1_success"]
    factor_yield = yield_info["factor_yield_mass"]

    strict_baseline = strict_null_baseline_fp_rate(t=t, a=a, N=N, trials=2048, seed=42)
    if strict_baseline > 0:
        yield_ratio = factor_yield / strict_baseline
    else:
        yield_ratio = float("inf") if factor_yield > 0 else 1.0

    # Exploratory layer (debugging context)
    exploratory_yield = _compute_exploratory_yield(counts=counts, t=t, a=a, N=N)
    exploratory_baseline = _run_exploratory_null_baseline(
        trials=2048, t=t, a=a, N=N, seed=42,
        postprocess_fn=exploratory_postprocess_y,
    )
    if exploratory_baseline > 0:
        exploratory_ratio = exploratory_yield / exploratory_baseline
    else:
        exploratory_ratio = float("inf") if exploratory_yield > 0 else 1.0

    # Histogram overlap with ideal peaks
    overlap = histogram_vs_ideal_overlap(counts=counts, a=a, N=N, t=t)
    near_peak_bins_count = int(overlap.get("near_peak_bins_count", 0) or 0)
    mass_near = overlap["mass_near_peaks"]

    # Uniform expectation based on the union of near-peak bins.
    uniform_peak_mass = near_peak_bins_count / (1 << t) if t > 0 and near_peak_bins_count > 0 else 0
    peak_enrichment = mass_near / uniform_peak_mass if uniform_peak_mass > 0 else 0

    factor_found_in_histogram = _as_bool(row.get("strict_success"))

    # Simple composite label used for quick filtering in plots/reports.
    dominant_outcome_success = (
        top1_success
        and yield_ratio > 1.5
        and peak_enrichment > 2.0
    )

    enriched.update({
        "top1_success": top1_success,
        "factor_yield_mass": round(factor_yield, 6),
        "factorable_shots": yield_info["factorable_shots"],
        "total_shots": yield_info["total_shots"],
        "unique_factorable_bitstrings": yield_info["unique_factorable_bitstrings"],
        "top1_bitstring": yield_info["top1_bitstring"],
        "top1_count": yield_info["top1_count"],
        "strict_baseline_fp_rate": round(strict_baseline, 6),
        "yield_vs_baseline_ratio": round(yield_ratio, 4),
        "exploratory_yield_mass": round(exploratory_yield, 6),
        "exploratory_baseline_fp_rate": round(exploratory_baseline, 6),
        "exploratory_yield_ratio": round(exploratory_ratio, 4),
        "factor_found_in_histogram": factor_found_in_histogram,
        "dominant_outcome_success": dominant_outcome_success,
        "near_peak_bins_count": near_peak_bins_count,
        "r_true": overlap["r_true"],
        "peaks_total": overlap["peaks_total"],
        "peaks_hit": overlap["peaks_hit"],
        "peak_hit_fraction": round(overlap["peak_hit_fraction"], 4),
        "mass_near_peaks": round(overlap["mass_near_peaks"], 6),
        "best_bit_order": overlap.get("best_bit_order"),
        "uniform_peak_mass": round(uniform_peak_mass, 6),
        "peak_enrichment": round(peak_enrichment, 4),
    })
    return enriched

def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in SUMMARY_COLUMNS})

def _plot_ideal_peak_overlay(row: dict[str, Any], figures_dir: Path) -> None:
    counts = row.get("counts", {})
    N = row.get("N")
    a = row.get("a")
    t = row.get("t")
    if not counts or N is None or a is None or t is None:
        return

    two_to_t = 1 << t
    ideal = compute_ideal_peaks(a=a, N=N, t=t)
    r_true = ideal[0]["r_true"] if ideal else "?"

    # Plot in the bit order that gave the better overlap score.
    best_order = row.get("best_bit_order", "raw")
    use_reverse = (best_order == "reversed")

    y_counts: dict[int, int] = {}
    for raw_bits, count in counts.items():
        bits = str(raw_bits).replace(" ", "")
        if not bits:
            continue
        y = int(bits[::-1], 2) if use_reverse else int(bits, 2)
        y_counts[y] = y_counts.get(y, 0) + count

    total = sum(y_counts.values())
    backend = row.get("backend", "unknown")

    ys = sorted(y_counts.keys())
    probs = [y_counts.get(y, 0) / total for y in ys]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(ys, probs, width=max(1, two_to_t // 200), alpha=0.6,
           color="#4C72B0", label="Observed", edgecolor="none")

    for i, peak in enumerate(ideal):
        label = "Ideal peaks (s/r)" if i == 0 else None
        ax.axvline(x=peak["y_ideal"], color="red", linestyle="--",
                    alpha=0.7, linewidth=1.2, label=label)

    uniform_prob = 1.0 / two_to_t
    ax.axhline(y=uniform_prob, color="gray", linestyle=":", alpha=0.5,
               linewidth=1.0, label=f"Uniform (1/{two_to_t})")

    fy = row.get("factor_yield_mass", 0) or 0
    bl = row.get("strict_baseline_fp_rate", 0) or 0
    pe = row.get("peak_enrichment", 0) or 0
    mn = row.get("mass_near_peaks", 0) or 0

    ax.set_xlabel(f"Measurement outcome y  (bit order: {best_order})")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"N={N}, a={a}, r_true={r_true} | backend={backend}\n"
        f"strict_yield={fy:.3f}  strict_baseline={bl:.3f}  "
        f"peak_enrich={pe:.1f}x  mass_near_peaks={mn:.4f}"
    )
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, two_to_t - 0.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(figures_dir / f"ideal_overlay_N{N}_a{a}.png", dpi=200)
    plt.close()

def _plot_two_layer_comparison(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    completed = [r for r in rows if _is_completed(r)]
    if not completed:
        return

    labels = [f'N={r["N"]}' for r in completed]
    x = np.arange(len(labels))

    strict_yield = [r.get("factor_yield_mass", 0) or 0 for r in completed]
    strict_bl = [r.get("strict_baseline_fp_rate", 0) or 0 for r in completed]
    expl_yield = [r.get("exploratory_yield_mass", 0) or 0 for r in completed]
    expl_bl = [r.get("exploratory_baseline_fp_rate", 0) or 0 for r in completed]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    width = 0.2

    # Strict layer
    ax1.bar(x - width/2, strict_yield, width, label="Strict yield (hardware)",
            color="#4C72B0", alpha=0.85)
    ax1.bar(x + width/2, strict_bl, width, label="Strict baseline (random)",
            color="#C44E52", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Fraction of shots")
    ax1.set_title("Strict layer: quality-filtered + bounded multiples (k≤4)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis="y")

    # Exploratory layer
    ax2.bar(x - width/2, expl_yield, width, label="Exploratory yield (hardware)",
            color="#4C72B0", alpha=0.85)
    ax2.bar(x + width/2, expl_bl, width, label="Exploratory baseline (random)",
            color="#C44E52", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Fraction of shots")
    ax2.set_title("Exploratory layer: all convergents + unbounded multiples")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(figures_dir / "two_layer_comparison.png", dpi=200)
    plt.close()

def _plot_peak_overlap_summary(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    completed = [r for r in rows if _is_completed(r) and r.get("peaks_total")]
    if not completed:
        return

    labels = [f'N={r["N"]}' for r in completed]
    mass_near = [r.get("mass_near_peaks", 0) for r in completed]
    uniform = [r.get("uniform_peak_mass", 0) for r in completed]
    enrichment = [r.get("peak_enrichment", 0) for r in completed]

    x = np.arange(len(labels))
    width = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Mass near peaks vs uniform expectation
    ax1.bar(x - width/2, mass_near, width, label="Observed mass near peaks",
            color="#55A868", alpha=0.85)
    ax1.bar(x + width/2, uniform, width, label="Uniform noise expectation",
            color="#CCCCCC", alpha=0.85, edgecolor="#999999")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Fraction of shots")
    ax1.set_title("Shot mass near ideal QPE peaks")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis="y")

    for i, r in enumerate(completed):
        ax1.annotate(f'r={r.get("r_true", "?")}',
                     (x[i], max(mass_near[i], uniform[i]) + 0.002),
                     ha="center", fontsize=8, color="#333")

    # Peak enrichment
    colors = ["#55A868" if e > 2.0 else "#DD8452" if e > 1.0 else "#C44E52"
              for e in enrichment]
    ax2.bar(x, enrichment, width=0.5, color=colors, alpha=0.85, edgecolor="black",
            linewidth=0.3)
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, label="Uniform = 1.0x")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Enrichment over uniform")
    ax2.set_title("Peak enrichment (observed / expected under uniform noise)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(figures_dir / "peak_overlap_summary.png", dpi=200)
    plt.close()

def _print_rigorous_report(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("RIGOROUS TWO-LAYER SUCCESS REPORT")
    print("=" * 80)

    completed = [r for r in rows if _is_completed(r)]

    for r in completed:
        N = r["N"]
        a = r["a"]
        backend = r.get("backend", "?")
        two_q = r.get("two_qubit_gates", "?")

        print(f"\n--- N={N} (a={a}) on {backend} | 2q_gates={two_q} ---")

        # Strict layer
        top1 = r.get("top1_success")
        top1_bits = r.get("top1_bitstring", "?")
        top1_count = r.get("top1_count", "?")
        fy = r.get("factor_yield_mass", 0)
        bl = r.get("strict_baseline_fp_rate", 0)
        ratio = r.get("yield_vs_baseline_ratio", 0)
        factorable = r.get("factorable_shots", 0)
        total = r.get("total_shots", 0)

        if fy is None:
            print("  No shot histogram stored for this run; skipping strict/exploratory metrics.")
            continue

        print(f"  STRICT LAYER (headline metrics):")
        print(f"    Top-1 success:          {top1}  (bitstring={top1_bits}, count={top1_count})")
        print(f"    Per-shot factor yield:  {fy:.4f}  ({factorable}/{total} shots)")
        print(f"    Null baseline rate:     {bl:.4f}")
        print(f"    Yield / baseline:       {ratio:.2f}x  {'(ENRICHED)' if ratio > 1.5 else '(NOT enriched)'}")

        # Exploratory layer
        efy = r.get("exploratory_yield_mass", 0)
        ebl = r.get("exploratory_baseline_fp_rate", 0)
        er = r.get("exploratory_yield_ratio", 0)

        print(f"  EXPLORATORY LAYER (debugging/transparency):")
        print(f"    Exploratory yield:      {efy:.4f}")
        print(f"    Exploratory baseline:   {ebl:.4f}")
        print(f"    Exploratory ratio:      {er:.2f}x")

        # Peak diagnostics
        r_true = r.get("r_true", "?")
        peaks_total = r.get("peaks_total", 0)
        peaks_hit = r.get("peaks_hit", 0)
        mass_near = r.get("mass_near_peaks", 0)
        uniform_pm = r.get("uniform_peak_mass", 0)
        peak_enrich = r.get("peak_enrichment", 0)
        best_order = r.get("best_bit_order", "?")

        print(f"  IDEAL PEAK COMPARISON (bit order: {best_order}):")
        print(f"    True order r:           {r_true}")
        print(f"    Ideal peaks hit:        {peaks_hit}/{peaks_total} ({r.get('peak_hit_fraction', 0):.2%}) [weak diagnostic]")
        print(f"    Mass near peaks:        {mass_near:.4f}  (uniform would give ~{uniform_pm:.4f})")
        print(f"    Peak enrichment:        {peak_enrich:.2f}x over uniform")

        # Verdict
        dominant = r.get("dominant_outcome_success")
        found = r.get("factor_found_in_histogram")

        print(f"  LABELS:")
        print(f"    factor_found_in_histogram (top-k scan): {found}")
        print(f"    dominant_outcome_success (composite):    {dominant}")

        if dominant:
            verdict = "GENUINE QUANTUM SIGNAL: top-1 factors, enriched yield, and period structure"
        elif top1 and ratio > 1.0 and peak_enrich > 1.5:
            verdict = "MARGINAL: some evidence of quantum signal, but below confidence thresholds"
        elif found and ratio <= 1.0:
            verdict = "EXPLORATORY ONLY: factors found by scanning, but yield not above random baseline"
        elif found:
            verdict = "WEAK: factors found, slight yield enrichment, but period structure absent"
        else:
            verdict = "NO FACTORS FOUND"

        print(f"    VERDICT: {verdict}")

    print("\n" + "=" * 80)
    print("NOTE: dominant_outcome_success thresholds (yield ratio > 1.5x,")
    print("peak enrichment > 2.0x) are heuristic screening filters, not")
    print("mathematically derived bounds.  Report them as such.")
    print("=" * 80)

def _analyze_input_file(
    *,
    input_path: Path,
    output_csv: Path,
    figures_dir: Path,
    run_label: str,
    required: bool,
) -> None:
    if not input_path.exists():
        if required:
            raise SystemExit(f"Input file does not exist: {input_path}")
        print(f"Skipped {run_label}: {input_path} not found")
        return

    raw_rows = _load_jsonl(input_path)
    if not raw_rows:
        if required:
            raise SystemExit(f"No rows found in {input_path}")
        print(f"Skipped {run_label}: empty file")
        return

    enriched_rows = [enrich_row(row) for row in raw_rows]

    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(rows=enriched_rows, output_path=output_csv)

    for row in enriched_rows:
        if _is_completed(row) and row.get("counts"):
            _plot_ideal_peak_overlay(row, figures_dir)

    _plot_two_layer_comparison(enriched_rows, figures_dir)
    _plot_peak_overlap_summary(enriched_rows, figures_dir)
    _print_rigorous_report(enriched_rows)

    print(f"\n[{run_label}] Loaded {len(enriched_rows)} rows from {input_path}")
    print(f"[{run_label}] Wrote summary CSV to {output_csv}")
    print(f"[{run_label}] Wrote figures to {figures_dir}")

    enriched_path = figures_dir / f"enriched_{input_path.name}"
    with enriched_path.open("w", encoding="utf-8") as f:
        for row in enriched_rows:
            slim = {k: v for k, v in row.items() if k != "counts"}
            f.write(json.dumps(slim) + "\n")
    print(f"[{run_label}] Wrote enriched results to {enriched_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Shor sweep results with rigorous two-layer success metrics."
    )
    parser.add_argument("--input", help="JSONL results file from run_sweep.py")
    parser.add_argument("--output-csv", default="data/summary/results_summary.csv")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    if input_path is None:
        raise SystemExit("Please provide --input path to your JSONL results file.")

    _analyze_input_file(
        input_path=input_path,
        output_csv=Path(args.output_csv),
        figures_dir=Path(args.figures_dir),
        run_label="primary",
        required=True,
    )


if __name__ == "__main__":
    main()
