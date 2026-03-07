# Shor Order-Finding Sweep (Qiskit 2.x)

This project runs small Shor-style order-finding experiments in Qiskit and summarizes them with strict post-processing metrics.

The workflow is:

1. Build QPE order-finding circuits for semiprimes `N`.
2. Run on Aer simulator or IBM hardware.
3. Post-process bitstrings with strict order checks.
4. Enrich and plot results with strict and exploratory diagnostics.

## What is in this repo

- `shor/modexp.py`: modular-multiplication oracle builders.
- `shor/qpe.py`: QPE order-finding circuit construction.
- `shor/postprocess.py`: strict and exploratory post-processing, plus peak-overlap helpers.
- `shor/runtime.py`: IBM Runtime execution helpers.
- `experiments/run_sweep.py`: sweep runner (simulator, noisy simulator, hardware).
- `experiments/analyze_results.py`: enriched CSV + plots + console report.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib numpy
```

For hardware, configure IBM credentials (or provide credentials JSON expected by `run_sweep.py`).

## Run sweeps

Run from `sometimes-too-slow-for-shor/`.

### Simulator

```bash
python experiments/run_sweep.py
```

Optional flags:

- `--include-10bit`
- `--noise-backend ibm_sherbrooke`
- `--n-values 15,21,35`
- `--shots 2048`

### Hardware

```bash
python experiments/run_sweep.py --hardware --backend-name auto
```

or pin a backend:

```bash
python experiments/run_sweep.py --hardware --backend-name ibm_sherbrooke
```

### Output behavior

Raw rows are JSONL in `data/raw/`:

- simulator: `results_simulator_*.jsonl`
- hardware: `results_hardware_*.jsonl`

If you do not pass `--output`, the runner resumes/appends to the latest matching file for that run type when available.

## Success and baseline semantics

### Strict success (raw run row)

`strict_success` in `run_sweep.py` is from strict histogram scan (`shor_postprocess_counts`, top-k search over measured bitstrings).

### Baselines

There are two distinct baseline notions by design:

- Canonical strict null baseline (used for analysis metrics):  
  `strict_null_baseline_fp_rate(...)` in `shor/postprocess.py`, computed from uniformly random `y in [0, 2^t - 1]` with `strict_postprocess_y`.
- Legacy histogram baseline (debug-only, stored in raw run rows):
  - `top1_histogram_baseline_trials`
  - `top1_histogram_baseline_false_positives`
  - `top1_histogram_baseline_fp_rate`
  - `top1_histogram_baseline_examples`

## Analyze results

`analyze_results.py` requires an explicit input file:

```bash
python experiments/analyze_results.py --input data/raw/results_simulator_20260228_181306.jsonl
```

Defaults:

- `--output-csv data/summary/results_summary.csv`
- `--figures-dir figures`

### What analysis writes

- Enriched summary CSV (default `data/summary/results_summary.csv`).
- Plot files in `figures/`:
  - `two_layer_comparison.png`
  - `peak_overlap_summary.png`
  - `ideal_overlay_N{N}_a{a}.png` (per completed row with counts)
- Enriched JSONL (without raw `counts`) in figures dir:
  - `enriched_<input_filename>.jsonl`

### Metrics used in analysis

- Strict layer:
  - `top1_success`
  - `factor_yield_mass`
  - `strict_baseline_fp_rate`
  - `yield_vs_baseline_ratio`
- Exploratory layer:
  - `exploratory_yield_mass`
  - `exploratory_baseline_fp_rate`
  - `exploratory_yield_ratio`
- Peak-structure diagnostics:
  - `mass_near_peaks`
  - `near_peak_bins_count`
  - `uniform_peak_mass` (computed from near-peak bin union size)
  - `peak_enrichment`
  - `peaks_hit` / `peak_hit_fraction` (treated as weaker diagnostics)

`dominant_outcome_success` is a heuristic composite label, not a proof.

## Notes

- This code is aimed at toy-scale order-finding studies and metric discipline.
- The exact permutation-oracle path is not scalable for large factoring instances.


## Citation

```bash
@misc{sharma2026shorhonestmetrics,
 author = {Monit Sharma},
 title = {Sometimes It's Too Slow… for Shor: Factoring Numbers on IBM Quantum Hardware with Honest Metrics},
 year = {2026},
 howpublished = {\url{https://github.com/MonitSharma/medium_articles/tree/main/sometimes-too-slow-for-shor}},
 note = {GitHub repository}
 ```