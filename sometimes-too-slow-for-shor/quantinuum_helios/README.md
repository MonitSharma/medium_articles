# Quantinuum Helios-1 Sweep

This subfolder mirrors the IBM hardware experiment, but moves the hardware boundary to Quantinuum Nexus.

The working Helios path is:

1. Build the same Qiskit Shor/QPE circuits from `../shor/`.
2. Convert Qiskit to TKET with `pytket-qiskit`.
3. Rebase/decompose to QIR-compatible gates and emit QIR with `pytket-qir`.
4. Submit through `qnexus` using `HeliosConfig(system_name="Helios-1")`.
5. Save raw JSONL locally and reuse the existing strict/exploratory analysis plots.

## Install

From the repository root, using the existing project virtualenv if available:

```bash
python -m pip install -r sometimes-too-slow-for-shor/quantinuum_helios/requirements_quantinuum.txt
```

You also need the original project requirements (`qiskit`, `matplotlib`, `numpy`, etc.).

## Authenticate

If you are not already logged into Nexus on this machine:

```bash
python sometimes-too-slow-for-shor/quantinuum_helios/run_helios_sweep.py --login browser --n-values 15 --shots 128
```

For username/password login, use `--login credentials`. Quantinuum's docs note that Nexus auth tokens are stored locally and typically last 30 days.

## Run Helios-1

The default sweep matches the IBM hardware-scale set in this project: `N=15,21,35`.

```bash
cd sometimes-too-slow-for-shor
python quantinuum_helios/run_helios_sweep.py --target Helios-1 --shots 1024
```

Useful smaller smoke test:

```bash
python quantinuum_helios/run_helios_sweep.py --target Helios-1 --n-values 15 --shots 128
```

For Helios, Nexus requires a maximum HQC cost guard. The runner estimates QIR cost first and uses a 25% margin automatically. To cap a run explicitly:

```bash
python quantinuum_helios/run_helios_sweep.py --target Helios-1 --n-values 15 --shots 128 --max-cost 250
```

Raw output is written to:

```text
quantinuum_helios/data/raw/results_helios_hardware_*.jsonl
```

## Analyze

```bash
python quantinuum_helios/analyze_helios_results.py \
  --input quantinuum_helios/data/raw/results_helios_hardware_YYYYMMDD_HHMMSS.jsonl
```

Analysis output:

- `quantinuum_helios/data/summary/results_summary_helios.csv`
- `quantinuum_helios/figures/two_layer_comparison.png`
- `quantinuum_helios/figures/peak_overlap_summary.png`
- `quantinuum_helios/figures/ideal_overlay_N{N}_a{a}.png`

## Notes

- `Helios-1` is the default hardware target. Emulator targets can be passed explicitly if your Nexus account exposes them, for example `--target Helios-1E`.
- The existing exact permutation oracle is still toy-scale only. Larger `N` values can produce expensive QIR and should be cost-checked before running many shots.
- No credentials are stored in this folder.
