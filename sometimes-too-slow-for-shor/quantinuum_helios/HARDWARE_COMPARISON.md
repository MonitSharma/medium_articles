# Shor on Real Hardware: IBM Quantum vs Quantinuum Helios

*Updated 2026-06-21 with the new `N=21, a=2` Helios-1E run (Nexus job `e03104e4-5a1c-46bb-aa89-5ed28a559f55`).*

This note is the consolidated, apples-to-apples comparison of every **completed** hardware/emulator run of the Shor / quantum-phase-estimation (QPE) order-finding circuit in this project, across two very different machines:

- **IBM Quantum** — superconducting transmons (`ibm_torino`, `ibm_fez`), billed by runtime minute.
- **Quantinuum Helios** — trapped ions (`Helios-1` device, `Helios-1E` noisy emulator), billed in Helios Quantum Credits (HQC) tied to circuit volume.

Everything below uses the project's own strict post-processing: top-1 success, per-shot factor yield, a strict false-positive null baseline, and ideal-peak enrichment. No metric is relaxed between providers.

> **The one-paragraph version.** On `N=15`, where the oracle is a hand-built `mod 15` permutation, Helios is in a different league — a genuine period signal (`15.6×` peak enrichment) versus IBM's "factors are somewhere in the histogram" (`1.0×`). On `N=21`, the story flips into a wall: the generic exact-permutation oracle compiles to ~10k two-qubit gates, and **neither machine produces a clean period signal**. IBM still scrapes factors out of the histogram by brute-force scanning; Helios does not, even with a mathematically favorable base. And the Helios `N=21` runs cost ~`27,200 HQC` each — roughly `30×` the `N=15` charge. The bottleneck is the oracle construction, not the qubit count.

---

## The problem and how each machine runs it

We run textbook Shor order-finding: a `t`-qubit counting register driving controlled-`a^{2^k} mod N` modular exponentiation on an `n`-qubit work register, then an inverse QFT and measurement of the counting register. Recovering the period `r` from the measured phase yields the factors via `gcd(a^{r/2} ± 1, N)`.

| Stage | IBM path | Quantinuum Helios path |
|---|---|---|
| Circuit source | Qiskit (`shor/qpe.py`) | same Qiskit circuit |
| Compilation | Qiskit transpile → ISA gates on the backend coupling map | Qiskit → TKET (`pytket-qiskit`) → QIR (`pytket-qir`, ADAPTIVE profile) |
| Oracle | exact permutation unitary (generic), `mod15` special-case for `N=15` | identical oracle, decomposed into the Helios native gate set (`RZZ`, `U1q`, …) |
| Execution | IBM Runtime sampler | Quantinuum Nexus `start_execute_job`, HQC cost-guarded |
| Hardware | superconducting, all-to-all via routing/SWAPs | trapped ion, native all-to-all, very high two-qubit fidelity |
| Billing | wall-clock runtime × plan rate | HQC = job overhead + shot-scaled weighted gate/measurement count |

The two compilers explain most of what follows. IBM's router inserts SWAP networks, so its two-qubit gate count explodes with `N`. Helios's all-to-all connectivity keeps the gate count far lower for the same circuit — but HQC pricing is *driven by* that gate count, so a low gate count is exactly what keeps Helios affordable, and the generic oracle does not stay low.

---

## Every completed run

![Hero comparison](comparison/hero_hardware_comparison.png)

| Provider | System | N | base `a` | order `r` | depth | 2q gates | strict success | factor yield | yield / null | peak enrichment | runtime | native cost |
|---|---|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---:|
| IBM | `ibm_torino` | 15 | 7 | 4 | 2,316 | 727 | ✅ | 0.108 | 1.88× | 1.00× | 32.4 s | ~$52 (PAYG) |
| IBM | `ibm_torino` | 21 | 19 | 6 | 73,170 | 23,054 | ✅ | 0.040 | 1.82× | 1.00× | 906.9 s | ~$1,451 |
| IBM | `ibm_fez` | 35 | 4 | 6 | 382,165 | 116,706 | ✅ | 0.029 | 2.61× | 1.56× | 6,569.7 s | ~$10,511 |
| **Helios** | `Helios-1` | 15 | 7 | 4 | 840 | 328 | ✅ | **0.665** | **11.54×** | **15.63×** | 5,673 s | **896 HQC** |
| **Helios** | `Helios-1E` | 21 | 4 | **3 (odd)** | 30,008 | 10,130 | ❌ | 0.000 | 1.00× | 2.67× | 23,530 s | 27,198 HQC |
| **Helios** | `Helios-1E` | 21 | 2 | 6 | 30,017 | 10,130 | ❌ | 0.031 | 1.42× | 1.28× | 15,865 s | **27,198 HQC** |

- "strict success" = the standard Shor post-processing recovered non-trivial factors.
- "peak enrichment" = measured mass near ideal QPE peaks ÷ the uniform-random expectation. `1.0×` means the histogram is statistically indistinguishable from random noise near the peaks, even if a factor is hiding in it.
- The `N=15` Helios run and both `N=21` runs are noisy state-vector emulation (`Helios-1` and `Helios-1E` respectively); IBM runs are physical QPU. The HQC charges are real.

---

## N=15: the clean Helios win

This is the one case where the oracle stays small (the `mod15` special-case compiles to **328** two-qubit gates on Helios vs **727** on IBM), and the difference in output quality is stark:

| Metric | IBM `ibm_torino` | Helios-1 | Read |
|---|---:|---:|---|
| Strict factor yield | 0.108 | **0.665** | `6.1×` more shots factor on Helios |
| Yield / strict null | 1.88× | **11.54×** | Helios separates far more cleanly from random post-processing |
| Mass near ideal peaks | 0.047 | **0.732** | IBM ≈ uniform expectation; Helios concentrates on the QPE peaks |
| Peak enrichment | 1.00× | **15.63×** | IBM has no period structure; Helios clearly does |
| Top-1 outcome factors? | No | **Yes** | Helios' single most-likely outcome already factors |

IBM's `N=15` result is real but weak: factors are recoverable by scanning the histogram, yet the distribution itself carries no detectable period structure (`1.0×` enrichment). Helios' result is qualitatively different — the dominant outcome factors, two-thirds of shots factor, and the distribution is sharply peaked on the ideal phases. **"Factors found somewhere in the histogram" and "the distribution clearly contains the period" are different claims, and only Helios satisfies the stronger one here.**

The trade-off: Helios took `5,673 s` (emulation) vs IBM's `32 s` of QPU time, and cost `896 HQC`.

---

## N=21: the wall, and why the base matters

![N=21 base comparison](comparison/n21_base_comparison.png)

`N=21` is where the generic oracle stops being cheap. The exact-permutation construction compiles to **10,130** two-qubit gates and depth **~30,000** on Helios. We ran it twice, and the two runs isolate two different failure modes.

**`a=4` — wrong base, mathematically dead on arrival.** The order of `4 mod 21` is `3`, which is *odd*. Shor's `a^{r/2} ± 1` trick requires an even order, so this base **cannot** factor `N=21` on any hardware, ideal or not. The run did show weak order-3 structure (`2.67×` enrichment, all 3 ideal peaks hit), but zero factors is the correct, expected outcome. This is a worked example of why base selection is part of the algorithm, not a hardware result.

**`a=2` — right base, but the signal drowns.** The order of `2 mod 21` is `6` (even), and `gcd(2^3 ± 1, 21) = {7, 3}` — a textbook factor-yielding base. In principle this should work. On Helios-1E it did **not**: factor yield was `0.031` (only `1.42×` the random null), and peak enrichment was `1.28×` — essentially noise. There is a visible spike at `y=512` (the `k=3` peak) in the spectrum, but the overall distribution is too washed out by the depth-30k circuit for strict post-processing to recover `r`.

For reference, IBM's `N=21, a=19` run (also order 6) *did* clear strict success — but only by histogram scanning, with its own enrichment at `1.0×` (no real period structure either), and it needed **23,054** two-qubit gates to get there. So the honest read of `N=21` is:

> Neither machine produces a genuine period signal at `N=21` with the generic oracle. IBM recovers factors by brute-force histogram scanning; Helios does not. Both are operating below the threshold where the QPE distribution is meaningfully period-structured.

---

## Resources and cost: two different economics

**Two-qubit gates (Panel C).** IBM's routed circuits balloon — `727 → 23,054 → 116,706` for `N = 15 → 21 → 35`. Helios's all-to-all connectivity keeps the same circuits far smaller (`328` at `N=15`, `10,130` at `N=21`). On gate count alone, Helios wins decisively.

**But HQC is priced on gate count.** Quantinuum's documented model is roughly `HQC ≈ 5 + C·(N_1q + 10·N_2q + 5·N_m)/5000` per shot-batch, with Helios adding dynamic cost for branching programs. So the same gate count that makes Helios efficient is what you pay for:

| Run | Two-qubit gates | HQC charged | ≈ USD @ $12.50/HQC* |
|---|---:|---:|---:|
| Helios `N=15`, `a=7` | 328 | 896 | ~$11,200 |
| Helios `N=21`, `a=4` | 10,130 | 27,198 | ~$340,000 |
| Helios `N=21`, `a=2` | 10,130 | 27,198 | ~$340,000 |

\*Azure H2 Standard-bundle-implied rate, used only for order-of-magnitude scale — **not** a Helios quote.

The `N=21` HQC charge is ~`30×` the `N=15` charge, tracking the `~31×` jump in two-qubit gates. Extrapolating the circuit volume, an `N=35` exact-oracle run would land near `130,000–150,000 HQC`. The generic oracle becomes economically impractical well before it becomes algorithmically interesting.

**IBM's runtime billing** is far cheaper at these scales (`~$52 / ~$1,451 / ~$10,511` at PAYG `$96/min`), because it charges wall-clock, not circuit volume. IBM's cost wall is *time* (and fidelity), not credits.

---

## What is best at what

| Dimension | Winner | Why |
|---|---|---|
| Signal quality where the oracle is compact (`N=15`) | **Helios** | `15.6×` enrichment, top-1 factors, two-thirds yield |
| Two-qubit gate efficiency | **Helios** | all-to-all connectivity, no SWAP networks |
| Cost at small scale | **IBM** | `~$52` vs `896 HQC` for the same `N=15` |
| Getting *a* completed run at larger `N` | **IBM** | `N=21` and `N=35` both completed; Helios `N=21` completes but `N=35` exceeds practical credit budgets |
| Honest period recovery at `N=21` | **Neither** | both below the period-signal threshold with the generic oracle |
| Base/algorithm hygiene | **N/A** | `a=4` failing on `N=21` is math, not hardware |

The cross-cutting lesson: **the limiting factor is the oracle compilation, not the chip.** Helios's hardware is clearly capable (see `N=15`), but the generic exact-permutation oracle expands into a circuit whose depth destroys the signal and whose HQC cost destroys the budget. A compact, hand-specialized oracle (as in `N=15`) is what unlocks Helios; without it, `N=21` is a wall on both machines.

---

## Reproducing this

From `sometimes-too-slow-for-shor/`:

```bash
# 1. Fetch the finished a=2 Helios job and write its raw row
python quantinuum_helios/fetch_helios_job_result.py \
  --job-id e03104e4-5a1c-46bb-aa89-5ed28a559f55 \
  --N 21 --base 2 --shots 1024 \
  --template quantinuum_helios/data/raw/results_helios_hardware_N21_Helios-1E_20260620_retry.jsonl \
  --submission quantinuum_helios/data/submissions/submit_N21_a2_Helios-1E_20260620.json \
  --output quantinuum_helios/data/raw/results_helios_hardware_N21_a2_Helios-1E_20260620.jsonl

# 2. Strict analysis (period/yield/enrichment) for that run
python quantinuum_helios/analyze_helios_results.py \
  --input quantinuum_helios/data/raw/results_helios_hardware_N21_a2_Helios-1E_20260620.jsonl \
  --output-csv quantinuum_helios/data/summary/results_summary_helios_N21_a2_Helios-1E_20260620.csv \
  --figures-dir quantinuum_helios/figures/N21_a2

# 3. Build the consolidated comparison figures + CSV
python quantinuum_helios/generate_full_comparison.py
```

Outputs:

- `comparison/full_hardware_comparison.csv` — one row per completed run
- `comparison/hero_hardware_comparison.png` / `.pdf` — 4-panel overview
- `comparison/n21_base_comparison.png` / `.pdf` — `N=21` base/period deep dive

## Pricing references

- IBM Quantum pricing: https://www.ibm.com/quantum/products
- Quantinuum HQC workflow: https://docs.quantinuum.com/systems/user_guide/hardware_user_guide/workflow.html
- Helios costing: https://docs.quantinuum.com/systems/trainings/helios/getting_started/costing.html
- Azure Quantinuum bundle pricing (USD/HQC scale only): https://learn.microsoft.com/en-us/azure/quantum/pricing
