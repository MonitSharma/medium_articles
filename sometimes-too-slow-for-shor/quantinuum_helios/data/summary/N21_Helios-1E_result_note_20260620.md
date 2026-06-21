# N=21 Helios-1E Result Note

Run date: 2026-06-20

## Completed Hardware-Emulator Run

- Device: Quantinuum Helios-1E
- Nexus execute job: `0f42f397-ad7f-477c-8bed-3be9dc7b90ce`
- Problem: `N=21`
- Base: `a=4`
- Shots: `1024`
- Run status: `completed`
- Final cost: `27197.72 HQC`
- Runtime reported by local client: `23530.35 s` (`6.54 h`)
- Circuit width: `15 qubits`
- Decomposed circuit depth: `30008`
- Two-qubit gates: `10130`
- Unique measured bitstrings: `502`

## Factoring Outcome

The run did not recover factors under strict Shor postprocessing.

- Strict success: `False`
- Recovered `p`: none
- Recovered `q`: none
- Recovered order: none
- Per-shot factor yield: `0 / 1024`

This result is not primarily a hardware-execution failure. The selected base was mathematically unfavorable for factoring `N=21`.

For `N=21` and `a=4`, the true order is:

- `ord_21(4) = 3`

Since the order is odd, standard Shor postprocessing cannot use `a^(r/2) +/- 1` to recover non-trivial factors. The analysis still showed phase-structure evidence:

- Ideal `r=3` peaks hit: `3 / 3`
- Mass near ideal peaks: `0.023438`
- Uniform expected peak mass: `0.008789`
- Peak enrichment: `2.6667x`

So the useful interpretation is: the run showed weak order-3 peak enrichment, but the chosen base could not produce factors.

## Better Follow-up Base

A better base for a follow-up `N=21` run is:

- Recommended base: `a=2`
- True order: `ord_21(2) = 6`
- `a^(r/2) = 2^3 = 8`
- `gcd(8 - 1, 21) = 7`
- `gcd(8 + 1, 21) = 3`

This is a factor-yielding base because the order is even and `a^(r/2)` is not congruent to `-1 mod N`.

Other factor-yielding bases include `a=8`, `a=10`, `a=13`, and `a=19`. Bases such as `a=4` and `a=16` have odd order for `N=21`, while bases such as `a=5` and `a=17` have even order but hit the `a^(r/2) = -1 mod N` failure case.

## HQC Budget Assessment

The completed `a=4` run cost `27197.72 HQC`. A follow-up run with `a=2` should have broadly similar cost because it uses the same `N`, the same counting-register size, and the same 1024-shot Helios-1E execution path. The exact HQC cost can still differ after decomposition/QIR conversion, but the previous run is the best empirical estimate.

With `131252 HQC` remaining:

- One `N=21, a=2, 1024-shot` run is realistic.
- At the previous cost, the remaining budget covers about `4.83` runs of this size.
- A practical execution guard would be `--max-cost 35000`, which is above the observed `27197.72 HQC` cost while still preventing an unexpectedly expensive submission.

No follow-up run has been submitted as part of this note.
