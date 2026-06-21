from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_sweep import (
    _configure_logging,
    _estimate_oracle_memory,
    _format_mb,
    _is_memory_failure,
    _log_memory_checkpoint,
    _log_run_outcome,
    _memory_guard_reason,
    _memory_snapshot,
    _pick_coprime_base,
    _result_row,
    _run_null_baseline,
)
from shor.postprocess import shor_postprocess_counts
from shor.qpe import build_qpe_order_finding_circuit

from quantinuum_runtime import DEFAULT_HELIOS_DEVICE, DEFAULT_NEXUS_PROJECT, run_on_quantinuum_nexus


LOGGER = logging.getLogger(__name__)
SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_HELIOS_NS = [15, 21, 35]
EXTENDED_TOY_NS = [91, 143, 221, 247, 299, 323]


def _parse_n_values(raw: str | None, extended: bool) -> list[int]:
    if raw:
        return [int(token.strip()) for token in raw.split(",") if token.strip()]

    values = list(DEFAULT_HELIOS_NS)
    if extended:
        values.extend(EXTENDED_TOY_NS)
    return values


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_ROOT / "data" / "raw" / f"results_helios_hardware_{timestamp}.jsonl"


def _latest_output_path() -> Path | None:
    candidates = sorted(
        (SCRIPT_ROOT / "data" / "raw").glob("results_helios_hardware_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_output_plan(output_arg: str | None, append: bool) -> tuple[Path, str, Path | None]:
    if output_arg:
        output_path = Path(output_arg)
        if output_path.exists():
            if append:
                return output_path, "a", output_path
            raise SystemExit(
                f"Refusing to overwrite existing output file: {output_path}. "
                "Use --append or provide a new --output path."
            )
        return output_path, "w", None

    latest_path = _latest_output_path()
    if latest_path is not None:
        return latest_path, "a", latest_path

    output_path = _default_output_path()
    return output_path, "w", None


def _load_completed_numbers(resume_path: Path | None, target: str) -> set[int]:
    if resume_path is None or not resume_path.exists():
        return set()

    completed: set[int] = set()
    with resume_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_status") != "completed":
                continue
            if row.get("backend") != target:
                continue
            try:
                completed.add(int(row["N"]))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def _quantinuum_failure_metadata(
    *,
    target: str,
    run_status: str,
    failure_reason: str,
    circuit_qubits: int | None,
    oracle_memory_estimate: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "backend": target,
        "job_id": None,
        "depth": None,
        "two_qubit_gates": None,
        "num_qubits": circuit_qubits,
        "runtime_sec": None,
        "run_status": run_status,
        "failure_reason": failure_reason,
        **_memory_snapshot(),
        "memory_before_transpile_mb": None,
        "memory_after_transpile_mb": None,
        "quantinuum_device": target,
    }
    metadata.update(oracle_memory_estimate)
    return metadata


def _attach_quantinuum_metadata(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    for key, value in metadata.items():
        if key.startswith("quantinuum_"):
            row[key] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Shor toy sweep on Quantinuum Helios hardware through Nexus."
    )
    parser.add_argument("--n-values", help="Comma-separated semiprimes. Defaults to 15,21,35.")
    parser.add_argument(
        "--extended-toy-sweep",
        action="store_true",
        help="Also include the larger toy semiprimes used by the simulator sweep.",
    )
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--base",
        type=int,
        help="Override the coprime base a. Intended for targeted hardware reruns.",
    )
    parser.add_argument("--t", type=int, help="Override counting-register size. Default is 2 * ceil(log2(N)).")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--optimisation-level",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Nexus/TKET optimisation level. Quantinuum docs define 0-2 for Nexus compile jobs.",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "standard", "permutation", "semi_compiled"],
        help="Oracle construction method. 'auto' uses the hand-optimized N=15 path when available.",
    )
    parser.add_argument("--baseline-trials", type=int, default=512)
    parser.add_argument("--target", default=DEFAULT_HELIOS_DEVICE, help="Quantinuum Nexus target device.")
    parser.add_argument("--project-name", default=DEFAULT_NEXUS_PROJECT)
    parser.add_argument(
        "--login",
        default="none",
        choices=["none", "browser", "credentials"],
        help="Optional qnexus login step before submission. Use 'none' when already authenticated.",
    )
    parser.add_argument("--simulator", default="state-vector", help="Quantinuum simulator mode for emulator targets.")
    parser.add_argument("--noisy-simulation", action="store_true", default=True)
    parser.add_argument("--no-noisy-simulation", action="store_false", dest="noisy_simulation")
    parser.add_argument("--target-2qb-gate", help="Optional Quantinuum target 2-qubit gate, e.g. ZZ.")
    parser.add_argument("--max-cost", type=int, help="Optional per-program maximum HQC cost guard.")
    parser.add_argument("--user-group", help="Optional Nexus/Quantinuum user group.")
    parser.add_argument(
        "--wait-timeout-seconds",
        type=int,
        help="Timeout passed to qnexus.jobs.wait_for. Omit to use qnexus default.",
    )
    parser.add_argument("--skip-cost-estimate", action="store_true")
    parser.add_argument(
        "--skip-qiskit-decompose",
        action="store_true",
        help=(
            "Skip local decomposition to u/cx before Qiskit-to-TKET conversion. "
            "The default decomposition is needed for the existing exact UnitaryGate oracles."
        ),
    )
    parser.add_argument("--output", help="JSONL output path. Defaults inside quantinuum_helios/data/raw/.")
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--memory-budget-fraction",
        type=float,
        default=0.90,
        help="Abort if predicted dense controlled-oracle cache exceeds this fraction of available RAM.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logging.getLogger("qnexus").setLevel(logging.WARNING)
    logging.getLogger("pytket").setLevel(logging.WARNING)

    rng = random.Random(args.seed)
    n_values = _parse_n_values(args.n_values, args.extended_toy_sweep)
    output_path, file_mode, resume_path = _resolve_output_plan(args.output, args.append)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing Quantinuum Helios results to %s", output_path)
    if resume_path is not None:
        LOGGER.info("Resuming from existing results file %s", resume_path)

    completed_numbers = _load_completed_numbers(resume_path, target=args.target)
    if completed_numbers:
        LOGGER.info("Skipping already completed N values from resume file: %s", sorted(completed_numbers))

    with output_path.open(file_mode, encoding="utf-8") as handle:
        for N in n_values:
            if N in completed_numbers:
                LOGGER.info("Skipping N=%d because it is already completed in %s", N, resume_path or output_path)
                continue

            n_work = math.ceil(math.log2(N))
            t = args.t if args.t is not None else 2 * n_work
            a = args.base if args.base is not None else _pick_coprime_base(N=N, rng=rng)
            if not (1 < a < N) or math.gcd(a, N) != 1:
                raise SystemExit(f"Base a={a} must satisfy 1 < a < N and gcd(a, N) = 1 for N={N}.")
            oracle_memory_estimate = _estimate_oracle_memory(N=N, a=a, t=t, n_work=n_work, method=args.method)
            prebuild_snapshot = _memory_snapshot()

            LOGGER.info(
                "Building QPE circuit for Quantinuum %s | N=%d bits=%d a=%d t=%d n_work=%d",
                args.target,
                N,
                N.bit_length(),
                a,
                t,
                n_work,
            )
            _log_memory_checkpoint(
                stage="pre-build",
                N=N,
                a=a,
                snapshot=prebuild_snapshot,
                estimate=oracle_memory_estimate,
            )

            guard_reason = _memory_guard_reason(
                snapshot=prebuild_snapshot,
                estimate=oracle_memory_estimate,
                budget_fraction=args.memory_budget_fraction,
            )
            if guard_reason is not None:
                LOGGER.warning("Aborting N=%d with a=%d before circuit build: %s", N, a, guard_reason)
                baseline = _run_null_baseline(args.baseline_trials, t, a, N, rng)
                metadata = _quantinuum_failure_metadata(
                    target=args.target,
                    run_status="aborted_predicted_memory_limit",
                    failure_reason=guard_reason,
                    circuit_qubits=None,
                    oracle_memory_estimate=oracle_memory_estimate,
                )
                row = _result_row(
                    N=N, a=a, t=t, n_work=n_work, shots=args.shots,
                    counts={}, metadata=metadata, post=None, baseline=baseline,
                )
                row = _attach_quantinuum_metadata(row, metadata)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                continue

            try:
                circuit = build_qpe_order_finding_circuit(N=N, a=a, t=t, n_work=n_work, method=args.method)
            except Exception as exc:
                run_status = "build_failed_memory" if _is_memory_failure(exc) else "build_failed"
                LOGGER.warning("Stopping N=%d with a=%d during circuit build: %s", N, a, exc)
                baseline = _run_null_baseline(args.baseline_trials, t, a, N, rng)
                metadata = _quantinuum_failure_metadata(
                    target=args.target,
                    run_status=run_status,
                    failure_reason=str(exc),
                    circuit_qubits=None,
                    oracle_memory_estimate=oracle_memory_estimate,
                )
                row = _result_row(
                    N=N, a=a, t=t, n_work=n_work, shots=args.shots,
                    counts={}, metadata=metadata, post=None, baseline=baseline,
                )
                row = _attach_quantinuum_metadata(row, metadata)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                continue

            postbuild_snapshot = _memory_snapshot()
            _log_memory_checkpoint(
                stage="post-build",
                N=N,
                a=a,
                snapshot=postbuild_snapshot,
                estimate=oracle_memory_estimate,
            )

            postbuild_guard_reason = _memory_guard_reason(
                snapshot=postbuild_snapshot,
                estimate=oracle_memory_estimate,
                budget_fraction=args.memory_budget_fraction,
            )
            if postbuild_guard_reason is not None:
                LOGGER.warning("Aborting N=%d with a=%d after circuit build: %s", N, a, postbuild_guard_reason)
                baseline = _run_null_baseline(args.baseline_trials, t, a, N, rng)
                metadata = _quantinuum_failure_metadata(
                    target=args.target,
                    run_status="aborted_post_build_memory_limit",
                    failure_reason=postbuild_guard_reason,
                    circuit_qubits=circuit.num_qubits,
                    oracle_memory_estimate=oracle_memory_estimate,
                )
                row = _result_row(
                    N=N, a=a, t=t, n_work=n_work, shots=args.shots,
                    counts={}, metadata=metadata, post=None, baseline=baseline,
                )
                row = _attach_quantinuum_metadata(row, metadata)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                continue

            baseline = _run_null_baseline(args.baseline_trials, t, a, N, rng)

            try:
                counts, metadata = run_on_quantinuum_nexus(
                    qc=circuit,
                    device_name=args.target,
                    project_name=args.project_name,
                    shots=args.shots,
                    optimisation_level=args.optimisation_level,
                    login_mode=args.login,
                    simulator=args.simulator,
                    noisy_simulation=args.noisy_simulation,
                    target_2qb_gate=args.target_2qb_gate,
                    max_cost=args.max_cost,
                    user_group=args.user_group,
                    wait_timeout_seconds=args.wait_timeout_seconds,
                    estimate_cost=not args.skip_cost_estimate,
                    decompose_qiskit=not args.skip_qiskit_decompose,
                )
            except Exception as exc:
                LOGGER.warning("Stopping N=%d with a=%d during Quantinuum execution: %s", N, a, exc)
                metadata = _quantinuum_failure_metadata(
                    target=args.target,
                    run_status="quantinuum_failed",
                    failure_reason=str(exc),
                    circuit_qubits=circuit.num_qubits,
                    oracle_memory_estimate=oracle_memory_estimate,
                )
                row = _result_row(
                    N=N, a=a, t=t, n_work=n_work, shots=args.shots,
                    counts={}, metadata=metadata, post=None, baseline=baseline,
                )
                row = _attach_quantinuum_metadata(row, metadata)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                continue

            metadata.update(oracle_memory_estimate)
            post = shor_postprocess_counts(counts=counts, t=t, a=a, N=N, top_k=args.top_k)
            _log_run_outcome(
                run_label="Quantinuum Helios",
                N=N,
                a=a,
                counts=counts,
                metadata=metadata,
                post=post,
            )
            row = _result_row(
                N=N, a=a, t=t, n_work=n_work, shots=args.shots,
                counts=counts, metadata=metadata, post=post, baseline=baseline,
            )
            row = _attach_quantinuum_metadata(row, metadata)
            handle.write(json.dumps(row) + "\n")
            handle.flush()

    print(f"Wrote Quantinuum Helios sweep results to {output_path}")
    print(
        "Analyze with: "
        f"python quantinuum_helios/analyze_helios_results.py --input {output_path}"
    )


if __name__ == "__main__":
    main()
