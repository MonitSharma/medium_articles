from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shor.modexp import estimate_permutation_unitary_memory_mb
from shor.postprocess import shor_postprocess_counts
from shor.qpe import build_qpe_order_finding_circuit
from shor.runtime import (
    DEFAULT_IBM_CREDENTIALS_PATH,
    get_available_memory_mb,
    get_current_memory_mb,
    get_peak_memory_mb,
    run_on_ibm,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_NS = [15, 21, 35, 91, 143, 221, 247, 299, 323]
OPTIONAL_10BIT_NS = [899, 1007]


def _backend_name(backend: Any) -> str:
    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        return str(name_attr())
    if name_attr is not None:
        return str(name_attr)
    return backend.__class__.__name__


def _count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    return sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)


def _build_pass_manager(backend: Any, opt_level: int):
    try:
        return generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    except TypeError:
        return generate_preset_pass_manager(optimization_level=opt_level, target=backend.target)


def _pick_coprime_base(N: int, rng: random.Random) -> int:
    while True:
        candidate = rng.randint(2, N - 2)
        if math.gcd(candidate, N) == 1:
            return candidate


def _parse_n_values(raw: str | None, include_10bit: bool) -> list[int]:
    if raw:
        return [int(token.strip()) for token in raw.split(",") if token.strip()]

    values = list(DEFAULT_NS)
    if include_10bit:
        values.extend(OPTIONAL_10BIT_NS)
    return values


def _default_output_path(use_hardware: bool) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_kind = "hardware" if use_hardware else "simulator"
    return Path("data/raw") / f"results_{run_kind}_{timestamp}.jsonl"


def _latest_output_path(use_hardware: bool) -> Path | None:
    run_kind = "hardware" if use_hardware else "simulator"
    candidates = sorted(
        Path("data/raw").glob(f"results_{run_kind}_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_output_plan(
    output_arg: str | None,
    append: bool,
    use_hardware: bool,
) -> tuple[Path, str, Path | None]:
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

    latest_path = _latest_output_path(use_hardware=use_hardware)
    if latest_path is not None:
        return latest_path, "a", latest_path

    output_path = _default_output_path(use_hardware=use_hardware)
    return output_path, "w", None


def _resume_mode_backend_matches(
    backend: str | None,
    *,
    use_hardware: bool,
    noise_backend_name: str | None,
) -> bool:
    if not backend:
        return False

    if use_hardware:
        return not backend.startswith("aer")

    if backend == "aer-ideal:matrix_product_state":
        return True

    if noise_backend_name:
        return backend == f"aer-noisy:{noise_backend_name}:matrix_product_state"

    return False


def _load_completed_numbers(
    resume_path: Path | None,
    *,
    use_hardware: bool,
    noise_backend_name: str | None,
) -> set[int]:
    if resume_path is None or not resume_path.exists():
        return set()

    completed_by_n: dict[int, set[str]] = {}

    with resume_path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.replace(b"\x00", b"").strip()
            if not raw_line:
                continue

            try:
                row = json.loads(raw_line.decode("utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue

            if row.get("run_status") != "completed":
                continue

            n_value = row.get("N")
            if n_value is None:
                continue

            try:
                n_int = int(n_value)
            except (TypeError, ValueError):
                continue

            backend = row.get("backend")
            if not _resume_mode_backend_matches(
                backend,
                use_hardware=use_hardware,
                noise_backend_name=noise_backend_name,
            ):
                continue

            completed_by_n.setdefault(n_int, set()).add(str(backend))

    completed_numbers: set[int] = set()
    for n_value, backends in completed_by_n.items():
        if use_hardware:
            if backends:
                completed_numbers.add(n_value)
            continue

        if "aer-ideal:matrix_product_state" not in backends:
            continue

        if noise_backend_name and f"aer-noisy:{noise_backend_name}:matrix_product_state" not in backends:
            continue

        completed_numbers.add(n_value)

    return completed_numbers


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Keep this script's logs at the requested level, but suppress Qiskit's
    # internal pass-manager timing noise unless it is a real warning/error.
    LOGGER.setLevel(getattr(logging, level_name))
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
    logging.getLogger("qiskit.transpiler").setLevel(logging.WARNING)
    logging.getLogger("qiskit_aer").setLevel(logging.WARNING)
    logging.getLogger("qiskit_ibm_runtime").setLevel(logging.WARNING)


def _log_run_outcome(
    run_label: str,
    N: int,
    a: int,
    counts: dict[str, int],
    metadata: dict[str, Any],
    post: dict[str, Any] | None,
) -> None:
    shots = sum(counts.values())
    unique_frac = (len(counts) / shots) if shots else 0.0
    backend = metadata.get("backend", "unknown")
    current_memory_mb = _format_mb(metadata.get("current_memory_mb"))
    peak_memory_mb = _format_mb(metadata.get("peak_memory_mb"))

    if post is None:
        LOGGER.info(
            "%s result for N=%d with a=%d on %s | strict_success=False unique_frac=%.4f top_outcome=%s current_mem=%s peak_mem=%s",
            run_label,
            N,
            a,
            backend,
            unique_frac,
            max(counts, key=counts.get) if counts else None,
            current_memory_mb,
            peak_memory_mb,
        )
        return

    LOGGER.info(
        "%s result for N=%d with a=%d on %s | strict_success=True factors=(%d,%d) r=%d bit_order=%s y=%d top_outcome=%s unique_frac=%.4f current_mem=%s peak_mem=%s",
        run_label,
        N,
        a,
        backend,
        post["p"],
        post["q"],
        post["r_min"],
        post["bit_order"],
        post["y"],
        post["raw_bitstring"],
        unique_frac,
        current_memory_mb,
        peak_memory_mb,
    )


def _memory_snapshot() -> dict[str, float | None]:
    return {
        "current_memory_mb": get_current_memory_mb(),
        "peak_memory_mb": get_peak_memory_mb(),
        "available_memory_mb": get_available_memory_mb(),
    }


def _format_mb(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.1f} MiB"


def _estimate_oracle_memory(N: int, a: int, t: int, n_work: int, method: str) -> dict[str, Any]:
    normalized = method.lower()
    if normalized == "standard":
        normalized = "auto"

    if normalized == "auto" and N == 15 and n_work == 4:
        return {
            "oracle_model": "mod15-specialized",
            "unique_oracles": 0,
            "per_oracle_matrix_mb": 0.0,
            "per_controlled_matrix_mb": 0.0,
            "estimated_cached_oracle_mb": 0.0,
            "estimated_cached_controlled_mb": 0.0,
        }

    if normalized not in {"auto", "permutation", "semi_compiled"}:
        return {
            "oracle_model": normalized,
            "unique_oracles": None,
            "per_oracle_matrix_mb": None,
            "per_controlled_matrix_mb": None,
            "estimated_cached_oracle_mb": None,
            "estimated_cached_controlled_mb": None,
        }

    unique_oracles = len({pow(a, 1 << exponent_index, N) for exponent_index in range(t)})
    per_oracle_matrix_mb = estimate_permutation_unitary_memory_mb(n_work)
    per_controlled_matrix_mb = estimate_permutation_unitary_memory_mb(n_work + 1)

    return {
        "oracle_model": "exact-permutation-unitary",
        "unique_oracles": unique_oracles,
        "per_oracle_matrix_mb": per_oracle_matrix_mb,
        "per_controlled_matrix_mb": per_controlled_matrix_mb,
        "estimated_cached_oracle_mb": unique_oracles * per_oracle_matrix_mb,
        "estimated_cached_controlled_mb": unique_oracles * per_controlled_matrix_mb,
    }


def _memory_guard_reason(
    snapshot: dict[str, float | None],
    estimate: dict[str, Any],
    budget_fraction: float,
) -> str | None:
    available_memory_mb = snapshot.get("available_memory_mb")
    predicted_memory_mb = estimate.get("estimated_cached_controlled_mb")

    if available_memory_mb is None or predicted_memory_mb in {None, 0}:
        return None

    threshold_mb = available_memory_mb * budget_fraction
    if predicted_memory_mb > threshold_mb:
        return (
            f"Predicted dense controlled-oracle cache footprint {_format_mb(predicted_memory_mb)} "
            f"exceeds the configured {budget_fraction:.0%} memory budget "
            f"({_format_mb(threshold_mb)} of {_format_mb(available_memory_mb)} available)."
        )

    return None


def _log_memory_checkpoint(
    stage: str,
    N: int,
    a: int,
    snapshot: dict[str, float | None],
    estimate: dict[str, Any],
    always_log: bool = False,
) -> None:
    predicted_memory_mb = estimate.get("estimated_cached_controlled_mb")
    if not always_log and N.bit_length() < 8 and (predicted_memory_mb or 0.0) < 128.0:
        return

    LOGGER.info(
        "Memory checkpoint (%s) for N=%d with a=%d | current=%s peak=%s available=%s predicted_controlled_cache=%s",
        stage,
        N,
        a,
        _format_mb(snapshot.get("current_memory_mb")),
        _format_mb(snapshot.get("peak_memory_mb")),
        _format_mb(snapshot.get("available_memory_mb")),
        _format_mb(predicted_memory_mb),
    )


def _is_memory_failure(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True

    name = exc.__class__.__name__.lower()
    if "memory" in name:
        return True

    message = str(exc).lower()
    return "unable to allocate" in message or "out of memory" in message


def _is_hardware_time_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    phrases = (
        "qpu time",
        "usage limit",
        "no usable ibm quantum credential",
        "no more time available",
        "time-related runtime error",
    )
    return any(phrase in message for phrase in phrases)


def _run_null_baseline(
    trials: int,
    t: int,
    a: int,
    N: int,
    rng: random.Random,
) -> dict[str, Any]:
    # Legacy histogram/top-1 baseline kept for debugging only.
    # This is intentionally distinct from the canonical strict null baseline
    # used in analysis (strict_postprocess_y on uniformly random y values).
    if trials <= 0:
        return {
            "top1_histogram_baseline_trials": 0,
            "top1_histogram_baseline_false_positives": 0,
            "top1_histogram_baseline_fp_rate": 0.0,
        }

    false_positives = 0
    examples: list[dict[str, Any]] = []

    for _ in range(trials):
        bits = "".join(rng.choice("01") for _ in range(t))
        post = shor_postprocess_counts({bits: 1}, t=t, a=a, N=N, top_k=1)
        if post is not None:
            false_positives += 1
            if len(examples) < 3:
                examples.append(
                    {
                        "bits": bits,
                        "bit_order": post["bit_order"],
                        "p": post["p"],
                        "q": post["q"],
                        "r_min": post["r_min"],
                    }
                )

    return {
        "top1_histogram_baseline_trials": trials,
        "top1_histogram_baseline_false_positives": false_positives,
        "top1_histogram_baseline_fp_rate": false_positives / trials,
        "top1_histogram_baseline_examples": examples,
    }


def _run_on_aer(
    qc: QuantumCircuit,
    shots: int,
    seed: int,
    opt_level: int,
    noise_backend_name: str | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("qiskit-aer is required for simulator sweeps") from exc

    if noise_backend_name:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "qiskit-ibm-runtime is required to build a noise model from a real backend"
            ) from exc

        service = QiskitRuntimeService()
        noise_backend = service.backend(noise_backend_name)
        backend = AerSimulator.from_backend(noise_backend)
        backend.set_options(method="matrix_product_state")
        backend_label = f"aer-noisy:{_backend_name(noise_backend)}:matrix_product_state"
    else:
        backend = AerSimulator(method="matrix_product_state")
        backend_label = "aer-ideal:matrix_product_state"

    memory_before_transpile = _memory_snapshot()
    pass_manager = _build_pass_manager(backend=backend, opt_level=opt_level)
    isa_circuit = pass_manager.run(qc)
    memory_after_transpile = _memory_snapshot()
    depth = isa_circuit.depth()
    two_qubit_gates = _count_two_qubit_gates(isa_circuit)

    LOGGER.info(
        "Running simulator backend %s | qubits=%d depth=%d 2q=%d shots=%d seed=%d",
        backend_label,
        isa_circuit.num_qubits,
        depth,
        two_qubit_gates,
        shots,
        seed,
    )

    started_at = time.perf_counter()
    job = backend.run(isa_circuit, shots=shots, seed_simulator=seed)
    result = job.result()
    runtime_sec = time.perf_counter() - started_at
    memory_after_run = _memory_snapshot()

    counts = dict(result.get_counts())
    metadata = {
        "backend": backend_label,
        "job_id": None,
        "depth": depth,
        "two_qubit_gates": two_qubit_gates,
        "num_qubits": isa_circuit.num_qubits,
        "runtime_sec": runtime_sec,
        "current_memory_mb": memory_after_run.get("current_memory_mb"),
        "available_memory_mb": memory_after_run.get("available_memory_mb"),
        "peak_memory_mb": memory_after_run.get("peak_memory_mb"),
        "memory_before_transpile_mb": memory_before_transpile.get("current_memory_mb"),
        "memory_after_transpile_mb": memory_after_transpile.get("current_memory_mb"),
    }
    return counts, metadata


def _result_row(
    *,
    N: int,
    a: int,
    t: int,
    n_work: int,
    shots: int,
    counts: dict[str, int],
    metadata: dict[str, Any],
    post: dict[str, Any] | None,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "N": N,
        "bits": N.bit_length(),
        "a": a,
        "t": t,
        "n_work": n_work,
        "backend": metadata.get("backend"),
        "depth": metadata.get("depth"),
        "two_qubit_gates": metadata.get("two_qubit_gates"),
        "shots": shots,
        "unique_frac": (len(counts) / shots) if shots else 0.0,
        "strict_success": post is not None,
        "p": post["p"] if post else None,
        "q": post["q"] if post else None,
        "r_min": post["r_min"] if post else None,
        "runtime_sec": metadata.get("runtime_sec"),
        "run_status": metadata.get("run_status", "completed"),
        "failure_reason": metadata.get("failure_reason"),
        "current_memory_mb": metadata.get("current_memory_mb"),
        "available_memory_mb": metadata.get("available_memory_mb"),
        "peak_memory_mb": metadata.get("peak_memory_mb"),
        "memory_before_transpile_mb": metadata.get("memory_before_transpile_mb"),
        "memory_after_transpile_mb": metadata.get("memory_after_transpile_mb"),
        "oracle_model": metadata.get("oracle_model"),
        "unique_oracles": metadata.get("unique_oracles"),
        "per_oracle_matrix_mb": metadata.get("per_oracle_matrix_mb"),
        "per_controlled_matrix_mb": metadata.get("per_controlled_matrix_mb"),
        "estimated_cached_oracle_mb": metadata.get("estimated_cached_oracle_mb"),
        "estimated_cached_controlled_mb": metadata.get("estimated_cached_controlled_mb"),
        "credential_label": metadata.get("credential_label"),
        "qpu_time_remaining_seconds": metadata.get("qpu_time_remaining_seconds"),
        "active_instance": metadata.get("active_instance"),
        "job_id": metadata.get("job_id"),
        "num_qubits": metadata.get("num_qubits"),
        "counts": counts or {},
        "raw_bitstring": post["raw_bitstring"] if post else None,
        "tested_bitstring": post["tested_bitstring"] if post else None,
        "bit_order": post["bit_order"] if post else None,
        "candidate_y": post["y"] if post else None,
    }
    row.update(baseline)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Shor order-finding sweep on simulator and/or hardware.")
    parser.add_argument("--n-values", help="Comma-separated semiprimes. Defaults to the built-in sweep list.")
    parser.add_argument(
        "--include-10bit",
        action="store_true",
        help="Include two optional 10-bit semiprimes (899 and 1007) in the default sweep.",
    )
    parser.add_argument("--shots", type=int, default=1024, help="Number of shots per run.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for base selection and simulator shots.")
    parser.add_argument(
        "--t",
        type=int,
        help="Override the counting-register size. Default is 2 * ceil(log2(N)).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many of the most likely bitstrings to scan in strict post-processing.",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Transpiler optimization level for simulator and hardware runs.",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "standard", "permutation", "semi_compiled"],
        help="Oracle construction method for modular multiplication.",
    )
    parser.add_argument(
        "--baseline-trials",
        type=int,
        default=512,
        help="Number of random t-bit strings used for the null baseline per N.",
    )
    parser.add_argument(
        "--noise-backend",
        help="Optional IBM backend name used to derive an Aer noise model (for example ibm_sherbrooke).",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Run each circuit on IBM Quantum hardware via SamplerV2 instead of the simulator.",
    )
    parser.add_argument(
        "--backend-name",
        default="auto",
        help="Hardware backend name, or 'auto' to use least_busy(...).",
    )
    parser.add_argument(
        "--disable-dd",
        action="store_true",
        help="Disable dynamical decoupling for Runtime hardware jobs.",
    )
    parser.add_argument(
        "--disable-twirling",
        action="store_true",
        help="Disable gate/measurement twirling for Runtime hardware jobs.",
    )
    parser.add_argument(
        "--credentials-path",
        default=str(DEFAULT_IBM_CREDENTIALS_PATH),
        help="Path to the IBM credential JSON file containing token/CRN entries for hardware rotation.",
    )
    parser.add_argument(
        "--min-qpu-seconds",
        type=float,
        default=60.0,
        help="Minimum remaining QPU seconds required before a hardware credential is allowed to run.",
    )
    parser.add_argument(
        "--output",
        help="JSONL output path for all run records. Defaults to a new timestamped file in data/raw/.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the JSONL file instead of overwriting it.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--memory-budget-fraction",
        type=float,
        default=0.90,
        help=(
            "Abort a run before or after circuit construction if the predicted dense controlled-oracle "
            "cache footprint exceeds this fraction of currently available RAM."
        ),
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)
    if args.hardware and args.noise_backend:
        LOGGER.warning("--noise-backend is ignored when --hardware is set because --hardware now runs hardware only.")

    rng = random.Random(args.seed)
    n_values = _parse_n_values(raw=args.n_values, include_10bit=args.include_10bit)
    output_path, file_mode, resume_path = _resolve_output_plan(
        output_arg=args.output,
        append=args.append,
        use_hardware=args.hardware,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing results to %s", output_path)
    if resume_path is not None:
        LOGGER.info("Resuming from existing results file %s", resume_path)

    completed_numbers = _load_completed_numbers(
        resume_path=resume_path,
        use_hardware=args.hardware,
        noise_backend_name=None if args.hardware else args.noise_backend,
    )
    if completed_numbers:
        LOGGER.info("Skipping already completed N values from resume file: %s", sorted(completed_numbers))

    with output_path.open(file_mode, encoding="utf-8") as handle:
        for N in n_values:
            if N in completed_numbers:
                LOGGER.info("Skipping N=%d because it is already completed in %s", N, resume_path or output_path)
                continue

            n_work = math.ceil(math.log2(N))
            t = args.t if args.t is not None else 2 * n_work
            a = _pick_coprime_base(N=N, rng=rng)
            oracle_memory_estimate = _estimate_oracle_memory(N=N, a=a, t=t, n_work=n_work, method=args.method)
            prebuild_snapshot = _memory_snapshot()

            LOGGER.info("Building QPE circuit for N=%d (bits=%d) with a=%d, t=%d, n_work=%d", N, N.bit_length(), a, t, n_work)
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
                baseline = _run_null_baseline(trials=args.baseline_trials, t=t, a=a, N=N, rng=rng)
                failure_metadata = {
                    "backend": "not_run",
                    "job_id": None,
                    "depth": None,
                    "two_qubit_gates": None,
                    "num_qubits": None,
                    "runtime_sec": None,
                    "run_status": "aborted_predicted_memory_limit",
                    "failure_reason": guard_reason,
                    "current_memory_mb": prebuild_snapshot.get("current_memory_mb"),
                    "available_memory_mb": prebuild_snapshot.get("available_memory_mb"),
                    "peak_memory_mb": prebuild_snapshot.get("peak_memory_mb"),
                    "memory_before_transpile_mb": None,
                    "memory_after_transpile_mb": None,
                }
                failure_metadata.update(oracle_memory_estimate)
                row = _result_row(
                    N=N,
                    a=a,
                    t=t,
                    n_work=n_work,
                    shots=args.shots,
                    counts={},
                    metadata=failure_metadata,
                    post=None,
                    baseline=baseline,
                )
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                continue

            try:
                circuit = build_qpe_order_finding_circuit(N=N, a=a, t=t, n_work=n_work, method=args.method)
            except Exception as exc:
                run_status = "build_failed_memory" if _is_memory_failure(exc) else "build_failed"
                LOGGER.warning("Stopping N=%d with a=%d during circuit build: %s", N, a, exc)
                baseline = _run_null_baseline(trials=args.baseline_trials, t=t, a=a, N=N, rng=rng)
                failure_metadata = {
                    "backend": "not_run",
                    "job_id": None,
                    "depth": None,
                    "two_qubit_gates": None,
                    "num_qubits": None,
                    "runtime_sec": None,
                    "run_status": run_status,
                    "failure_reason": str(exc),
                    **_memory_snapshot(),
                    "memory_before_transpile_mb": None,
                    "memory_after_transpile_mb": None,
                }
                failure_metadata.update(oracle_memory_estimate)
                row = _result_row(
                    N=N,
                    a=a,
                    t=t,
                    n_work=n_work,
                    shots=args.shots,
                    counts={},
                    metadata=failure_metadata,
                    post=None,
                    baseline=baseline,
                )
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
                baseline = _run_null_baseline(trials=args.baseline_trials, t=t, a=a, N=N, rng=rng)
                failure_metadata = {
                    "backend": "not_run",
                    "job_id": None,
                    "depth": None,
                    "two_qubit_gates": None,
                    "num_qubits": circuit.num_qubits,
                    "runtime_sec": None,
                    "run_status": "aborted_post_build_memory_limit",
                    "failure_reason": postbuild_guard_reason,
                    "current_memory_mb": postbuild_snapshot.get("current_memory_mb"),
                    "available_memory_mb": postbuild_snapshot.get("available_memory_mb"),
                    "peak_memory_mb": postbuild_snapshot.get("peak_memory_mb"),
                    "memory_before_transpile_mb": None,
                    "memory_after_transpile_mb": None,
                }
                failure_metadata.update(oracle_memory_estimate)
                row = _result_row(
                    N=N,
                    a=a,
                    t=t,
                    n_work=n_work,
                    shots=args.shots,
                    counts={},
                    metadata=failure_metadata,
                    post=None,
                    baseline=baseline,
                )
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                continue

            baseline = _run_null_baseline(trials=args.baseline_trials, t=t, a=a, N=N, rng=rng)

            if args.hardware:
                try:
                    hardware_counts, hardware_metadata = run_on_ibm(
                        qc=circuit,
                        backend_name=args.backend_name,
                        shots=args.shots,
                        opt_level=args.opt_level,
                        seed=args.seed,
                        use_dd=not args.disable_dd,
                        use_twirling=not args.disable_twirling,
                        credentials_path=args.credentials_path,
                        min_qpu_seconds=args.min_qpu_seconds,
                    )
                except Exception as exc:
                    run_status = "hardware_failed_quota" if _is_hardware_time_error(exc) else "hardware_failed"
                    LOGGER.warning("Stopping N=%d with a=%d during hardware execution: %s", N, a, exc)
                    failure_metadata = {
                        "backend": "hardware_not_run",
                        "job_id": None,
                        "depth": None,
                        "two_qubit_gates": None,
                        "num_qubits": circuit.num_qubits,
                        "runtime_sec": None,
                        "run_status": run_status,
                        "failure_reason": str(exc),
                        **_memory_snapshot(),
                        "memory_before_transpile_mb": None,
                        "memory_after_transpile_mb": None,
                    }
                    failure_metadata.update(oracle_memory_estimate)
                    hardware_row = _result_row(
                        N=N,
                        a=a,
                        t=t,
                        n_work=n_work,
                        shots=args.shots,
                        counts={},
                        metadata=failure_metadata,
                        post=None,
                        baseline=baseline,
                    )
                    handle.write(json.dumps(hardware_row) + "\n")
                    handle.flush()
                    continue

                hardware_post = shor_postprocess_counts(
                    counts=hardware_counts,
                    t=t,
                    a=a,
                    N=N,
                    top_k=args.top_k,
                )
                hardware_metadata.update(oracle_memory_estimate)
                _log_run_outcome(
                    run_label="Hardware",
                    N=N,
                    a=a,
                    counts=hardware_counts,
                    metadata=hardware_metadata,
                    post=hardware_post,
                )
                hardware_row = _result_row(
                    N=N,
                    a=a,
                    t=t,
                    n_work=n_work,
                    shots=args.shots,
                    counts=hardware_counts,
                    metadata=hardware_metadata,
                    post=hardware_post,
                    baseline=baseline,
                )
                handle.write(json.dumps(hardware_row) + "\n")
                handle.flush()
                continue

            try:
                counts, metadata = _run_on_aer(
                    qc=circuit,
                    shots=args.shots,
                    seed=args.seed,
                    opt_level=args.opt_level,
                )
            except Exception as exc:
                run_status = "sim_failed_memory" if _is_memory_failure(exc) else "sim_failed"
                LOGGER.warning("Stopping N=%d with a=%d during simulator execution: %s", N, a, exc)
                failure_metadata = {
                    "backend": "aer-ideal:matrix_product_state",
                    "job_id": None,
                    "depth": None,
                    "two_qubit_gates": None,
                    "num_qubits": circuit.num_qubits,
                    "runtime_sec": None,
                    "run_status": run_status,
                    "failure_reason": str(exc),
                    **_memory_snapshot(),
                    "memory_before_transpile_mb": None,
                    "memory_after_transpile_mb": None,
                }
                failure_metadata.update(oracle_memory_estimate)
                row = _result_row(
                    N=N,
                    a=a,
                    t=t,
                    n_work=n_work,
                    shots=args.shots,
                    counts={},
                    metadata=failure_metadata,
                    post=None,
                    baseline=baseline,
                )
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                continue

            metadata.update(oracle_memory_estimate)
            post = shor_postprocess_counts(counts=counts, t=t, a=a, N=N, top_k=args.top_k)
            _log_run_outcome(
                run_label="Simulator",
                N=N,
                a=a,
                counts=counts,
                metadata=metadata,
                post=post,
            )
            row = _result_row(
                N=N,
                a=a,
                t=t,
                n_work=n_work,
                shots=args.shots,
                counts=counts,
                metadata=metadata,
                post=post,
                baseline=baseline,
            )
            handle.write(json.dumps(row) + "\n")
            handle.flush()

            if args.noise_backend:
                noisy_counts, noisy_metadata = _run_on_aer(
                    qc=circuit,
                    shots=args.shots,
                    seed=args.seed,
                    opt_level=args.opt_level,
                    noise_backend_name=args.noise_backend,
                )
                noisy_post = shor_postprocess_counts(
                    counts=noisy_counts,
                    t=t,
                    a=a,
                    N=N,
                    top_k=args.top_k,
                )
                noisy_metadata.update(oracle_memory_estimate)
                _log_run_outcome(
                    run_label="Noisy simulator",
                    N=N,
                    a=a,
                    counts=noisy_counts,
                    metadata=noisy_metadata,
                    post=noisy_post,
                )
                noisy_row = _result_row(
                    N=N,
                    a=a,
                    t=t,
                    n_work=n_work,
                    shots=args.shots,
                    counts=noisy_counts,
                    metadata=noisy_metadata,
                    post=noisy_post,
                    baseline=baseline,
                )
                handle.write(json.dumps(noisy_row) + "\n")
                handle.flush()

    print(f"Wrote sweep results to {output_path}")


if __name__ == "__main__":
    main()
