from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiskit import transpile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from shor.qpe import build_qpe_order_finding_circuit

from quantinuum_runtime import (
    DEFAULT_HELIOS_DEVICE,
    DEFAULT_NEXUS_PROJECT,
    _compiled_circuit_metrics,
    _qir_text_to_helios_bitcode,
)


def _login(qnx: Any, mode: str) -> None:
    if mode == "none":
        return
    if mode == "browser":
        qnx.login()
        return
    if mode == "credentials":
        qnx.login_with_credentials()
        return
    raise ValueError(f"Unknown login mode: {mode}")


def _ref_id(ref: Any) -> str:
    for attr in ("id", "uuid", "name"):
        value = getattr(ref, attr, None)
        if value is not None:
            return str(value)
    return repr(ref)


def _status_row(status: Any) -> dict[str, Any]:
    if status is None:
        return {}
    return {
        "status": str(getattr(status, "status", None)),
        "message": getattr(status, "message", None),
        "cost": getattr(status, "cost", None),
        "queued_time": str(getattr(status, "queued_time", None)),
        "submitted_time": str(getattr(status, "submitted_time", None)),
        "running_time": str(getattr(status, "running_time", None)),
        "completed_time": str(getattr(status, "completed_time", None)),
        "error_time": str(getattr(status, "error_time", None)),
        "error_detail": getattr(status, "error_detail", None),
    }


def _write_submission(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a Helios QIR job and exit once Nexus reports it running.")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--base", type=int, required=True)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--t", type=int, help="Counting register size. Defaults to 2 * ceil(log2(N)).")
    parser.add_argument("--method", default="auto", choices=["auto", "standard", "permutation", "semi_compiled"])
    parser.add_argument("--target", default=DEFAULT_HELIOS_DEVICE)
    parser.add_argument("--project-name", default=DEFAULT_NEXUS_PROJECT)
    parser.add_argument("--max-cost", type=float)
    parser.add_argument("--login", default="none", choices=["none", "browser", "credentials"])
    parser.add_argument("--poll-until-running-seconds", type=int, default=600)
    parser.add_argument("--poll-interval-seconds", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-result-output", type=Path)
    args = parser.parse_args()

    if not (1 < args.base < args.N) or math.gcd(args.base, args.N) != 1:
        raise SystemExit(f"Base a={args.base} must satisfy 1 < a < N and gcd(a, N) = 1 for N={args.N}.")

    import qnexus as qnx
    from pytket.extensions.qiskit import qiskit_to_tk
    from pytket.passes import FlattenRelabelRegistersPass
    from pytket.qir.conversion.api import QIRFormat, QIRProfile, pytket_to_qir

    started_at = time.perf_counter()
    n_work = math.ceil(math.log2(args.N))
    t = args.t if args.t is not None else 2 * n_work
    suffix = uuid.uuid4().hex[:10]

    print(f"Building QPE circuit N={args.N} a={args.base} t={t} n_work={n_work}")
    circuit = build_qpe_order_finding_circuit(N=args.N, a=args.base, t=t, n_work=n_work, method=args.method)
    original_depth = circuit.depth()
    original_instruction_count = len(circuit.data)

    decomposed = transpile(
        circuit,
        basis_gates=["rx", "ry", "rz", "cx"],
        optimization_level=1,
    )
    tket_circuit = qiskit_to_tk(decomposed)
    FlattenRelabelRegistersPass("q").apply(tket_circuit)
    metrics = _compiled_circuit_metrics(tket_circuit)

    qir_text = pytket_to_qir(
        tket_circuit,
        name=f"shor_qpe_N{args.N}_a{args.base}_{suffix}",
        qir_format=QIRFormat.STRING,
        int_type=64,
        profile=QIRProfile.ADAPTIVE,
        cut_pytket_register=True,
    )
    qir_bitcode = _qir_text_to_helios_bitcode(qir_text)

    _login(qnx, args.login)
    project = qnx.projects.get_or_create(args.project_name)
    qnx.context.set_active_project(project)

    qir_ref = qnx.qir.upload(
        qir=qir_bitcode,
        name=f"qir-shor-qpe-N{args.N}-a{args.base}-{suffix}",
        project=project,
    )

    helios_config_cls = qnx.models.HeliosConfig
    config_kwargs: dict[str, Any] = {"system_name": args.target}
    if args.target.upper().endswith("E"):
        config_kwargs["emulator_config"] = qnx.models.HeliosEmulatorConfig(n_qubits=circuit.num_qubits)
    if args.max_cost is not None:
        config_kwargs["max_cost"] = float(args.max_cost)
    backend_config = helios_config_cls(**config_kwargs)

    execute_kwargs: dict[str, Any] = {}
    if args.max_cost is not None:
        execute_kwargs["max_cost"] = [float(args.max_cost)]

    job_name = f"execute-shor-qpe-N{args.N}-a{args.base}-{suffix}"
    execute_job_ref = qnx.start_execute_job(
        programs=[qir_ref],
        backend_config=backend_config,
        n_shots=[args.shots],
        name=job_name,
        **execute_kwargs,
    )

    row: dict[str, Any] = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "N": args.N,
        "a": args.base,
        "t": t,
        "n_work": n_work,
        "shots": args.shots,
        "target": args.target,
        "project_name": args.project_name,
        "max_cost_hqc": args.max_cost,
        "job_name": job_name,
        "execute_job_id": _ref_id(execute_job_ref),
        "qir_ref": _ref_id(qir_ref),
        "expected_result_output": str(args.expected_result_output) if args.expected_result_output else None,
        "original_depth": original_depth,
        "original_instruction_count": original_instruction_count,
        "qiskit_decomposed_depth": decomposed.depth(),
        "qiskit_decomposed_instruction_count": len(decomposed.data),
        "num_qubits": circuit.num_qubits,
        "submit_runtime_sec": time.perf_counter() - started_at,
        **metrics,
    }

    deadline = time.monotonic() + args.poll_until_running_seconds
    terminal_fragments = ("COMPLETED", "ERROR", "CANCELLED", "TERMINATED")
    last_status = None
    while True:
        last_status = qnx.jobs.status(execute_job_ref)
        status_text = str(getattr(last_status, "status", ""))
        row["last_status"] = _status_row(last_status)
        _write_submission(args.output, row)
        print(f"{datetime.now().isoformat()} | {status_text} | cost={getattr(last_status, 'cost', None)}")

        if "RUNNING" in status_text or any(fragment in status_text for fragment in terminal_fragments):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(args.poll_interval_seconds)

    print(json.dumps(row, indent=2, sort_keys=True))
    print(f"Wrote submission metadata to {args.output}")


if __name__ == "__main__":
    main()
