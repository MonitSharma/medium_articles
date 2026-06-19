from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from datetime import datetime
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

from quantinuum_runtime import DEFAULT_HELIOS_DEVICE, DEFAULT_NEXUS_PROJECT, _qir_text_to_helios_bitcode


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


def _compiled_circuit_metrics(tket_circuit: Any) -> dict[str, int | None]:
    depth = tket_circuit.depth()
    commands = tket_circuit.get_commands()
    return {
        "depth": int(depth) if depth is not None else None,
        "two_qubit_gates": sum(1 for command in commands if len(getattr(command, "qubits", ())) == 2),
        "compiled_commands": len(commands),
    }


def _default_output_path(N: int, shots: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_ROOT / "data" / "cost_estimates" / f"helios_qir_cost_N{N}_{shots}shots_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Helios-1 QIR HQC cost without executing hardware.")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--a", type=int, required=True)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--t", type=int, help="Counting register size. Defaults to 2 * ceil(log2(N)).")
    parser.add_argument("--method", default="auto", choices=["auto", "standard", "permutation", "semi_compiled"])
    parser.add_argument("--project-name", default=DEFAULT_NEXUS_PROJECT)
    parser.add_argument("--device", default=DEFAULT_HELIOS_DEVICE)
    parser.add_argument("--login", default="none", choices=["none", "browser", "credentials"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import qnexus as qnx
    from pytket.extensions.qiskit import qiskit_to_tk
    from pytket.passes import FlattenRelabelRegistersPass
    from pytket.qir.conversion.api import QIRFormat, QIRProfile, pytket_to_qir

    started_at = time.perf_counter()
    n_work = math.ceil(math.log2(args.N))
    t = args.t if args.t is not None else 2 * n_work

    _login(qnx, args.login)
    project = qnx.projects.get_or_create(args.project_name)
    qnx.context.set_active_project(project)

    circuit = build_qpe_order_finding_circuit(N=args.N, a=args.a, t=t, n_work=n_work, method=args.method)
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

    suffix = uuid.uuid4().hex[:10]
    qir_text = pytket_to_qir(
        tket_circuit,
        name=f"shor_qpe_N{args.N}_a{args.a}_{suffix}",
        qir_format=QIRFormat.STRING,
        int_type=64,
        profile=QIRProfile.ADAPTIVE,
        cut_pytket_register=True,
    )
    qir_ref = qnx.qir.upload(
        qir=_qir_text_to_helios_bitcode(qir_text),
        name=f"qir-cost-shor-qpe-N{args.N}-a{args.a}-{suffix}",
        project=project,
    )

    row = {
        "N": args.N,
        "a": args.a,
        "t": t,
        "n_work": n_work,
        "shots": args.shots,
        "device": args.device,
        "project_name": args.project_name,
        "program_format": "qir",
        "qir_profile": "ADAPTIVE",
        "qir_ref": str(getattr(qir_ref, "id", qir_ref)),
        "original_depth": original_depth,
        "original_instruction_count": original_instruction_count,
        "qiskit_decomposed_depth": decomposed.depth(),
        "qiskit_decomposed_instruction_count": len(decomposed.data),
        "runtime_sec": time.perf_counter() - started_at,
        **metrics,
    }

    costing_job_ref = None
    try:
        costing_job_ref = qnx.start_execute_job(
            programs=[qir_ref],
            n_shots=[args.shots],
            backend_config=qnx.QuantinuumConfig(device_name=f"{args.device}SC"),
            project=project,
            name=f"QIR cost estimate N{args.N} a{args.a} {args.shots} shots",
        )
        row["costing_job_ref"] = str(getattr(costing_job_ref, "id", costing_job_ref))
        qnx.jobs.wait_for(costing_job_ref)
        cost_hqc, confidence = qnx.jobs.cost_confidence(costing_job_ref)[0]
        row["cost_hqc"] = float(cost_hqc)
        row["cost_confidence"] = float(confidence)
        row["run_status"] = "cost_estimated"
    except Exception as exc:
        row["costing_job_ref"] = str(getattr(costing_job_ref, "id", costing_job_ref)) if costing_job_ref else None
        row["run_status"] = "cost_estimate_failed"
        row["failure_type"] = exc.__class__.__name__
        row["failure_reason"] = str(exc)

    output_path = args.output or _default_output_path(args.N, args.shots)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2, sort_keys=True))
    print(f"Wrote cost estimate to {output_path}")


if __name__ == "__main__":
    main()
