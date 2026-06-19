from __future__ import annotations

import logging
import math
import time
import uuid
from collections import Counter
from typing import Any

from qiskit import QuantumCircuit, transpile

from shor.runtime import get_available_memory_mb, get_current_memory_mb, get_peak_memory_mb


LOGGER = logging.getLogger(__name__)
DEFAULT_HELIOS_DEVICE = "Helios-1"
DEFAULT_NEXUS_PROJECT = "Sometimes Too Slow for Shor - Helios"


def _ref_identifier(ref: Any) -> str:
    for attr in ("id", "uuid", "name"):
        value = getattr(ref, attr, None)
        if value is not None:
            return str(value)
    return repr(ref)


def _maybe_call(obj: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    return method(*args, **kwargs)


def _login(qnx: Any, mode: str) -> None:
    if mode == "none":
        return
    if mode == "browser":
        qnx.login()
        return
    if mode == "credentials":
        qnx.login_with_credentials()
        return
    raise ValueError(f"Unknown Nexus login mode: {mode}")


def _is_helios_target(name: str) -> bool:
    return name.lower().startswith("helios")


def _backend_config(
    qnx: Any,
    *,
    device_name: str,
    simulator: str,
    noisy_simulation: bool,
    target_2qb_gate: str | None,
    max_cost: int | None,
    user_group: str | None,
) -> Any:
    if _is_helios_target(device_name):
        config_cls = getattr(qnx, "HeliosConfig", None)
        if config_cls is None:
            config_cls = qnx.models.HeliosConfig
        kwargs: dict[str, Any] = {"system_name": device_name}
        if max_cost is not None:
            kwargs["max_cost"] = float(max_cost)
        return config_cls(**kwargs)

    config_cls = getattr(qnx, "QuantinuumConfig", None)
    if config_cls is None:
        config_cls = qnx.models.QuantinuumConfig

    kwargs: dict[str, Any] = {
        "device_name": device_name,
        "simulator": simulator,
        "noisy_simulation": noisy_simulation,
    }
    if target_2qb_gate:
        kwargs["target_2qb_gate"] = target_2qb_gate
    if max_cost is not None:
        kwargs["max_cost"] = max_cost
    if user_group:
        kwargs["user_group"] = user_group

    return config_cls(**kwargs)


def _compiled_circuit_metrics(compiled_circuit: Any) -> dict[str, int | None]:
    depth = _maybe_call(compiled_circuit, "depth")
    commands = _maybe_call(compiled_circuit, "get_commands") or []
    two_qubit_gates = 0
    for command in commands:
        qubits = getattr(command, "qubits", ())
        if len(qubits) == 2:
            two_qubit_gates += 1

    return {
        "depth": int(depth) if depth is not None else None,
        "two_qubit_gates": two_qubit_gates,
        "compiled_commands": len(commands),
    }


def _bitstring_from_key(key: Any, width: int | None = None) -> str:
    if isinstance(key, int):
        return format(key, f"0{width}b") if width else format(key, "b")

    if isinstance(key, str):
        if key.isdecimal() and width:
            return format(int(key), f"0{width}b")
        return key.replace(" ", "")

    if isinstance(key, (tuple, list)):
        return "".join(str(int(bit)) for bit in key)

    to_readouts = getattr(key, "to_readouts", None)
    if callable(to_readouts):
        readouts = to_readouts()
        if len(readouts):
            return "".join(str(int(bit)) for bit in readouts[0])

    tolist = getattr(key, "tolist", None)
    if callable(tolist):
        values = tolist()
        if values and isinstance(values[0], (tuple, list)):
            values = values[0]
        return "".join(str(int(bit)) for bit in values)

    text = str(key).strip()
    if text.startswith("(") and text.endswith(")"):
        tokens = [token.strip() for token in text[1:-1].split(",") if token.strip()]
        if tokens and all(token in {"0", "1"} for token in tokens):
            return "".join(tokens)

    return text.replace(" ", "")


def _counts_from_backend_result(result: Any, shots: int, width: int | None = None) -> dict[str, int]:
    raw_counts = None
    get_counts = getattr(result, "get_counts", None)
    if callable(get_counts):
        raw_counts = get_counts()

    if raw_counts is None:
        collated_counts = getattr(result, "collated_counts", None)
        if callable(collated_counts):
            raw_counts = collated_counts()

    if raw_counts is None:
        qir_results = getattr(result, "results", None)
        if isinstance(qir_results, str):
            raw_counts = _counts_from_qir_result_text(qir_results, max_outputs=shots)

    if raw_counts is None:
        get_distribution = getattr(result, "get_distribution", None)
        if not callable(get_distribution):
            raise RuntimeError("Quantinuum result exposes neither get_counts() nor get_distribution()")
        distribution = get_distribution()
        raw_counts = {
            key: int(round(probability * shots))
            for key, probability in distribution.items()
        }

    if isinstance(raw_counts, dict) and raw_counts and all(isinstance(value, dict) for value in raw_counts.values()):
        raw_counts = raw_counts.get("meas") or next(iter(raw_counts.values()))

    counts: Counter[str] = Counter()
    for key, count in raw_counts.items():
        bitstring = _bitstring_from_key(key, width=width)
        if bitstring:
            counts[bitstring] += int(count)
    return dict(counts)


def _counts_from_qir_result_text(result_text: str, max_outputs: int | None = None) -> dict[int, int]:
    counts: Counter[int] = Counter()
    outputs_seen = 0
    for raw_line in result_text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 4:
            continue
        if parts[0] != "OUTPUT" or parts[1] != "INT":
            continue
        try:
            counts[int(parts[2])] += 1
            outputs_seen += 1
        except ValueError:
            continue
        if max_outputs is not None and outputs_seen >= max_outputs:
            break
    return dict(counts)


def _qir_text_to_helios_bitcode(qir_text: str) -> bytes:
    import pyqir

    # pytket-qir 2.0 emits a QIS-scoped read_result helper. Quantinuum's QIR
    # docs and Helios checker expect this as a runtime helper.
    compatible_qir = qir_text.replace(
        "__quantum__qis__read_result__body",
        "__quantum__rt__read_result",
    )
    return pyqir.Module.from_ir(pyqir.Context(), compatible_qir).bitcode


def run_on_quantinuum_nexus(
    qc: QuantumCircuit,
    *,
    device_name: str = DEFAULT_HELIOS_DEVICE,
    project_name: str = DEFAULT_NEXUS_PROJECT,
    shots: int = 1024,
    optimisation_level: int = 2,
    login_mode: str = "none",
    simulator: str = "state-vector",
    noisy_simulation: bool = True,
    target_2qb_gate: str | None = None,
    max_cost: int | None = None,
    user_group: str | None = None,
    wait_timeout_seconds: int | None = None,
    estimate_cost: bool = True,
    decompose_qiskit: bool = True,
    decompose_optimisation_level: int = 1,
) -> tuple[dict[str, int], dict[str, Any]]:
    """
    Convert a Qiskit circuit to TKET, compile through Nexus, execute, and return counts.

    Quantinuum's current Qiskit pathway is Qiskit -> TKET -> Nexus. This adapter
    keeps the rest of the Shor experiment unchanged while moving the hardware
    boundary from IBM Runtime to Quantinuum Nexus.
    """
    try:
        import qnexus as qnx
        from pytket.extensions.qiskit import qiskit_to_tk
    except ImportError as exc:  # pragma: no cover - local environment dependent
        raise RuntimeError(
            "Quantinuum runs require qnexus, pytket, and pytket-qiskit. "
            "Install them with: pip install qnexus pytket pytket-qiskit"
        ) from exc

    _login(qnx=qnx, mode=login_mode)

    suffix = uuid.uuid4().hex[:10]
    started_at = time.perf_counter()

    input_circuit = qc
    decompose_runtime_sec = 0.0
    original_depth = qc.depth()
    original_instruction_count = len(qc.data)
    if decompose_qiskit:
        decompose_started_at = time.perf_counter()
        input_circuit = transpile(
            qc,
            basis_gates=["rx", "ry", "rz", "cx"],
            optimization_level=decompose_optimisation_level,
        )
        decompose_runtime_sec = time.perf_counter() - decompose_started_at

    convert_started_at = time.perf_counter()
    tket_circuit = qiskit_to_tk(input_circuit)
    conversion_runtime_sec = time.perf_counter() - convert_started_at

    project = qnx.projects.get_or_create(project_name)
    qnx.context.set_active_project(project)

    compile_job_ref = None
    compile_runtime_sec = 0.0
    circuit_ref = None
    qir_ref = None
    qir_profile = None
    qir_conversion_runtime_sec = None
    cost_hqc = None
    effective_max_cost = max_cost

    if _is_helios_target(device_name):
        try:
            from pytket.passes import FlattenRelabelRegistersPass
            from pytket.qir.conversion.api import QIRFormat, QIRProfile, pytket_to_qir
        except ImportError as exc:  # pragma: no cover - local environment dependent
            raise RuntimeError(
                "Helios runs require pytket-qir for QIR bitcode generation. "
                "Install it with: pip install pytket-qir"
            ) from exc

        qir_started_at = time.perf_counter()
        FlattenRelabelRegistersPass("q").apply(tket_circuit)
        qir_profile = "ADAPTIVE"
        qir_text = pytket_to_qir(
            tket_circuit,
            name=f"shor_qpe_{qc.name}_{suffix}",
            qir_format=QIRFormat.STRING,
            int_type=64,
            profile=QIRProfile.ADAPTIVE,
            cut_pytket_register=True,
        )
        qir_conversion_runtime_sec = time.perf_counter() - qir_started_at
        if not isinstance(qir_text, str):
            raise RuntimeError("pytket-qir did not return QIR text")
        qir_bitcode = _qir_text_to_helios_bitcode(qir_text)

        qir_ref = qnx.qir.upload(
            qir=qir_bitcode,
            name=f"qir-shor-qpe-{qc.name}-{suffix}",
            project=project,
        )
        compiled_ref = qir_ref
        metrics = _compiled_circuit_metrics(tket_circuit)

        if estimate_cost:
            try:
                cost_hqc = qnx.qir.cost(
                    programs=[qir_ref],
                    n_shots=[shots],
                    project=project,
                    system_name="Helios-1",
                )
            except Exception as exc:  # pragma: no cover - remote/version dependent
                LOGGER.warning("Could not estimate Quantinuum Helios QIR HQC cost: %s", exc)

        if effective_max_cost is None and cost_hqc is not None:
            effective_max_cost = max(1, math.ceil(float(cost_hqc) * 1.25))

        config = _backend_config(
            qnx,
            device_name=device_name,
            simulator=simulator,
            noisy_simulation=noisy_simulation,
            target_2qb_gate=target_2qb_gate,
            max_cost=effective_max_cost,
            user_group=user_group,
        )
    else:
        config = _backend_config(
            qnx,
            device_name=device_name,
            simulator=simulator,
            noisy_simulation=noisy_simulation,
            target_2qb_gate=target_2qb_gate,
            max_cost=effective_max_cost,
            user_group=user_group,
        )
        circuit_ref = qnx.circuits.upload(
            circuit=tket_circuit,
            name=f"shor-qpe-{qc.name}-{suffix}",
        )
        compile_level = max(0, min(2, int(optimisation_level)))
        compile_started_at = time.perf_counter()
        compile_job_ref = qnx.start_compile_job(
            programs=[circuit_ref],
            optimisation_level=compile_level,
            backend_config=config,
            name=f"compile-shor-qpe-{qc.name}-{suffix}",
        )
        qnx.jobs.wait_for(compile_job_ref, timeout=wait_timeout_seconds)
        compiled_ref = qnx.jobs.results(compile_job_ref)[0].get_output()
        compiled_circuit = compiled_ref.download_circuit()
        compile_runtime_sec = time.perf_counter() - compile_started_at
        metrics = _compiled_circuit_metrics(compiled_circuit)
        if estimate_cost:
            try:
                cost_hqc = qnx.client.circuits.cost(
                    compiled_ref,
                    n_shots=shots,
                    backend_config=config,
                )
            except Exception as exc:  # pragma: no cover - remote/version dependent
                LOGGER.warning("Could not estimate Quantinuum HQC cost: %s", exc)

    LOGGER.info(
        "Submitting to Quantinuum %s | qubits=%d depth=%s 2q=%s shots=%d estimated_cost=%s HQC max_cost=%s",
        device_name,
        qc.num_qubits,
        metrics.get("depth"),
        metrics.get("two_qubit_gates"),
        shots,
        cost_hqc,
        effective_max_cost,
    )

    execute_started_at = time.perf_counter()
    execute_kwargs: dict[str, Any] = {}
    if _is_helios_target(device_name) and effective_max_cost is not None:
        execute_kwargs["max_cost"] = [float(effective_max_cost)]

    execute_job_ref = qnx.start_execute_job(
        programs=[compiled_ref],
        backend_config=config,
        n_shots=[shots],
        name=f"execute-shor-qpe-{qc.name}-{suffix}",
        **execute_kwargs,
    )
    final_execute_status = qnx.jobs.wait_for(execute_job_ref, timeout=wait_timeout_seconds)
    result_ref = qnx.jobs.results(execute_job_ref)[0]
    result = result_ref.download_result()
    execute_runtime_sec = time.perf_counter() - execute_started_at
    total_runtime_sec = time.perf_counter() - started_at

    counts = _counts_from_backend_result(result=result, shots=shots, width=len(qc.clbits))
    metadata: dict[str, Any] = {
        "backend": device_name,
        "job_id": _ref_identifier(execute_job_ref),
        "num_qubits": qc.num_qubits,
        "runtime_sec": total_runtime_sec,
        "current_memory_mb": get_current_memory_mb(),
        "available_memory_mb": get_available_memory_mb(),
        "peak_memory_mb": get_peak_memory_mb(),
        "quantinuum_project": project_name,
        "quantinuum_device": device_name,
        "quantinuum_backend_config_type": config.__class__.__name__,
        "quantinuum_direct_execute": _is_helios_target(device_name),
        "quantinuum_compile_job_id": _ref_identifier(compile_job_ref) if compile_job_ref is not None else None,
        "quantinuum_execute_job_id": _ref_identifier(execute_job_ref),
        "quantinuum_circuit_ref": _ref_identifier(circuit_ref) if circuit_ref is not None else None,
        "quantinuum_compiled_circuit_ref": _ref_identifier(compiled_ref),
        "quantinuum_result_ref": _ref_identifier(result_ref),
        "quantinuum_cost_hqc": cost_hqc,
        "quantinuum_actual_cost_hqc": getattr(final_execute_status, "cost", None),
        "quantinuum_result_cost_hqc": getattr(result_ref, "cost", None),
        "quantinuum_effective_max_cost_hqc": effective_max_cost,
        "quantinuum_program_format": "qir" if qir_ref is not None else "pytket-circuit",
        "quantinuum_qir_ref": _ref_identifier(qir_ref) if qir_ref is not None else None,
        "quantinuum_qir_profile": qir_profile,
        "quantinuum_qir_conversion_runtime_sec": qir_conversion_runtime_sec,
        "quantinuum_original_depth": original_depth,
        "quantinuum_original_instruction_count": original_instruction_count,
        "quantinuum_qiskit_decomposed": decompose_qiskit,
        "quantinuum_qiskit_decompose_runtime_sec": decompose_runtime_sec,
        "quantinuum_qiskit_decomposed_depth": input_circuit.depth(),
        "quantinuum_qiskit_decomposed_instruction_count": len(input_circuit.data),
        "quantinuum_conversion_runtime_sec": conversion_runtime_sec,
        "quantinuum_compile_runtime_sec": compile_runtime_sec,
        "quantinuum_execute_runtime_sec": execute_runtime_sec,
        "quantinuum_login_mode": login_mode,
        "quantinuum_simulator": simulator,
        "quantinuum_noisy_simulation": noisy_simulation,
        "quantinuum_target_2qb_gate": target_2qb_gate,
        "quantinuum_user_group": user_group,
        **metrics,
    }
    return counts, metadata


__all__ = [
    "DEFAULT_HELIOS_DEVICE",
    "DEFAULT_NEXUS_PROJECT",
    "run_on_quantinuum_nexus",
]
