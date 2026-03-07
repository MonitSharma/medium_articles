from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


LOGGER = logging.getLogger(__name__)

_MIB = 1024 * 1024
DEFAULT_IBM_CREDENTIALS_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ibm_credentials.json"
_EXHAUSTED_QPU_CREDENTIAL_FINGERPRINTS: set[str] = set()


def get_peak_memory_mb() -> float | None:
    """
    Return the process peak resident-set size in MiB, if the platform exposes it.

    On macOS, ru_maxrss is reported in bytes.
    On Linux, ru_maxrss is reported in KiB.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - platform-dependent
        return None

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / _MIB
    return rss / 1024


def get_current_memory_mb() -> float | None:
    """Return the current process RSS in MiB when available."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - optional dependency
        return get_peak_memory_mb()

    return psutil.Process().memory_info().rss / _MIB


def get_available_memory_mb() -> float | None:
    """Return available system memory in MiB when it can be detected."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - optional dependency
        pass
    else:
        return psutil.virtual_memory().available / _MIB

    try:
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):  # pragma: no cover - platform-dependent
        return None

    return (available_pages * page_size) / _MIB


def _backend_name(backend: Any) -> str:
    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        return str(name_attr())
    if name_attr is not None:
        return str(name_attr)
    return backend.__class__.__name__


def _count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    return sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)


def _unique_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for name in names:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)

    return ordered


def _candidate_data_names(data: Any, circuit: QuantumCircuit) -> list[str]:
    names = [creg.name for creg in circuit.cregs]

    keys_method = getattr(data, "keys", None)
    if callable(keys_method):
        try:
            names.extend(str(key) for key in keys_method())
        except TypeError:
            pass

    names.extend(name for name in dir(data) if not name.startswith("_"))
    return _unique_names(names)


def _extract_counts_from_sampler_result(pub_result: Any, circuit: QuantumCircuit) -> dict[str, int]:
    data = pub_result.data

    for name in _candidate_data_names(data, circuit):
        maybe_register = getattr(data, name, None)
        if hasattr(maybe_register, "get_counts"):
            return dict(maybe_register.get_counts())

    raise RuntimeError("Could not locate a classical register payload with get_counts() in SamplerV2 result")


def _build_pass_manager(backend: Any, opt_level: int):
    try:
        return generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    except TypeError:
        return generate_preset_pass_manager(optimization_level=opt_level, target=backend.target)


def _apply_sampler_options(sampler: Any, use_dd: bool, use_twirling: bool) -> None:
    if not (use_dd or use_twirling):
        return

    try:
        options = sampler.options

        if use_dd:
            dd = getattr(options, "dynamical_decoupling", None)
            if dd is None:
                raise AttributeError("dynamical_decoupling options are unavailable")
            if hasattr(dd, "enable"):
                dd.enable = True
            if hasattr(dd, "sequence_type"):
                dd.sequence_type = "XpXm"

        if use_twirling:
            twirling = getattr(options, "twirling", None)
            if twirling is None:
                raise AttributeError("twirling options are unavailable")
            if hasattr(twirling, "enable_gates"):
                twirling.enable_gates = True
            if hasattr(twirling, "enable_measure"):
                twirling.enable_measure = True
            if hasattr(twirling, "num_randomizations"):
                twirling.num_randomizations = "auto"

    except Exception as exc:  # pragma: no cover - version-dependent runtime API
        warnings.warn(
            f"Could not apply requested SamplerV2 options (DD/twirling). Continuing without them: {exc}",
            stacklevel=2,
        )


def _credential_fingerprint(credential: dict[str, Any], index: int) -> str:
    token = str(credential.get("token") or "")
    crn = str(credential.get("crn") or "")
    raw = f"{crn}|{token}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_runtime_credentials(
    credentials_path: Path,
    attempted_fingerprints: set[str],
) -> list[dict[str, Any]]:
    if not credentials_path.exists():
        return []

    payload = json.loads(credentials_path.read_text(encoding="utf-8"))
    raw_credentials = payload.get("credentials", [])
    if not isinstance(raw_credentials, list):
        raise RuntimeError(f"Invalid credentials payload in {credentials_path}: 'credentials' must be a list")

    fresh_credentials: list[dict[str, Any]] = []
    for index, item in enumerate(raw_credentials):
        if not isinstance(item, dict):
            continue

        token = item.get("token")
        crn = item.get("crn")
        if not token or not crn:
            continue

        label = str(item.get("label") or f"credential-{index + 1}")
        fingerprint = _credential_fingerprint(item, index)
        if fingerprint in attempted_fingerprints or fingerprint in _EXHAUSTED_QPU_CREDENTIAL_FINGERPRINTS:
            continue

        fresh_credentials.append(
            {
                "label": label,
                "token": str(token),
                "crn": str(crn),
                "fingerprint": fingerprint,
            }
        )

    return fresh_credentials


def _build_service_from_credential(QiskitRuntimeService: Any, credential: dict[str, Any]) -> Any:
    last_error: Exception | None = None

    for kwargs in (
        {"channel": "ibm_quantum_platform", "token": credential["token"], "instance": credential["crn"]},
        {"channel": "ibm_cloud", "token": credential["token"], "instance": credential["crn"]},
        {"token": credential["token"], "instance": credential["crn"]},
    ):
        try:
            return QiskitRuntimeService(**kwargs)
        except Exception as exc:  # pragma: no cover - version/environment dependent
            last_error = exc

    if last_error is None:
        raise RuntimeError(f"Could not initialize IBM Runtime service for credential {credential['label']}")
    raise last_error


def _usage_remaining_seconds(usage_dict: dict[str, Any]) -> float | None:
    remaining = usage_dict.get("usage_remaining_seconds")
    if remaining is not None:
        try:
            return float(remaining)
        except (TypeError, ValueError):
            return None

    if usage_dict.get("usage_limit_reached"):
        return 0.0

    return None


def _has_minimum_qpu_time(service: Any, min_qpu_seconds: float) -> tuple[bool, float | None, dict[str, Any]]:
    usage_dict = service.usage()
    remaining_seconds = _usage_remaining_seconds(usage_dict)

    if remaining_seconds is None:
        # If the service cannot quantify a remaining quota but also does not report
        # that the limit is exhausted, treat the credential as usable.
        return True, None, usage_dict

    return remaining_seconds >= min_qpu_seconds, remaining_seconds, usage_dict


def _resolve_backend_for_service(service: Any, backend_name: str, min_num_qubits: int) -> Any:
    if backend_name == "auto":
        return service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=min_num_qubits,
        )
    return service.backend(backend_name)


def _is_time_quota_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    phrases = (
        "no more time available",
        "usage limit",
        "usage_remaining",
        "usage remaining",
        "time is made available",
        "met its usage limit",
        "no more time available for this instance",
    )
    return any(phrase in message for phrase in phrases)


def _execute_sampler_job(
    *,
    qc: QuantumCircuit,
    backend: Any,
    SamplerV2: Any,
    shots: int,
    opt_level: int,
    seed: int,
    use_dd: bool,
    use_twirling: bool,
    credential_label: str,
    qpu_time_remaining_seconds: float | None,
    active_instance: str | None,
) -> tuple[dict[str, int], dict[str, Any]]:
    pass_manager = _build_pass_manager(backend=backend, opt_level=opt_level)
    isa_circuit = pass_manager.run(qc)
    depth = isa_circuit.depth()
    two_qubit_gates = _count_two_qubit_gates(isa_circuit)
    backend_label = _backend_name(backend)

    LOGGER.info(
        "Submitting ISA circuit to %s with credential=%s | qubits=%d depth=%d 2q=%d shots=%d seed=%d qpu_time_remaining=%s",
        backend_label,
        credential_label,
        isa_circuit.num_qubits,
        depth,
        two_qubit_gates,
        shots,
        seed,
        f"{qpu_time_remaining_seconds:.1f}s" if qpu_time_remaining_seconds is not None else "unknown",
    )

    try:
        sampler = SamplerV2(mode=backend)
    except TypeError:  # pragma: no cover - constructor names vary slightly by version
        sampler = SamplerV2(backend=backend)

    _apply_sampler_options(sampler=sampler, use_dd=use_dd, use_twirling=use_twirling)

    started_at = time.perf_counter()
    job = sampler.run([isa_circuit], shots=shots)
    primitive_result = job.result()
    runtime_sec = time.perf_counter() - started_at

    counts = _extract_counts_from_sampler_result(primitive_result[0], isa_circuit)
    job_id_attr = getattr(job, "job_id", None)
    job_id = job_id_attr() if callable(job_id_attr) else job_id_attr

    metadata = {
        "backend": backend_label,
        "job_id": job_id,
        "depth": depth,
        "two_qubit_gates": two_qubit_gates,
        "num_qubits": isa_circuit.num_qubits,
        "runtime_sec": runtime_sec,
        "current_memory_mb": get_current_memory_mb(),
        "available_memory_mb": get_available_memory_mb(),
        "peak_memory_mb": get_peak_memory_mb(),
        "credential_label": credential_label,
        "qpu_time_remaining_seconds": qpu_time_remaining_seconds,
        "active_instance": active_instance,
    }
    return counts, metadata


def run_on_ibm(
    qc: QuantumCircuit,
    backend_name: str = "auto",
    shots: int = 1024,
    opt_level: int = 1,
    seed: int = 7,
    use_dd: bool = True,
    use_twirling: bool = True,
    credentials_path: str | os.PathLike[str] | None = None,
    min_qpu_seconds: float = 60.0,
) -> tuple[dict[str, int], dict[str, Any]]:
    """
    Transpile to ISA, execute with SamplerV2, and return counts plus metadata.

    Credential behavior:
    - load token/CRN pairs from the credentials JSON file before each selection pass
    - require at least ``min_qpu_seconds`` of remaining instance time when the API exposes it
    - try credentials serially
    - if a time-quota-related error occurs during execution, switch to the next credential
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "qiskit-ibm-runtime >= 0.40 is required for hardware runs. "
            "Install it before using run_on_ibm()."
        ) from exc

    attempted_fingerprints: set[str] = set()
    credentials_file = Path(credentials_path) if credentials_path else DEFAULT_IBM_CREDENTIALS_PATH
    last_error: Exception | None = None

    while True:
        fresh_credentials = _load_runtime_credentials(
            credentials_path=credentials_file,
            attempted_fingerprints=attempted_fingerprints,
        )

        if not fresh_credentials:
            break

        for credential in fresh_credentials:
            attempted_fingerprints.add(credential["fingerprint"])
            label = credential["label"]

            try:
                service = _build_service_from_credential(QiskitRuntimeService, credential)
            except Exception as exc:  # pragma: no cover - depends on remote auth/version
                LOGGER.warning("Skipping credential %s because the service could not be initialized: %s", label, exc)
                last_error = exc
                continue

            try:
                has_time, remaining_seconds, usage_dict = _has_minimum_qpu_time(
                    service=service,
                    min_qpu_seconds=min_qpu_seconds,
                )
            except Exception as exc:  # pragma: no cover - depends on network/service response
                LOGGER.warning("Skipping credential %s because instance usage could not be queried: %s", label, exc)
                last_error = exc
                continue

            if not has_time:
                _EXHAUSTED_QPU_CREDENTIAL_FINGERPRINTS.add(credential["fingerprint"])
                LOGGER.info(
                    "Skipping credential %s because only %s of QPU time remains (< %.1fs required).",
                    label,
                    f"{remaining_seconds:.1f}s" if remaining_seconds is not None else "unknown",
                    min_qpu_seconds,
                )
                last_error = RuntimeError(
                    f"Credential {label} does not have the required remaining QPU time ({min_qpu_seconds:.1f}s)."
                )
                continue

            try:
                backend = _resolve_backend_for_service(
                    service=service,
                    backend_name=backend_name,
                    min_num_qubits=qc.num_qubits,
                )
            except Exception as exc:
                LOGGER.warning("Skipping credential %s because backend selection failed: %s", label, exc)
                last_error = exc
                continue

            try:
                active_instance_attr = getattr(service, "active_instance", None)
                active_instance = active_instance_attr() if callable(active_instance_attr) else credential["crn"]
                return _execute_sampler_job(
                    qc=qc,
                    backend=backend,
                    SamplerV2=SamplerV2,
                    shots=shots,
                    opt_level=opt_level,
                    seed=seed,
                    use_dd=use_dd,
                    use_twirling=use_twirling,
                    credential_label=label,
                    qpu_time_remaining_seconds=remaining_seconds,
                    active_instance=active_instance,
                )
            except Exception as exc:
                if _is_time_quota_error(exc):
                    _EXHAUSTED_QPU_CREDENTIAL_FINGERPRINTS.add(credential["fingerprint"])
                    LOGGER.warning(
                        "Credential %s hit a time-related runtime error; switching to the next credential: %s",
                        label,
                        exc,
                    )
                    last_error = exc
                    continue
                raise

        # Reload the credentials file so newly added credentials become available
        # in the next pass. If nothing new was added, the next loop breaks.

    if credentials_file.exists():
        error_message = (
            f"No usable IBM Quantum credential was able to run the workload from {credentials_file}. "
            f"At least {min_qpu_seconds:.1f}s of QPU time is required."
        )
        if last_error is not None:
            raise RuntimeError(f"{error_message} Last error: {last_error}") from last_error
        raise RuntimeError(error_message)

    # Fallback: if the credentials file is absent, preserve the previous behavior
    # and rely on the locally saved default account.
    LOGGER.warning(
        "Credentials file %s was not found. Falling back to the locally saved default IBM Runtime account.",
        credentials_file,
    )
    service = QiskitRuntimeService()
    try:
        has_time, remaining_seconds, _ = _has_minimum_qpu_time(service=service, min_qpu_seconds=min_qpu_seconds)
    except Exception:
        has_time, remaining_seconds = True, None
    if not has_time:
        raise RuntimeError(
            f"The default IBM Runtime account does not have at least {min_qpu_seconds:.1f}s of QPU time available."
        )

    backend = _resolve_backend_for_service(
        service=service,
        backend_name=backend_name,
        min_num_qubits=qc.num_qubits,
    )
    active_instance_attr = getattr(service, "active_instance", None)
    active_instance = active_instance_attr() if callable(active_instance_attr) else None
    return _execute_sampler_job(
        qc=qc,
        backend=backend,
        SamplerV2=SamplerV2,
        shots=shots,
        opt_level=opt_level,
        seed=seed,
        use_dd=use_dd,
        use_twirling=use_twirling,
        credential_label="default-account",
        qpu_time_remaining_seconds=remaining_seconds,
        active_instance=active_instance,
    )


__all__ = [
    "DEFAULT_IBM_CREDENTIALS_PATH",
    "get_available_memory_mb",
    "get_current_memory_mb",
    "get_peak_memory_mb",
    "run_on_ibm",
]
