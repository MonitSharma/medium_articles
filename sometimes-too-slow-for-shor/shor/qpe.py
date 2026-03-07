from __future__ import annotations

import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .modexp import build_controlled_modular_multiply


def _append_inverse_qft(circuit: QuantumCircuit, qubits: list) -> None:
    """Append an inverse QFT without final swaps."""
    for target_index, target in enumerate(qubits):
        for control_index in range(target_index):
            angle = -math.pi / (2 ** (target_index - control_index))
            circuit.cp(angle, qubits[control_index], target)
        circuit.h(target)


def _normalize_oracle_method(method: str) -> str:
    normalized = method.lower()
    if normalized == "standard":
        return "auto"
    if normalized in {"auto", "permutation", "semi_compiled"}:
        return normalized
    raise ValueError(f"Unknown QPE/oracle method: {method}")


def build_qpe_order_finding_circuit(
    N: int,
    a: int,
    t: int,
    n_work: int,
    method: str = "standard",
) -> QuantumCircuit:
    """
    Build the textbook QPE circuit for order finding of a modulo N.

    Counting-register convention:
    - counting[0] controls U^(2^0), counting[1] controls U^(2^1), ...
    - the inverse QFT is emitted without final swaps

    Because of that no-swap choice, the observed bitstring may appear in either
    the "raw" order or the reversed order depending on how one interprets the
    register layout. The strict post-processing layer checks both.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    if math.gcd(a, N) != 1:
        raise ValueError("a must be coprime with N")

    required_work = math.ceil(math.log2(N))
    if n_work < required_work:
        raise ValueError(f"n_work={n_work} is too small for N={N}")

    strategy = _normalize_oracle_method(method)
    counting = QuantumRegister(t, "count")
    work = QuantumRegister(n_work, "work")
    meas = ClassicalRegister(t, "meas")
    circuit = QuantumCircuit(counting, work, meas, name=f"order_find_{N}")

    # Qiskit's little-endian basis means work[0] is the least-significant bit.
    # Flipping only that qubit prepares the integer basis state |1>.
    circuit.x(work[0])
    circuit.h(counting)

    controlled_oracle_cache: dict[int, object] = {}

    for exponent_index, control in enumerate(counting):
        power = 1 << exponent_index
        multiplier = pow(a, power, N)
        if multiplier not in controlled_oracle_cache:
            controlled_oracle_cache[multiplier] = build_controlled_modular_multiply(
                N=N,
                a=a,
                power=power,
                n_work=n_work,
                strategy=strategy,
            )
        circuit.append(controlled_oracle_cache[multiplier], [control, *work])

    _append_inverse_qft(circuit, list(counting))
    circuit.measure(counting, meas)
    return circuit


__all__ = ["build_qpe_order_finding_circuit"]
