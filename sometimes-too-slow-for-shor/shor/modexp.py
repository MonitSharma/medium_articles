from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import UnitaryGate

_MIB = 1024 * 1024
_COMPLEX128_BYTES = 16


def _validate_modmul_inputs(N: int, multiplier: int, n_work: int) -> int:
    if N <= 2:
        raise ValueError("N must be composite and greater than 2")
    if n_work < math.ceil(math.log2(N)):
        raise ValueError("n_work is too small to encode residues modulo N")
    if n_work > 10:
        raise ValueError("The explicit permutation oracle is intentionally limited to n_work <= 10")

    multiplier %= N
    if math.gcd(multiplier, N) != 1:
        raise ValueError("Multiplier must be coprime with N")

    return multiplier


@lru_cache(maxsize=256)
def _build_permutation_modmul_gate(N: int, multiplier: int, n_work: int) -> Gate:
    multiplier = _validate_modmul_inputs(N=N, multiplier=multiplier, n_work=n_work)

    warnings.warn(
        "Using the exact permutation UnitaryGate oracle. It is clear and correct, "
        "but scales poorly and is only practical for toy-scale experiments.",
        stacklevel=2,
    )

    dimension = 1 << n_work
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)

    for basis_state in range(dimension):
        mapped_state = (basis_state * multiplier) % N if basis_state < N else basis_state
        matrix[mapped_state, basis_state] = 1.0

    return UnitaryGate(matrix, label=f"x{multiplier} mod {N}")


def estimate_permutation_unitary_memory_mb(n_qubits: int) -> float:
    """Estimate the dense complex128 matrix size for a UnitaryGate on n_qubits."""
    dimension = 1 << n_qubits
    return (dimension * dimension * _COMPLEX128_BYTES) / _MIB


@lru_cache(maxsize=64)
def _build_mod15_rotation_gate(multiplier: int) -> Gate | None:
    """
    Hand-optimized exact cases for N=15.

    These are the standard bit-rotation identities:
    - x -> 2x mod 15
    - x -> 4x mod 15
    - x -> 8x mod 15

    They are exact over the full 4-qubit computational basis because they are
    pure wire permutations (so |0> and |15> remain fixed as required).
    """
    circuit = QuantumCircuit(4, name=f"mul_{multiplier}_mod15")

    if multiplier == 1:
        return circuit.to_gate(label="x1 mod 15")
    if multiplier == 2:
        circuit.swap(2, 3)
        circuit.swap(1, 2)
        circuit.swap(0, 1)
        return circuit.to_gate(label="x2 mod 15")
    if multiplier == 4:
        circuit.swap(1, 3)
        circuit.swap(0, 2)
        return circuit.to_gate(label="x4 mod 15")
    if multiplier == 8:
        circuit.swap(0, 1)
        circuit.swap(1, 2)
        circuit.swap(2, 3)
        return circuit.to_gate(label="x8 mod 15")

    return None


def _build_semi_compiled_modmul_gate(N: int, multiplier: int, n_work: int) -> Gate:
    multiplier = _validate_modmul_inputs(N=N, multiplier=multiplier, n_work=n_work)

    if N == 15 and n_work == 4:
        gate = _build_mod15_rotation_gate(multiplier)
        if gate is not None:
            return gate

    warnings.warn(
        "No hand-optimized semi-compiled oracle is available for this case. "
        "Falling back to the exact permutation oracle.",
        stacklevel=2,
    )
    return _build_permutation_modmul_gate(N=N, multiplier=multiplier, n_work=n_work)


def build_modular_multiply_gate(
    N: int,
    multiplier: int,
    n_work: int,
    strategy: str = "permutation",
) -> Gate:
    """
    Build an in-place multiplication gate |x> -> |multiplier * x mod N>.

    Strategy choices:
    - auto: use a hand-optimized toy oracle when available, otherwise use the
      exact permutation oracle fallback
    - permutation: exact explicit permutation matrix (best for correctness)
    - semi_compiled: exact hand-optimized toy cases when available, else fallback
    """
    normalized = strategy.lower()

    if normalized == "auto":
        if N == 15 and n_work == 4:
            gate = _build_mod15_rotation_gate(multiplier % N)
            if gate is not None:
                return gate

        warnings.warn(
            "Auto oracle mode is falling back to the exact permutation UnitaryGate. "
            "This is exact, but memory and transpilation cost can grow quickly at larger sizes.",
            stacklevel=2,
        )
        return _build_permutation_modmul_gate(N=N, multiplier=multiplier, n_work=n_work)

    if normalized == "permutation":
        return _build_permutation_modmul_gate(N=N, multiplier=multiplier, n_work=n_work)
    if normalized == "semi_compiled":
        return _build_semi_compiled_modmul_gate(N=N, multiplier=multiplier, n_work=n_work)

    raise ValueError(f"Unknown modular multiplication strategy: {strategy}")


def build_controlled_modular_multiply(
    N: int,
    a: int,
    power: int,
    n_work: int,
    strategy: str = "permutation",
) -> Instruction:
    """
    Build controlled multiplication by a**power mod N.

    The returned instruction acts on one control qubit plus the work register.
    """
    if math.gcd(a, N) != 1:
        raise ValueError("a must be coprime with N for order finding")

    multiplier = pow(a, power, N)
    base_gate = build_modular_multiply_gate(N=N, multiplier=multiplier, n_work=n_work, strategy=strategy)
    controlled_gate = base_gate.control(1)
    controlled_gate.label = f"c*x{multiplier} mod {N}"
    return controlled_gate

__all__ = [
    "build_modular_multiply_gate",
    "build_controlled_modular_multiply",
    "estimate_permutation_unitary_memory_mb",
]
