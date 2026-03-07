from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Iterable, Iterator


def _continued_fraction_terms(value: Fraction) -> list[int]:
    """Return the exact continued-fraction expansion for a rational value."""
    terms: list[int] = []
    numerator = value.numerator
    denominator = value.denominator

    while denominator:
        quotient, remainder = divmod(numerator, denominator)
        terms.append(quotient)
        numerator, denominator = denominator, remainder

    return terms


def _convergents(terms: list[int]) -> Iterator[Fraction]:
    """Yield convergents from a continued-fraction term list."""
    h_nm2, h_nm1 = 0, 1
    k_nm2, k_nm1 = 1, 0

    for term in terms:
        h_n = term * h_nm1 + h_nm2
        k_n = term * k_nm1 + k_nm2
        yield Fraction(h_n, k_n)
        h_nm2, h_nm1 = h_nm1, h_n
        k_nm2, k_nm1 = k_nm1, k_n


def _ordered_unique(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []

    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)

    return ordered


def _prime_factors(n: int) -> list[int]:
    factors: list[int] = []
    candidate = 2
    remaining = n

    while candidate * candidate <= remaining:
        if remaining % candidate == 0:
            factors.append(candidate)
            while remaining % candidate == 0:
                remaining //= candidate
        candidate += 1 if candidate == 2 else 2

    if remaining > 1:
        factors.append(remaining)

    return factors


def _reduce_to_minimal_order(r: int, a: int, N: int) -> int:
    """Shrink a verified order candidate to the minimal order when possible."""
    r_min = r

    for factor in _prime_factors(r):
        while r_min % factor == 0 and pow(a, r_min // factor, N) == 1:
            r_min //= factor

    return r_min


def _factor_from_verified_order(r: int, a: int, N: int) -> tuple[int, int, int] | None:
    """Attempt to derive non-trivial factors from a verified multiplicative order."""
    if r % 2 != 0:
        return None

    half_power = pow(a, r // 2, N)
    if half_power in {1, N - 1}:
        return None

    for factor in _ordered_unique([math.gcd(half_power - 1, N), math.gcd(half_power + 1, N)]):
        if 1 < factor < N and N % factor == 0:
            other = N // factor
            if 1 < other < N and factor * other == N:
                p, q = sorted((factor, other))
                return p, q, r

    return None

def _good_convergent_denominators(phase: Fraction, N: int, t: int) -> list[int]:
    """
    Return convergent denominators that actually fit the measured phase.
    """
    threshold = Fraction(1, 2 * (1 << t))
    good: list[int] = []
    terms = _continued_fraction_terms(phase)

    for convergent in _convergents(terms):
        if convergent.denominator > N:
            continue
        if abs(phase - convergent) <= threshold:
            good.append(convergent.denominator)

    # Also include the best rational with denominator <= N if it passes the bound.
    best = phase.limit_denominator(max_denominator=N)
    if abs(phase - best) <= threshold:
        good.append(best.denominator)

    return good


def _expand_with_multiples(candidates: list[int], N: int, max_multiplier: int = 4) -> list[int]:
    """
    Try small multiples (k*q) for denominator candidates.
    """
    expanded: list[int] = []
    for q in candidates:
        if q <= 0:
            continue
        for k in range(1, max_multiplier + 1):
            r = k * q
            if r <= N:
                expanded.append(r)
    return expanded


def strict_postprocess_y(y: int, t: int, a: int, N: int) -> tuple[int, int, int] | None:
    """
    Strict post-processing for one measured y.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    if N <= 2:
        raise ValueError("N must be composite and greater than 2")
    if not 0 <= y < (1 << t):
        raise ValueError(f"y={y} is outside the {t}-bit counting range")

    phase = Fraction(y, 1 << t)

    # Only keep convergents that match the measured phase quality bound.
    base_candidates = _good_convergent_denominators(phase, N, t)

    # Bounded expansion keeps this path intentionally conservative.
    all_candidates = _expand_with_multiples(base_candidates, N, max_multiplier=4)

    for r in _ordered_unique(all_candidates):
        if r <= 0 or r > N:
            continue
        if pow(a, r, N) != 1:
            continue

        r_min = _reduce_to_minimal_order(r, a, N)
        factors = _factor_from_verified_order(r_min, a, N)
        if factors is not None:
            return factors

    return None


def strict_null_baseline_fp_rate(
    t: int,
    a: int,
    N: int,
    trials: int = 2048,
    seed: int = 42,
) -> float:
    """
    Canonical strict null baseline:
    false-positive rate of strict_postprocess_y on uniformly random y in [0, 2^t - 1].
    """
    if trials <= 0:
        return 0.0
    if t <= 0:
        raise ValueError("t must be positive")

    rng = random.Random(seed)
    false_positives = 0
    upper = 1 << t
    for _ in range(trials):
        y = rng.randrange(upper)
        if strict_postprocess_y(y=y, t=t, a=a, N=N) is not None:
            false_positives += 1

    return false_positives / trials


def exploratory_postprocess_y(y: int, t: int, a: int, N: int) -> tuple[int, int, int] | None:
    """
    Broad post-processing path for debugging.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    if N <= 2:
        raise ValueError("N must be composite and greater than 2")
    if not 0 <= y < (1 << t):
        raise ValueError(f"y={y} is outside the {t}-bit counting range")

    phase = Fraction(y, 1 << t)
    base_candidates: list[int] = []

    for convergent in _convergents(_continued_fraction_terms(phase)):
        base_candidates.append(convergent.denominator)

    base_candidates.append(phase.limit_denominator(max_denominator=N).denominator)

    # Try all multiples up to N in exploratory mode.
    expanded: list[int] = []
    for q in base_candidates:
        if q <= 0:
            continue
        k = 1
        while k * q <= N:
            expanded.append(k * q)
            k += 1

    for r in _ordered_unique(expanded):
        if r <= 0 or r > N:
            continue
        if pow(a, r, N) != 1:
            continue

        r_min = _reduce_to_minimal_order(r, a, N)
        factors = _factor_from_verified_order(r_min, a, N)
        if factors is not None:
            return factors

    return None

def per_shot_factor_yield(
    counts: dict[str, int],
    t: int,
    a: int,
    N: int,
    try_reversed_bitorder: bool = True,
) -> dict[str, int | float]:
    """
    Fraction of all shots that yield factors under strict post-processing.
    """
    total_shots = sum(counts.values())
    factorable_shots = 0
    unique_factorable = 0

    top1_bitstring = max(counts, key=counts.get) if counts else ""
    top1_count = counts.get(top1_bitstring, 0)
    top1_success = False

    for raw_bits, count in counts.items():
        bits = str(raw_bits).replace(" ", "")
        if not bits:
            continue

        candidates = [("raw", bits, int(bits, 2))]
        if try_reversed_bitorder and bits[::-1] != bits:
            candidates.append(("reversed", bits[::-1], int(bits[::-1], 2)))

        found = False
        for _, _, y_val in candidates:
            if strict_postprocess_y(y=y_val, t=t, a=a, N=N) is not None:
                found = True
                break

        if found:
            factorable_shots += count
            unique_factorable += 1
            if bits == top1_bitstring:
                top1_success = True

    return {
        "total_shots": total_shots,
        "factorable_shots": factorable_shots,
        "factor_yield_mass": factorable_shots / total_shots if total_shots > 0 else 0.0,
        "unique_factorable_bitstrings": unique_factorable,
        "top1_success": top1_success,
        "top1_bitstring": top1_bitstring,
        "top1_count": top1_count,
    }

def compute_ideal_peaks(a: int, N: int, t: int) -> list[dict]:
    """
    Compute ideal QPE peak locations y = round(s * 2^t / r).
    """
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            break

    two_to_t = 1 << t
    peaks = []
    for s in range(r):
        y_ideal = round(s * two_to_t / r)
        peaks.append({
            "s": s,
            "r_true": r,
            "y_ideal": y_ideal,
            "y_ideal_bitstring": format(y_ideal, f"0{t}b"),
            "phase_ideal": s / r,
        })
    return peaks


def _build_y_counts_single_order(counts: dict[str, int], reverse: bool) -> dict[int, int]:
    """Convert bitstring counts to y-value counts under ONE bit interpretation."""
    y_counts: dict[int, int] = {}
    for raw_bits, count in counts.items():
        bits = str(raw_bits).replace(" ", "")
        if not bits:
            continue
        if reverse:
            y = int(bits[::-1], 2)
        else:
            y = int(bits, 2)
        y_counts[y] = y_counts.get(y, 0) + count
    return y_counts


def _overlap_for_single_order(
    y_counts: dict[int, int],
    ideal: list[dict],
    two_to_t: int,
    tolerance: int,
    total_shots: int,
) -> dict[str, float | int]:
    """Compute peak overlap for one bit-order interpretation."""
    peaks_hit = 0
    mass_near_peaks = 0

    # Build the union of bins near any ideal peak.
    near_peak_ys: set[int] = set()
    for peak in ideal:
        y_ideal = peak["y_ideal"]
        for dy in range(-tolerance, tolerance + 1):
            near_peak_ys.add((y_ideal + dy) % two_to_t)

    # Count how many ideal peaks have at least one observed hit in tolerance.
    for peak in ideal:
        y_ideal = peak["y_ideal"]
        for dy in range(-tolerance, tolerance + 1):
            y_check = (y_ideal + dy) % two_to_t
            if y_check in y_counts:
                peaks_hit += 1
                break

    # Sum observed probability mass inside the near-peak union.
    for y_check in near_peak_ys:
        mass_near_peaks += y_counts.get(y_check, 0)

    mass_fraction = mass_near_peaks / total_shots if total_shots > 0 else 0.0

    return {
        "peaks_hit": peaks_hit,
        "mass_near_peaks": mass_fraction,
        "near_peak_bins_count": len(near_peak_ys),
    }


def histogram_vs_ideal_overlap(
    counts: dict[str, int],
    a: int,
    N: int,
    t: int,
    tolerance: int = 1,
    try_reversed_bitorder: bool = True,
) -> dict[str, float | int]:
    """
    Compare measured histogram to ideal peaks and pick the better bit-order view.
    """
    ideal = compute_ideal_peaks(a=a, N=N, t=t)
    if not ideal:
        return {"peaks_total": 0, "peaks_hit": 0, "peak_hit_fraction": 0.0,
                "mass_near_peaks": 0.0, "near_peak_bins_count": 0, "r_true": 0, "best_bit_order": "raw"}

    r_true = ideal[0]["r_true"]
    two_to_t = 1 << t
    total_shots = sum(counts.values())

    # Raw bit-order interpretation.
    raw_y = _build_y_counts_single_order(counts, reverse=False)
    raw_overlap = _overlap_for_single_order(raw_y, ideal, two_to_t, tolerance, total_shots)

    best = raw_overlap
    best_order = "raw"

    if try_reversed_bitorder:
        rev_y = _build_y_counts_single_order(counts, reverse=True)
        rev_overlap = _overlap_for_single_order(rev_y, ideal, two_to_t, tolerance, total_shots)

        if rev_overlap["mass_near_peaks"] > raw_overlap["mass_near_peaks"]:
            best = rev_overlap
            best_order = "reversed"

    return {
        "peaks_total": len(ideal),
        "peaks_hit": best["peaks_hit"],
        "peak_hit_fraction": best["peaks_hit"] / len(ideal) if ideal else 0.0,
        "mass_near_peaks": best["mass_near_peaks"],
        "near_peak_bins_count": best["near_peak_bins_count"],
        "r_true": r_true,
        "best_bit_order": best_order,
    }

def shor_postprocess_counts(
    counts: dict[str, int],
    t: int,
    a: int,
    N: int,
    top_k: int = 20,
    try_reversed_bitorder: bool = True,
) -> dict[str, int | str] | None:
    """
    Run strict post-processing on the top-k most likely bitstrings.
    """
    if top_k <= 0:
        return None

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]

    for raw_bits, count in ranked:
        bits = str(raw_bits).replace(" ", "")
        if not bits:
            continue

        candidates: list[tuple[str, str, int]] = [("raw", bits, int(bits, 2))]
        if try_reversed_bitorder and bits[::-1] != bits:
            reversed_bits = bits[::-1]
            candidates.append(("reversed", reversed_bits, int(reversed_bits, 2)))

        for bit_order, tested_bits, y in candidates:
            factors = strict_postprocess_y(y=y, t=t, a=a, N=N)
            if factors is None:
                continue

            p, q, r_min = factors
            return {
                "p": p,
                "q": q,
                "r_min": r_min,
                "count": count,
                "raw_bitstring": bits,
                "tested_bitstring": tested_bits,
                "bit_order": bit_order,
                "y": y,
            }

    return None


__all__ = [
    "strict_postprocess_y",
    "strict_null_baseline_fp_rate",
    "exploratory_postprocess_y",
    "shor_postprocess_counts",
    "per_shot_factor_yield",
    "compute_ideal_peaks",
    "histogram_vs_ideal_overlap",
]
