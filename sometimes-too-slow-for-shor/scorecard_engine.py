#!/usr/bin/env python3
"""
Scorecard Engine for Evaluating Shor's Algorithm Factoring Claims
================================================================

Computes null-baseline false-positive rates for every published (N, a, t)
combination. The key insight: when the control register has only t qubits,
there are only 2^t possible measurement outcomes. The continued-fractions
post-processing pipeline can extract correct factors from random noise
with surprisingly high probability for small t.

This script implements:
1. Strict-layer FP rate: P(random y -> correct r via continued fractions)
2. Exploratory-layer FP rate: P(random y -> factors via gcd scanning)
3. Full pipeline success rate against uniform random input
4. Comparison with reported experimental success rates
"""

import math
import os
from fractions import Fraction
from collections import defaultdict
import json

# =============================================================================
# Core Number Theory Functions
# =============================================================================

def gcd(a, b):
    """Euclidean GCD."""
    while b:
        a, b = b, a % b
    return a

def is_coprime(a, N):
    return gcd(a, N) == 1

def multiplicative_order(a, N):
    """Compute the multiplicative order of a modulo N."""
    if gcd(a, N) != 1:
        return None
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            return None  # Safety
    return r

def continued_fraction_convergents(numerator, denominator):
    """
    Compute all convergents of numerator/denominator using
    the continued fraction expansion.
    Returns list of (p, q) tuples where p/q is a convergent.
    """
    convergents = []
    a = numerator
    b = denominator
    
    # Previous convergents
    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0
    
    while b != 0:
        q = a // b
        a, b = b, a % b
        
        h_prev, h_curr = h_curr, q * h_curr + h_prev
        k_prev, k_curr = k_curr, q * k_curr + k_prev
        
        convergents.append((h_curr, k_curr))
    
    return convergents

def attempt_factor_from_order(r, a, N):
    """
    Given a candidate order r, attempt to extract factors of N.
    Returns (p, q) if successful, None otherwise.
    
    Handles:
    1. Standard case: r even → gcd(a^(r/2) ± 1, N)
    2. Perfect square base: if a = b^2, then a^(r/2) = b^r even for odd r
    3. General GCD approach: try gcd(a^k - 1, N) for divisors of r
    """
    if r is None or r <= 0:
        return None
    
    # Check: a^r mod N == 1?
    if pow(a, r, N) != 1:
        return None
    
    # Strategy 1: Standard even-r approach
    if r % 2 == 0:
        x = pow(a, r // 2, N)
        if x != N - 1:  # Not trivial
            p = gcd(x - 1, N)
            q = gcd(x + 1, N)
            result = _check_factors(p, q, N)
            if result:
                return result
    
    # Strategy 2: Perfect square base (handles N=21, a=4, r=3)
    sqrt_a = math.isqrt(a)
    if sqrt_a * sqrt_a == a and sqrt_a > 1:
        x = pow(sqrt_a, r, N)
        if x != N - 1 and x != 1:
            p = gcd(x - 1, N)
            q = gcd(x + 1, N)
            result = _check_factors(p, q, N)
            if result:
                return result
    
    # Strategy 3: Try gcd(a^k - 1, N) for proper divisors of r
    for k in range(1, r):
        if r % k == 0:
            x = pow(a, k, N)
            if x != 1:
                p = gcd(x - 1, N)
                if 1 < p < N:
                    return (min(p, N // p), max(p, N // p))
                p = gcd(x + 1, N)
                if 1 < p < N:
                    return (min(p, N // p), max(p, N // p))
    
    return None

def _check_factors(p, q, N):
    """Helper to validate factor candidates."""
    if 1 < p < N and 1 < q < N:
        return (min(p, q), max(p, q))
    if 1 < p < N:
        return (min(p, N // p), max(p, N // p))
    if 1 < q < N:
        return (min(q, N // q), max(q, N // q))
    return None

def attempt_factor_from_measurement(y, t, a, N, max_denominator=None):
    """
    Full Shor post-processing pipeline for a single measurement outcome y.
    
    Given measurement y from a t-qubit register:
    1. Compute phase phi = y / 2^t
    2. Find continued fraction convergents
    3. For each convergent denominator r_candidate:
       - Check if it's a valid order
       - Try to extract factors
    
    Returns (factors, r_found) or (None, None).
    """
    if max_denominator is None:
        max_denominator = N
    
    two_t = 2 ** t
    
    if y == 0:
        return None, None
    
    # Get convergents of y / 2^t
    convergents = continued_fraction_convergents(y, two_t)
    
    for p_conv, q_conv in convergents:
        if q_conv <= 0 or q_conv > max_denominator:
            continue
        
        r_candidate = q_conv
        
        # Check if a^r_candidate mod N == 1
        if pow(a, r_candidate, N) == 1:
            factors = attempt_factor_from_order(r_candidate, a, N)
            if factors is not None:
                return factors, r_candidate
            
            # Also try multiples of r_candidate
            for mult in range(2, N // r_candidate + 1):
                r_mult = r_candidate * mult
                if r_mult > N:
                    break
                if pow(a, r_mult, N) == 1:
                    factors = attempt_factor_from_order(r_mult, a, N)
                    if factors is not None:
                        return factors, r_mult
    
    return None, None

def exploratory_factor_attempt(y, t, a, N):
    """
    ADVERSARIAL expanded post-processing: tries convergent denominators
    AND their small multiples, AND nearby values of y.
    
    NOTE: This is NOT a faithful reconstruction of any single paper's pipeline.
    It is an adversarial stress-test modeling the worst-case scenario where
    an experimenter aggressively scans nearby outcomes and multiples.
    Results from this function should be labeled as "adversarial expanded
    post-processing" rates, not as the literal published pipeline.
    """
    factors, r = attempt_factor_from_measurement(y, t, a, N)
    if factors is not None:
        return factors, r
    
    # Try y ± 1, y ± 2 (adjacent peak scanning)
    two_t = 2 ** t
    for delta in [-1, 1, -2, 2]:
        y_adj = (y + delta) % two_t
        if y_adj == 0:
            continue
        factors, r = attempt_factor_from_measurement(y_adj, t, a, N)
        if factors is not None:
            return factors, r
    
    return None, None


# =============================================================================
# Null Baseline Computation
# =============================================================================

def compute_strict_null_fp_rate(N, a, t):
    """
    Compute the LIBERAL null false-positive rate:
    Uses the full recovery pipeline including perfect-square-base trick,
    divisor scanning, and multiples of candidate orders.
    
    This is MORE generous than textbook Shor. See compute_textbook_null_fp_rate
    for the narrower baseline.
    """
    two_t = 2 ** t
    true_factors = factorize_small(N)
    if true_factors is None:
        return None
    
    successes = 0
    for y in range(two_t):
        factors, _ = attempt_factor_from_measurement(y, t, a, N)
        if factors is not None and set(factors) == set(true_factors):
            successes += 1
    
    return successes / two_t

def compute_textbook_null_fp_rate(N, a, t):
    """
    Compute the TEXTBOOK-ONLY null false-positive rate:
    
    Pure textbook Shor post-processing:
    1. Continued fraction expansion of y/2^t
    2. Check if convergent denominator r satisfies a^r ≡ 1 (mod N)
    3. If r is ODD: fail (textbook says restart with new base)
    4. If r is even: compute gcd(a^(r/2) ± 1, N)
    5. No multiples, no divisor scanning, no perfect-square tricks
    
    This is the narrowest defensible interpretation of Shor's post-processing.
    """
    two_t = 2 ** t
    true_factors = factorize_small(N)
    if true_factors is None:
        return None
    
    successes = 0
    for y in range(two_t):
        if y == 0:
            continue
        convergents = continued_fraction_convergents(y, two_t)
        found = False
        for _, q_conv in convergents:
            if q_conv <= 0 or q_conv > N:
                continue
            # Must be valid order
            if pow(a, q_conv, N) != 1:
                continue
            # Textbook: odd r → fail, restart with new base
            if q_conv % 2 != 0:
                continue
            # Standard even-r extraction
            x = pow(a, q_conv // 2, N)
            if x == N - 1:  # trivial case
                continue
            p_cand = gcd(x - 1, N)
            q_cand = gcd(x + 1, N)
            for c in [p_cand, q_cand]:
                if 1 < c < N:
                    other = N // c
                    if set((c, other)) == set(true_factors):
                        found = True
                        break
            if found:
                break
        if found:
            successes += 1
    
    return successes / two_t

def compute_exploratory_null_fp_rate(N, a, t):
    """
    Compute the exploratory-layer false positive rate:
    What fraction of uniformly random measurements yield correct factors
    when using the extended scanning approach (adjacent outcomes, multiples)?
    """
    two_t = 2 ** t
    true_factors = factorize_small(N)
    if true_factors is None:
        return None
    
    successes = 0
    for y in range(two_t):
        factors, _ = exploratory_factor_attempt(y, t, a, N)
        if factors is not None and set(factors) == set(true_factors):
            successes += 1
    
    return successes / two_t

def qpe_probability(y, s, r, two_t):
    """
    Exact QPE probability of measuring y given eigenvalue phase s/r.
    
    Starting from the standard QPE amplitude:
      P(y|s) = (1/2^{2t}) * |sum_{x=0}^{2^t-1} exp(2πi·Δ·x)|^2
    
    where Δ = s/r - y/2^t. Evaluating the geometric series gives:
      P(y|s) = sin²(π·Δ·2^t) / (2^{2t} · sin²(π·Δ))
    
    when Δ is not an integer, and P(y|s) = 1 at exact peaks (Δ integer).
    
    Reference: Nielsen & Chuang, Eq. 5.26-5.27.
    """
    import math
    
    # Phase difference
    delta = s / r - y / two_t
    
    # Check if delta is (near) an integer → exact peak
    if abs(delta - round(delta)) < 1e-12:
        return 1.0
    
    # Geometric series formula
    numerator = math.sin(math.pi * delta * two_t) ** 2
    denominator = two_t * (math.sin(math.pi * delta) ** 2)
    
    if denominator < 1e-15:
        return 1.0
    
    return numerator / (two_t * denominator)


def compute_ideal_probability_distribution(a, N, t):
    """
    Compute the exact ideal QPE probability distribution P(y) for all y.
    
    The state after QPE is a mixture over eigenvalue indices s = 0, ..., r-1,
    each with weight 1/r. So:
    P(y) = (1/r) * sum_{s=0}^{r-1} P(y|s)
    """
    import math
    
    r = multiplicative_order(a, N)
    if r is None:
        return None, None
    
    two_t = 2 ** t
    probs = []
    
    for y in range(two_t):
        p_y = 0.0
        for s in range(r):
            p_y += qpe_probability(y, s, r, two_t)
        p_y /= r
        probs.append(p_y)
    
    return probs, r


def compute_ideal_success_rate(N, a, t):
    """
    Compute the ideal quantum success rate using the EXACT QPE probability
    distribution, not peak counting.
    
    This weights each outcome y by its true QPE probability and checks
    whether that outcome yields correct factors.
    """
    probs, r = compute_ideal_probability_distribution(a, N, t)
    if probs is None:
        return None, None
    
    two_t = 2 ** t
    true_factors = factorize_small(N)
    if true_factors is None:
        return None, None
    
    weighted_success = 0.0
    for y in range(two_t):
        factors, _ = attempt_factor_from_measurement(y, t, a, N)
        if factors is not None and set(factors) == set(true_factors):
            weighted_success += probs[y]
    
    return weighted_success, r

def factorize_small(N):
    """Classically factorize a small semiprime."""
    for p in range(2, int(math.isqrt(N)) + 1):
        if N % p == 0:
            return (p, N // p)
    return None


# =============================================================================
# Smolin-Smith-Vargo Analysis
# =============================================================================

def smolin_analysis(N, a, t):
    """
    Smolin-Smith-Vargo test: Does this compiled experiment reduce to
    a trivial verification circuit?
    
    Key question: Given that the compiler knows the factors,
    could a 2-qubit circuit achieve the same result?
    
    For order-2 bases: always reduces to 2 qubits (Smolin et al. 2013)
    For any compiled circuit: the circuit structure embeds the answer
    """
    r = multiplicative_order(a, N)
    
    # If r = 2, this is exactly the Smolin case
    if r == 2:
        return {
            "reduces_to_smolin": True,
            "min_qubits_smolin": 2,
            "order": r,
            "note": "Order-2 base: trivially reduces to 2-qubit Smolin circuit"
        }
    
    # For compiled circuits with known factors:
    # The modular exponentiation is pre-computed classically
    # Only the phase estimation + post-processing matters
    return {
        "reduces_to_smolin": True,  # All compiled circuits embed the answer
        "min_qubits_smolin": 2,
        "order": r,
        "note": f"Compiled circuit with r={r}: oracle embeds factor knowledge"
    }


# =============================================================================
# Published Experiments Database
# =============================================================================

EXPERIMENTS = [
    {
        "year": 2001,
        "authors": "Vandersypen et al.",
        "N": 15,
        "a": 7,
        "t": 3,  # effective control register bits
        "qubits": 7,
        "platform": "Liquid-state NMR",
        "method": "Gate-model (compiled)",
        "cx_gates": "~12 (NMR pulse equiv.)",
        "oracle_type": "Compiled/hardcoded",
        "success_metric": "Single run",
        "null_baseline_reported": False,
        "raw_histogram": False,
        "category": "A",
        "notes": "First proof-of-concept. NMR ensemble average."
    },
    {
        "year": 2007,
        "authors": "Lu et al.",
        "N": 15,
        "a": 2,
        "t": 2,
        "qubits": 4,
        "platform": "Photonic",
        "method": "Gate-model (compiled)",
        "cx_gates": "~4 (photonic equiv.)",
        "oracle_type": "Fully compiled",
        "success_metric": "Coincidence counts",
        "null_baseline_reported": False,
        "raw_histogram": False,
        "category": "A",
        "notes": "4-photon cluster state. Heavily compiled."
    },
    {
        "year": 2007,
        "authors": "Lanyon et al.",
        "N": 15,
        "a": 2,
        "t": 2,
        "qubits": 4,
        "platform": "Photonic",
        "method": "Gate-model (compiled)",
        "cx_gates": "~4",
        "oracle_type": "Fully compiled",
        "success_metric": "Coincidence counts",
        "null_baseline_reported": False,
        "raw_histogram": False,
        "category": "A",
        "notes": "Compiled photonic demonstration."
    },
    {
        "year": 2012,
        "authors": "Martin-Lopez et al.",
        "N": 15,
        "a": 2,
        "t": 3,
        "qubits": "2 (recycled)",
        "platform": "Photonic",
        "method": "Gate-model (iterative/compiled)",
        "cx_gates": "~6",
        "oracle_type": "Compiled + qubit recycling",
        "success_metric": "Photon coincidences",
        "null_baseline_reported": False,
        "raw_histogram": True,
        "category": "A",
        "notes": "Qubit recycling via semiclassical QFT. Architecturally important."
    },
    {
        "year": 2012,
        "authors": "Lucero et al.",
        "N": 21,
        "a": 4,
        "t": 3,
        "qubits": 4,
        "platform": "Superconducting (Josephson)",
        "method": "Gate-model (compiled)",
        "cx_gates": "~11",
        "oracle_type": "Fully compiled",
        "success_metric": "Repeated measurements",
        "null_baseline_reported": False,
        "raw_histogram": True,
        "category": "A",
        "notes": "First factorization of 21 on superconducting hardware."
    },
    {
        "year": 2019,
        "authors": "Amico, Saleem & Kumph",
        "N": 15,
        "a": 11,
        "t": 3,
        "qubits": 6,
        "platform": "IBM ibmqx5 (Superconducting)",
        "method": "Gate-model (compiled, iterative split)",
        "cx_gates": "~15 per sub-circuit",
        "oracle_type": "Compiled + semiclassical QFT (split circuits)",
        "success_metric": "Statistical overlap (SSO)",
        "null_baseline_reported": False,
        "raw_histogram": True,
        "category": "A",
        "notes": "Split-circuit iterative approach. Used SSO instead of continued fractions."
    },
    {
        "year": 2019,
        "authors": "Amico, Saleem & Kumph",
        "N": 21,
        "a": 4,
        "t": 3,
        "qubits": 6,
        "platform": "IBM ibmqx5 (Superconducting)",
        "method": "Gate-model (compiled, iterative split)",
        "cx_gates": "~20 per sub-circuit",
        "oracle_type": "Compiled + semiclassical QFT (split circuits)",
        "success_metric": "Statistical overlap (SSO)",
        "null_baseline_reported": False,
        "raw_histogram": True,
        "category": "A",
        "notes": "Split-circuit iterative approach for N=21."
    },
    {
        "year": 2019,
        "authors": "Amico, Saleem & Kumph",
        "N": 35,
        "a": 4,
        "t": 3,
        "qubits": 6,
        "platform": "IBM ibmqx5 (Superconducting)",
        "method": "Gate-model (compiled, iterative split)",
        "cx_gates": "~30 per sub-circuit",
        "oracle_type": "Compiled + semiclassical QFT (split circuits)",
        "success_metric": "Statistical overlap (SSO)",
        "null_baseline_reported": False,
        "raw_histogram": True,
        "category": "A",
        "notes": "N=35 attempted but largely failed (~14% success). Highest N attempted with compiled gate-model Shor on hardware."
    },
    {
        "year": 2016,
        "authors": "Monz et al.",
        "N": 15,
        "a": 2,
        "t": 4,
        "qubits": 5,
        "platform": "Trapped Ion",
        "method": "Gate-model (compiled, iterative)",
        "cx_gates": "~10 (MS gates)",
        "oracle_type": "Compiled + in-sequence measurement",
        "success_metric": "Success probability",
        "null_baseline_reported": False,
        "raw_histogram": True,
        "category": "A",
        "notes": "Claimed 'scalable' architecture."
    },
    {
        "year": 2021,
        "authors": "Skosana & Tame",
        "N": 21,
        "a": 4,
        "t": 3,
        "qubits": 5,
        "platform": "IBM Q (Superconducting)",
        "method": "Gate-model (compiled)",
        "cx_gates": 25,
        "oracle_type": "Compiled + Margolus gates",
        "success_metric": "Kolmogorov distance + state tomography",
        "null_baseline_reported": False,
        "raw_histogram": True,
        "category": "A",
        "notes": "Most rigorous N=21 demo. Verified entanglement. Fidelity ~0.70."
    },
    {
        "year": 2013,
        "authors": "Geller & Zhou",
        "N": 51,
        "a": None,  # Multiple bases
        "t": 4,
        "qubits": 8,
        "platform": "Theoretical",
        "method": "Gate-model (compressed)",
        "cx_gates": "4 (CNOT only)",
        "oracle_type": "Compressed modular exponentiation",
        "success_metric": "Theoretical",
        "null_baseline_reported": False,
        "raw_histogram": False,
        "category": "A",
        "notes": "Theoretical only. Fermat prime structure exploited."
    },
    {
        "year": 2013,
        "authors": "Geller & Zhou",
        "N": 85,
        "a": None,
        "t": 4,
        "qubits": 8,
        "platform": "Theoretical",
        "method": "Gate-model (compressed)",
        "cx_gates": "4 (CNOT only)",
        "oracle_type": "Compressed modular exponentiation",
        "success_metric": "Theoretical",
        "null_baseline_reported": False,
        "raw_histogram": False,
        "category": "A",
        "notes": "Theoretical only. Products of Fermat primes."
    },
    # Adiabatic / Ising stunts
    {
        "year": 2012,
        "authors": "Xu et al.",
        "N": 143,
        "a": None,
        "t": None,
        "qubits": 4,
        "platform": "NMR (adiabatic)",
        "method": "Adiabatic/Ising",
        "cx_gates": "N/A",
        "oracle_type": "Ising Hamiltonian",
        "success_metric": "Ground state energy",
        "null_baseline_reported": False,
        "raw_histogram": False,
        "category": "B",
        "notes": "11×13. Factors differ by 2 bits. NOT Shor's algorithm."
    },
    {
        "year": 2014,
        "authors": "Dattani & Bryans",
        "N": 56153,
        "a": None,
        "t": None,
        "qubits": 4,
        "platform": "Same as Xu (theoretical)",
        "method": "Adiabatic/Ising",
        "cx_gates": "N/A",
        "oracle_type": "Same Ising Hamiltonian as N=143",
        "success_metric": "Mathematical equivalence",
        "null_baseline_reported": False,
        "raw_histogram": False,
        "category": "B",
        "notes": "233×241. SAME Hamiltonian as 143. Pure math trick."
    },
    {
        "year": 2018,
        "authors": "Dash et al.",
        "N": 4088459,
        "a": None,
        "t": None,
        "qubits": 2,
        "platform": "IBM Q",
        "method": "Exact search (Ising)",
        "cx_gates": "~2",
        "oracle_type": "Bit-proximity exploitation",
        "success_metric": "Measurement",
        "null_baseline_reported": False,
        "raw_histogram": False,
        "category": "B",
        "notes": "2017×2027. Twin-prime-like proximity."
    },
    {
        "year": 2019,
        "authors": "Q2B Conference",
        "N": 1099551473989,
        "a": None,
        "t": None,
        "qubits": 3,
        "platform": "Unspecified",
        "method": "Combinatorial/Ising",
        "cx_gates": "N/A",
        "oracle_type": "Hyper-specific classical preprocessing",
        "success_metric": "Claimed factorization",
        "null_baseline_reported": False,
        "raw_histogram": False,
        "category": "B",
        "notes": "Maximum stunt. Classical preprocessing does all work."
    },
    # Lattice/QAOA
    {
        "year": 2022,
        "authors": "Yan et al.",
        "N": 261980999226229,
        "a": None,
        "t": None,
        "qubits": 10,
        "platform": "Superconducting",
        "method": "QAOA + Schnorr lattice",
        "cx_gates": "~50 (QAOA layers)",
        "oracle_type": "Lattice-CVP hybrid",
        "success_metric": "Factorization output",
        "null_baseline_reported": False,
        "raw_histogram": False,
        "category": "C",
        "notes": "48-bit number. Classical lattice method does heavy lifting. Scaling claims disputed."
    },
]


# =============================================================================
# Main Scorecard Computation
# =============================================================================

def compute_full_scorecard():
    """
    Compute the complete scorecard for all published experiments.
    """
    results = []
    
    for exp in EXPERIMENTS:
        result = {**exp}
        
        if exp["category"] == "A" and exp["a"] is not None and exp["t"] is not None:
            N = exp["N"]
            a = exp["a"]
            t = exp["t"]
            
            # Compute null baseline rates
            strict_fp = compute_strict_null_fp_rate(N, a, t)
            textbook_fp = compute_textbook_null_fp_rate(N, a, t)
            exploratory_fp = compute_exploratory_null_fp_rate(N, a, t)
            ideal_success, true_order = compute_ideal_success_rate(N, a, t)
            smolin = smolin_analysis(N, a, t)
            
            result["true_order"] = true_order
            result["textbook_null_fp_rate"] = textbook_fp
            result["strict_null_fp_rate"] = strict_fp
            result["exploratory_null_fp_rate"] = exploratory_fp
            result["ideal_success_rate"] = ideal_success
            result["smolin_analysis"] = smolin
            result["outcome_space_size"] = 2 ** t
            
            # Compute: how many of the 2^t outcomes yield correct factors?
            two_t = 2 ** t
            true_factors = factorize_small(N)
            strict_successes = 0
            exploratory_successes = 0
            for y in range(two_t):
                f1, _ = attempt_factor_from_measurement(y, t, a, N)
                if f1 is not None and set(f1) == set(true_factors):
                    strict_successes += 1
                f2, _ = exploratory_factor_attempt(y, t, a, N)
                if f2 is not None and set(f2) == set(true_factors):
                    exploratory_successes += 1
            
            result["strict_success_count"] = strict_successes
            result["exploratory_success_count"] = exploratory_successes
            
            # Detailed outcome analysis
            outcome_analysis = []
            for y in range(two_t):
                convergents = continued_fraction_convergents(y, two_t)
                factors, r_found = attempt_factor_from_measurement(y, t, a, N)
                outcome_analysis.append({
                    "y": y,
                    "y_binary": format(y, f'0{t}b'),
                    "phase": f"{y}/{two_t}",
                    "convergent_denominators": [q for _, q in convergents],
                    "factors_found": factors,
                    "order_found": r_found,
                    "success": factors is not None and set(factors) == set(true_factors)
                })
            result["outcome_analysis"] = outcome_analysis
            
        elif exp["category"] == "A" and exp["a"] is None:
            # For theoretical papers (Geller & Zhou), compute for representative bases
            N = exp["N"]
            t = exp["t"]
            
            # Find all coprime bases and their orders
            orders = {}
            for a_test in range(2, min(N, 50)):
                if is_coprime(a_test, N):
                    r = multiplicative_order(a_test, N)
                    if r is not None:
                        if r not in orders:
                            orders[r] = []
                        orders[r].append(a_test)
            
            result["order_distribution"] = {str(k): len(v) for k, v in sorted(orders.items())}
            result["representative_bases"] = {str(k): v[:3] for k, v in sorted(orders.items())}
            
            # Compute FP rates for a few representative bases
            fp_rates = {}
            for r_val, bases in orders.items():
                a_rep = bases[0]
                strict_fp = compute_strict_null_fp_rate(N, a_rep, t)
                fp_rates[f"a={a_rep},r={r_val}"] = strict_fp
            result["representative_fp_rates"] = fp_rates
        
        elif exp["category"] == "B":
            # Ising stunt analysis
            N = exp["N"]
            factors = factorize_small(N)
            if factors:
                p, q = factors
                p_bits = p.bit_length()
                q_bits = q.bit_length()
                bit_diff = abs(p_bits - q_bits)
                
                # Compute how many binary multiplication variables
                # and how many are trivially determined
                total_vars = p_bits + q_bits
                # Both factors are odd (known): saves 2 variables
                # MSBs are 1 (known): saves 2 variables
                known_bits = 4  # minimum: both odd + both MSB=1
                
                result["factors"] = factors
                result["p_bits"] = p_bits
                result["q_bits"] = q_bits
                result["bit_difference"] = bit_diff
                result["total_binary_vars"] = total_vars
                result["known_bits_minimum"] = known_bits
                result["remaining_vars_upper_bound"] = total_vars - known_bits
                result["classically_trivial"] = (total_vars - known_bits) <= 10
                
                # Check if Fermat's method works easily
                s = math.isqrt(N)
                fermat_steps = 0
                while s * s < N:
                    s += 1
                while True:
                    t2 = s * s - N
                    sqrt_t2 = math.isqrt(t2)
                    fermat_steps += 1
                    if sqrt_t2 * sqrt_t2 == t2:
                        break
                    s += 1
                    if fermat_steps > 10000:
                        break
                result["fermat_steps"] = fermat_steps
        
        results.append(result)
    
    return results


def print_scorecard(results):
    """Pretty-print the scorecard results."""
    
    print("=" * 100)
    print("SCORECARD: Evaluating Published Shor's Algorithm Factoring Claims")
    print("=" * 100)
    
    # Category A: Gate-model
    print("\n" + "=" * 100)
    print("CATEGORY A: Gate-Model Shor's Algorithm (Compiled Oracles)")
    print("=" * 100)
    
    for r in results:
        if r["category"] != "A":
            continue
        
        print(f"\n{'─' * 80}")
        print(f"  {r['year']} | {r['authors']} | N={r['N']} | {r['platform']}")
        print(f"  Qubits: {r['qubits']} | CX gates: {r['cx_gates']} | Oracle: {r['oracle_type']}")
        print(f"  Success metric: {r['success_metric']}")
        print(f"  Null baseline reported: {'Yes' if r['null_baseline_reported'] else 'NO'}")
        print(f"  Raw histogram shown: {'Yes' if r['raw_histogram'] else 'No'}")
        
        if "strict_null_fp_rate" in r:
            print(f"\n  *** NULL BASELINE ANALYSIS ***")
            print(f"  Control register: t={r['t']} qubits → {r['outcome_space_size']} possible outcomes")
            print(f"  True order: r={r['true_order']}")
            print(f"  Textbook-only FP rate (even-r, standard gcd): {r['textbook_null_fp_rate']:.4f} ({r['textbook_null_fp_rate']*100:.1f}%)")
            print(f"  Liberal FP rate (+ sqrt trick, divisors):      {r['strict_null_fp_rate']:.4f} ({r['strict_null_fp_rate']*100:.1f}%)")
            print(f"  Adversarial expanded rate (+ adjacent y):      {r['exploratory_null_fp_rate']:.4f} ({r['exploratory_null_fp_rate']*100:.1f}%)")
            print(f"  Ideal quantum success rate: {r['ideal_success_rate']:.4f} ({r['ideal_success_rate']*100:.1f}%)" if r['ideal_success_rate'] else "")
            print(f"  Outcomes yielding correct factors: {r['strict_success_count']}/{r['outcome_space_size']} (strict)")
            print(f"  Outcomes yielding correct factors: {r['exploratory_success_count']}/{r['outcome_space_size']} (exploratory)")
            
            # The key comparison
            if r['strict_null_fp_rate'] > 0.15:
                print(f"\n  ⚠️  HIGH FALSE POSITIVE RATE: A random number generator has a")
                print(f"     {r['strict_null_fp_rate']*100:.1f}% chance of 'factoring' N={r['N']} through")
                print(f"     the same classical post-processing pipeline.")
            
            # Print outcome table
            print(f"\n  Outcome-by-outcome analysis:")
            print(f"  {'y':>4} | {'binary':>6} | {'phase':>8} | {'convergent denoms':>20} | {'factors':>10} | {'success':>7}")
            print(f"  {'─'*4}─┼─{'─'*6}─┼─{'─'*8}─┼─{'─'*20}─┼─{'─'*10}─┼─{'─'*7}")
            for oa in r.get("outcome_analysis", []):
                denoms = str(oa["convergent_denominators"])
                if len(denoms) > 20:
                    denoms = denoms[:17] + "..."
                factors_str = str(oa["factors_found"]) if oa["factors_found"] else "—"
                success_str = "✓" if oa["success"] else "✗"
                print(f"  {oa['y']:>4} | {oa['y_binary']:>6} | {oa['phase']:>8} | {denoms:>20} | {factors_str:>10} | {success_str:>7}")
        
        elif "representative_fp_rates" in r:
            print(f"\n  *** THEORETICAL ANALYSIS (Multiple bases) ***")
            print(f"  Order distribution: {r['order_distribution']}")
            print(f"  Representative FP rates:")
            for key, fp in r["representative_fp_rates"].items():
                print(f"    {key}: {fp:.4f} ({fp*100:.1f}%)")
        
        print(f"  Notes: {r['notes']}")
    
    # Category B: Ising stunts
    print("\n\n" + "=" * 100)
    print("CATEGORY B: Adiabatic / Ising Stunt Factorizations (NOT Shor's Algorithm)")
    print("=" * 100)
    
    for r in results:
        if r["category"] != "B":
            continue
        
        print(f"\n{'─' * 80}")
        print(f"  {r['year']} | {r['authors']} | N={r['N']:,} | {r['platform']}")
        print(f"  Qubits: {r['qubits']} | Method: {r['method']}")
        
        if "factors" in r:
            p, q = r["factors"]
            print(f"  Factors: {p} × {q}")
            print(f"  Factor bit lengths: {r['p_bits']} and {r['q_bits']} bits (difference: {r['bit_difference']})")
            print(f"  Total binary variables: {r['total_binary_vars']}")
            print(f"  Known bits (minimum): {r['known_bits_minimum']}")
            print(f"  Remaining variables (upper bound): {r['remaining_vars_upper_bound']}")
            print(f"  Classically trivial (≤10 free vars): {'YES' if r['classically_trivial'] else 'No'}")
            print(f"  Fermat's method steps: {r['fermat_steps']}")
            if r['fermat_steps'] <= 100:
                print(f"  ⚠️  Fermat's classical method factors this in {r['fermat_steps']} steps!")
        
        print(f"  Notes: {r['notes']}")
    
    # Category C: Lattice/QAOA
    print("\n\n" + "=" * 100)
    print("CATEGORY C: Lattice/QAOA Hybrid Claims")
    print("=" * 100)
    
    for r in results:
        if r["category"] != "C":
            continue
        
        print(f"\n{'─' * 80}")
        print(f"  {r['year']} | {r['authors']} | N={r['N']:,} | {r['platform']}")
        print(f"  Qubits: {r['qubits']} | Method: {r['method']}")
        print(f"  Notes: {r['notes']}")


def generate_csv_table(results, filepath):
    """Generate a CSV summary table."""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Year", "Authors", "N", "Method", "Qubits", "CX Gates",
            "Oracle Type", "t (control bits)", "True Order",
            "Strict Null FP Rate", "Exploratory Null FP Rate",
            "Ideal Success Rate", "Null Baseline Reported?",
            "Passes Smolin Test?", "Category", "Verdict"
        ])
        
        for r in results:
            strict_fp = r.get("strict_null_fp_rate", "N/A")
            exploratory_fp = r.get("exploratory_null_fp_rate", "N/A")
            ideal = r.get("ideal_success_rate", "N/A")
            true_order = r.get("true_order", "N/A")
            
            if isinstance(strict_fp, float):
                strict_fp = f"{strict_fp:.4f}"
            if isinstance(exploratory_fp, float):
                exploratory_fp = f"{exploratory_fp:.4f}"
            if isinstance(ideal, float):
                ideal = f"{ideal:.4f}"
            
            # Verdict
            if r["category"] == "B":
                verdict = "Stunt (not Shor's)"
            elif r["category"] == "C":
                verdict = "Disputed scaling"
            elif isinstance(r.get("strict_null_fp_rate"), float) and r["strict_null_fp_rate"] > 0.20:
                verdict = "High FP risk"
            elif r.get("null_baseline_reported", False):
                verdict = "Properly controlled"
            else:
                verdict = "No null baseline"
            
            writer.writerow([
                r["year"], r["authors"], r["N"], r["method"], r["qubits"],
                r["cx_gates"], r["oracle_type"], r.get("t", "N/A"),
                true_order, strict_fp, exploratory_fp, ideal,
                "Yes" if r["null_baseline_reported"] else "No",
                "No" if r["category"] == "A" else "N/A",
                r["category"], verdict
            ])


if __name__ == "__main__":
    print("Computing scorecard for all published Shor's algorithm experiments...\n")
    results = compute_full_scorecard()
    print_scorecard(results)
    
    # Output directory next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "scorecard.csv")
    generate_csv_table(results, csv_path)
    print(f"\n\nCSV table saved to {csv_path}")
    
    # Save detailed JSON
    json_path = os.path.join(output_dir, "scorecard_detailed.json")
    # Make JSON serializable
    json_results = []
    for r in results:
        jr = {}
        for k, v in r.items():
            if k == "outcome_analysis":
                # Simplify for JSON
                jr[k] = [
                    {kk: vv for kk, vv in oa.items() if kk != "convergent_denominators"}
                    for oa in v
                ]
            else:
                jr[k] = v
        json_results.append(jr)
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Detailed JSON saved to {json_path}")