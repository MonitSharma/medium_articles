"""
Resource Estimation Engine for Shor's Algorithm at Scale
=========================================================

Computes physical resource requirements for factoring n-bit integers
using Shor's algorithm with surface code quantum error correction.

Based on resource estimates from:
- Gidney & Ekerå (2021): "How to factor 2048 bit RSA integers in 8 hours"
- Litinski (2019): "How to compute a 256-bit elliptic curve private key with only 14 magic states"  
- Webster et al. (2015): "Reducing the overhead for quantum computation when noise is low"
- Fowler et al. (2012): "Surface codes: Towards practical large-scale quantum computation"
- Häner et al. (2017): "Factoring using 2n+2 qubits with Toffoli based modular multiplication"

This script provides a transparent, reproducible resource model for scaling analysis.
"""

import json
import math
import os
import numpy as np

# Base directory: same folder as this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Physical parameters (state of the art, ~2024-2025)
# ============================================================

# Current best two-qubit gate error rates on superconducting hardware
CURRENT_2Q_ERROR_RATE = 1e-3       # ~0.1% (IBM Heron, Google Willow ballpark)
CURRENT_1Q_ERROR_RATE = 1e-4       # ~0.01%
CURRENT_MEASUREMENT_ERROR = 5e-3   # ~0.5%
CURRENT_T1_US = 300                # T1 in microseconds
CURRENT_T2_US = 200                # T2 in microseconds
CURRENT_GATE_TIME_NS = 60          # Two-qubit gate time in nanoseconds
CURRENT_MEAS_TIME_NS = 600         # Measurement time in nanoseconds
CURRENT_CYCLE_TIME_US = 1.0        # QEC cycle time in microseconds

# Surface code parameters
SURFACE_CODE_THRESHOLD = 1e-2      # ~1% threshold for surface code


def logical_error_rate(p_phys, d):
    """
    Surface code logical error rate per QEC round.
    
    p_L ≈ 0.1 * (100 * p_phys)^((d+1)/2)
    
    Valid when p_phys << threshold (~1%).
    """
    if p_phys >= SURFACE_CODE_THRESHOLD:
        return 1.0  # Above threshold, no error suppression
    return 0.1 * (100 * p_phys) ** ((d + 1) / 2)


def required_code_distance(p_phys, target_logical_error, num_logical_qubits, num_rounds):
    """
    Compute minimum surface code distance to achieve target total failure probability.
    
    We need: num_logical_qubits * num_rounds * p_L(d) < target_logical_error
    So: p_L(d) < target / (n_qubits * n_rounds)
    """
    p_L_target = target_logical_error / (num_logical_qubits * num_rounds)
    
    for d in range(3, 101, 2):  # odd distances only
        if logical_error_rate(p_phys, d) < p_L_target:
            return d
    return 101  # Couldn't achieve target


def physical_qubits_per_logical(d):
    """
    Number of physical qubits per logical qubit in a rotated surface code.
    Includes data qubits and syndrome qubits.
    
    For a distance-d rotated surface code: 2d^2 - 1 data+syndrome qubits
    Plus routing/ancilla overhead: we use ~2d^2 total per logical qubit.
    """
    return 2 * d * d


# ============================================================
# Logical resource estimates for Shor's algorithm
# ============================================================

def shors_logical_resources_basic(n_bits):
    """
    Basic (textbook) Shor's algorithm resource estimates.
    Uses 2n+3 logical qubits, O(n^3) Toffoli gates.
    
    Based on Beauregard (2003) circuit construction.
    """
    n_logical = 2 * n_bits + 3
    
    # Modular exponentiation requires O(n^2) modular multiplications,
    # each requiring O(n) Toffoli gates → O(n^3) total
    n_toffoli = 40 * n_bits ** 3  # Empirical constant from circuit analysis
    
    # Each Toffoli decomposes to ~7 T gates in Clifford+T
    n_t_gates = 7 * n_toffoli
    
    # Circuit depth (sequential): O(n^3) for controlled modular exponentiation
    circuit_depth = 50 * n_bits ** 3
    
    return {
        'method': 'Beauregard (textbook)',
        'n_bits': n_bits,
        'logical_qubits': n_logical,
        'toffoli_gates': n_toffoli,
        't_gates': n_t_gates,
        'circuit_depth': circuit_depth,
    }


def shors_logical_resources_optimized(n_bits):
    """
    Optimized Shor's algorithm resource estimates.
    Based on Gidney & Ekerå (2021) - windowed arithmetic + measurement-based uncomputation.
    
    Key improvements:
    - Uses ~2n + O(n/log n) logical qubits
    - Reduces Toffoli count to O(n^2 * log n) via windowed multiplication
    - Reduces depth via parallelism
    """
    # Gidney-Ekerå specific numbers for RSA-2048 (n=2048):
    # ~20 million noisy qubits, ~8 hours, using 3n + 0.002n^2 logical qubits
    # But the key scaling: ~0.3 * n^2 * log2(n) Toffoli gates
    
    n_logical = 3 * n_bits + max(1, int(0.002 * n_bits ** 2))
    
    # Windowed modular exponentiation: O(n^2 * log n)
    n_toffoli = int(0.3 * n_bits ** 2 * math.log2(max(2, n_bits)))
    
    # Toffoli → T gate decomposition (catalyzed): ~4 T gates per Toffoli
    n_t_gates = 4 * n_toffoli
    
    # Depth with parallelism
    circuit_depth = int(2 * n_bits ** 2 * math.log2(max(2, n_bits)))
    
    return {
        'method': 'Gidney-Ekerå (optimized)',
        'n_bits': n_bits,
        'logical_qubits': n_logical,
        'toffoli_gates': n_toffoli,
        't_gates': n_t_gates,
        'circuit_depth': circuit_depth,
    }


def gidney_ekera_rsa2048():
    """
    Exact numbers from Gidney & Ekerå (2021) for RSA-2048.
    This was the standard reference until 2025.
    """
    return {
        'method': 'Gidney-Ekerå (2021)',
        'n_bits': 2048,
        'logical_qubits': 4099,  # 2n+3 with optimizations
        'toffoli_gates': int(2.7e9),  # 2.7 billion
        't_gates': int(1.08e10),  # ~10.8 billion
        'physical_qubits': int(20e6),  # 20 million
        'runtime_hours': 8,
        'code_distance': 27,
        'qec_rounds': int(2.7e9 * 3),  # ~3 QEC rounds per Toffoli (lattice surgery)
        'assumed_physical_error_rate': 1e-3,
    }


def gidney_2025_rsa2048():
    """
    Updated numbers from Gidney (2025): "How to factor 2048 bit RSA integers
    with less than a million noisy qubits" (arXiv:2505.15917).
    
    20x fewer qubits than 2021, trading time for space.
    Same hardware assumptions as 2021.
    """
    return {
        'method': 'Gidney (2025)',
        'n_bits': 2048,
        'logical_qubits': 1399,
        'toffoli_gates': int(6.5e9),  # ~6.5 billion (more than 2021 due to space-time tradeoff)
        'physical_qubits': int(1e6),  # < 1 million
        'runtime_hours': 5 * 24,      # < 1 week
        'assumed_physical_error_rate': 1e-3,
    }


def pinnacle_2026_rsa2048():
    """
    Numbers from Webster et al. (2026): "The Pinnacle Architecture"
    (arXiv:2602.11457). Uses QLDPC codes instead of surface codes.
    
    Further 10x reduction over Gidney 2025, but requires QLDPC hardware
    that does not yet exist.
    """
    return {
        'method': 'Pinnacle / Webster et al. (2026)',
        'n_bits': 2048,
        'physical_qubits': int(1e5),  # < 100,000
        'runtime_hours': 30 * 24,     # ~1 month
        'assumed_physical_error_rate': 1e-3,
        'note': 'Requires QLDPC codes; not yet demonstrated in hardware',
    }


# ============================================================
# Full physical resource estimation
# ============================================================

def full_resource_estimate(n_bits, p_phys=CURRENT_2Q_ERROR_RATE, target_success=0.99,
                           method='optimized'):
    """
    Complete physical resource estimate for factoring an n-bit number.
    
    Returns dict with logical qubits, physical qubits, code distance,
    estimated runtime, and all intermediate quantities.
    """
    if method == 'optimized':
        logical = shors_logical_resources_optimized(n_bits)
    else:
        logical = shors_logical_resources_basic(n_bits)
    
    n_logical = logical['logical_qubits']
    depth = logical['circuit_depth']
    
    # Target logical error: we want overall success probability > target_success
    target_logical_error = 1 - target_success
    
    # Code distance needed
    d = required_code_distance(p_phys, target_logical_error, n_logical, depth)
    
    # Physical qubits
    phys_per_logical = physical_qubits_per_logical(d)
    
    # Magic state distillation factories add ~20-50% overhead
    factory_overhead = 1.3  # 30% additional qubits for T-factories
    
    total_physical = int(n_logical * phys_per_logical * factory_overhead)
    
    # Runtime estimate
    # Each QEC round takes ~d * cycle_time 
    # Total QEC rounds ≈ circuit_depth
    qec_round_time_us = d * CURRENT_CYCLE_TIME_US
    total_time_us = depth * qec_round_time_us
    total_time_hours = total_time_us / (3600 * 1e6)
    
    result = {
        **logical,
        'physical_error_rate': p_phys,
        'code_distance': d,
        'physical_per_logical': phys_per_logical,
        'total_physical_qubits': total_physical,
        'runtime_hours': total_time_hours,
        'runtime_days': total_time_hours / 24,
        'runtime_years': total_time_hours / (24 * 365.25),
        'target_success_prob': target_success,
    }
    
    return result


# ============================================================
# Published experimental parameters for comparison
# ============================================================

PUBLISHED_EXPERIMENTS = [
    {
        'paper': 'Vandersypen et al. (2001)',
        'N': 15,
        'n_bits': 4,
        'qubits_used': 7,
        'platform': 'NMR (7-qubit)',
        'gates': 300,   # approximate
        'depth': 300,    # approximate (NMR, no parallelism)
        'year': 2001,
    },
    {
        'paper': 'Lucero et al. (2012)',
        'N': 15,
        'n_bits': 4,
        'qubits_used': 3,
        'platform': 'Superconducting',
        'gates': 15,     # compiled/simplified circuit
        'depth': 10,
        'year': 2012,
    },
    {
        'paper': 'Monz et al. (2016)',
        'N': 15,
        'n_bits': 4,
        'qubits_used': 5,
        'platform': 'Trapped ions',
        'gates': 200,    # approximate
        'depth': 100,
        'year': 2016,
    },
    {
        'paper': 'Amico et al. (2019)',
        'N': 15,
        'n_bits': 4,
        'qubits_used': 7,
        'platform': 'IBM superconducting',
        'gates': 80,     # highly compiled
        'depth': 40,
        'year': 2019,
    },
    {
        'paper': 'Skosana & Tame (2021)',
        'N': 21,
        'n_bits': 5,
        'qubits_used': 5,
        'platform': 'IBM superconducting',
        'gates': 40,     # heavily simplified
        'depth': 20,
        'year': 2021,
    },
]


# RSA key sizes for analysis
RSA_TARGETS = [
    {'name': 'RSA-256', 'bits': 256},
    {'name': 'RSA-512', 'bits': 512},
    {'name': 'RSA-768', 'bits': 768},
    {'name': 'RSA-1024', 'bits': 1024},
    {'name': 'RSA-2048', 'bits': 2048},
    {'name': 'RSA-4096', 'bits': 4096},
]


def compute_all_estimates():
    """Compute resource estimates for all target sizes."""
    results = []
    
    bit_sizes = [4, 5, 8, 10, 16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096]
    
    for n_bits in bit_sizes:
        est_basic = full_resource_estimate(n_bits, method='basic')
        est_opt = full_resource_estimate(n_bits, method='optimized')
        results.append({
            'n_bits': n_bits,
            'N_approx': f'~2^{n_bits}',
            'basic': est_basic,
            'optimized': est_opt,
        })
    
    return results


def improvement_rate_analysis():
    """
    Analyze historical improvement rates in quantum hardware
    and project when RSA-2048 factoring might become feasible.
    """
    # Historical qubit counts — separating achieved from planned/roadmap
    # 'achieved' means the chip existed and was publicly benchmarked
    # 'planned' means announced on a roadmap but not yet demonstrated at scale
    achieved = [
        (2016, 9, 'Google', True),
        (2017, 20, 'IBM', True),
        (2018, 72, 'Google Bristlecone', True),
        (2019, 53, 'Google Sycamore', True),
        (2020, 65, 'IBM Hummingbird', True),
        (2021, 127, 'IBM Eagle', True),
        (2022, 433, 'IBM Osprey', True),
        (2023, 1121, 'IBM Condor', True),
    ]
    
    planned = [
        (2025, 4158, 'IBM Flamingo (roadmap)', False),
    ]
    
    history = achieved + planned
    
    years = np.array([h[0] for h in achieved])
    qubits = np.array([h[1] for h in achieved])
    
    # Fit exponential growth using ONLY achieved systems
    log_qubits = np.log(qubits)
    coeffs = np.polyfit(years - years[0], log_qubits, 1)
    growth_rate = coeffs[0]
    doubling_time = np.log(2) / growth_rate
    
    # Error rate improvement history (approximate)
    error_history = [
        (2016, 5e-2),    # ~5% two-qubit errors
        (2018, 1e-2),    # ~1%
        (2020, 5e-3),    # ~0.5%
        (2022, 2e-3),    # ~0.2%
        (2024, 5e-4),    # ~0.05% (best reported)
    ]
    
    error_years = np.array([h[0] for h in error_history])
    error_rates = np.array([h[1] for h in error_history])
    
    log_errors = np.log(error_rates)
    error_coeffs = np.polyfit(error_years - error_years[0], log_errors, 1)
    error_improvement_rate = -error_coeffs[0]
    error_halving_time = np.log(2) / error_improvement_rate
    
    # Project when we might hit targets at needed error rates
    # Use Gidney 2025 target (<1M qubits) as the more current reference
    target_qubits_2021 = 20e6
    target_qubits_2025 = 1e6
    years_to_target_qubits_2021 = (np.log(target_qubits_2021) - coeffs[1]) / coeffs[0] + years[0]
    years_to_target_qubits_2025 = (np.log(target_qubits_2025) - coeffs[1]) / coeffs[0] + years[0]
    
    target_error = 1e-4  # Below which surface code overhead becomes manageable
    years_to_target_error = (np.log(target_error) - error_coeffs[1]) / error_coeffs[0] + error_years[0]
    
    return {
        'achieved': achieved,
        'planned': planned,
        'qubit_history': history,
        'qubit_doubling_time_years': doubling_time,
        'error_history': error_history,
        'error_halving_time_years': error_halving_time,
        'projected_year_qubits_2021': years_to_target_qubits_2021,
        'projected_year_qubits_2025': years_to_target_qubits_2025,
        'projected_year_errors': years_to_target_error,
        'fit_coeffs': coeffs,
        'fit_base_year': years[0],
    }


def gap_analysis():
    """
    Compute the quantitative gap between current capabilities and RSA-2048 factoring
    across multiple dimensions.
    
    Note: We use Gidney-Ekerå 2021 as the primary reference point for consistency,
    but also report the Gidney 2025 and Pinnacle 2026 estimates.
    """
    # Current state of the art
    # Note: We distinguish high-performance chips (~100-156 qubits, e.g. IBM Heron)
    # from largest-scale chips (~1000+ qubits, e.g. IBM Condor, which trades 
    # qubit count for connectivity/fidelity). We report the performance-oriented
    # range here.
    current = {
        'physical_qubits_performance': 156,  # IBM Heron (performance-oriented)
        'physical_qubits_scale': 1121,        # IBM Condor (scale-oriented)
        'best_2q_error': 5e-4,               # Best reported on select qubit pairs
        'typical_2q_error': 1e-3,
        'max_circuit_depth': 100,             # Before decoherence kills signal
        'largest_factored_N': 21,             # Skosana & Tame (though vacuous)
        'largest_factored_bits': 5,
        'coherence_time_us': 300,
    }
    
    # Required for RSA-2048 — using Gidney-Ekerå 2021 as reference
    required_2021 = gidney_ekera_rsa2048()
    required_2025 = gidney_2025_rsa2048()
    required_2026 = pinnacle_2026_rsa2048()
    
    # For circuit depth: the actual sequential depth in QEC rounds
    # Each Toffoli requires ~3 QEC rounds via lattice surgery
    required_qec_depth = required_2021['qec_rounds']
    
    # Compute gap ratios (using 2021 reference and scale-oriented qubit count)
    gaps = {
        'qubits': {
            'current': current['physical_qubits_scale'],
            'required_2021': required_2021['physical_qubits'],
            'required_2025': required_2025['physical_qubits'],
            'required_2026': required_2026['physical_qubits'],
            'ratio_2021': required_2021['physical_qubits'] / current['physical_qubits_scale'],
            'ratio_2025': required_2025['physical_qubits'] / current['physical_qubits_scale'],
            'ratio_2026': required_2026['physical_qubits'] / current['physical_qubits_scale'],
        },
        'qec_depth': {
            'current_max': current['max_circuit_depth'],
            'required': required_qec_depth,
            'ratio': required_qec_depth / current['max_circuit_depth'],
            'orders_of_magnitude': math.log10(required_qec_depth / current['max_circuit_depth']),
        },
        'number_size': {
            'current_bits': current['largest_factored_bits'],
            'target_bits': 2048,
            'ratio': 2048 / current['largest_factored_bits'],
        },
        'toffoli_gates': {
            'current_max': 40,  # gates in best toy experiment
            'required': required_2021['toffoli_gates'],
            'ratio': required_2021['toffoli_gates'] / 40,
            'orders_of_magnitude': math.log10(required_2021['toffoli_gates'] / 40),
        },
    }
    
    return {
        'current': current,
        'required_2021': required_2021,
        'required_2025': required_2025,
        'required_2026': required_2026,
        'gaps': gaps,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("RESOURCE ESTIMATES FOR QUANTUM FACTORING")
    print("=" * 70)
    
    # Scaling table
    print("\n--- Scaling Analysis (Optimized Shor's, p_phys=0.001) ---\n")
    print(f"{'n_bits':>8} {'Logical Q':>12} {'Toffoli':>15} {'Code Dist':>10} {'Phys Q':>15} {'Time':>15}")
    print("-" * 80)
    
    for n_bits in [4, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        est = full_resource_estimate(n_bits, method='optimized')
        
        if est['runtime_years'] > 1:
            time_str = f"{est['runtime_years']:.1f} years"
        elif est['runtime_hours'] > 1:
            time_str = f"{est['runtime_hours']:.1f} hours"
        else:
            time_str = f"{est['runtime_hours']*60:.1f} min"
        
        print(f"{n_bits:>8} {est['logical_qubits']:>12,} {est['toffoli_gates']:>15,} "
              f"{est['code_distance']:>10} {est['total_physical_qubits']:>15,} {time_str:>15}")
    
    # Gap analysis
    print("\n\n--- Gap Analysis: Current vs RSA-2048 ---\n")
    gaps = gap_analysis()
    print("  Using Gidney-Ekerå 2021 as reference:")
    for dim, data in gaps['gaps'].items():
        if 'ratio' in data:
            print(f"    {dim}: {data['ratio']:,.0f}x gap", end="")
            if 'orders_of_magnitude' in data:
                print(f" ({data['orders_of_magnitude']:.1f} orders of magnitude)")
            else:
                print()
        elif 'ratio_2021' in data:
            print(f"    {dim}: {data['ratio_2021']:,.0f}x (2021), {data['ratio_2025']:,.0f}x (2025), {data['ratio_2026']:,.0f}x (2026)")
    
    # Timeline
    print("\n\n--- Improvement Rate Analysis ---\n")
    timeline = improvement_rate_analysis()
    print(f"  Qubit count doubling time: {timeline['qubit_doubling_time_years']:.1f} years")
    print(f"  Error rate halving time:   {timeline['error_halving_time_years']:.1f} years")
    print(f"  Projected year (20M qubits, 2021 est): {timeline['projected_year_qubits_2021']:.0f}")
    print(f"  Projected year (1M qubits, 2025 est):  {timeline['projected_year_qubits_2025']:.0f}")
    print(f"  Projected year (errors):   {timeline['projected_year_errors']:.0f}")
    
    # Save results
    all_estimates = compute_all_estimates()
    output = {
        'gap_analysis_2021': {
            'current_qubits_scale': gaps['current']['physical_qubits_scale'],
            'current_qubits_performance': gaps['current']['physical_qubits_performance'],
            'required_2021': gaps['required_2021']['physical_qubits'],
            'required_2025': gaps['required_2025']['physical_qubits'],
            'required_2026': gaps['required_2026']['physical_qubits'],
        },
        'timeline': {
            'qubit_doubling_years': timeline['qubit_doubling_time_years'],
            'error_halving_years': timeline['error_halving_time_years'],
            'projected_year_2021': timeline['projected_year_qubits_2021'],
            'projected_year_2025': timeline['projected_year_qubits_2025'],
        },
    }
    
    output_dir = os.path.join(_SCRIPT_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'resource_estimates.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n\nResults saved to {output_path}")
