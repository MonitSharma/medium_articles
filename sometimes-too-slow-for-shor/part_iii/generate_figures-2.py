"""
Figure Generation for Blog 3: "The Fault in Our Qubits"
========================================================

Generates publication-quality figures for the scaling analysis blog post.
All figures use consistent styling and are saved as PNG at 300 DPI.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import math
import os

from resource_estimation import (
    full_resource_estimate, 
    PUBLISHED_EXPERIMENTS, 
    gap_analysis, 
    improvement_rate_analysis,
    gidney_ekera_rsa2048,
    gidney_2025_rsa2048,
    pinnacle_2026_rsa2048,
    shors_logical_resources_basic,
    shors_logical_resources_optimized,
    logical_error_rate,
)

# ============================================================
# Style configuration
# ============================================================

COLORS = {
    'primary': '#2563EB',       # Blue
    'secondary': '#DC2626',     # Red
    'accent': '#059669',        # Green
    'warning': '#D97706',       # Amber
    'purple': '#7C3AED',        # Purple
    'gray': '#6B7280',          # Gray
    'light_gray': '#E5E7EB',
    'dark': '#1F2937',
    'bg': '#FFFFFF',
    'current': '#3B82F6',       # Current state blue
    'required': '#EF4444',      # Required state red
    'gap': '#FEF3C7',           # Gap fill (light amber)
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def save_fig(fig, name, dpi=300):
    path = os.path.join(FIGDIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 1: The Exponential Wall - Resource Scaling
# ============================================================

def fig1_exponential_wall():
    """
    Log-scale plot showing how physical qubits, gate count, and runtime
    scale with the number being factored.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    bit_sizes = [4, 5, 8, 10, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    qubits = []
    gates = []
    times_hours = []
    
    for n in bit_sizes:
        est = full_resource_estimate(n, method='optimized')
        qubits.append(est['total_physical_qubits'])
        gates.append(est['toffoli_gates'])
        times_hours.append(est['runtime_hours'])
    
    # Panel 1: Physical qubits
    ax = axes[0]
    ax.semilogy(bit_sizes, qubits, 'o-', color=COLORS['primary'], 
                linewidth=2, markersize=6, zorder=5)
    ax.axhline(y=1121, color=COLORS['secondary'], linestyle='--', alpha=0.7, label='Current max scale (~1,121)')
    ax.axhline(y=20e6, color=COLORS['warning'], linestyle=':', alpha=0.7, label='Gidney-Ekerå 2021 target')
    ax.axhline(y=1e6, color=COLORS['accent'], linestyle=':', alpha=0.7, label='Gidney 2025 target')
    ax.fill_between([0, 5000], 1, 1121, alpha=0.08, color=COLORS['current'])
    ax.set_xlabel('Number Size (bits)')
    ax.set_ylabel('Physical Qubits Required')
    ax.set_title('Physical Qubits')
    ax.set_xlim(0, 4500)
    ax.legend(fontsize=8, loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.0e}'))
    
    # Panel 2: Gate count (Toffolis)
    ax = axes[1]
    ax.semilogy(bit_sizes, gates, 's-', color=COLORS['accent'], 
                linewidth=2, markersize=6, zorder=5)
    ax.axhline(y=100, color=COLORS['secondary'], linestyle='--', alpha=0.7, 
               label='Current reliable depth (~100)')
    ax.set_xlabel('Number Size (bits)')
    ax.set_ylabel('Toffoli Gate Count')
    ax.set_title('Toffoli Gates Required')
    ax.set_xlim(0, 4500)
    ax.legend(fontsize=8, loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.0e}'))
    
    # Panel 3: Runtime
    ax = axes[2]
    times_display = [max(t, 1e-6) for t in times_hours]
    ax.semilogy(bit_sizes, times_display, 'D-', color=COLORS['purple'], 
                linewidth=2, markersize=6, zorder=5)
    # Reference lines
    ax.axhline(y=1, color=COLORS['gray'], linestyle=':', alpha=0.5, label='1 hour')
    ax.axhline(y=24, color=COLORS['gray'], linestyle='--', alpha=0.5, label='1 day')
    ax.axhline(y=8760, color=COLORS['secondary'], linestyle='--', alpha=0.5, label='1 year')
    ax.set_xlabel('Number Size (bits)')
    ax.set_ylabel('Runtime (hours)')
    ax.set_title('Estimated Runtime')
    ax.set_xlim(0, 4500)
    ax.legend(fontsize=8, loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.0e}'))
    
    fig.suptitle('The Scaling Wall: Resource Growth for Shor\'s Algorithm', 
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig1_scaling_wall')


# ============================================================
# Figure 2: The Gap Visualization
# ============================================================

def fig2_the_gap():
    """
    Bar chart showing current capabilities vs RSA-2048 requirements
    across multiple dimensions, on log scale.
    
    Uses Gidney-Ekerå 2021 as the reference point for consistency.
    Circuit depth is measured in QEC rounds (~3 per Toffoli via lattice surgery),
    distinct from raw Toffoli count.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    gap_data = gap_analysis()
    
    dimensions = ['Physical\nQubits', 'QEC Round\nDepth', 'Number\nSize (bits)', 'Toffoli\nGate Count']
    
    ge = gidney_ekera_rsa2048()
    
    current_vals = [
        gap_data['current']['physical_qubits_scale'],  # 1121 (Condor)
        gap_data['current']['max_circuit_depth'],       # 100 operations
        gap_data['current']['largest_factored_bits'],   # 5
        40,                                              # gates in best toy experiment
    ]
    
    required_vals = [
        ge['physical_qubits'],      # 20M
        ge['qec_rounds'],           # ~8.1B QEC rounds (distinct from Toffoli count)
        2048,                       # bits
        ge['toffoli_gates'],        # 2.7B Toffolis
    ]
    
    x = np.arange(len(dimensions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, current_vals, width, label='Current Best (2024)', 
                   color=COLORS['current'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, required_vals, width, label='Required for RSA-2048\n(Gidney-Ekerå 2021)', 
                   color=COLORS['required'], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    ax.set_yscale('log')
    ax.set_ylabel('Value (log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_title('Mind the Gap: Current State vs. RSA-2048 Requirements (Gidney-Ekerå 2021)', 
                 fontsize=13, fontweight='bold')
    
    # Add gap annotations
    for i in range(len(dimensions)):
        ratio = required_vals[i] / current_vals[i]
        oom = math.log10(ratio)
        y_pos = max(current_vals[i], required_vals[i]) * 2
        ax.annotate(f'{oom:.1f} orders\nof magnitude', 
                    xy=(x[i], y_pos), ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=COLORS['dark'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['gap'], 
                             edgecolor=COLORS['warning'], alpha=0.9))
    
    ax.set_ylim(1, 1e12)
    fig.tight_layout()
    save_fig(fig, 'fig2_the_gap')


# ============================================================
# Figure 3: Error Correction Overhead
# ============================================================

def fig3_error_correction():
    """
    Shows how surface code distance and physical qubit overhead
    scale with target error rate and physical error rate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel 1: Logical error rate vs code distance for different physical error rates
    ax = axes[0]
    distances = list(range(3, 41, 2))
    
    for p_phys, color, label in [
        (1e-2, COLORS['secondary'], '$p = 10^{-2}$ (threshold)'),
        (5e-3, COLORS['warning'], '$p = 5 \\times 10^{-3}$'),
        (1e-3, COLORS['primary'], '$p = 10^{-3}$ (current best)'),
        (1e-4, COLORS['accent'], '$p = 10^{-4}$ (target)'),
    ]:
        p_L = [logical_error_rate(p_phys, d) for d in distances]
        ax.semilogy(distances, p_L, 'o-', color=color, label=label, 
                    markersize=4, linewidth=2)
    
    ax.axhline(y=1e-15, color=COLORS['gray'], linestyle=':', alpha=0.5)
    ax.text(35, 2e-15, 'RSA-2048\ntarget', ha='center', fontsize=8, color=COLORS['gray'])
    ax.set_xlabel('Surface Code Distance $d$')
    ax.set_ylabel('Logical Error Rate per Round')
    ax.set_title('Error Suppression vs Code Distance')
    ax.legend(fontsize=9)
    ax.set_ylim(1e-30, 1)
    
    # Panel 2: Physical qubits per logical qubit vs code distance
    ax = axes[1]
    phys_per_logical = [2 * d * d for d in distances]
    
    ax.plot(distances, phys_per_logical, 's-', color=COLORS['purple'], 
            linewidth=2, markersize=5)
    
    # Annotate key points
    key_distances = [7, 13, 17, 27, 31]
    for d in key_distances:
        n_phys = 2 * d * d
        ax.annotate(f'd={d}\n{n_phys:,} phys/logical', 
                    xy=(d, n_phys), fontsize=8,
                    xytext=(10, 15), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray'], alpha=0.6),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', 
                             edgecolor=COLORS['gray'], alpha=0.8))
    
    ax.set_xlabel('Surface Code Distance $d$')
    ax.set_ylabel('Physical Qubits per Logical Qubit')
    ax.set_title('Error Correction Overhead')
    
    fig.suptitle('The Error Correction Tax: More Protection = More Qubits', 
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig3_error_correction')


# ============================================================
# Figure 4: Historical Progress & Projection
# ============================================================

def fig4_timeline():
    """
    Historical qubit counts and error rates with extrapolation
    to RSA-2048 requirements.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    timeline = improvement_rate_analysis()
    
    # Panel 1: Qubit count trajectory
    ax = axes[0]
    
    # Plot achieved points
    ach_years = [h[0] for h in timeline['achieved']]
    ach_qubits = [h[1] for h in timeline['achieved']]
    ach_labels = [h[2] for h in timeline['achieved']]
    
    ax.semilogy(ach_years, ach_qubits, 'o-', color=COLORS['primary'], 
                linewidth=2, markersize=8, zorder=5, label='Achieved')
    
    # Plot planned/roadmap points distinctly
    if timeline['planned']:
        plan_years = [h[0] for h in timeline['planned']]
        plan_qubits = [h[1] for h in timeline['planned']]
        plan_labels = [h[2] for h in timeline['planned']]
        ax.semilogy(plan_years, plan_qubits, 'D', color=COLORS['warning'], 
                    markersize=10, zorder=5, label='Roadmap / Planned',
                    markeredgecolor='white', markeredgewidth=1.5)
    
    # Extrapolation (fitted to achieved only)
    future_years = np.arange(2024, 2056)
    coeffs = timeline['fit_coeffs']
    base_year = timeline['fit_base_year']
    projected = np.exp(coeffs[0] * (future_years - base_year) + coeffs[1])
    ax.semilogy(future_years, projected, '--', color=COLORS['primary'], 
                alpha=0.4, linewidth=2, label='Extrapolation (achieved only)')
    
    # Target lines — both 2021 and 2025 estimates
    ax.axhline(y=20e6, color=COLORS['secondary'], linestyle='-', linewidth=2, alpha=0.7)
    ax.text(2055, 25e6, 'Gidney-Ekerå 2021: 20M', ha='right', fontsize=9,
            color=COLORS['secondary'], fontweight='bold')
    
    ax.axhline(y=1e6, color=COLORS['accent'], linestyle='-', linewidth=2, alpha=0.7)
    ax.text(2055, 1.3e6, 'Gidney 2025: <1M', ha='right', fontsize=9,
            color=COLORS['accent'], fontweight='bold')
    
    ax.axhline(y=1e5, color=COLORS['purple'], linestyle='--', linewidth=1.5, alpha=0.6)
    ax.text(2055, 1.3e5, 'Pinnacle 2026: <100K', ha='right', fontsize=8,
            color=COLORS['purple'])
    
    # Cross-year annotation for Gidney 2025 target
    cross_year_2025 = timeline['projected_year_qubits_2025']
    ax.axvline(x=cross_year_2025, color=COLORS['accent'], linestyle=':', alpha=0.5)
    ax.text(cross_year_2025, 5, f'~{cross_year_2025:.0f}', ha='center', fontsize=9, 
            color=COLORS['accent'], fontweight='bold')
    
    # Label some achieved data points
    for i in [0, 3, 5, 6, 7]:
        if i < len(ach_years):
            offset = (10, 10) if i % 2 == 0 else (10, -20)
            ax.annotate(ach_labels[i], xy=(ach_years[i], ach_qubits[i]), fontsize=7,
                       xytext=offset, textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray'], alpha=0.4))
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Physical Qubit Count')
    ax.set_title('Qubit Count Trajectory')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(2015, 2057)
    ax.set_ylim(1, 1e9)
    
    # Panel 2: Error rate trajectory
    ax = axes[1]
    error_years = [h[0] for h in timeline['error_history']]
    error_rates = [h[1] for h in timeline['error_history']]
    
    ax.semilogy(error_years, error_rates, 's-', color=COLORS['accent'], 
                linewidth=2, markersize=8, zorder=5, label='Best reported 2Q error')
    
    # Extrapolation
    future_years_e = np.arange(2025, 2051)
    e_coeffs = np.polyfit(np.array(error_years) - error_years[0], np.log(error_rates), 1)
    projected_errors = np.exp(e_coeffs[0] * (future_years_e - error_years[0]) + e_coeffs[1])
    ax.semilogy(future_years_e, projected_errors, '--', color=COLORS['accent'], 
                alpha=0.4, linewidth=2, label='Extrapolation')
    
    # Threshold line
    ax.axhline(y=1e-3, color=COLORS['warning'], linestyle='--', alpha=0.7)
    ax.text(2050, 1.3e-3, 'Gidney-Ekerå assumption: $10^{-3}$', ha='right', fontsize=9,
            color=COLORS['warning'])
    
    ax.axhline(y=1e-4, color=COLORS['secondary'], linestyle='-', alpha=0.5)
    ax.text(2050, 1.3e-4, 'More comfortable: $10^{-4}$', ha='right', fontsize=9,
            color=COLORS['secondary'])
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Two-Qubit Gate Error Rate')
    ax.set_title('Error Rate Trajectory')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(2015, 2052)
    ax.set_ylim(1e-7, 0.1)
    
    fig.suptitle('When Might We Get There? Historical Trends and (Optimistic) Extrapolation', 
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig4_timeline')


# ============================================================
# Figure 5: What Toy Experiments Actually Demonstrate
# ============================================================

def fig5_what_they_show():
    """
    A clear visualization of what published experiments actually demonstrate
    vs what would be needed for meaningful factoring.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Published experiments
    experiments = PUBLISHED_EXPERIMENTS
    
    exp_names = [e['paper'].split('(')[0].strip() for e in experiments]
    exp_qubits = [e['qubits_used'] for e in experiments]
    exp_gates = [e['gates'] for e in experiments]
    exp_years = [e['year'] for e in experiments]
    
    # Scatter: x = qubits, y = gates, size = year (larger = more recent)
    sizes = [(y - 1998) * 15 for y in exp_years]
    scatter = ax.scatter(exp_qubits, exp_gates, s=sizes, c=exp_years,
                        cmap='Blues', edgecolors=COLORS['primary'], 
                        linewidth=1.5, zorder=5, alpha=0.8,
                        vmin=2000, vmax=2025)
    
    # Label each point
    offsets = [(15, 10), (15, -15), (15, 10), (-20, 15), (15, 10)]
    for i, (name, x, y) in enumerate(zip(exp_names, exp_qubits, exp_gates)):
        ax.annotate(f'{name}\n({experiments[i]["year"]})', 
                    xy=(x, y), fontsize=8,
                    xytext=offsets[i % len(offsets)], textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray'], alpha=0.5),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', 
                             edgecolor=COLORS['gray'], alpha=0.7))
    
    # Mark the RSA-2048 targets (multiple estimates)
    ax.scatter([20e6], [2.7e9], s=400, c=COLORS['secondary'], marker='*', 
              zorder=6, edgecolors='darkred', linewidth=1)
    ax.annotate('RSA-2048\n(G-E 2021:\n20M qubits)', xy=(20e6, 2.7e9), fontsize=9,
                fontweight='bold', color=COLORS['secondary'],
                xytext=(-60, -45), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2))
    
    # Gidney 2025 target
    ax.scatter([1e6], [6.5e9], s=300, c=COLORS['accent'], marker='*', 
              zorder=6, edgecolors='darkgreen', linewidth=1)
    ax.annotate('RSA-2048\n(Gidney 2025:\n<1M qubits)', xy=(1e6, 6.5e9), fontsize=9,
                fontweight='bold', color=COLORS['accent'],
                xytext=(-80, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))
    
    # Mark intermediate targets
    for n_bits, label, marker_color in [
        (16, '16-bit\nfactoring', COLORS['warning']),
        (64, '64-bit\nfactoring', COLORS['purple']),
        (256, '256-bit\nfactoring', COLORS['accent']),
    ]:
        est = full_resource_estimate(n_bits, method='optimized')
        ax.scatter([est['total_physical_qubits']], [est['toffoli_gates']], 
                  s=150, c=marker_color, marker='D', zorder=5, alpha=0.7,
                  edgecolors='white', linewidth=1)
        ax.annotate(label, xy=(est['total_physical_qubits'], est['toffoli_gates']),
                   fontsize=8, color=marker_color,
                   xytext=(15, 5), textcoords='offset points')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical Qubits', fontsize=12)
    ax.set_ylabel('Gate Count', fontsize=12)
    ax.set_title('Published Experiments vs. Meaningful Factoring Requirements', 
                 fontsize=14, fontweight='bold')
    
    # Draw the "meaningful factoring frontier"
    ax.plot([100, 1e8], [1e4, 1e12], '--', color=COLORS['gray'], alpha=0.3, linewidth=1)
    ax.text(5e4, 3e8, 'Classical simulation\nboundary (approx.)', 
            rotation=35, fontsize=8, color=COLORS['gray'], alpha=0.6,
            style='italic')
    
    ax.set_xlim(1, 1e9)
    ax.set_ylim(5, 1e11)
    
    plt.colorbar(scatter, ax=ax, label='Year', shrink=0.8)
    
    fig.tight_layout()
    save_fig(fig, 'fig5_what_they_show')


# ============================================================
# Figure 6: The Honest Scorecard
# ============================================================

def fig6_honest_scorecard():
    """
    Radar/spider chart showing current quantum hardware capabilities
    as a fraction of what's needed for RSA-2048.
    
    NOTE: Several axes use editorial estimates rather than computed values.
    The figure is labeled as illustrative accordingly.
    """
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    categories = [
        'Qubit Count\n(log ratio)',
        'Gate Fidelity\n(log ratio)', 
        'Circuit Depth\n(log ratio)',
        'Connectivity\n(estimated)',
        'Error Correction\n(estimated)',
        'Coherence Time\n(log ratio)',
    ]
    
    # Current capability as fraction of RSA-2048 requirement
    # Where computable, we use log10(current/required) mapped to [0, 1] where 1 = sufficient
    # Where not directly computable, we note the value is an editorial estimate
    current_fractions = [
        math.log10(1121) / math.log10(20e6),           # Qubits: log(1121)/log(20M) ≈ 0.42
        0.85,                                            # Gate fidelity: editorial (close to threshold)
        math.log10(100) / math.log10(8.1e9),            # Depth: log(100)/log(8.1B QEC rounds) ≈ 0.20
        0.15,                                            # Connectivity: editorial (far from full)
        0.05,                                            # Error correction: editorial (barely started)
        math.log10(300) / math.log10(1e8),              # Coherence: log(300µs)/log(100s effective) ≈ 0.31
    ]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    current_fractions += current_fractions[:1]
    
    # Required (all 1.0)
    required = [1.0] * N + [1.0]
    
    # Plot
    ax.plot(angles, required, '-', color=COLORS['secondary'], linewidth=2, alpha=0.5)
    ax.fill(angles, required, alpha=0.05, color=COLORS['secondary'])
    
    ax.plot(angles, current_fractions, 'o-', color=COLORS['primary'], 
            linewidth=2.5, markersize=8)
    ax.fill(angles, current_fractions, alpha=0.15, color=COLORS['primary'])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    
    # Set y-axis
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color=COLORS['gray'])
    
    # Legend
    ax.plot([], [], '-', color=COLORS['primary'], linewidth=2.5, label='Current State (2024)')
    ax.plot([], [], '-', color=COLORS['secondary'], linewidth=2, alpha=0.5, label='RSA-2048 Requirement')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    ax.set_title('An Illustrative Scorecard:\nCurrent Hardware vs. RSA-2048 Requirements', 
                 fontsize=14, fontweight='bold', pad=30)
    
    fig.tight_layout()
    save_fig(fig, 'fig6_honest_scorecard')


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("Generating figures for Blog 3: 'The Fault in Our Qubits'")
    print("=" * 60)
    
    print("\n[1/6] The Scaling Wall...")
    fig1_exponential_wall()
    
    print("\n[2/6] The Gap...")
    fig2_the_gap()
    
    print("\n[3/6] Error Correction Overhead...")
    fig3_error_correction()
    
    print("\n[4/6] Historical Timeline & Projection...")
    fig4_timeline()
    
    print("\n[5/6] What Experiments Actually Show...")
    fig5_what_they_show()
    
    print("\n[6/6] The Honest Scorecard...")
    fig6_honest_scorecard()
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {FIGDIR}")
