"""
Quantum Key Distribution (QKD) - Blog Post Figures & Simulations
=================================================================
Accompanies: "The Key to the Quantum Garden"

Generates:
  1. BB84 protocol simulation with and without eavesdropper
  2. QBER vs eavesdropping fraction
  3. Secure key rate vs distance (fiber loss model)
  4. Global QKD network comparison (deployment scale)
  5. Attack taxonomy visualization
  6. India's NQM timeline and progress

References:
  - Bennett & Brassard, 1984 (BB84 protocol)
  - Scarani et al., Rev. Mod. Phys. 81, 1301 (2009)
  - Chen et al., Nature 589, 214 (2021) - China 4600 km network
  - Pirandola et al., Nat. Commun. 8, 15043 (2017) - PLOB bound
  - Lucamarini et al., Nature 557, 400 (2018) - TF-QKD
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#0d1117',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'font.family': 'monospace',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
})

COLORS = {
    'cyan': '#58a6ff',
    'green': '#3fb950',
    'red': '#f85149',
    'orange': '#d29922',
    'purple': '#bc8cff',
    'pink': '#f778ba',
    'yellow': '#e3b341',
    'white': '#c9d1d9',
    'dim': '#8b949e',
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('QKD_FIGURES_OUT', os.path.join(SCRIPT_DIR, 'figures'))
os.makedirs(OUT, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# FIGURE 1: BB84 Protocol Simulation
# ══════════════════════════════════════════════════════════════════

def simulate_bb84(n_bits=10000, eve_present=False, eve_fraction=1.0):
    """
    Simulate BB84 with optional eavesdropper.
    
    Parameters:
        n_bits: number of raw qubits sent
        eve_present: whether Eve intercept-resends
        eve_fraction: fraction of qubits Eve intercepts (0-1)
    
    Returns dict with QBER, sifted key length, etc.
    """
    # Alice prepares random bits and bases
    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)  # 0=rectilinear(+), 1=diagonal(×)
    
    # Eve intercept-resend (if present)
    if eve_present:
        eve_bases = np.random.randint(0, 2, n_bits)
        eve_intercepts = np.random.random(n_bits) < eve_fraction
        
        # Eve measures in her basis → collapses state
        # If Eve's basis matches Alice's, she gets the correct bit
        # If not, she gets a random bit and re-sends in wrong state
        eve_bits = np.where(
            eve_bases == alice_bases,
            alice_bits,  # correct measurement
            np.random.randint(0, 2, n_bits)  # random result
        )
        
        # What Bob receives: original if Eve didn't intercept,
        # Eve's re-prepared state if she did
        transmitted_bits = np.where(eve_intercepts, eve_bits, alice_bits)
        transmitted_bases = np.where(eve_intercepts, eve_bases, alice_bases)
        # Actually Bob measures what Eve sent. If Eve used wrong basis,
        # the state is disturbed even when Bob uses Alice's basis.
        # Probability of error when Eve intercepts with wrong basis AND Bob uses Alice's basis: 50%
    else:
        transmitted_bits = alice_bits.copy()
    
    # Bob chooses random bases and measures
    bob_bases = np.random.randint(0, 2, n_bits)
    
    if eve_present:
        # Bob's measurement result depends on whether Eve disturbed the state
        bob_bits = np.zeros(n_bits, dtype=int)
        for i in range(n_bits):
            if not eve_intercepts[i]:
                # No interception
                if bob_bases[i] == alice_bases[i]:
                    bob_bits[i] = alice_bits[i]  # correct
                else:
                    bob_bits[i] = np.random.randint(0, 2)  # random
            else:
                # Eve intercepted
                if bob_bases[i] == eve_bases[i]:
                    bob_bits[i] = eve_bits[i]  # Bob reads Eve's state correctly
                else:
                    bob_bits[i] = np.random.randint(0, 2)  # random
    else:
        bob_bits = np.where(
            bob_bases == alice_bases,
            alice_bits,
            np.random.randint(0, 2, n_bits)
        )
    
    # Sifting: keep only where bases match
    matching_bases = alice_bases == bob_bases
    sifted_alice = alice_bits[matching_bases]
    sifted_bob = bob_bits[matching_bases]
    
    # QBER on sifted key
    errors = np.sum(sifted_alice != sifted_bob)
    qber = errors / len(sifted_alice) if len(sifted_alice) > 0 else 0
    
    return {
        'n_bits': n_bits,
        'sifted_length': len(sifted_alice),
        'errors': errors,
        'qber': qber,
        'sifted_alice': sifted_alice,
        'sifted_bob': sifted_bob,
        'eve_present': eve_present,
    }


# Run simulations
np.random.seed(42)
result_no_eve = simulate_bb84(n_bits=100000, eve_present=False)
result_with_eve = simulate_bb84(n_bits=100000, eve_present=True, eve_fraction=1.0)

print(f"=== BB84 Simulation Results ===")
print(f"No eavesdropper:  QBER = {result_no_eve['qber']:.4f}, Sifted bits = {result_no_eve['sifted_length']}")
print(f"Full eavesdrop:   QBER = {result_with_eve['qber']:.4f}, Sifted bits = {result_with_eve['sifted_length']}")
print(f"Theoretical QBER with full intercept-resend: 25%")
print()

# Figure 1: QBER histogram comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sample check: compute QBER on random subsets to show distribution
n_samples = 1000
sample_size = 200

qbers_clean = []
qbers_eve = []
for _ in range(n_samples):
    idx = np.random.choice(len(result_no_eve['sifted_alice']), sample_size, replace=False)
    q = np.mean(result_no_eve['sifted_alice'][idx] != result_no_eve['sifted_bob'][idx])
    qbers_clean.append(q)
    
    idx = np.random.choice(len(result_with_eve['sifted_alice']), sample_size, replace=False)
    q = np.mean(result_with_eve['sifted_alice'][idx] != result_with_eve['sifted_bob'][idx])
    qbers_eve.append(q)

axes[0].hist(qbers_clean, bins=30, color=COLORS['green'], alpha=0.8, edgecolor='#0d1117')
axes[0].axvline(x=0.11, color=COLORS['red'], linestyle='--', linewidth=2, label='Security threshold (11%)')
axes[0].set_title('No Eavesdropper', color=COLORS['green'])
axes[0].set_xlabel('Quantum Bit Error Rate (QBER)')
axes[0].set_ylabel('Frequency')
axes[0].legend(fontsize=9)
axes[0].set_xlim(-0.02, 0.40)

axes[1].hist(qbers_eve, bins=30, color=COLORS['red'], alpha=0.8, edgecolor='#0d1117')
axes[1].axvline(x=0.11, color=COLORS['red'], linestyle='--', linewidth=2, label='Security threshold (11%)')
axes[1].axvline(x=0.25, color=COLORS['orange'], linestyle=':', linewidth=2, label='Theoretical (25%)')
axes[1].set_title('Full Intercept-Resend Attack', color=COLORS['red'])
axes[1].set_xlabel('Quantum Bit Error Rate (QBER)')
axes[1].set_ylabel('Frequency')
axes[1].legend(fontsize=9)
axes[1].set_xlim(-0.02, 0.40)

fig.suptitle('BB84 Eavesdropping Detection via QBER Sampling\n(1000 random samples of 200 sifted bits each)',
             fontsize=14, fontweight='bold', color=COLORS['cyan'])
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_bb84_qber_detection.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 1 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2: QBER vs Eavesdropping Fraction
# ══════════════════════════════════════════════════════════════════

eve_fractions = np.linspace(0, 1, 50)
qbers_vs_fraction = []

for frac in eve_fractions:
    r = simulate_bb84(n_bits=50000, eve_present=True, eve_fraction=frac)
    qbers_vs_fraction.append(r['qber'])

# Theoretical: QBER = f/4 where f is fraction intercepted
# (Eve wrong basis 50% of time × Bob gets wrong bit 50% of those = 25% per intercepted bit)
theoretical_qber = eve_fractions * 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(eve_fractions, qbers_vs_fraction, color=COLORS['cyan'], s=20, alpha=0.8, label='Simulated', zorder=3)
ax.plot(eve_fractions, theoretical_qber, color=COLORS['orange'], linewidth=2, linestyle='--', label='Theoretical (f/4)')
ax.axhline(y=0.11, color=COLORS['red'], linestyle=':', linewidth=2, alpha=0.7, label='BB84 security threshold (~11%)')
ax.fill_between(eve_fractions, 0.11, 0.30, alpha=0.1, color=COLORS['red'])
ax.text(0.05, 0.13, 'INSECURE ZONE\n(abort protocol)', color=COLORS['red'], fontsize=10, fontstyle='italic')
ax.text(0.05, 0.03, 'SECURE ZONE\n(proceed with privacy amplification)', color=COLORS['green'], fontsize=10, fontstyle='italic')

ax.set_xlabel('Fraction of Qubits Intercepted by Eve')
ax.set_ylabel('Quantum Bit Error Rate (QBER)')
ax.set_title('How Eavesdropping Reveals Itself:\nQBER Scales Linearly with Interception Fraction',
             color=COLORS['cyan'])
ax.legend(loc='upper left')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.01, 0.30)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_qber_vs_eavesdropping.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3: Secure Key Rate vs Distance
# ══════════════════════════════════════════════════════════════════

def bb84_key_rate(distance_km, alpha_db_per_km=0.2, dark_count=1e-6, 
                  detector_eff=0.1, f_error=1.16, clock_rate=1e9):
    """
    Simplified BB84 decoy-state key rate model.
    
    Based on: Scarani et al., Rev. Mod. Phys. 81, 1301 (2009)
    
    R ≈ clock_rate × η × [1 - 2h(QBER)]
    where η is overall transmittance and h is binary entropy
    """
    # Channel transmittance (fiber loss)
    loss_db = alpha_db_per_km * distance_km
    eta_channel = 10 ** (-loss_db / 10)
    eta_total = eta_channel * detector_eff
    
    # QBER from dark counts
    signal = eta_total
    noise = 2 * dark_count  # two detectors
    qber = noise / (signal + noise) if (signal + noise) > 0 else 0.5
    qber = min(qber, 0.5)
    
    # Binary entropy
    if qber == 0:
        h = 0
    elif qber >= 0.5:
        return 0
    else:
        h = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber)
    
    # Secure key rate (bits per pulse)
    rate_per_pulse = max(0, eta_total * (1 - f_error * h - h))
    
    # If QBER > 11%, no secure key
    if qber > 0.11:
        return 0
    
    return rate_per_pulse * clock_rate  # bits per second


def plob_bound(distance_km, alpha_db_per_km=0.2):
    """
    PLOB bound (Pirandola-Laurenza-Ottaviani-Banchi, 2017)
    Ultimate rate-loss tradeoff for point-to-point QKD.
    R_PLOB = -log2(1 - η)
    """
    loss_db = alpha_db_per_km * distance_km
    eta = 10 ** (-loss_db / 10)
    if eta <= 0 or eta >= 1:
        return 0
    return -np.log2(1 - eta)


def tf_qkd_rate(distance_km, alpha_db_per_km=0.2):
    """
    Twin-Field QKD approximate scaling.
    Lucamarini et al., Nature 557, 400 (2018)
    Scales as √η instead of η, enabling much longer distances.
    """
    loss_db = alpha_db_per_km * distance_km
    eta = 10 ** (-loss_db / 10)
    # TF-QKD rate scales as O(√η)
    rate = 1e-2 * np.sqrt(eta)  # approximate scaling with prefactor
    return max(rate, 1e-15)


distances = np.linspace(1, 600, 500)

bb84_rates = [bb84_key_rate(d) for d in distances]
plob_rates = [plob_bound(d) for d in distances]
tf_rates = [tf_qkd_rate(d) for d in distances]

fig, ax = plt.subplots(figsize=(12, 7))

ax.semilogy(distances, bb84_rates, color=COLORS['cyan'], linewidth=2.5, label='BB84 Decoy-State')
ax.semilogy(distances, plob_rates, color=COLORS['dim'], linewidth=2, linestyle='--', label='PLOB Bound (ultimate limit)')
ax.semilogy(distances, tf_rates, color=COLORS['purple'], linewidth=2.5, label='TF-QKD (√η scaling)')

# Mark key deployments
deployments = [
    (100, 'Tokyo QKD\nTestbed', COLORS['orange']),
    (200, 'QNu Labs\nARMOS', COLORS['green']),
    (421, 'TF-QKD record\n(Chen et al. 2021)', COLORS['purple']),
    (509, 'TF-QKD record\n(Liu et al. 2023)', COLORS['pink']),
]

for dist, label, color in deployments:
    rate = bb84_key_rate(dist) if dist <= 300 else tf_qkd_rate(dist)
    rate = max(rate, 1e-1)
    ax.scatter([dist], [rate], color=color, s=100, zorder=5, edgecolors='white', linewidth=1.5)
    ax.annotate(label, (dist, rate), textcoords="offset points", xytext=(10, 15),
                fontsize=8, color=color, fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

ax.set_xlabel('Distance (km)')
ax.set_ylabel('Secure Key Rate (bits/s or bits/pulse)')
ax.set_title('QKD Key Rate vs Distance:\nThe Fiber Loss Wall and How to Scale It',
             color=COLORS['cyan'])
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(1e-2, 1e10)
ax.set_xlim(0, 600)
ax.grid(True, alpha=0.3)

# Annotate trusted relay region
ax.axvspan(80, 200, alpha=0.05, color=COLORS['yellow'])
ax.text(130, 2e9, 'Trusted relay\nspacing range', color=COLORS['yellow'], 
        fontsize=9, ha='center', fontstyle='italic')

plt.tight_layout()
plt.savefig(f'{OUT}/fig3_key_rate_vs_distance.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 3 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4: Global QKD Network Comparison
# ══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 7))

networks = [
    ('China CN-QCN\n(2025)', 12000, 145, COLORS['red'], '12,000+ km\n145 backbone nodes\n20 metro networks\n800+ user nodes'),
    ('China BSBN +\nMicius (2021)', 4600, 32, COLORS['orange'], '4,600 km\n32 trusted relays\n150+ users\nSatellite + fiber'),
    ('India NQM\nQNu Labs (2026)', 1000, 6, COLORS['green'], '1,000 km\n~6 nodes (est.)\nIndigenous tech\nTarget: 2,000 km'),
    ('India NQM\nPhase 1 (2025)', 500, 4, COLORS['cyan'], '500 km\n4 nodes\n60% fewer nodes\nthan conventional'),
    ('EuroQCI\n(planned 2030)', 0, 0, COLORS['purple'], 'Pan-EU network\n27 member states\nEagle-1 satellite\nFiber + space'),
    ('UK UKQN\n(2024)', 410, 8, COLORS['pink'], '410 km\n8 nodes\nCambridge-London\nBT partnership'),
    ('S. Korea\nSKT (2025)', 800, 12, COLORS['yellow'], '800 km\nSeoul corridor\nSK Telecom\nCommercial'),
]

# Sort by distance
networks.sort(key=lambda x: x[1], reverse=True)

bars = ax.barh(range(len(networks)), [n[1] for n in networks], 
               color=[n[3] for n in networks], alpha=0.8, height=0.6,
               edgecolor='#0d1117', linewidth=1.5)

ax.set_yticks(range(len(networks)))
ax.set_yticklabels([n[0] for n in networks], fontsize=10)
ax.set_xlabel('Network Span (km)')
ax.set_title('Global QKD Network Deployments: Where India Stands',
             color=COLORS['cyan'])

# Add annotations
for i, (name, dist, nodes, color, desc) in enumerate(networks):
    if dist > 0:
        ax.text(dist + 150, i, f'{dist:,} km', va='center', fontsize=10, 
                color=color, fontweight='bold')

ax.set_xlim(0, 15000)
ax.grid(True, axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUT}/fig4_global_qkd_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 4 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5: QKD Attack Taxonomy
# ══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'QKD Attack Taxonomy: Theory vs Practice', 
        ha='center', fontsize=16, fontweight='bold', color=COLORS['cyan'])

# Three columns
cols = [
    {
        'title': 'PROTOCOL-LEVEL\n(Theoretical)',
        'color': COLORS['green'],
        'x': 2.3,
        'attacks': [
            'Intercept-Resend',
            'Man-in-the-Middle',
            'PNS Attack',
            'USD Attack',
            'Beam Splitting',
        ],
        'defense': 'Detected by\nQBER monitoring\n& authentication'
    },
    {
        'title': 'IMPLEMENTATION\n(Hardware)',
        'color': COLORS['orange'],
        'x': 7,
        'attacks': [
            'Detector Blinding',
            'Laser Damage (LDA)',
            'Trojan Horse',
            'Phase Remapping',
            'Timing Side-Channel',
        ],
        'defense': 'Requires\ndevice certification\n& monitoring'
    },
    {
        'title': 'INFRASTRUCTURE\n(Network)',
        'color': COLORS['red'],
        'x': 11.7,
        'attacks': [
            'Trusted Relay Breach',
            'Denial of Service',
            'Classical Channel Attack',
            'Insider Threat',
            'Supply Chain Compromise',
        ],
        'defense': 'Mitigated by\nMDI-QKD, DI-QKD\n& network redundancy'
    }
]

for col in cols:
    # Column header
    box = mpatches.FancyBboxPatch((col['x']-1.8, 7.8), 3.6, 1.2,
                                   boxstyle="round,pad=0.1",
                                   facecolor=col['color'], alpha=0.2,
                                   edgecolor=col['color'], linewidth=2)
    ax.add_patch(box)
    ax.text(col['x'], 8.4, col['title'], ha='center', va='center',
            fontsize=11, fontweight='bold', color=col['color'])
    
    # Attack items
    for j, atk in enumerate(col['attacks']):
        y = 7.2 - j * 0.9
        box = mpatches.FancyBboxPatch((col['x']-1.6, y-0.3), 3.2, 0.55,
                                       boxstyle="round,pad=0.05",
                                       facecolor='#161b22', edgecolor=col['color'],
                                       linewidth=1, alpha=0.8)
        ax.add_patch(box)
        ax.text(col['x'], y, atk, ha='center', va='center',
                fontsize=9, color=COLORS['white'])
    
    # Defense note
    y_def = 7.2 - len(col['attacks']) * 0.9 - 0.3
    ax.text(col['x'], y_def, col['defense'], ha='center', va='center',
            fontsize=8, color=col['color'], fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', 
                     edgecolor=col['color'], alpha=0.5, linewidth=0.5))

# Severity arrow
ax.annotate('', xy=(12.5, 9.2), xytext=(1.5, 9.2),
            arrowprops=dict(arrowstyle='->', color=COLORS['dim'], lw=2))
ax.text(7, 9.0, 'Increasing practical severity →', ha='center', fontsize=9,
        color=COLORS['dim'], fontstyle='italic')

plt.tight_layout()
plt.savefig(f'{OUT}/fig5_attack_taxonomy.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 5 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6: India NQM Timeline & Progress
# ══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(2023.5, 2033)
ax.set_ylim(-1, 3)
ax.axis('off')

# Timeline base
ax.plot([2024, 2032], [0, 0], color=COLORS['dim'], linewidth=3, zorder=1)

milestones = [
    (2024, 'NQM Launch\n(Oct 2024)', COLORS['cyan'], 1, True),
    (2025, '500 km QKD\nQNu Labs\nESTIC 2025', COLORS['green'], 1.5, True),
    (2026, '1,000 km QKD\nDemonstrated\n(Apr 2026)', COLORS['yellow'], 2, True),
    (2028, 'Target:\n1,500 km\n(projected)', COLORS['purple'], 1.2, False),
    (2032, 'Target:\n2,000 km\n(mission goal)', COLORS['orange'], 1.5, False),
]

for year, label, color, height, achieved in milestones:
    marker = 'o' if achieved else 'D'
    alpha = 1.0 if achieved else 0.5
    ax.scatter([year], [0], color=color, s=150, zorder=5, marker=marker, 
              edgecolors='white', linewidth=1.5, alpha=alpha)
    ax.plot([year, year], [0, height * (1 if True else -1)], 
           color=color, linewidth=1.5, linestyle='-' if achieved else ':', alpha=alpha)
    ax.text(year, height + 0.15, label, ha='center', va='bottom',
            fontsize=9, color=color, fontweight='bold' if achieved else 'normal',
            alpha=alpha)

# Progress bar
progress = (1000 / 2000) * 100
bar_y = -0.6
ax.barh(bar_y, 2032-2024, left=2024, height=0.25, color='#21262d', zorder=1)
ax.barh(bar_y, (2026-2024), left=2024, height=0.25, color=COLORS['green'], alpha=0.7, zorder=2)
ax.text(2028, bar_y, f'{progress:.0f}% of 2,000 km target achieved', 
        ha='center', va='center', fontsize=10, color=COLORS['green'], fontweight='bold')

ax.set_title("India's National Quantum Mission: QKD Network Progress",
             fontsize=14, fontweight='bold', color=COLORS['cyan'], pad=20)

plt.tight_layout()
plt.savefig(f'{OUT}/fig6_india_nqm_timeline.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 6 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 7: QKD vs PQC — The Global Divide
# ══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)

ax.text(6, 7.5, 'The Global Quantum Security Divide', ha='center',
        fontsize=16, fontweight='bold', color=COLORS['cyan'])

# QKD camp
qkd_box = mpatches.FancyBboxPatch((0.3, 1), 5.2, 5.5,
                                    boxstyle="round,pad=0.2",
                                    facecolor=COLORS['green'], alpha=0.08,
                                    edgecolor=COLORS['green'], linewidth=2)
ax.add_patch(qkd_box)
ax.text(2.9, 6.1, 'QKD-Forward', ha='center', fontsize=13, 
        fontweight='bold', color=COLORS['green'])

qkd_countries = [
    '[CN] China — 12,000+ km deployed',
    '[IN] India — 1,000 km, targeting 2,000',
    '[EU] Europe — EuroQCI (27 nations)',
    '[KR] South Korea — 800 km commercial',
    '[JP] Japan — Tokyo testbed',
    '[SG] Singapore — NQSN+ network',
]
for i, c in enumerate(qkd_countries):
    ax.text(1, 5.3 - i*0.7, c, fontsize=9, color=COLORS['white'])

# PQC camp
pqc_box = mpatches.FancyBboxPatch((6.5, 1), 5.2, 5.5,
                                    boxstyle="round,pad=0.2",
                                    facecolor=COLORS['red'], alpha=0.08,
                                    edgecolor=COLORS['red'], linewidth=2)
ax.add_patch(pqc_box)
ax.text(9.1, 6.1, 'PQC-First', ha='center', fontsize=13,
        fontweight='bold', color=COLORS['red'])

pqc_countries = [
    '[US] USA — NSA: "does not support QKD"',
    '[UK] UK — NCSC: PQC preferred',
    '[FR] France — ANSSI: PQC priority',
    '[DE] Germany — BSI: QKD "not mature"',
    '[NL] Netherlands — PQC + caution on QKD',
    '[SE] Sweden — PQC standard track',
]
for i, c in enumerate(pqc_countries):
    ax.text(7.2, 5.3 - i*0.7, c, fontsize=9, color=COLORS['white'])

# Middle ground
ax.text(6, 0.5, 'Growing consensus: QKD + PQC hybrid approach offers defense-in-depth',
        ha='center', fontsize=10, color=COLORS['orange'], fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1117', 
                 edgecolor=COLORS['orange'], linewidth=1))

plt.tight_layout()
plt.savefig(f'{OUT}/fig7_qkd_vs_pqc_divide.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 7 saved")


print(f"\n{'='*50}")
print(f"All figures saved to {OUT}/")
print(f"{'='*50}")
