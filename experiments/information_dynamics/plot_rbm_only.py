"""Plot RBM results separately."""
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load RBM results
with open('results/mnist_real_results_20251117_014743.pkl', 'rb') as f:
    results = pickle.load(f)

epochs = results['epochs']
bayesian_mi = np.array(results['bayesian_mi'])

# Get V-information for different budgets
v_info_10 = np.array(results['v_info_10'])
v_info_50 = np.array(results['v_info_50'])
v_info_100 = np.array(results['v_info_100'])

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RBM: Information Dynamics on MNIST (20 epochs)', 
             fontsize=16, fontweight='bold')

# Plot 1: All information measures
ax = axes[0, 0]
ax.plot(epochs, bayesian_mi, 'o-', linewidth=3, markersize=10,
        color='#E63946', label='Bayesian MI', zorder=10)
ax.plot(epochs, v_info_10, 's-', linewidth=2, markersize=7,
        color='#F77F00', label='V-info (10 steps)', alpha=0.8)
ax.plot(epochs, v_info_50, 'd-', linewidth=2, markersize=7,
        color='#06AED5', label='V-info (50 steps)', alpha=0.8)
ax.plot(epochs, v_info_100, '^-', linewidth=2, markersize=7,
        color='#118AB2', label='V-info (100 steps)', alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Information (bits)', fontsize=12, fontweight='bold')
ax.set_title('Information Content vs Extractable', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Information gaps
ax = axes[0, 1]
gap_10 = np.maximum(0, bayesian_mi - v_info_10)
gap_50 = np.maximum(0, bayesian_mi - v_info_50)
gap_100 = np.maximum(0, bayesian_mi - v_info_100)
ax.plot(epochs, gap_10, 'o-', linewidth=2, markersize=8,
        color='#F77F00', label='Gap @ 10 steps')
ax.plot(epochs, gap_50, 's-', linewidth=2, markersize=8,
        color='#06AED5', label='Gap @ 50 steps')
ax.plot(epochs, gap_100, '^-', linewidth=2, markersize=8,
        color='#118AB2', label='Gap @ 100 steps')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Information Gap (bits)', fontsize=12, fontweight='bold')
ax.set_title('Unusable Information (Computational Constraint)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Extraction efficiency
ax = axes[1, 0]
eff_10 = np.array([v/m*100 if m > 0 else 0 for v, m in zip(v_info_10, bayesian_mi)])
eff_50 = np.array([v/m*100 if m > 0 else 0 for v, m in zip(v_info_50, bayesian_mi)])
eff_100 = np.array([v/m*100 if m > 0 else 0 for v, m in zip(v_info_100, bayesian_mi)])
ax.plot(epochs, eff_10, 'o-', linewidth=2, markersize=8,
        color='#F77F00', label='10 steps')
ax.plot(epochs, eff_50, 's-', linewidth=2, markersize=8,
        color='#06AED5', label='50 steps')
ax.plot(epochs, eff_100, '^-', linewidth=2, markersize=8,
        color='#118AB2', label='100 steps')
ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Extraction Efficiency (%)', fontsize=12, fontweight='bold')
ax.set_title('% of Information Extractable by Budget', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Summary stats
ax = axes[1, 1]
ax.axis('off')

# Summary text
summary = f"""
RBM Training Summary (20 epochs)

Configuration:
  • Visible: 130 units (100 pixels + 30 labels)
  • Hidden: 100 units
  • Training samples: 2000
  • Learning rate: 0.05
  • CD steps: 1

Final Results:
  • Bayesian MI: {bayesian_mi[-1]:.4f} bits
  • V-info (10 steps): {v_info_10[-1]:.4f} bits
  • V-info (50 steps): {v_info_50[-1]:.4f} bits
  • V-info (100 steps): {v_info_100[-1]:.4f} bits

Information Capacity:
  • Maximum possible: log₂(3) ≈ 1.585 bits
  • Achieved: {bayesian_mi[-1]/1.585*100:.1f}% of maximum

Training Characteristics:
  • Highly variable MI across epochs
  • Some epochs exceed 100% extractability
  • Reflects stochastic Gibbs sampling
"""

ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig('results/rbm_only_20epochs.png', dpi=300, bbox_inches='tight')
print("✅ RBM visualization saved to: results/rbm_only_20epochs.png")
