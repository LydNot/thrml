"""Plot just the Bayesian MI for RBM."""
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load RBM results
with open('results/mnist_real_results_20251117_014743.pkl', 'rb') as f:
    results = pickle.load(f)

epochs = np.array(results['epochs'])
bayesian_mi = np.array(results['bayesian_mi'])

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot Bayesian MI
ax.plot(epochs, bayesian_mi, 'o-', linewidth=3, markersize=12,
        color='#E63946', label='Bayesian MI: I(H; Y)', zorder=10)

# Add reference lines
ax.axhline(y=1.585, color='green', linestyle='--', linewidth=2, 
           label='Theoretical Maximum (log₂(3) = 1.585 bits)', alpha=0.7)
ax.axhline(y=0.03, color='gray', linestyle=':', linewidth=2,
           label='Random Baseline (~0.03 bits)', alpha=0.5)

# Calculate mean and show it
mean_mi = np.mean(bayesian_mi)
ax.axhline(y=mean_mi, color='red', linestyle='--', linewidth=1.5,
           label=f'Mean MI = {mean_mi:.3f} bits', alpha=0.5)

# Styling
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Mutual Information (bits)', fontsize=14, fontweight='bold')
ax.set_title('RBM: Mutual Information Between Hidden States and Digit Classes\n' + 
             'I(H; Y) - Information captured by 100 hidden units about 3 MNIST digit classes',
             fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([0, 1.8])

# Add annotation for final value
final_mi = bayesian_mi[-1]
ax.annotate(f'Final: {final_mi:.3f} bits\n({final_mi/1.585*100:.1f}% of max)',
            xy=(epochs[-1], final_mi), 
            xytext=(epochs[-1]-3, final_mi+0.3),
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Add statistics box
stats_text = f"""Training Statistics:
  Min MI:  {np.min(bayesian_mi):.3f} bits (epoch {epochs[np.argmin(bayesian_mi)]})
  Max MI:  {np.max(bayesian_mi):.3f} bits (epoch {epochs[np.argmax(bayesian_mi)]})
  Mean MI: {mean_mi:.3f} bits
  Std Dev: {np.std(bayesian_mi):.3f} bits
  
  Gain over random: {mean_mi - 0.03:.3f} bits
  % of maximum achieved: {final_mi/1.585*100:.1f}%"""

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace')

plt.tight_layout()
plt.savefig('results/rbm_mi_clean.png', dpi=300, bbox_inches='tight')
print("✅ Clean RBM MI plot saved to: results/rbm_mi_clean.png")
print(f"\nSummary:")
print(f"  Final MI: {final_mi:.4f} bits")
print(f"  Mean MI:  {mean_mi:.4f} bits")
print(f"  Range:    [{np.min(bayesian_mi):.4f}, {np.max(bayesian_mi):.4f}] bits")
