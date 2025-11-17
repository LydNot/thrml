"""
Compare RBM vs MLP: Information Dynamics on MNIST

This script runs both experiments and creates side-by-side comparisons to highlight
differences in information dynamics between:
- Energy-based model (RBM with Gibbs sampling)
- Discriminative model (MLP with backprop)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from mnist_real_training import run_real_mnist_experiment as run_rbm_experiment
from mnist_mlp_baseline import run_mlp_mnist_experiment


def run_comparison_experiments(
    # Shared parameters
    n_epochs=5,
    n_hidden=50,
    n_samples=500,
    n_pixel_features=100,
    info_sampling_chains=50,

    # Quick test mode
    quick_test=False,
):
    """
    Run both RBM and MLP experiments with comparable parameters.

    Args:
        quick_test: If True, use minimal parameters for fast testing
    """

    if quick_test:
        print("\n" + "⚡" * 30)
        print("QUICK TEST MODE - Using minimal parameters")
        print("⚡" * 30 + "\n")
        n_epochs = 3
        n_hidden = 20
        n_samples = 200
        n_pixel_features = 50
        info_sampling_chains = 30

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 70)
    print("COMPARISON EXPERIMENT: RBM vs MLP on MNIST")
    print("=" * 70)
    print(f"\nShared Configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Hidden units: {n_hidden}")
    print(f"  Training samples: {n_samples}")
    print(f"  Pixel features: {n_pixel_features}")
    print(f"  Info sampling chains: {info_sampling_chains}")
    print()

    # ====================
    # 1. Run RBM Experiment
    # ====================
    print("\n" + "🔥" * 35)
    print("EXPERIMENT 1/2: RBM (Energy-Based Model)")
    print("🔥" * 35 + "\n")

    rbm_budgets = [10, 50] if quick_test else [10, 50, 100]

    rbm_results, rbm_timestamp = run_rbm_experiment(
        n_epochs=n_epochs,
        n_hidden=n_hidden,
        n_samples=n_samples,
        n_pixel_features=n_pixel_features,
        info_sampling_chains=info_sampling_chains,
        budgets=rbm_budgets,
        learning_rate=0.05,
        cd_steps=1,
        batch_size=100,
    )

    print("\n✅ RBM experiment complete!")

    # ====================
    # 2. Run MLP Experiment
    # ====================
    print("\n" + "🧠" * 35)
    print("EXPERIMENT 2/2: MLP (Discriminative Baseline)")
    print("🧠" * 35 + "\n")

    mlp_budgets = [0.0, 0.2] if quick_test else [0.0, 0.1, 0.5]

    mlp_results, mlp_timestamp = run_mlp_mnist_experiment(
        n_epochs=n_epochs,
        n_hidden=n_hidden,
        n_samples=n_samples,
        n_pixel_features=n_pixel_features,
        info_sampling_chains=info_sampling_chains,
        budgets=mlp_budgets,
        learning_rate=0.001,
        batch_size=100,
    )

    print("\n✅ MLP experiment complete!")

    return rbm_results, mlp_results, timestamp


def plot_comparison(rbm_results, mlp_results, timestamp, save_dir="results"):
    """Create side-by-side comparison plots."""

    Path(save_dir).mkdir(exist_ok=True)

    epochs = rbm_results['epochs']

    # Extract data
    rbm_mi = np.array(rbm_results['bayesian_mi'])
    mlp_mi = np.array(mlp_results['bayesian_mi'])
    mlp_loss = np.array(mlp_results['train_loss'])

    # Get budget keys
    rbm_budget_keys = sorted([k for k in rbm_results.keys() if k.startswith('v_info_')])
    mlp_budget_keys = sorted([k for k in mlp_results.keys() if k.startswith('v_info_')])

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle(f'RBM vs MLP: Information Dynamics Comparison\n{timestamp}',
                 fontsize=18, fontweight='bold')

    colors_rbm = ['#FF6B6B', '#EE5A6F', '#C73E1D']
    colors_mlp = ['#4ECDC4', '#45B7AF', '#3A9D95']

    # ======================
    # Row 1: Bayesian MI Comparison
    # ======================

    # Plot 1: RBM Bayesian MI
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, rbm_mi, 'o-', linewidth=3, markersize=10,
             color='#FF6B6B', label='RBM Bayesian MI')

    for i, budget_key in enumerate(rbm_budget_keys):
        budget = budget_key.split('_')[-1]
        v_info = rbm_results[budget_key]
        ax1.plot(epochs, v_info, 's--', linewidth=2, markersize=6,
                 color=colors_rbm[i], label=f'V-info ({budget} steps)', alpha=0.7)

    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Information (bits)', fontsize=11, fontweight='bold')
    ax1.set_title('🔥 RBM: Information Content vs Extractable',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: MLP Bayesian MI + Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()

    ax2.plot(epochs, mlp_mi, 'o-', linewidth=3, markersize=10,
             color='#4ECDC4', label='MLP Bayesian MI')

    for i, budget_key in enumerate(mlp_budget_keys):
        budget = budget_key.split('_')[-1]
        v_info = mlp_results[budget_key]
        ax2.plot(epochs, v_info, 's--', linewidth=2, markersize=6,
                 color=colors_mlp[i], label=f'V-info (σ={budget})', alpha=0.7)

    ax2_twin.plot(epochs, mlp_loss, '^:', linewidth=2, markersize=6,
                  color='gray', label='Train Loss', alpha=0.6)

    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Information (bits)', fontsize=11, fontweight='bold')
    ax2_twin.set_ylabel('Loss', fontsize=11, fontweight='bold', color='gray')
    ax2_twin.tick_params(axis='y', labelcolor='gray')
    ax2.set_title('🧠 MLP: Information Content vs Extractable',
                  fontsize=13, fontweight='bold')

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ======================
    # Row 2: Direct MI Comparison
    # ======================

    # Plot 3: Direct MI comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, rbm_mi, 'o-', linewidth=3, markersize=10,
             color='#FF6B6B', label='RBM', zorder=3)
    ax3.plot(epochs, mlp_mi, 's-', linewidth=3, markersize=10,
             color='#4ECDC4', label='MLP', zorder=3)

    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Bayesian MI (bits)', fontsize=11, fontweight='bold')
    ax3.set_title('Direct Comparison: Bayesian MI', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3)

    # Plot 4: MI Ratio (MLP / RBM)
    ax4 = fig.add_subplot(gs[1, 1])

    # Compute ratio safely
    ratio = np.array([mlp / rbm if rbm > 1e-6 else 0
                      for mlp, rbm in zip(mlp_mi, rbm_mi)])

    ax4.plot(epochs, ratio, 'D-', linewidth=3, markersize=8,
             color='#9B59B6', label='MLP/RBM Ratio')
    ax4.axhline(y=1, color='black', linestyle='--', linewidth=2,
                alpha=0.5, label='Equal (ratio=1)')

    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('MI Ratio (MLP/RBM)', fontsize=11, fontweight='bold')
    ax4.set_title('Information Ratio: MLP vs RBM', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    # Add annotation
    if any(ratio > 0):
        final_ratio = ratio[-1] if ratio[-1] > 0 else np.nan
        if not np.isnan(final_ratio):
            ax4.text(0.05, 0.95, f'Final ratio: {final_ratio:.2f}',
                    transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))

    # ======================
    # Row 3: Information Gaps
    # ======================

    # Plot 5: RBM Information Gaps
    ax5 = fig.add_subplot(gs[2, 0])

    for i, budget_key in enumerate(rbm_budget_keys):
        budget = budget_key.split('_')[-1]
        v_info = np.array(rbm_results[budget_key])
        gap = np.maximum(0, rbm_mi - v_info)
        ax5.plot(epochs, gap, 'o-', linewidth=2, markersize=7,
                 color=colors_rbm[i], label=f'{budget} steps')

    ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Information Gap (bits)', fontsize=11, fontweight='bold')
    ax5.set_title('🔥 RBM: Unusable Information', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 6: MLP Information Gaps
    ax6 = fig.add_subplot(gs[2, 1])

    for i, budget_key in enumerate(mlp_budget_keys):
        budget = budget_key.split('_')[-1]
        v_info = np.array(mlp_results[budget_key])
        gap = np.maximum(0, mlp_mi - v_info)
        ax6.plot(epochs, gap, 's-', linewidth=2, markersize=7,
                 color=colors_mlp[i], label=f'σ={budget}')

    ax6.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Information Gap (bits)', fontsize=11, fontweight='bold')
    ax6.set_title('🧠 MLP: Unusable Information', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Save
    output_path = f"{save_dir}/comparison_rbm_vs_mlp_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison visualization saved to: {output_path}")

    return output_path


def print_comparison_summary(rbm_results, mlp_results):
    """Print summary statistics comparing both models."""

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    rbm_mi_final = rbm_results['bayesian_mi'][-1]
    mlp_mi_final = mlp_results['bayesian_mi'][-1]

    print(f"\n📊 Final Bayesian MI:")
    print(f"  RBM (Energy-Based):     {rbm_mi_final:.4f} bits")
    print(f"  MLP (Discriminative):   {mlp_mi_final:.4f} bits")

    if rbm_mi_final > 1e-6 and mlp_mi_final > 1e-6:
        ratio = mlp_mi_final / rbm_mi_final
        print(f"  Ratio (MLP/RBM):        {ratio:.2f}x")
        if ratio > 1:
            print(f"  → MLP learned {ratio:.1f}x more mutual information")
        else:
            print(f"  → RBM learned {1/ratio:.1f}x more mutual information")

    print(f"\n🎯 Information Extraction:")

    # RBM extraction
    print(f"\n  RBM (by sampling budget):")
    rbm_budget_keys = sorted([k for k in rbm_results.keys() if k.startswith('v_info_')])
    for budget_key in rbm_budget_keys:
        budget = budget_key.split('_')[-1]
        v_info = rbm_results[budget_key][-1]
        gap = max(0, rbm_mi_final - v_info)
        pct = (v_info / rbm_mi_final * 100) if rbm_mi_final > 0 else 0
        print(f"    {budget:>3s} steps: {v_info:.4f} bits ({pct:>3.0f}% extractable, gap: {gap:.4f})")

    # MLP extraction
    print(f"\n  MLP (by noise level):")
    mlp_budget_keys = sorted([k for k in mlp_results.keys() if k.startswith('v_info_')])
    for budget_key in mlp_budget_keys:
        budget = budget_key.split('_')[-1]
        v_info = mlp_results[budget_key][-1]
        gap = max(0, mlp_mi_final - v_info)
        pct = (v_info / mlp_mi_final * 100) if mlp_mi_final > 0 else 0
        print(f"    σ={budget:>4s}: {v_info:.4f} bits ({pct:>3.0f}% extractable, gap: {gap:.4f})")

    if 'train_loss' in mlp_results:
        print(f"\n📉 MLP Training Loss:")
        print(f"  Initial: {mlp_results['train_loss'][0]:.4f}")
        print(f"  Final:   {mlp_results['train_loss'][-1]:.4f}")
        improvement = mlp_results['train_loss'][0] - mlp_results['train_loss'][-1]
        print(f"  Improvement: {improvement:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare RBM vs MLP on MNIST')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with minimal parameters')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--hidden', type=int, default=50,
                        help='Number of hidden units (default: 50)')
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of training samples (default: 500)')

    args = parser.parse_args()

    print("\n" + "🔬" * 35)
    print("RBM vs MLP COMPARISON EXPERIMENT")
    print("Comparing information dynamics between energy-based and discriminative models")
    print("🔬" * 35)

    # Run experiments
    rbm_results, mlp_results, timestamp = run_comparison_experiments(
        n_epochs=args.epochs,
        n_hidden=args.hidden,
        n_samples=args.samples,
        quick_test=args.quick,
    )

    # Generate comparison plots
    print("\n" + "📊" * 35)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("📊" * 35)

    plot_path = plot_comparison(rbm_results, mlp_results, timestamp)

    # Print summary
    print_comparison_summary(rbm_results, mlp_results)

    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Visualization: {plot_path}")
    print("\n💡 Key Insights:")
    print("  • RBM uses sampling-based training (Contrastive Divergence)")
    print("  • MLP uses gradient-based training (Backpropagation)")
    print("  • V-information budget means different things:")
    print("    - RBM: Number of Gibbs sampling steps")
    print("    - MLP: Amount of noise added to representations")
    print("  • Compare the information gaps to see computational constraints!")
    print()
