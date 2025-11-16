"""
Visualize information dynamics during EBM training.

Creates publication-quality plots showing:
1. Training accuracy over time
2. Bayesian MI evolution
3. V-information at different budgets
4. Information gap (unusable information)
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path


def load_tracker(path: str):
    """Load saved tracking data."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_information_dynamics(tracker_data, save_path=None):
    """
    Create comprehensive visualization of information dynamics.
    
    Args:
        tracker_data: Dictionary with tracking data or InfoTracker object
        save_path: Optional path to save figure
    """
    
    # Handle both dict and object
    if hasattr(tracker_data, '__dict__'):
        data = tracker_data.__dict__
    else:
        data = tracker_data
    
    epochs = data['epoch']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Information Dynamics in Discrete EBM Training', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Accuracy
    ax = axes[0, 0]
    ax.plot(epochs, data['accuracy'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classification Accuracy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 2: Bayesian MI with uncertainty
    ax = axes[0, 1]
    mi_mean = np.array(data['bayesian_mi_mean'])
    mi_std = np.array(data['bayesian_mi_std'])
    ax.plot(epochs, mi_mean, 'o-', linewidth=2, markersize=8, color='#A23B72', label='Bayesian MI')
    ax.fill_between(epochs, mi_mean - mi_std, mi_mean + mi_std, alpha=0.3, color='#A23B72')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Information (bits)', fontsize=12)
    ax.set_title('Bayesian Mutual Information', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: V-information at different budgets
    ax = axes[1, 0]
    ax.plot(epochs, data['v_info_10_steps'], 'o-', linewidth=2, markersize=6,
            color='#F18F01', label='10 Gibbs steps', alpha=0.8)
    ax.plot(epochs, data['v_info_100_steps'], 's-', linewidth=2, markersize=6,
            color='#C73E1D', label='100 Gibbs steps', alpha=0.8)
    ax.plot(epochs, data['v_info_1000_steps'], '^-', linewidth=2, markersize=6,
            color='#6A994E', label='1000 Gibbs steps', alpha=0.8)
    ax.plot(epochs, mi_mean, '--', linewidth=2, color='gray', label='Bayesian MI (limit)', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Information (bits)', fontsize=12)
    ax.set_title('V-Information: Extractable Information', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Information Gap (unusable information)
    ax = axes[1, 1]
    ax.plot(epochs, data['info_gap_10'], 'o-', linewidth=2, markersize=6,
            color='#F18F01', label='Gap @ 10 steps', alpha=0.8)
    ax.plot(epochs, data['info_gap_100'], 's-', linewidth=2, markersize=6,
            color='#C73E1D', label='Gap @ 100 steps', alpha=0.8)
    ax.plot(epochs, data['info_gap_1000'], '^-', linewidth=2, markersize=6,
            color='#6A994E', label='Gap @ 1000 steps', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Unusable Information (bits)', fontsize=12)
    ax.set_title('Information Gap: Bayesian MI - V-Info', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_v_information_budget_curve(budgets, v_infos, bayesian_mi, save_path=None):
    """
    Plot how V-information grows with computational budget.
    
    Args:
        budgets: List of sampling budgets (num Gibbs steps)
        v_infos: List of V-information values corresponding to budgets
        bayesian_mi: The theoretical maximum (Bayesian MI)
        save_path: Optional path to save figure
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.semilogx(budgets, v_infos, 'o-', linewidth=3, markersize=10, 
                color='#2E86AB', label='V-Information')
    ax.axhline(y=bayesian_mi, color='#A23B72', linestyle='--', linewidth=2,
               label=f'Bayesian MI (limit) = {bayesian_mi:.2f} bits')
    
    # Fill area between curve and limit
    ax.fill_between(budgets, v_infos, bayesian_mi, alpha=0.2, color='red',
                    label='Unusable information')
    
    ax.set_xlabel('Computational Budget (Gibbs sampling steps)', fontsize=13)
    ax.set_ylabel('Information (bits)', fontsize=13)
    ax.set_title('V-Information vs. Computational Budget', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add percentage annotations
    for budget, v_info in zip(budgets[::2], v_infos[::2]):  # Annotate every other point
        pct = (v_info / bayesian_mi) * 100 if bayesian_mi > 0 else 0
        ax.annotate(f'{pct:.0f}%', xy=(budget, v_info), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def create_demo_visualization():
    """Create demo visualization with synthetic data."""
    
    print("Creating demo visualization with synthetic data...")
    
    # Synthetic data showing expected behavior
    epochs = list(range(10))
    
    # Simulate: MI increases during training
    bayesian_mi = [0.5 + 0.3 * e for e in epochs]
    bayesian_std = [0.1] * len(epochs)
    
    # Simulate: V-info lags behind and depends on budget
    v_10 = [mi * 0.3 for mi in bayesian_mi]
    v_100 = [mi * 0.7 for mi in bayesian_mi]
    v_1000 = [mi * 0.95 for mi in bayesian_mi]
    
    # Accuracy increases
    accuracy = [0.33 + 0.06 * e for e in epochs]
    
    # Information gaps
    gap_10 = [mi - v for mi, v in zip(bayesian_mi, v_10)]
    gap_100 = [mi - v for mi, v in zip(bayesian_mi, v_100)]
    gap_1000 = [mi - v for mi, v in zip(bayesian_mi, v_1000)]
    
    tracker_data = {
        'epoch': epochs,
        'accuracy': accuracy,
        'bayesian_mi_mean': bayesian_mi,
        'bayesian_mi_std': bayesian_std,
        'v_info_10_steps': v_10,
        'v_info_100_steps': v_100,
        'v_info_1000_steps': v_1000,
        'info_gap_10': gap_10,
        'info_gap_100': gap_100,
        'info_gap_1000': gap_1000,
    }
    
    # Create plots
    fig1 = plot_information_dynamics(tracker_data, 
                                     save_path='experiments/information_dynamics/demo_dynamics.png')
    plt.show()
    
    # Budget curve
    budgets = [1, 5, 10, 50, 100, 500, 1000]
    final_mi = bayesian_mi[-1]
    v_curve = [final_mi * (1 - np.exp(-b / 150)) for b in budgets]
    
    fig2 = plot_v_information_budget_curve(budgets, v_curve, final_mi,
                                          save_path='experiments/information_dynamics/demo_budget_curve.png')
    plt.show()
    
    print("\nDemo visualizations created!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Load and plot real results
        tracker_path = sys.argv[1]
        print(f"Loading results from {tracker_path}...")
        data = load_tracker(tracker_path)
        
        output_dir = Path(tracker_path).parent
        plot_information_dynamics(data, 
                                 save_path=str(output_dir / "information_dynamics.png"))
        plt.show()
    else:
        # Create demo
        create_demo_visualization()

