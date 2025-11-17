"""
Compute layer-wise (group-wise) MI for RBM hidden units.

In an RBM, all hidden units are in a single layer, but we can group them
(e.g., early, middle, late indices) to see if different groups capture
different amounts of information about the labels.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import sys
from pathlib import Path

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))
from compute_bayesian_mi import estimate_discrete_mi

# Import RBM components
from mnist_real_training import (
    load_and_prepare_mnist,
    create_rbm_model,
    subsample_mnist,
    sample_hidden_given_visible
)

print(f"JAX devices: {jax.devices()}")


def compute_grouped_mi(results_file, n_groups=3):
    """
    Re-run sampling on saved RBM model and compute MI for groups of hidden units.

    Args:
        results_file: Path to saved RBM results (contains final model parameters)
        n_groups: Number of groups to split hidden units into
    """

    # Load previous results to get model configuration
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print("=" * 70)
    print("COMPUTING GROUPED MI FOR RBM HIDDEN UNITS")
    print("=" * 70)
    print(f"\nUsing saved results from: {results_file}")
    print(f"Number of groups: {n_groups}")

    # We need to recreate and retrain the model or use a saved checkpoint
    # For now, let's train a fresh model and track group-wise MI

    # Configuration (match the RBM experiment)
    n_epochs = 20
    n_hidden = 50
    n_samples = 2000
    n_pixel_features = 100
    info_sampling_chains = 50
    learning_rate = 0.01
    cd_steps = 1
    batch_size = 500  # Increased for speed

    print(f"\nConfiguration:")
    print(f"  Hidden units: {n_hidden}")
    print(f"  Groups: {n_groups}")

    # Calculate group sizes
    group_size = n_hidden // n_groups
    groups = []
    for i in range(n_groups):
        start = i * group_size
        end = (i + 1) * group_size if i < n_groups - 1 else n_hidden
        groups.append((start, end))
        print(f"  Group {i+1}: units {start}-{end-1} ({end-start} units)")

    # Load data
    pixels, labels = load_and_prepare_mnist()

    # Subsample
    key = jax.random.key(42)
    key, k_sub = jax.random.split(key)

    n_label_features = labels.shape[1]
    n_visible = n_pixel_features + n_label_features

    visible_data = subsample_mnist(pixels, labels, n_samples, n_pixel_features, k_sub)

    # Create model
    key, k_model = jax.random.split(key)
    model, visible_block, hidden_block = create_rbm_model(n_visible, n_hidden, k_model)

    print(f"\nModel structure:")
    print(f"  Visible: {len(visible_block)} nodes")
    print(f"  Hidden: {len(hidden_block)} nodes")

    # Track results including group-wise MI
    from mnist_real_training import compute_cd_gradients
    from thrml.models.ising import IsingEBM
    import time

    results_new = {
        'epochs': [],
        'bayesian_mi': [],  # MI using all hidden units
    }

    # Add tracking for each group
    for i in range(n_groups):
        results_new[f'group_{i+1}_mi'] = []

    print("\n" + "=" * 70)
    print("TRAINING WITH GROUP-WISE MI TRACKING")
    print("=" * 70)

    for epoch in range(n_epochs):
        epoch_start = time.time()

        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{n_epochs}")
        print(f"{'='*70}")

        # Training with CD
        print(f"\nTraining with CD-{cd_steps}...")

        key, k_shuffle = jax.random.split(key)
        perm = jax.random.permutation(k_shuffle, n_samples)
        shuffled_data = visible_data[perm]

        n_batches = max(1, n_samples // batch_size)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = shuffled_data[start_idx:end_idx]

            key, k_cd = jax.random.split(key)
            weight_grad, bias_grad = compute_cd_gradients(
                model, visible_block, hidden_block,
                batch_data, cd_steps, k_cd
            )

            # Update parameters
            new_weights = model.weights + learning_rate * weight_grad
            new_biases = model.biases + learning_rate * bias_grad

            model = IsingEBM(
                model.nodes, model.edges,
                new_biases, new_weights, model.beta
            )

        # Compute MI for full hidden layer (baseline)
        print(f"\nComputing MI...")
        key, k_mi = jax.random.split(key)

        n_mi_samples = min(info_sampling_chains, n_samples)
        mi_data = visible_data[:n_mi_samples]

        # Sample all hidden units
        hidden_samples = sample_hidden_given_visible(
            model, visible_block, hidden_block, mi_data,
            n_mi_samples, n_gibbs_steps=100, key=k_mi
        )

        label_data = mi_data[:, n_pixel_features:]

        # Full MI
        mi_full = estimate_discrete_mi(key, hidden_samples, label_data)
        print(f"  Full MI (all {n_hidden} units): {mi_full:.4f} bits")
        results_new['bayesian_mi'].append(float(mi_full))

        # Group-wise MI
        print(f"\n  Group-wise MI:")
        for i, (start, end) in enumerate(groups):
            # Extract just this group of hidden units
            hidden_group = hidden_samples[:, start:end]

            # Compute MI for this group
            mi_group = estimate_discrete_mi(key, hidden_group, label_data)
            results_new[f'group_{i+1}_mi'].append(float(mi_group))

            print(f"    Group {i+1} (units {start:2d}-{end-1:2d}): {mi_group:.4f} bits")

        results_new['epochs'].append(epoch)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f"rbm_grouped_mi_{timestamp}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results_new, f)

    print(f"\nResults saved to: {output_file}")

    return results_new, timestamp, n_groups


def plot_grouped_mi(results, timestamp, n_groups):
    """Plot MI for different groups of RBM hidden units."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = np.array(results['epochs'])
    bayesian_mi = np.array(results['bayesian_mi'])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color scheme for groups
    colors = ['#E63946', '#F77F00', '#06AED5', '#118AB2', '#073B4C']

    # Plot each group
    for i in range(n_groups):
        group_mi = np.array(results[f'group_{i+1}_mi'])
        ax.plot(epochs, group_mi, 'o-',
                linewidth=2.5,
                markersize=8,
                color=colors[i % len(colors)],
                label=f'Group {i+1}',
                zorder=10-i)

        print(f"Group {i+1}: {group_mi[0]:.3f} → {group_mi[-1]:.3f} bits")

    # Also plot full MI for reference (dashed line)
    ax.plot(epochs, bayesian_mi, '--',
            linewidth=2,
            color='black',
            alpha=0.5,
            label='Full (all units)',
            zorder=1)

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mutual Information I(Hidden Group; Labels) [bits]', fontsize=14, fontweight='bold')
    ax.set_title('Information About Labels in RBM Hidden Unit Groups',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')

    plt.tight_layout()

    output_path = f"results/rbm_grouped_mi_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"\n✅ Plot saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("RBM GROUPED MI ANALYSIS")
    print("Analyzing information in different groups of hidden units")
    print("🔬 " * 20 + "\n")

    # Train and compute group-wise MI
    results, timestamp, n_groups = compute_grouped_mi(
        results_file="results/mnist_real_results_20251117_014914.pkl",
        n_groups=5  # Split into 5 groups like the MLP
    )

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    plot_path = plot_grouped_mi(results, timestamp, n_groups)

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)

    # Print summary
    final_full_mi = results['bayesian_mi'][-1]
    print(f"\nFinal Results:")
    print(f"  Full MI (all units): {final_full_mi:.4f} bits")
    for i in range(n_groups):
        group_mi = results[f'group_{i+1}_mi'][-1]
        pct = (group_mi / final_full_mi * 100) if final_full_mi > 0 else 0
        print(f"  Group {i+1} MI: {group_mi:.4f} bits ({pct:.0f}% of full)")
