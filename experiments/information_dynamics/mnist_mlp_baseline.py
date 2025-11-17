"""
MLP Baseline for MNIST with Information Dynamics Tracking

This script trains a standard feedforward MLP on MNIST and tracks:
1. Bayesian MI between hidden representations and labels (deterministic)
2. V-information at different "budgets" (using dropout/noise for approximation)
3. Comparison with RBM energy-based approach

This provides a baseline to compare against the RBM's information dynamics.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Callable

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from compute_bayesian_mi import estimate_discrete_mi

# Check GPU availability
print(f"JAX devices: {jax.devices()}")
print(f"GPU count: {jax.device_count()}")
print()


class MLP(eqx.Module):
    """Multi-layer MLP for MNIST classification with layer-wise information tracking."""
    layers: list
    n_layers: int

    def __init__(self, n_input, n_hidden, n_output, key, n_layers=2):
        """
        Args:
            n_input: Input dimension
            n_hidden: Hidden layer size (int or list of ints for each layer)
            n_output: Output dimension
            key: Random key
            n_layers: Total number of layers (2 = input→hidden→output)
        """
        keys = jax.random.split(key, n_layers)
        self.n_layers = n_layers

        # Make n_hidden a list if it isn't already
        if isinstance(n_hidden, int):
            hidden_sizes = [n_hidden] * (n_layers - 1)
        else:
            hidden_sizes = list(n_hidden)

        self.layers = []

        # Input to first hidden
        self.layers.append(eqx.nn.Linear(n_input, hidden_sizes[0], key=keys[0]))

        # Hidden to hidden (if multiple hidden layers)
        for i in range(1, n_layers - 1):
            self.layers.append(eqx.nn.Linear(hidden_sizes[i-1], hidden_sizes[i], key=keys[i]))

        # Last hidden to output
        self.layers.append(eqx.nn.Linear(hidden_sizes[-1], n_output, key=keys[-1]))

    def __call__(self, x, return_hidden=False, return_all_layers=False):
        """Forward pass through network.

        Args:
            x: Input features
            return_hidden: If True, return (output, last_hidden)
            return_all_layers: If True, return (output, all_activations) list
        """
        activations = []  # Store activations at each layer

        # Forward through hidden layers
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = jax.nn.tanh(layer(h))
            activations.append(h)

        # Output layer (no activation)
        output = self.layers[-1](h)

        if return_all_layers:
            return output, activations
        elif return_hidden:
            return output, activations[-1]  # Last hidden layer
        return output


def load_and_prepare_mnist():
    """Load real MNIST data."""
    print("Loading MNIST data...")
    data = jnp.load('../../tests/mnist_test_data/train_data_filtered.npy')

    # Separate pixels and labels
    n_pixels_full = 28 * 28
    pixels = data[:, :n_pixels_full].astype(jnp.float32)
    labels = data[:, n_pixels_full:].astype(jnp.float32)

    print(f"  Loaded {data.shape[0]} samples")
    print(f"  Pixels: {pixels.shape}")
    print(f"  Labels: {labels.shape}")

    return pixels, labels


def subsample_mnist(pixels, labels, n_samples, n_pixel_features, key):
    """Subsample MNIST to manageable size."""
    n_total = pixels.shape[0]

    # Sample examples
    key, k1 = jax.random.split(key)
    indices = jax.random.choice(k1, n_total, (n_samples,), replace=False)

    pixels_sub = pixels[indices]
    labels_sub = labels[indices]

    # Subsample pixels (take every Nth pixel for speed)
    if n_pixel_features < pixels.shape[1]:
        stride = int(np.ceil(np.sqrt(pixels.shape[1] / n_pixel_features)))
        pixel_indices = jnp.arange(0, pixels.shape[1], stride)[:n_pixel_features]
        pixels_sub = pixels_sub[:, pixel_indices]

    print(f"  Subsampled to {n_samples} examples × {pixels_sub.shape[1]} pixels × {labels_sub.shape[1]} labels")

    return pixels_sub, labels_sub


def compute_loss(model, x, y):
    """Compute cross-entropy loss."""
    logits = jax.vmap(model)(x)
    # Binary cross-entropy for multi-label classification
    loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss


def train_step(model, opt_state, x_batch, y_batch, optimizer):
    """Single training step."""
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def get_all_layer_activations(model, x_data):
    """Get activations at all hidden layers."""
    def get_activations(x):
        _, activations = model(x, return_all_layers=True)
        return activations

    # Process all samples
    all_activations = jax.vmap(get_activations)(x_data)
    return all_activations  # List of [n_samples, layer_size] arrays


def get_hidden_representations(model, x_data, key=None, noise_std=0.0):
    """Extract hidden layer representations.

    Args:
        model: MLP model
        x_data: Input data
        key: Random key for adding noise (optional)
        noise_std: Standard deviation of Gaussian noise to add
    """
    def get_hidden(x):
        _, hidden = model(x, return_hidden=True)
        return hidden

    hidden = jax.vmap(get_hidden)(x_data)

    # Optionally add noise to simulate "sampling budget"
    if noise_std > 0 and key is not None:
        noise = jax.random.normal(key, hidden.shape) * noise_std
        hidden = hidden + noise

    return hidden


def hidden_to_binary(hidden, use_median=True):
    """Convert continuous hidden activations to binary for MI estimation.

    Args:
        hidden: Hidden activations [n_samples, n_hidden]
        use_median: If True, threshold at median per feature (adaptive)
                    If False, use 0.0 threshold (for tanh)

    Using median helps avoid imbalanced binary representations when
    activations are small or biased.
    """
    if use_median:
        # Threshold at median per feature
        medians = jnp.median(hidden, axis=0, keepdims=True)
        return (hidden > medians).astype(jnp.bool_)
    else:
        # Fixed threshold at 0 (midpoint of tanh)
        return (hidden > 0.0).astype(jnp.bool_)


def continuous_mi_knn(hidden, labels, k=3):
    """Estimate continuous MI using k-nearest neighbors.

    Uses scikit-learn's mutual_info_regression which estimates
    differential entropy via KNN density estimation.

    Args:
        hidden: Hidden activations [n_samples, n_hidden]
        labels: Labels [n_samples, n_labels]
        k: Number of neighbors for KNN estimation

    Returns:
        MI estimate in bits
    """
    from sklearn.feature_selection import mutual_info_regression
    import numpy as np

    hidden_np = np.array(hidden)
    labels_np = np.array(labels)

    # Decode labels to classes
    label_sums = labels_np.sum(axis=1)
    if np.std(label_sums) < 0.1:  # One-hot style
        if labels_np.shape[1] == 30:  # MNIST replicated
            pattern = labels_np[:, :3]
            label_classes = pattern[:, 0] * 0 + pattern[:, 1] * 1 + pattern[:, 2] * 2
        else:
            label_classes = np.argmax(labels_np, axis=1)
    else:
        label_classes = np.argmax(labels_np, axis=1)

    # Compute MI for continuous hidden vs discrete labels
    mi_values = mutual_info_regression(
        hidden_np,
        label_classes,
        n_neighbors=k,
        random_state=42
    )

    # Sum MI across all hidden dimensions (upper bound on total MI)
    total_mi = float(np.sum(mi_values)) / np.log(2)  # Convert nats to bits

    return total_mi


def linear_probe_mi(hidden, labels, key):
    """Estimate MI using linear probe as a stable lower bound.

    Trains a linear classifier from hidden states to labels and uses
    its performance as a proxy for mutual information.

    Args:
        hidden: Hidden activations [n_samples, n_hidden]
        labels: One-hot labels [n_samples, n_labels]
        key: Random key for train/test split

    Returns:
        MI estimate in bits (based on cross-entropy of linear probe)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Convert to numpy
    hidden_np = np.array(hidden)
    labels_np = np.array(labels)

    # Get label classes (decode one-hot or replicated encoding)
    label_sums = labels_np.sum(axis=1)
    if np.std(label_sums) < 0.1:  # One-hot style
        if labels_np.shape[1] == 30:  # MNIST replicated
            pattern = labels_np[:, :3]
            label_classes = pattern[:, 0] * 0 + pattern[:, 1] * 1 + pattern[:, 2] * 2
        else:
            label_classes = np.argmax(labels_np, axis=1)
    else:
        label_classes = np.argmax(labels_np, axis=1)

    # Train linear probe
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(hidden_np, label_classes)

    # Get predicted probabilities
    probs = clf.predict_proba(hidden_np)

    # Compute cross-entropy: H(Y|X) = -sum(p(y) * log(q(y|x)))
    n_samples = len(label_classes)
    cross_entropy = 0.0
    for i in range(n_samples):
        true_class = int(label_classes[i])
        pred_prob = probs[i, true_class]
        cross_entropy += -np.log2(pred_prob + 1e-10)
    cross_entropy /= n_samples

    # MI = H(Y) - H(Y|X)
    # H(Y) from empirical distribution
    unique, counts = np.unique(label_classes, return_counts=True)
    p_y = counts / n_samples
    h_y = -np.sum(p_y * np.log2(p_y + 1e-10))

    mi = h_y - cross_entropy
    mi = max(0.0, mi)  # Clamp to non-negative

    return float(mi)


def run_mlp_mnist_experiment(
    n_epochs=10,
    n_hidden=50,
    n_samples=1000,
    n_pixel_features=100,
    info_sampling_chains=50,
    budgets=[0.0, 0.1, 0.5],  # Noise levels for V-info approximation
    learning_rate=0.001,
    batch_size=100,
):
    """
    Main experiment: MLP training with information tracking.

    Args:
        budgets: Noise standard deviations for V-info (0 = deterministic)
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("MLP BASELINE: MNIST WITH INFORMATION DYNAMICS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Hidden units: {n_hidden}")
    print(f"  Training samples: {n_samples}")
    print(f"  Pixel features: {n_pixel_features}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Noise budgets: {budgets}")
    print()

    # Load data
    pixels, labels = load_and_prepare_mnist()

    # Subsample
    key = jax.random.key(42)
    key, k_sub = jax.random.split(key)

    pixels_sub, labels_sub = subsample_mnist(pixels, labels, n_samples, n_pixel_features, k_sub)

    n_input = pixels_sub.shape[1]
    n_output = labels_sub.shape[1]

    # Create model with 5 hidden layers for deep information tracking
    key, k_model = jax.random.split(key)
    # Gradually decreasing layer sizes: 100 → 80 → 60 → 40 → 20
    hidden_sizes = [n_hidden, 80, 60, 40, 20]
    model = MLP(n_input, hidden_sizes, n_output, k_model, n_layers=6)

    print(f"\nModel structure (5 hidden layers):")
    print(f"  Input: {n_input} features")
    for i, size in enumerate(hidden_sizes):
        print(f"  Hidden {i+1}: {size} units (tanh)")
    print(f"  Output: {n_output} labels (sigmoid)")

    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Track results
    results = {
        'epochs': [],
        'train_loss': [],
        'bayesian_mi': [],
    }
    for budget in budgets:
        results[f'v_info_{budget}'] = []

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(n_epochs):
        epoch_start = time.time()

        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{n_epochs}")
        print(f"{'='*70}")

        # Training with mini-batches
        print(f"\nTraining...")
        key, k_shuffle = jax.random.split(key)
        perm = jax.random.permutation(k_shuffle, n_samples)
        pixels_shuffled = pixels_sub[perm]
        labels_shuffled = labels_sub[perm]

        n_batches = max(1, n_samples // batch_size)
        epoch_loss = 0.0

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            x_batch = pixels_shuffled[start_idx:end_idx]
            y_batch = labels_shuffled[start_idx:end_idx]

            model, opt_state, loss = train_step(model, opt_state, x_batch, y_batch, optimizer)
            epoch_loss += loss

        avg_loss = epoch_loss / n_batches
        print(f"  Avg train loss: {avg_loss:.4f}")
        results['train_loss'].append(float(avg_loss))

        # Compute Bayesian MI at each layer
        print(f"\nComputing Bayesian MI at each layer...")
        key, k_mi = jax.random.split(key)

        n_mi_samples = min(info_sampling_chains, n_samples)
        mi_pixels = pixels_sub[:n_mi_samples]
        mi_labels = labels_sub[:n_mi_samples]

        # Get activations at all layers
        layer_activations = get_all_layer_activations(model, mi_pixels)

        # Compute MI for each hidden layer using linear probe (most stable)
        for layer_idx, layer_acts in enumerate(layer_activations):
            layer_mi = linear_probe_mi(layer_acts, mi_labels, k_mi)

            # Initialize storage if first epoch
            if f'layer_{layer_idx+1}_mi' not in results:
                results[f'layer_{layer_idx+1}_mi'] = []

            results[f'layer_{layer_idx+1}_mi'].append(float(layer_mi))
            print(f"  Layer {layer_idx+1} MI: {layer_mi:.4f} bits")

        # Also keep track of last layer for backward compatibility
        hidden_reps = layer_activations[-1]  # Last hidden layer
        mi_linear = results[f'layer_{len(layer_activations)}_mi'][-1]

        results['bayesian_mi'].append(float(mi_linear))
        if 'bayesian_mi_linear' not in results:
            results['bayesian_mi_linear'] = []
        results['bayesian_mi_linear'].append(float(mi_linear))

        # Compute V-information at different noise levels
        print(f"\nComputing V-information (with noise):")

        for budget in budgets:
            key, k_v = jax.random.split(key)

            # Use fewer samples for V-info
            n_v_samples = min(30, n_samples)
            v_pixels = pixels_sub[:n_v_samples]
            v_labels = labels_sub[:n_v_samples]

            # Add noise to hidden representations
            hidden_v = get_hidden_representations(model, v_pixels, k_v, noise_std=budget)
            hidden_v_binary = hidden_to_binary(hidden_v, use_median=False)
            labels_v_binary = (v_labels > 0.5).astype(jnp.bool_)

            v_info = estimate_discrete_mi(key, hidden_v_binary, labels_v_binary)

            gap = max(0, mi_linear - v_info)
            pct = (v_info / mi_linear * 100) if mi_linear > 0 else 0

            print(f"  Noise σ={budget:.2f}: {v_info:.4f} bits (gap: {gap:.4f}, {pct:.0f}% extractable)")
            results[f'v_info_{budget}'].append(float(v_info))

        results['epochs'].append(epoch)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    import pickle
    results_file = results_dir / f"mnist_mlp_results_{timestamp}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {results_file}")

    return results, timestamp


def plot_mlp_results(results, timestamp, save_dir="results"):
    """Create plots from MLP experimental results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Path(save_dir).mkdir(exist_ok=True)

    epochs = results['epochs']
    bayesian_mi = np.array(results['bayesian_mi'])
    train_loss = np.array(results['train_loss'])

    # Extract budget keys
    budget_keys = [k for k in results.keys() if k.startswith('v_info_')]
    budgets = [float(k.split('_')[-1]) for k in budget_keys]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MLP Baseline: Information Dynamics\n{timestamp}',
                 fontsize=16, fontweight='bold')

    colors = ['#F18F01', '#C73E1D', '#6A994E']

    # Plot 1: Layer-wise MI showing information flow through network
    ax = axes[0, 0]
    ax2 = ax.twinx()

    # Plot MI at each layer (5 colors for 5 layers)
    layer_colors = ['#E63946', '#F77F00', '#FCBF49', '#06AED5', '#118AB2']
    layer_keys = [k for k in results.keys() if k.startswith('layer_') and k.endswith('_mi')]
    layer_keys = sorted(layer_keys, key=lambda x: int(x.split('_')[1]))

    for i, layer_key in enumerate(layer_keys):
        layer_num = int(layer_key.split('_')[1])
        layer_mi = np.array(results[layer_key])
        ax.plot(epochs, layer_mi, 'o-', linewidth=2.5, markersize=8,
                color=layer_colors[i % len(layer_colors)],
                label=f'Layer {layer_num}', zorder=10-i)

    # Loss on right axis
    ax2.plot(epochs, train_loss, 's--', linewidth=2, markersize=6,
             color='gray', label='Train Loss', alpha=0.6)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Information (bits)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax.set_title('Layer-wise Information Flow', fontsize=13, fontweight='bold')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: V-info only
    ax = axes[0, 1]
    for i, (budget_key, budget) in enumerate(zip(budget_keys, budgets)):
        v_info = results[budget_key]
        ax.plot(epochs, v_info, 'o-', linewidth=2, markersize=8,
                color=colors[i % len(colors)], label=f'σ={budget:.2f}')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('V-Information (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Extractable Information by Noise Level', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Information gaps
    ax = axes[1, 0]
    for i, (budget_key, budget) in enumerate(zip(budget_keys, budgets)):
        v_info = np.array(results[budget_key])
        gap = np.maximum(0, bayesian_mi - v_info)
        ax.plot(epochs, gap, 'o-', linewidth=2, markersize=8,
                color=colors[i % len(colors)], label=f'Gap @ σ={budget:.2f}')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unusable Information (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Information Gap', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 4: Extraction efficiency
    ax = axes[1, 1]
    for i, (budget_key, budget) in enumerate(zip(budget_keys, budgets)):
        v_info = np.array(results[budget_key])
        eff = np.array([v/m*100 if m > 0 else 0 for v, m in zip(v_info, bayesian_mi)])
        ax.plot(epochs, eff, 'o-', linewidth=2, markersize=8,
                color=colors[i % len(colors)], label=f'σ={budget:.2f}')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Extraction Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('% of Information Extractable', fontsize=13, fontweight='bold')
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()

    output_path = f"{save_dir}/mnist_mlp_baseline_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("MLP BASELINE: MNIST INFORMATION DYNAMICS")
    print("Standard feedforward network with backpropagation")
    print("🔬 " * 20 + "\n")

    # Run experiment
    results, timestamp = run_mlp_mnist_experiment(
        n_epochs=20,  # Longer training to see information dynamics
        n_hidden=100,
        n_samples=2000,
        n_pixel_features=100,
        info_sampling_chains=50,
        budgets=[0.0, 0.1, 0.5],  # Noise levels: deterministic, low noise, high noise
        learning_rate=0.001,
        batch_size=100,
    )

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_path = plot_mlp_results(results, timestamp)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ EXPERIMENT COMPLETE - MLP BASELINE RESULTS!")
    print("=" * 70)

    final_mi = results['bayesian_mi'][-1]
    print(f"\nFinal Results:")
    print(f"  Bayesian MI: {final_mi:.4f} bits")
    print(f"  Train Loss: {results['train_loss'][-1]:.4f}")

    for budget in [0.0, 0.1, 0.5]:
        if f'v_info_{budget}' in results:
            v = results[f'v_info_{budget}'][-1]
            pct = (v / final_mi * 100) if final_mi > 0 else 0
            print(f"  V-info @ σ={budget:.2f}: {v:.4f} bits ({pct:.0f}% extractable)")

    print(f"\n📊 Visualization: {plot_path}")
    print("\n🔬 Compare this with the RBM results to see differences in information dynamics!")
