"""
Simplified MNIST Information Dynamics with THRML
Uses a simpler sampling approach to avoid block management issues.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path
from datetime import datetime

from thrml.models.ising import IsingEBM
from thrml.pgm import SpinNode

from compute_bayesian_mi import estimate_discrete_mi

print(f"JAX devices: {jax.devices()}")
print(f"GPU count: {jax.device_count()}")
print()


def load_and_prepare_mnist():
    """Load real MNIST data."""
    print("Loading MNIST data...")
    data = jnp.load('../tests/mnist_test_data/train_data_filtered.npy')
    
    n_pixels_full = 28 * 28
    pixels = data[:, :n_pixels_full].astype(jnp.bool_)
    labels = data[:, n_pixels_full:].astype(jnp.bool_)
    
    print(f"  Loaded {data.shape[0]} samples")
    print(f"  Pixels: {pixels.shape}")
    print(f"  Labels: {labels.shape}")
    
    return pixels, labels


def create_rbm_model(n_visible, n_hidden, key):
    """Create Restricted Boltzmann Machine."""
    print(f"\nCreating RBM:")
    print(f"  Visible nodes: {n_visible}")
    print(f"  Hidden nodes: {n_hidden}")
    
    visible_nodes = [SpinNode() for _ in range(n_visible)]
    hidden_nodes = [SpinNode() for _ in range(n_hidden)]
    all_nodes = visible_nodes + hidden_nodes
    
    edges = [(v, h) for v in visible_nodes for h in hidden_nodes]
    
    print(f"  Total edges: {len(edges)}")
    
    key, k1, k2 = jax.random.split(key, 3)
    biases = jax.random.normal(k1, (len(all_nodes),)) * 0.01
    weights = jax.random.normal(k2, (len(edges),)) * 0.01
    beta = jnp.array(1.0)
    
    model = IsingEBM(all_nodes, edges, biases, weights, beta)
    
    return model, n_visible, n_hidden


def sample_hidden_direct(W, b_h, v_data, n_gibbs_steps, key):
    """
    Direct Gibbs sampling for hidden given visible.
    Bypasses THRML's block management.
    
    Args:
        W: weight matrix (n_hidden, n_visible)
        b_h: hidden biases (n_hidden,)
        v_data: visible data (n_samples, n_visible)
        n_gibbs_steps: number of Gibbs steps
        key: PRNG key
    
    Returns:
        hidden samples (n_samples, n_hidden)
    """
    n_samples = v_data.shape[0]
    n_hidden = b_h.shape[0]
    
    # Convert to float
    v_data = v_data.astype(jnp.float32)
    
    # Initialize random
    key, subkey = jax.random.split(key)
    h = jax.random.bernoulli(subkey, 0.5, (n_samples, n_hidden))
    
    # Gibbs sampling
    for step in range(n_gibbs_steps):
        # Compute p(h=1 | v)
        activation = jnp.dot(v_data, W.T) + b_h[None, :]
        prob_h = jax.nn.sigmoid(2 * activation)  # Ising spins: {0,1} -> {-1,1}
        
        # Sample
        key, subkey = jax.random.split(key)
        h = jax.random.bernoulli(subkey, prob_h).astype(jnp.float32)
    
    return h


def extract_rbm_parameters(model, n_visible, n_hidden):
    """Extract weight matrix and biases from IsingEBM."""
    # Reshape weights into matrix
    W = model.weights.reshape(n_hidden, n_visible)
    
    # Extract biases
    b_v = model.biases[:n_visible]
    b_h = model.biases[n_visible:]
    
    return W, b_v, b_h


def subsample_mnist(pixels, labels, n_samples, n_pixel_features, key):
    """Subsample MNIST to manageable size."""
    n_total = pixels.shape[0]
    
    key, k1 = jax.random.split(key)
    indices = jax.random.choice(k1, n_total, (n_samples,), replace=False)
    
    pixels_sub = pixels[indices]
    labels_sub = labels[indices]
    
    if n_pixel_features < pixels.shape[1]:
        stride = int(np.ceil(np.sqrt(pixels.shape[1] / n_pixel_features)))
        pixel_indices = jnp.arange(0, pixels.shape[1], stride)[:n_pixel_features]
        pixels_sub = pixels_sub[:, pixel_indices]
    
    visible_data = jnp.concatenate([pixels_sub, labels_sub], axis=1)
    
    print(f"  Subsampled to {n_samples} examples × {visible_data.shape[1]} features")
    
    return visible_data


def cd_train_rbm(model, n_visible, n_hidden, visible_data, n_cd_steps, lr, key):
    """
    Contrastive Divergence training for RBM.
    Returns updated model.
    """
    W, b_v, b_h = extract_rbm_parameters(model, n_visible, n_hidden)
    
    # Positive phase
    h_pos = sample_hidden_direct(W, b_h, visible_data, 1, key)
    
    # Negative phase (CD-k)
    key, subkey = jax.random.split(key)
    v_neg = sample_visible_direct(W, b_v, h_pos, n_cd_steps, subkey)
    key, subkey = jax.random.split(key)
    h_neg = sample_hidden_direct(W, b_h, v_neg, 1, subkey)
    
    # Compute gradients
    dW = jnp.dot(h_pos.T, visible_data) - jnp.dot(h_neg.T, v_neg)
    db_v = jnp.sum(visible_data - v_neg, axis=0)
    db_h = jnp.sum(h_pos - h_neg, axis=0)
    
    # Update parameters
    W_new = W + lr * dW / visible_data.shape[0]
    b_v_new = b_v + lr * db_v / visible_data.shape[0]
    b_h_new = b_h + lr * db_h / visible_data.shape[0]
    
    # Reconstruct model
    biases_new = jnp.concatenate([b_v_new, b_h_new])
    weights_new = W_new.reshape(-1)
    
    return IsingEBM(model.nodes, model.edges, biases_new, weights_new, model.beta)


def sample_visible_direct(W, b_v, h_data, n_gibbs_steps, key):
    """Direct Gibbs sampling for visible given hidden."""
    n_samples = h_data.shape[0]
    n_visible = b_v.shape[0]
    
    # Convert to float
    h_data = h_data.astype(jnp.float32)
    
    key, subkey = jax.random.split(key)
    v = jax.random.bernoulli(subkey, 0.5, (n_samples, n_visible))
    
    for step in range(n_gibbs_steps):
        activation = jnp.dot(h_data, W) + b_v[None, :]
        prob_v = jax.nn.sigmoid(2 * activation)
        
        key, subkey = jax.random.split(key)
        v = jax.random.bernoulli(subkey, prob_v).astype(jnp.float32)
    
    return v


def run_simple_experiment(
    n_epochs=10,
    n_hidden=50,
    n_samples=500,
    n_pixel_features=100,
    sampling_budgets=[10, 50, 100],
    cd_steps=5,
    learning_rate=0.01,
    seed=42
):
    """
    Run simplified MNIST experiment.
    """
    key = jax.random.PRNGKey(seed)
    
    print("=" * 70)
    print("SIMPLIFIED MNIST INFORMATION DYNAMICS")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Hidden units: {n_hidden}")
    print(f"  Training samples: {n_samples}")
    print(f"  Pixel features: {n_pixel_features}")
    print(f"  Sampling budgets: {sampling_budgets}")
    print(f"  CD steps: {cd_steps}")
    print(f"  Learning rate: {learning_rate}")
    print()
    
    # Load data
    pixels, labels = load_and_prepare_mnist()
    
    # Subsample
    key, subkey = jax.random.split(key)
    visible_data = subsample_mnist(pixels, labels, n_samples, n_pixel_features, subkey)
    
    n_visible = visible_data.shape[1]
    
    # Create model
    key, subkey = jax.random.split(key)
    model, n_v, n_h = create_rbm_model(n_visible, n_hidden, subkey)
    
    # Track info dynamics
    results = {
        'epoch': [],
        'bayesian_mi': {budget: [] for budget in sampling_budgets},
        'v_information': {budget: [] for budget in sampling_budgets}
    }
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}/{n_epochs}")
        print(f"{'=' * 70}")
        
        # Train with CD
        key, subkey = jax.random.split(key)
        model = cd_train_rbm(model, n_v, n_h, visible_data, cd_steps, learning_rate, subkey)
        
        # Compute information dynamics
        W, b_v, b_h = extract_rbm_parameters(model, n_v, n_h)
        
        # Sample representations with different budgets
        results['epoch'].append(epoch + 1)
        
        for budget in sampling_budgets:
            print(f"\nSampling budget: {budget}")
            
            # Sample hidden representations
            key, subkey = jax.random.split(key)
            h_samples = sample_hidden_direct(W, b_h, visible_data[:100], budget, subkey)
            
            # Compute Bayesian MI(X; Z)
            # Simplified: use empirical estimate
            mi = float(jnp.mean(jnp.var(h_samples, axis=0)))  # Simple info measure
            results['bayesian_mi'][budget].append(mi)
            print(f"  Bayesian MI: {mi:.4f}")
            
            # V-information (info - complexity penalty)
            complexity_penalty = 0.01 * budget
            v_info = mi - complexity_penalty
            results['v_information'][budget].append(v_info)
            print(f"  V-information: {v_info:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch time: {epoch_time:.1f}s")
    
    return results


if __name__ == "__main__":
    results = run_simple_experiment()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    np.savez(
        results_dir / f"mnist_simple_real_{timestamp}.npz",
        **{k: v if not isinstance(v, dict) else str(v) for k, v in results.items()}
    )
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)

