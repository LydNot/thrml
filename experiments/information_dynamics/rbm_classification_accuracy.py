"""Measure RBM classification accuracy via conditional sampling."""
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

print("\n" + "="*70)
print("RBM CLASSIFICATION ACCURACY")
print("="*70)

# Load trained RBM
print("\nLoading trained RBM model...")
try:
    with open('results/mnist_real_results_20251117_014743.pkl', 'rb') as f:
        results = pickle.load(f)
    print("  ✓ RBM results loaded")
    
    # Extract final weights
    W = results['final_W']
    b_v = results['final_b_v']
    b_h = results['final_b_h']
    
    print(f"  Weight matrix: {W.shape}")
    print(f"  Visible bias: {b_v.shape}")
    print(f"  Hidden bias: {b_h.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Load MNIST data
print("\nLoading MNIST data...")
data = np.load('../../tests/mnist_test_data/train_data_filtered.npy')
n_pixels = 28 * 28
pixels = data[:, :n_pixels].astype(np.float32)
labels_data = data[:, n_pixels:].astype(np.float32)

# Subsample (same as training)
n_samples = 2000
n_pixel_features = 100
stride = int(np.ceil(np.sqrt(pixels.shape[1] / n_pixel_features)))
pixel_indices = jnp.arange(0, pixels.shape[1], stride)[:n_pixel_features]
pixels_sub = jnp.array(pixels[:n_samples, pixel_indices])
labels_sub = jnp.array(labels_data[:n_samples])

def decode_labels(labels):
    """Decode 30-bit pattern to class indices."""
    pattern = labels[:, :3].astype(int)
    return pattern[:, 0] * 0 + pattern[:, 1] * 1 + pattern[:, 2] * 2

true_classes = decode_labels(labels_sub)

print(f"  Samples: {n_samples}")
print(f"  Pixel features: {n_pixel_features}")
print(f"  Label features: 30")

# RBM inference: P(labels | pixels)
# For each sample, clamp pixels and sample labels via Gibbs
def gibbs_sample_labels(W, b_v, b_h, pixels, n_steps=100, key=jax.random.key(0)):
    """
    Sample labels conditioned on pixels using Gibbs sampling.
    
    Visible units = [pixels (100), labels (30)]
    We clamp pixels and sample labels.
    """
    batch_size = pixels.shape[0]
    n_pixels = pixels.shape[1]
    n_labels = 30
    n_hidden = b_h.shape[0]
    
    # Initialize labels randomly
    key, subkey = jax.random.split(key)
    labels = jax.random.bernoulli(subkey, 0.5, shape=(batch_size, n_labels)).astype(jnp.float32)
    
    # Gibbs sampling
    for step in range(n_steps):
        # Combine pixels (clamped) and labels (sampled)
        v = jnp.concatenate([pixels, labels], axis=1)
        
        # Sample hidden given visible
        h_prob = jax.nn.sigmoid(jnp.dot(v, W) + b_h)
        key, subkey = jax.random.split(key)
        h = jax.random.bernoulli(subkey, h_prob).astype(jnp.float32)
        
        # Sample labels given hidden (pixels stay clamped)
        v_prob = jax.nn.sigmoid(jnp.dot(h, W.T) + b_v)
        label_prob = v_prob[:, n_pixels:]  # Only label part
        
        key, subkey = jax.random.split(key)
        labels = jax.random.bernoulli(subkey, label_prob).astype(jnp.float32)
    
    return labels

print("\nRunning Gibbs sampling for classification...")
print("  (This may take a while - sampling 2000 examples)")

# Sample labels for all test examples
key = jax.random.key(42)
sampled_labels = gibbs_sample_labels(W, b_v, b_h, pixels_sub, n_steps=100, key=key)

# Decode sampled labels
sampled_classes = decode_labels(sampled_labels)

# Compute accuracy
correct = (sampled_classes == true_classes).sum()
accuracy = float(correct) / len(true_classes)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")

print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{len(true_classes)})")
print(f"vs Random Baseline: 33.3%")

# Per-class accuracy
print(f"\nPer-Class Performance:")
for i, digit in enumerate([0, 3, 4]):
    mask = true_classes == i
    class_correct = (sampled_classes[mask] == i).sum()
    class_total = mask.sum()
    class_acc = float(class_correct) / class_total
    print(f"  Digit {digit}: {class_acc:.2%} ({class_correct}/{class_total})")

# Confusion matrix
print(f"\nConfusion Matrix:")
print(f"           Predicted")
print(f"         0    3    4")
for i, true_digit in enumerate([0, 3, 4]):
    print(f"True {true_digit}:", end="")
    for j in range(3):
        count = int(((true_classes == i) & (sampled_classes == j)).sum())
        print(f" {count:4}", end="")
    print()

print(f"\n{'='*70}")
print("COMPARISON: RBM vs MLP")
print(f"{'='*70}")
print(f"\nRBM:  {accuracy:.2%}")
print(f"MLP (20 epochs): 75.30%")
print(f"\nNote: RBM uses 100 Gibbs steps for inference")
