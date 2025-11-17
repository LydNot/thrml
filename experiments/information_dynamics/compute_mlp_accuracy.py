"""Compute actual MLP classification accuracy."""
import jax
import jax.numpy as jnp
import equinox as eqx
import pickle
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

print("="*70)
print("MLP CLASSIFICATION ACCURACY")
print("="*70)

# Load the trained MLP model (multi-layer version)
print("\nLoading trained MLP model...")
try:
    # Try to load from our multi-layer experiment
    with open('results/mnist_mlp_results_20251117_033314.pkl', 'rb') as f:
        results = pickle.load(f)
    print("  ✓ Model results loaded")
except:
    print("  ✗ Could not load model file")
    sys.exit(1)

# Load MNIST data
print("\nLoading MNIST test data...")
data = np.load('../../tests/mnist_test_data/train_data_filtered.npy')
n_pixels = 28 * 28
pixels = jnp.array(data[:, :n_pixels].astype(np.float32))
labels = jnp.array(data[:, n_pixels:].astype(np.float32))

print(f"  Data shape: {pixels.shape}")
print(f"  Labels shape: {labels.shape}")

# Subsample to match training
n_samples = 2000
n_pixel_features = 100
pixels_sub = pixels[:n_samples]
labels_sub = labels[:n_samples]

# Subsample pixels (same as training)
stride = int(np.ceil(np.sqrt(pixels.shape[1] / n_pixel_features)))
pixel_indices = jnp.arange(0, pixels.shape[1], stride)[:n_pixel_features]
pixels_sub = pixels_sub[:, pixel_indices]

print(f"  Using {n_samples} samples with {n_pixel_features} pixel features")

# Decode labels (30-bit one-hot → 3 classes)
def decode_labels(labels):
    """Decode 30-bit replicated one-hot to class indices (0, 1, 2)."""
    pattern = labels[:, :3].astype(int)
    # [1,0,0]→0, [0,1,0]→1, [0,0,1]→2
    return pattern[:, 0] * 0 + pattern[:, 1] * 1 + pattern[:, 2] * 2

true_classes = decode_labels(labels_sub)
print(f"\n  True class distribution:")
for i in range(3):
    count = jnp.sum(true_classes == i)
    print(f"    Class {i}: {count} samples ({count/len(true_classes)*100:.1f}%)")

# Problem: We saved results but not the trained model itself!
# We only have the training trajectory, not the final model weights
print("\n⚠ Issue: Model weights were not saved, only training metrics!")
print("   The pickle file contains training history, not the model.")
print("\n   To get real accuracy, we'd need to:")
print("   1. Re-train the model (or save weights during training)")
print("   2. Load the saved model")
print("   3. Run inference")

# Instead, let's estimate from the linear probe MI
if 'layer_3_mi' in results:
    final_mi = results['layer_3_mi'][-1]  # Layer 3 had highest MI
    print(f"\n   Best available metric: Layer 3 MI = {final_mi:.4f} bits")
    est_acc = 1/3 + (1 - 1/3) * (final_mi / 1.585)
    print(f"   Estimated accuracy: ~{est_acc*100:.1f}%")

print("\n" + "="*70)
print("SOLUTION: Re-run with model saving enabled")
print("="*70)
