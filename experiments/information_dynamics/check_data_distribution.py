"""Check class distribution in MNIST data."""
import numpy as np

data = np.load('../../tests/mnist_test_data/train_data_filtered.npy')
n_pixels = 28 * 28
labels = data[:, n_pixels:]

# Decode labels
def decode_labels(labels):
    pattern = labels[:, :3].astype(int)
    return pattern[:, 0] * 0 + pattern[:, 1] * 1 + pattern[:, 2] * 2

all_classes = decode_labels(labels)

print("Full Dataset Class Distribution:")
print("="*50)
for i, digit in enumerate([0, 3, 4]):
    count = np.sum(all_classes == i)
    pct = count / len(all_classes) * 100
    print(f"  Digit {digit} (class {i}): {count:5d} samples ({pct:.1f}%)")

print(f"\n  Total: {len(all_classes)} samples")

# Check first 2000 (what we use for training)
train_classes = all_classes[:2000]
print("\nFirst 2000 Samples (Training Set):")
print("="*50)
for i, digit in enumerate([0, 3, 4]):
    count = np.sum(train_classes == i)
    pct = count / len(train_classes) * 100
    print(f"  Digit {digit} (class {i}): {count:5d} samples ({pct:.1f}%)")
