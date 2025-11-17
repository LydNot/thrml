"""Train MLP and measure both MI and classification accuracy."""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from compute_bayesian_mi import estimate_discrete_mi
import time

print("\n" + "🧠"*35)
print("MLP: Measuring Both MI and Classification Accuracy")
print("🧠"*35 + "\n")

# Multi-layer MLP
class MLP(eqx.Module):
    layers: list
    
    def __init__(self, n_input, hidden_sizes, n_output, key):
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        self.layers = []
        
        # Input → first hidden
        self.layers.append(eqx.nn.Linear(n_input, hidden_sizes[0], key=keys[0]))
        
        # Hidden → hidden
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(eqx.nn.Linear(hidden_sizes[i], hidden_sizes[i+1], key=keys[i+1]))
        
        # Last hidden → output
        self.layers.append(eqx.nn.Linear(hidden_sizes[-1], n_output, key=keys[-1]))
    
    def __call__(self, x, return_all=False):
        activations = []
        h = x
        for layer in self.layers[:-1]:
            h = jax.nn.tanh(layer(h))
            activations.append(h)
        output = self.layers[-1](h)
        
        if return_all:
            return output, activations
        return output

def linear_probe_mi(hidden, labels):
    """Quick MI via linear probe."""
    from sklearn.linear_model import LogisticRegression
    hidden_np = np.array(hidden)
    labels_np = np.array(labels)
    
    # Decode labels
    pattern = labels_np[:, :3].astype(int)
    label_classes = pattern[:, 0] * 0 + pattern[:, 1] * 1 + pattern[:, 2] * 2
    
    # Train linear probe
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(hidden_np, label_classes)
    probs = clf.predict_proba(hidden_np)
    
    # Compute MI from cross-entropy
    h_y = np.log2(3)  # 3 classes
    ce = -np.mean([np.log2(probs[i, label_classes[i]] + 1e-10) for i in range(len(label_classes))])
    mi = h_y - ce
    return max(0.0, mi)

# Load data
print("Loading MNIST...")
data = np.load('../../tests/mnist_test_data/train_data_filtered.npy')
n_pixels_full = 28 * 28
pixels = data[:, :n_pixels_full].astype(np.float32)
labels_data = data[:, n_pixels_full:].astype(np.float32)

# Subsample (same as experiments)
n_samples = 2000
n_pixel_features = 100
stride = int(np.ceil(np.sqrt(pixels.shape[1] / n_pixel_features)))
pixel_indices = jnp.arange(0, pixels.shape[1], stride)[:n_pixel_features]
pixels_sub = jnp.array(pixels[:n_samples, pixel_indices])
labels_sub = jnp.array(labels_data[:n_samples])

# Decode to class indices
def decode_labels(labels):
    pattern = labels[:, :3].astype(int)
    return pattern[:, 0] * 0 + pattern[:, 1] * 1 + pattern[:, 2] * 2

true_classes = decode_labels(labels_sub)
labels_onehot = jax.nn.one_hot(true_classes, 3)

print(f"  Samples: {n_samples}, Features: {n_pixel_features}, Classes: 3")

# Create model (3 hidden layers like our best experiment)
key = jax.random.key(42)
hidden_sizes = [100, 50, 25]
model = MLP(n_pixel_features, hidden_sizes, 3, key)

print(f"\nArchitecture:")
print(f"  Input: {n_pixel_features}")
for i, h in enumerate(hidden_sizes):
    print(f"  Hidden {i+1}: {h} units")
print(f"  Output: 3 classes")

# Training
optimizer = optax.adam(0.001)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def loss_fn(model, x, y):
    logits = jax.vmap(model)(x)
    return optax.softmax_cross_entropy(logits, y).mean()

@eqx.filter_jit  
def train_step(model, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

print(f"\n{'='*70}")
print("TRAINING (10 epochs)")
print(f"{'='*70}\n")

start = time.time()
for epoch in range(10):
    model, opt_state, loss = train_step(model, opt_state, pixels_sub, labels_onehot)
    print(f"  Epoch {epoch+1:2d}: loss = {loss:.4f}")

training_time = time.time() - start
print(f"\nTraining completed in {training_time:.1f}s")

# Get all layer activations
print(f"\n{'='*70}")
print("MEASURING PERFORMANCE")  
print(f"{'='*70}\n")

def get_all_activations(model, x):
    _, acts = jax.vmap(lambda x: model(x, return_all=True))(x)
    return [jnp.array(acts[i]) for i in range(len(acts))]

layer_activations = get_all_activations(model, pixels_sub)

# Compute MI for each layer
print("Layer-wise Mutual Information (MI with labels):")
layer_mis = []
for i, acts in enumerate(layer_activations):
    mi = linear_probe_mi(acts, labels_sub)
    layer_mis.append(mi)
    print(f"  Layer {i+1} ({acts.shape[1]} units): MI = {mi:.4f} bits ({mi/1.585*100:.1f}% of max)")

# Compute classification accuracy
logits = jax.vmap(model)(pixels_sub)
predictions = jnp.argmax(logits, axis=1)
correct = (predictions == true_classes).sum()
accuracy = float(correct) / len(true_classes)

print(f"\n{'='*70}")
print("CLASSIFICATION ACCURACY")
print(f"{'='*70}")
print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{len(true_classes)})")
print(f"vs Random Baseline: 33.3%")
print(f"Improvement: +{(accuracy - 0.333)*100:.1f} percentage points")

# Per-class accuracy
print(f"\nPer-Class Performance:")
for i in range(3):
    mask = true_classes == i
    class_correct = (predictions[mask] == i).sum()
    class_total = mask.sum()
    class_acc = float(class_correct) / class_total
    digit = ['0', '3', '4'][i]
    print(f"  Digit {digit}: {class_acc:.2%} ({class_correct}/{class_total} correct)")

# Confusion matrix  
print(f"\nConfusion Matrix:")
print(f"           Predicted")
print(f"         0    3    4")
for i, true_digit in enumerate(['0', '3', '4']):
    print(f"True {true_digit}:", end="")
    for j in range(3):
        count = int(((true_classes == i) & (predictions == j)).sum())
        print(f" {count:3}", end="")
    print()

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"\nBest Layer: Layer {np.argmax(layer_mis)+1} with MI = {max(layer_mis):.4f} bits")
print(f"Classification Accuracy: {accuracy:.2%}")
print(f"Training Time: {training_time:.1f}s")
print(f"\nConclusion: MLP achieves {accuracy:.1%} accuracy with {max(layer_mis):.2f} bits")
print(f"of mutual information between hidden states and labels.")
