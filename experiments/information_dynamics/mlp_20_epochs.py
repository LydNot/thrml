"""Train MLP for 20 epochs to see if it learns digit 4."""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

print("\n🧠 MLP: 20 Epochs Training\n")

# Multi-layer MLP
class MLP(eqx.Module):
    layers: list
    
    def __init__(self, n_input, hidden_sizes, n_output, key):
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        self.layers = []
        
        self.layers.append(eqx.nn.Linear(n_input, hidden_sizes[0], key=keys[0]))
        
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(eqx.nn.Linear(hidden_sizes[i], hidden_sizes[i+1], key=keys[i+1]))
        
        self.layers.append(eqx.nn.Linear(hidden_sizes[-1], n_output, key=keys[-1]))
    
    def __call__(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = jax.nn.tanh(layer(h))
        return self.layers[-1](h)

# Load data
data = np.load('../../tests/mnist_test_data/train_data_filtered.npy')
n_pixels_full = 28 * 28
pixels = data[:, :n_pixels_full].astype(np.float32)
labels_data = data[:, n_pixels_full:].astype(np.float32)

n_samples = 2000
n_pixel_features = 100
stride = int(np.ceil(np.sqrt(pixels.shape[1] / n_pixel_features)))
pixel_indices = jnp.arange(0, pixels.shape[1], stride)[:n_pixel_features]
pixels_sub = jnp.array(pixels[:n_samples, pixel_indices])
labels_sub = jnp.array(labels_data[:n_samples])

def decode_labels(labels):
    pattern = labels[:, :3].astype(int)
    return pattern[:, 0] * 0 + pattern[:, 1] * 1 + pattern[:, 2] * 2

true_classes = decode_labels(labels_sub)
labels_onehot = jax.nn.one_hot(true_classes, 3)

# Create model
key = jax.random.key(42)
hidden_sizes = [100, 50, 25]
model = MLP(n_pixel_features, hidden_sizes, 3, key)

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

def compute_accuracy(model, x, y_true):
    logits = jax.vmap(model)(x)
    predictions = jnp.argmax(logits, axis=1)
    return float((predictions == y_true).sum()) / len(y_true)

print("Training 20 epochs...")
for epoch in range(20):
    model, opt_state, loss = train_step(model, opt_state, pixels_sub, labels_onehot)
    if (epoch + 1) % 5 == 0:
        acc = compute_accuracy(model, pixels_sub, true_classes)
        print(f"  Epoch {epoch+1:2d}: loss = {loss:.4f}, accuracy = {acc:.2%}")

# Final evaluation
logits = jax.vmap(model)(pixels_sub)
predictions = jnp.argmax(logits, axis=1)
accuracy = float((predictions == true_classes).sum()) / len(true_classes)

print(f"\n{'='*60}")
print("FINAL RESULTS (20 epochs)")
print(f"{'='*60}")
print(f"\nOverall Accuracy: {accuracy:.2%}")

print(f"\nPer-Class Performance:")
for i, digit in enumerate([0, 3, 4]):
    mask = true_classes == i
    class_correct = (predictions[mask] == i).sum()
    class_total = mask.sum()
    class_acc = float(class_correct) / class_total
    print(f"  Digit {digit}: {class_acc:.2%} ({class_correct}/{class_total})")

print(f"\nConfusion Matrix:")
print(f"           Predicted")
print(f"         0    3    4")
for i, true_digit in enumerate([0, 3, 4]):
    print(f"True {true_digit}:", end="")
    for j in range(3):
        count = int(((true_classes == i) & (predictions == j)).sum())
        print(f" {count:4}", end="")
    print()
