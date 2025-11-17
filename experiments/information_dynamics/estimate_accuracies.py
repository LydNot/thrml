"""Estimate classification accuracies from information content."""
import numpy as np

print("="*70)
print("ESTIMATING CLASSIFICATION ACCURACY FROM MUTUAL INFORMATION")
print("="*70)

# Relationship between MI and accuracy for 3-class problem
# MI = H(Y) - H(Y|X) where H(Y|X) is conditional entropy
# For perfect classification: H(Y|X) = 0, so MI = H(Y) = log2(3) = 1.585
# For random: H(Y|X) = H(Y), so MI = 0

def mi_to_accuracy_approx(mi, max_mi=1.585):
    """
    Rough approximation: assume uniform error distribution
    If we have fraction f of information, we get roughly that fraction correct
    Plus 1/3 baseline (random guessing among 3 classes)
    """
    info_fraction = mi / max_mi
    # Better than random guessing by this fraction
    random_acc = 1/3
    perfect_acc = 1.0
    return random_acc + (perfect_acc - random_acc) * info_fraction

def mi_to_cross_entropy(mi, n_classes=3):
    """
    MI = H(Y) - H(Y|X)
    H(Y) = log2(3) = 1.585 for uniform 3 classes
    H(Y|X) = H(Y) - MI
    
    Cross-entropy relates to H(Y|X)
    """
    h_y = np.log2(n_classes)
    h_y_given_x = h_y - mi
    return h_y_given_x

print("\nMutual Information → Estimated Accuracy (3 classes):\n")

models = {
    "Random Baseline": 0.03,
    "RBM (trained)": 0.82,
    "MLP (original comparison)": 0.57,
    "MLP (multi-layer, linear probe)": 1.30,
    "Perfect Classifier": 1.585,
}

print(f"{'Model':<35} {'MI (bits)':<12} {'Est. Accuracy':<15} {'H(Y|X)'}")
print("-"*70)

for name, mi in models.items():
    acc = mi_to_accuracy_approx(mi)
    h_y_x = mi_to_cross_entropy(mi)
    print(f"{name:<35} {mi:>6.3f}       {acc*100:>5.1f}%          {h_y_x:.3f} bits")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print("""
For 3-class classification:
  • Random guessing: 33.3% accuracy
  • RBM (0.82 bits): ~54-58% accuracy (estimated)
  • MLP original (0.57 bits): ~49-52% accuracy (estimated)
  • MLP improved (1.30 bits): ~70-75% accuracy (estimated)
  • Perfect (1.585 bits): 100% accuracy

These are rough estimates! Actual accuracy depends on:
  - How information is distributed across hidden units
  - Quality of the decoder/classifier
  - Class imbalance and confusion patterns

To get real accuracy, we'd need to:
  • MLP: Forward pass → argmax → compare to labels
  • RBM: Gibbs sample labels|pixels → compare to true labels
""")

# For MLP, we can estimate from loss
print("\nMLP Loss-based estimate:")
mlp_final_loss = 0.217
print(f"  Final loss: {mlp_final_loss:.3f}")
print(f"  Cross-entropy loss ≈ -log(p_correct)")
print(f"  Estimated p_correct ≈ exp(-{mlp_final_loss:.3f}) = {np.exp(-mlp_final_loss):.3f}")
print(f"  Estimated accuracy: ~{np.exp(-mlp_final_loss)*100:.1f}%")
