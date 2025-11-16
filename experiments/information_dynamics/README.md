# Information Dynamics in Energy-Based Models

Research project studying how **usable information** evolves during training of discrete EBMs, accounting for:
- **Bayesian uncertainty** from finite training data
- **Computational constraints** from limited sampling budgets

## Key Questions

1. How much information do latent representations encode about labels? (Bayesian MI)
2. How much of that information is actually extractable with limited sampling? (V-information)
3. How does this gap evolve during training?

## Experiments

- `train_mnist_with_info_tracking.py` - Main training script with information tracking
- `compute_bayesian_mi.py` - Bayesian mutual information computation
- `compute_v_information.py` - V-information with sampling budget constraints
- `visualize_results.py` - Plot information dynamics over training

## Expected Results

We expect to see:
- Bayesian MI increases during training (latents learn about labels)
- V-information depends heavily on sampling budget
- Gap between Bayesian MI and V-information reveals "unusable" information
- This gap may explain why EBMs are hard to work with in practice


