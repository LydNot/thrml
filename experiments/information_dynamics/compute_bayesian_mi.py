"""
Compute Bayesian Mutual Information for discrete EBMs.

Bayesian MI accounts for uncertainty in the model parameters given finite data:
    I_Bayes(X; Y | D) = E_{p(θ|D)}[I(X; Y | θ)]

For discrete distributions, this requires:
1. Sampling model parameters from posterior p(θ|D)
2. For each sample, estimate I(X; Y | θ) via sampling
3. Average over parameter samples
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key
from typing import Callable

from thrml.block_sampling import BlockSamplingProgram, SamplingSchedule
from thrml.models.ising import IsingEBM, IsingSamplingProgram, estimate_moments


def estimate_discrete_mi(
    key: Key[Array, ""],
    latent_samples: Array,  # [n_samples, n_latent_vars]
    label_samples: Array,   # [n_samples, n_label_vars]
    n_bins: int = 2,  # Binary for spin models
) -> float:
    """
    Estimate mutual information between discrete latent and label variables.
    
    I(X; Y) = H(X) + H(Y) - H(X, Y)
    
    where H is the entropy.
    
    Args:
        key: JAX random key
        latent_samples: Binary samples from latent variables [n_samples, n_latent]
        label_samples: Binary samples from label variables [n_samples, n_label]
        n_bins: Number of discrete states (2 for binary/spin)
    
    Returns:
        Estimated mutual information in bits
    """
    n_samples = latent_samples.shape[0]
    
    # Convert boolean to int for easier processing
    latent_int = latent_samples.astype(jnp.int32)
    label_int = label_samples.astype(jnp.int32)
    
    # Compute empirical distributions
    # For simplicity, compute MI between aggregated latent and label states
    # (full joint over all variables would be exponentially large)
    
    # Aggregate by summing (proxy for joint state)
    latent_sum = jnp.sum(latent_int, axis=-1)  # [n_samples]
    label_sum = jnp.sum(label_int, axis=-1)    # [n_samples]
    
    # Compute empirical probabilities
    def compute_entropy(values: Array, max_val: int) -> float:
        """Compute entropy of discrete distribution."""
        counts = jnp.array([jnp.sum(values == i) for i in range(max_val + 1)])
        probs = counts / n_samples
        # Add small epsilon to avoid log(0)
        probs = jnp.clip(probs, 1e-10, 1.0)
        entropy = -jnp.sum(probs * jnp.log2(probs))
        return entropy
    
    max_latent = latent_samples.shape[1]
    max_label = label_samples.shape[1]
    
    h_latent = compute_entropy(latent_sum, max_latent)
    h_label = compute_entropy(label_sum, max_label)
    
    # Joint entropy - compute 2D histogram
    joint_counts = jnp.zeros((max_latent + 1, max_label + 1))
    for i in range(n_samples):
        l_val = latent_sum[i]
        lab_val = label_sum[i]
        joint_counts = joint_counts.at[l_val, lab_val].add(1)
    
    joint_probs = joint_counts / n_samples
    joint_probs = jnp.clip(joint_probs, 1e-10, 1.0)
    h_joint = -jnp.sum(joint_probs * jnp.log2(joint_probs))
    
    mi = h_latent + h_label - h_joint
    
    return float(mi)


def bayesian_mi_with_laplace_approximation(
    key: Key[Array, ""],
    model: IsingEBM,
    program: IsingSamplingProgram,
    schedule: SamplingSchedule,
    latent_blocks: list,
    label_blocks: list,
    data_samples: Array,  # [n_data_samples, n_visible]
    n_parameter_samples: int = 10,
    parameter_noise_scale: float = 0.1,
) -> tuple[float, float]:
    """
    Estimate Bayesian MI using Laplace approximation to posterior.
    
    Instead of full Bayesian inference, we:
    1. Use current parameters as MAP estimate
    2. Add Gaussian noise to simulate posterior uncertainty
    3. Compute MI for each parameter sample
    4. Return mean and std
    
    Args:
        key: JAX random key
        model: Current Ising EBM
        program: Sampling program
        schedule: Sampling schedule
        latent_blocks: Blocks representing latent variables
        label_blocks: Blocks representing labels
        data_samples: Data to condition on
        n_parameter_samples: Number of posterior samples
        parameter_noise_scale: Scale of Gaussian noise for parameters
        
    Returns:
        (mean_mi, std_mi): Mean and standard deviation of MI estimates
    """
    
    mis = []
    
    for i in range(n_parameter_samples):
        key, subkey = jax.random.split(key)
        
        # Sample parameters from approximate posterior
        # (Gaussian around current parameters)
        noise_w = jax.random.normal(subkey, model.weights.shape) * parameter_noise_scale
        noise_b = jax.random.normal(subkey, model.biases.shape) * parameter_noise_scale
        
        perturbed_weights = model.weights + noise_w
        perturbed_biases = model.biases + noise_b
        
        # Create perturbed model
        perturbed_model = IsingEBM(
            model.nodes,
            model.edges,
            perturbed_biases,
            perturbed_weights,
            model.beta
        )
        
        # TODO: Sample from model and compute MI
        # For now, return placeholder
        mi_estimate = 0.0
        mis.append(mi_estimate)
    
    mis = jnp.array(mis)
    return float(jnp.mean(mis)), float(jnp.std(mis))


def compute_conditional_entropy(
    latent_samples: Array,  # [n_samples, n_latent]
    label_samples: Array,   # [n_samples, n_label]
) -> float:
    """
    Compute H(Y|X) = H(X,Y) - H(X)
    
    This gives us how much uncertainty remains in Y after observing X.
    """
    # Simplified version using aggregated states
    n_samples = latent_samples.shape[0]
    
    latent_sum = jnp.sum(latent_samples.astype(jnp.int32), axis=-1)
    label_sum = jnp.sum(label_samples.astype(jnp.int32), axis=-1)
    
    max_latent = latent_samples.shape[1]
    max_label = label_samples.shape[1]
    
    # Compute H(X)
    latent_counts = jnp.array([jnp.sum(latent_sum == i) for i in range(max_latent + 1)])
    latent_probs = latent_counts / n_samples
    latent_probs = jnp.clip(latent_probs, 1e-10, 1.0)
    h_latent = -jnp.sum(latent_probs * jnp.log2(latent_probs))
    
    # Compute H(X,Y)
    joint_counts = jnp.zeros((max_latent + 1, max_label + 1))
    for i in range(n_samples):
        l_val = latent_sum[i]
        lab_val = label_sum[i]
        joint_counts = joint_counts.at[l_val, lab_val].add(1)
    
    joint_probs = joint_counts / n_samples
    joint_probs = jnp.clip(joint_probs, 1e-10, 1.0)
    h_joint = -jnp.sum(joint_probs * jnp.log2(joint_probs))
    
    conditional_entropy = h_joint - h_latent
    
    return float(conditional_entropy)


if __name__ == "__main__":
    # Quick test with synthetic data
    key = jax.random.key(42)
    
    # Generate correlated binary samples
    n_samples = 1000
    n_latent = 10
    n_label = 5
    
    key, k1, k2 = jax.random.split(key, 3)
    
    # Latents
    latent_samples = jax.random.bernoulli(k1, 0.5, (n_samples, n_latent))
    
    # Labels partially determined by latents (with noise)
    label_probs = jax.nn.sigmoid(jnp.sum(latent_samples[:, :n_label], axis=1, keepdims=True) - 2.5)
    labels = jax.random.bernoulli(k2, label_probs, (n_samples, n_label))
    
    mi = estimate_discrete_mi(key, latent_samples, labels)
    print(f"Estimated MI: {mi:.3f} bits")
    
    cond_ent = compute_conditional_entropy(latent_samples, labels)
    print(f"Conditional entropy H(Y|X): {cond_ent:.3f} bits")

