"""
Compute V-Information: information under computational constraints.

V-information measures how much information can be EXTRACTED given 
computational budget. For Gibbs sampling, the budget is the number of 
sampling steps.

Key idea from "A Theory of Usable Information Under Computational Constraints":
    V_C(X; Y) = max_{f ∈ C} I(f(X); Y)
    
where C is the class of functions computable within budget.

For Gibbs sampling with budget k steps, we define:
    V_{Gibbs-k}(Latent; Label) = information extractable from latent 
                                  using k Gibbs sampling steps
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key
from typing import Callable

from thrml.block_sampling import BlockSamplingProgram, SamplingSchedule, sample_states
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.block_management import Block

from compute_bayesian_mi import estimate_discrete_mi


def estimate_v_information(
    key: Key[Array, ""],
    model: IsingEBM,
    sampling_program: IsingSamplingProgram,
    latent_blocks: list[Block],
    label_blocks: list[Block],
    data_blocks: list[Block],
    data_samples: Array,  # [n_data, n_visible]
    n_gibbs_steps: int,
    n_chains: int = 100,
) -> float:
    """
    Estimate V-information with a budget of n_gibbs_steps.
    
    Process:
    1. Start from random initialization
    2. Run exactly n_gibbs_steps of Gibbs sampling
    3. Extract samples of latents and labels
    4. Compute MI between extracted latent and label samples
    
    This MI represents information extractable with given budget.
    
    Args:
        key: JAX random key
        model: Ising EBM
        sampling_program: Sampling program
        latent_blocks: Blocks of latent variables
        label_blocks: Blocks of label variables
        data_blocks: Blocks that are clamped to data
        data_samples: Data to clamp visible variables to
        n_gibbs_steps: Number of Gibbs steps (computational budget)
        n_chains: Number of parallel chains to run
        
    Returns:
        V-information in bits
    """
    
    # Create schedule with no warmup - we count every step
    schedule = SamplingSchedule(
        n_warmup=0,
        n_samples=1,  # Just one sample at the end
        steps_per_sample=n_gibbs_steps
    )
    
    # Sample a batch of data
    key, k_data = jax.random.split(key)
    n_data = data_samples.shape[0]
    data_indices = jax.random.choice(k_data, n_data, (n_chains,), replace=True)
    batch_data = data_samples[data_indices]
    
    # Initialize chains randomly
    key, k_init = jax.random.split(key)
    free_blocks = [b for b in [*latent_blocks, *label_blocks] if b not in data_blocks]
    init_states = hinton_init(k_init, model, free_blocks, (n_chains,))
    
    # Run Gibbs sampling for exactly n_gibbs_steps
    key, k_sample = jax.random.split(key)
    keys = jax.random.split(k_sample, n_chains)
    
    # Sample latents and labels together
    all_observed_blocks = latent_blocks + label_blocks
    
    samples = jax.vmap(
        lambda k, init, data: sample_states(
            k, sampling_program, schedule, init, [data], all_observed_blocks
        )
    )(keys, init_states, batch_data)
    
    # samples shape: [n_chains, n_samples=1, n_vars]
    # Extract latent and label parts
    n_latent_vars = sum(len(b) for b in latent_blocks)
    
    samples = samples[:, 0, :]  # Remove sample dimension [n_chains, n_vars]
    
    latent_samples = samples[:, :n_latent_vars]
    label_samples = samples[:, n_latent_vars:]
    
    # Compute MI between extracted samples
    v_info = estimate_discrete_mi(key, latent_samples, label_samples)
    
    return v_info


def compute_v_information_curve(
    key: Key[Array, ""],
    model: IsingEBM,
    sampling_program: IsingSamplingProgram,
    latent_blocks: list[Block],
    label_blocks: list[Block],
    data_blocks: list[Block],
    data_samples: Array,
    step_budgets: list[int],
    n_chains: int = 100,
) -> dict[int, float]:
    """
    Compute V-information for multiple sampling budgets.
    
    This shows how extractable information grows with computational budget.
    
    Returns:
        Dictionary mapping budget (num steps) to V-information (bits)
    """
    results = {}
    
    for budget in step_budgets:
        key, subkey = jax.random.split(key)
        v_info = estimate_v_information(
            subkey,
            model,
            sampling_program,
            latent_blocks,
            label_blocks,
            data_blocks,
            data_samples,
            n_gibbs_steps=budget,
            n_chains=n_chains,
        )
        results[budget] = v_info
        print(f"Budget: {budget} steps -> V-info: {v_info:.3f} bits")
    
    return results


def information_gap(bayesian_mi: float, v_info: float) -> float:
    """
    Compute the gap between total information and extractable information.
    
    This represents "unusable" information - it's encoded but can't be
    extracted with available computation.
    """
    return max(0.0, bayesian_mi - v_info)


if __name__ == "__main__":
    print("Testing V-information computation...")
    print("=" * 60)
    
    # This will be integrated with actual MNIST training
    # For now, just demonstrate the concept
    
    key = jax.random.key(42)
    
    # Simulate: as we increase sampling budget, we extract more information
    print("\nExpected behavior:")
    print("As Gibbs steps increase, V-information should approach true MI")
    print()
    
    # True MI (if we had infinite samples)
    true_mi = 2.5  # bits
    
    # V-information at different budgets
    budgets = [1, 5, 10, 50, 100, 500, 1000]
    
    print(f"True MI: {true_mi:.2f} bits")
    print()
    print("Budget (steps) | V-info (bits) | Gap (bits) | % Extractable")
    print("-" * 60)
    
    for budget in budgets:
        # Model: V-info approaches true MI logarithmically
        v_info = true_mi * (1 - jnp.exp(-budget / 100))
        gap = true_mi - v_info
        pct = (v_info / true_mi) * 100
        print(f"{budget:14d} | {v_info:13.2f} | {gap:10.2f} | {pct:13.1f}%")
    
    print()
    print("Key insight: Even if latents encode information (high Bayesian MI),")
    print("we can only extract it with sufficient computational budget!")


