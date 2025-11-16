"""
Simplified demo: Compute information metrics on synthetic samples.

This bypasses the complex sampling setup and directly demonstrates:
1. Computing Bayesian MI from samples  
2. Computing V-information at different budgets
3. Visualizing the information gap

We simulate what would happen during EBM training without actually running full training.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from compute_bayesian_mi import estimate_discrete_mi, compute_conditional_entropy


def generate_synthetic_latent_label_samples(
    n_samples, n_latent, n_label, correlation_strength, key
):
    """
    Generate synthetic samples where latents and labels are correlated.
    
    The latents partially determine the labels with some noise.
    """
    key, k1, k2 = jax.random.split(key, 3)
    
    # Sample latents
    latent_samples = jax.random.bernoulli(k1, 0.5, (n_samples, n_latent))
    
    # Labels depend on latents with some correlation
    # Use sum of first few latents to determine label probabilities
    n_relevant = min(5, n_latent)
    latent_sum = jnp.sum(latent_samples[:, :n_relevant], axis=1)
    
    # Sigmoid function to get label probabilities
    label_logits = (latent_sum - n_relevant/2) * correlation_strength
    label_probs = jax.nn.sigmoid(label_logits)
    
    # Sample labels
    label_samples = jax.random.bernoulli(
        k2, 
        label_probs[:, None], 
        (n_samples, n_label)
    )
    
    return latent_samples, label_samples


def simulate_limited_sampling(latent_samples, label_samples, n_steps, key):
    """
    Simulate the effect of limited Gibbs sampling steps.
    
    With fewer steps, we get noisier estimates of the latent-label relationship.
    Model: extracted info = true latents + noise that decreases with more steps
    """
    n_samples = latent_samples.shape[0]
    
    # Noise decreases with more sampling steps (logarithmically)
    noise_scale = 1.0 / jnp.sqrt(1 + n_steps / 10.0)
    
    # Add noise to latents (simulating imperfect sampling)
    noise = jax.random.normal(key, latent_samples.shape) * noise_scale
    noisy_latents = jax.nn.sigmoid(
        (latent_samples.astype(float) * 2 - 1) + noise
    )
    
    # Convert back to binary
    extracted_latents = (noisy_latents > 0.5).astype(jnp.bool_)
    
    return extracted_latents


def run_information_dynamics_simulation(n_epochs=10):
    """
    Simulate information dynamics over training.
    
    We simulate:
    - Latents learn more about labels over time (increasing MI)
    - V-information depends on sampling budget
    - Information gap persists even as total information grows
    """
    
    print("=" * 70)
    print("SIMULATED INFORMATION DYNAMICS EXPERIMENT")
    print("=" * 70)
    
    n_latent = 30
    n_label = 10
    n_samples = 200
    
    results = {
        'epochs': [],
        'bayesian_mi': [],
        'bayesian_mi_std': [],
        'v_info_10': [],
        'v_info_50': [],
        'v_info_200': [],
        'gap_10': [],
        'gap_50': [],
        'gap_200': [],
    }
    
    key = jax.random.key(42)
    
    print(f"\nSetup: {n_latent} latent vars, {n_label} label vars")
    print(f"Sampling {n_samples} data points per epoch\n")
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print("-" * 50)
        
        # Correlation increases during training (learning)
        correlation_strength = 0.5 + epoch * 0.3
        
        key, k_data = jax.random.split(key)
        
        # Generate samples at current training level
        latent_samples, label_samples = generate_synthetic_latent_label_samples(
            n_samples, n_latent, n_label, correlation_strength, k_data
        )
        
        # Compute Bayesian MI (with uncertainty from finite samples)
        key, k_mi = jax.random.split(key)
        bayesian_mi = estimate_discrete_mi(k_mi, latent_samples, label_samples)
        
        # Estimate uncertainty by bootstrap
        key, k_boot = jax.random.split(key)
        n_bootstrap = 5
        mi_bootstrap = []
        for _ in range(n_bootstrap):
            key, k_resample = jax.random.split(key)
            indices = jax.random.choice(k_resample, n_samples, (n_samples,), replace=True)
            mi_boot = estimate_discrete_mi(
                k_resample,
                latent_samples[indices],
                label_samples[indices]
            )
            mi_bootstrap.append(mi_boot)
        
        mi_std = float(jnp.std(jnp.array(mi_bootstrap)))
        
        print(f"  Bayesian MI: {bayesian_mi:.3f} ± {mi_std:.3f} bits")
        
        # Compute V-information at different budgets
        print(f"  V-information at different sampling budgets:")
        v_infos = {}
        
        for budget in [10, 50, 200]:
            key, k_budget = jax.random.split(key)
            
            # Simulate limited sampling
            extracted_latents = simulate_limited_sampling(
                latent_samples, label_samples, budget, k_budget
            )
            
            # Compute MI with extracted (noisy) latents
            v_info = estimate_discrete_mi(k_budget, extracted_latents, label_samples)
            v_infos[budget] = v_info
            
            gap = max(0, bayesian_mi - v_info)
            pct = (v_info / bayesian_mi * 100) if bayesian_mi > 0 else 0
            
            print(f"    {budget:3d} steps: {v_info:.3f} bits "
                  f"(gap: {gap:.3f}, {pct:.0f}% extractable)")
        
        # Record results
        results['epochs'].append(epoch)
        results['bayesian_mi'].append(bayesian_mi)
        results['bayesian_mi_std'].append(mi_std)
        results['v_info_10'].append(v_infos[10])
        results['v_info_50'].append(v_infos[50])
        results['v_info_200'].append(v_infos[200])
        results['gap_10'].append(max(0, bayesian_mi - v_infos[10]))
        results['gap_50'].append(max(0, bayesian_mi - v_infos[50]))
        results['gap_200'].append(max(0, bayesian_mi - v_infos[200]))
        
        print()
    
    print("=" * 70)
    print("Simulation complete!")
    print("=" * 70)
    
    return results


def plot_results(results, save_dir="experiments/information_dynamics/results"):
    """Create publication-quality plots."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    epochs = results['epochs']
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Information Dynamics During EBM Training\n(Simulated Experiment)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: All information curves
    ax = axes[0, 0]
    mi_mean = np.array(results['bayesian_mi'])
    mi_std = np.array(results['bayesian_mi_std'])
    
    ax.plot(epochs, mi_mean, 'o-', linewidth=3, markersize=10,
            color='#A23B72', label='Bayesian MI (total)', zorder=10)
    ax.fill_between(epochs, mi_mean - mi_std, mi_mean + mi_std, 
                    alpha=0.2, color='#A23B72')
    
    ax.plot(epochs, results['v_info_10'], 's-', linewidth=2, markersize=7,
            color='#F18F01', label='V-info (10 Gibbs steps)', alpha=0.8)
    ax.plot(epochs, results['v_info_50'], '^-', linewidth=2, markersize=7,
            color='#C73E1D', label='V-info (50 Gibbs steps)', alpha=0.8)
    ax.plot(epochs, results['v_info_200'], 'd-', linewidth=2, markersize=7,
            color='#6A994E', label='V-info (200 Gibbs steps)', alpha=0.8)
    
    ax.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Information (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Information Content vs. Extractable Information', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoom on V-information
    ax = axes[0, 1]
    ax.plot(epochs, results['v_info_10'], 's-', linewidth=2, markersize=7,
            color='#F18F01', label='10 steps', alpha=0.8)
    ax.plot(epochs, results['v_info_50'], '^-', linewidth=2, markersize=7,
            color='#C73E1D', label='50 steps', alpha=0.8)
    ax.plot(epochs, results['v_info_200'], 'd-', linewidth=2, markersize=7,
            color='#6A994E', label='200 steps', alpha=0.8)
    
    ax.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('V-Information (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Extractable Information by Budget', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Information gaps
    ax = axes[1, 0]
    ax.plot(epochs, results['gap_10'], 's-', linewidth=2, markersize=7,
            color='#F18F01', label='Gap @ 10 steps', alpha=0.8)
    ax.plot(epochs, results['gap_50'], '^-', linewidth=2, markersize=7,
            color='#C73E1D', label='Gap @ 50 steps', alpha=0.8)
    ax.plot(epochs, results['gap_200'], 'd-', linewidth=2, markersize=7,
            color='#6A994E', label='Gap @ 200 steps', alpha=0.8)
    
    ax.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unusable Information (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Information Gap: Bayesian MI - V-Information', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot 4: Extraction efficiency
    ax = axes[1, 1]
    eff_10 = np.array([v/m*100 if m > 0 else 0 
                       for v, m in zip(results['v_info_10'], mi_mean)])
    eff_50 = np.array([v/m*100 if m > 0 else 0 
                       for v, m in zip(results['v_info_50'], mi_mean)])
    eff_200 = np.array([v/m*100 if m > 0 else 0 
                        for v, m in zip(results['v_info_200'], mi_mean)])
    
    ax.plot(epochs, eff_10, 's-', linewidth=2, markersize=7,
            color='#F18F01', label='10 steps', alpha=0.8)
    ax.plot(epochs, eff_50, '^-', linewidth=2, markersize=7,
            color='#C73E1D', label='50 steps', alpha=0.8)
    ax.plot(epochs, eff_200, 'd-', linewidth=2, markersize=7,
            color='#6A994E', label='200 steps', alpha=0.8)
    
    ax.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Extraction Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('% of Information Extractable', 
                fontsize=13, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = f"{save_dir}/information_dynamics_simulation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("\nRunning information dynamics simulation...")
    print("(Simulating what happens during EBM training)\n")
    
    # Run simulation
    results = run_information_dynamics_simulation(n_epochs=10)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_path = plot_results(results)
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"\nFinal epoch:")
    final_mi = results['bayesian_mi'][-1]
    final_v10 = results['v_info_10'][-1]
    final_v50 = results['v_info_50'][-1]
    final_v200 = results['v_info_200'][-1]
    
    print(f"  Bayesian MI: {final_mi:.3f} bits (total information encoded)")
    print(f"  V-info @ 10 steps: {final_v10:.3f} bits ({final_v10/final_mi*100:.0f}% extractable)")
    print(f"  V-info @ 50 steps: {final_v50:.3f} bits ({final_v50/final_mi*100:.0f}% extractable)")
    print(f"  V-info @ 200 steps: {final_v200:.3f} bits ({final_v200/final_mi*100:.0f}% extractable)")
    
    print(f"\n🎯 Key Insight:")
    print(f"  Even though latents encode {final_mi:.2f} bits about labels,")
    print(f"  only {final_v10:.2f} bits are extractable with 10 Gibbs steps!")
    print(f"  This is the computational bottleneck EBMs face.")
    
    print(f"\n📊 Visualization: {plot_path}")
    print("\n✅ Demo complete! This demonstrates the core research idea.")

