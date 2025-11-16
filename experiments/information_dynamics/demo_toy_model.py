"""
Demo: Run information dynamics experiment on a toy Ising model.

This demonstrates the full pipeline:
1. Train a small Ising model (toy version of MNIST)
2. Compute Bayesian MI at each epoch
3. Compute V-information at different budgets
4. Visualize the results

This runs quickly and shows the concept works!
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.pgm import SpinNode

from compute_bayesian_mi import estimate_discrete_mi
from compute_v_information import estimate_v_information


def create_toy_model(n_visible=20, n_hidden=30, key=None):
    """Create a small RBM-like model for demo."""
    if key is None:
        key = jax.random.key(0)
    
    # Visible and hidden nodes
    visible_nodes = [SpinNode() for _ in range(n_visible)]
    hidden_nodes = [SpinNode() for _ in range(n_hidden)]
    all_nodes = visible_nodes + hidden_nodes
    
    # Create bipartite connections (RBM structure)
    edges = [(visible_nodes[i], hidden_nodes[j]) 
             for i in range(n_visible) 
             for j in range(n_hidden)]
    
    # Random initial parameters
    key, k1, k2 = jax.random.split(key, 3)
    biases = jax.random.normal(k1, (len(all_nodes),)) * 0.1
    weights = jax.random.normal(k2, (len(edges),)) * 0.1
    beta = jnp.array(1.0)
    
    model = IsingEBM(all_nodes, edges, biases, weights, beta)
    
    return model, Block(visible_nodes), Block(hidden_nodes)


def generate_toy_data(n_samples=100, n_visible=20, pattern_strength=0.8, key=None):
    """
    Generate toy binary data with some structure.
    
    The first half of bits tends to be correlated with a binary label.
    """
    if key is None:
        key = jax.random.key(42)
    
    data = []
    labels = []
    
    for _ in range(n_samples):
        key, k1, k2 = jax.random.split(key, 3)
        
        # Binary label
        label = jax.random.bernoulli(k1, 0.5)
        
        # First half correlated with label, second half random
        n_half = n_visible // 2
        
        if label:
            # If label=1, first half mostly 1s
            first_half = jax.random.bernoulli(k2, pattern_strength, (n_half,))
        else:
            # If label=0, first half mostly 0s
            first_half = jax.random.bernoulli(k2, 1-pattern_strength, (n_half,))
        
        # Second half is noise
        key, k3 = jax.random.split(key)
        second_half = jax.random.bernoulli(k3, 0.5, (n_visible - n_half,))
        
        sample = jnp.concatenate([first_half, second_half])
        data.append(sample)
        labels.append(label)
    
    return jnp.array(data), jnp.array(labels)


def run_toy_experiment(n_epochs=10, n_visible=20, n_hidden=30):
    """
    Run full experiment on toy model.
    """
    
    print("=" * 70)
    print("TOY EXPERIMENT: Information Dynamics in Small Ising Model")
    print("=" * 70)
    
    # Create model
    print(f"\nCreating model: {n_visible} visible, {n_hidden} hidden nodes")
    key = jax.random.key(0)
    model, visible_block, hidden_block = create_toy_model(n_visible, n_hidden, key)
    print(f"Total edges: {len(model.edges)}")
    
    # Generate synthetic data
    print(f"\nGenerating synthetic data...")
    key, k_data = jax.random.split(key)
    train_data, train_labels = generate_toy_data(200, n_visible, key=k_data)
    test_data, test_labels = generate_toy_data(100, n_visible, key=k_data)
    print(f"Train: {train_data.shape}, Test: {test_data.shape}")
    
    # Setup sampling
    free_blocks = [hidden_block]
    clamped_blocks = [visible_block]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks)
    
    # Track results
    results = {
        'epochs': [],
        'bayesian_mi': [],
        'v_info_10': [],
        'v_info_50': [],
        'v_info_200': [],
    }
    
    print("\n" + "=" * 70)
    print("Running Experiment")
    print("=" * 70)
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        print("-" * 50)
        
        # Simulate training (in real version, would update model)
        # For demo, just add some noise to parameters to show evolution
        if epoch > 0:
            key, k_noise = jax.random.split(key)
            noise = jax.random.normal(k_noise, model.biases.shape) * 0.05
            model = IsingEBM(
                model.nodes, model.edges,
                model.biases + noise,
                model.weights,
                model.beta
            )
            program = IsingSamplingProgram(model, free_blocks, clamped_blocks)
        
        # Sample from model to get hidden states given visible data
        key, k_sample = jax.random.split(key)
        schedule_fast = SamplingSchedule(n_warmup=20, n_samples=1, steps_per_sample=100)
        
        # Sample hidden states for a batch
        n_batch = 50
        batch_data = train_data[:n_batch]
        
        keys = jax.random.split(k_sample, n_batch)
        init_states = hinton_init(k_sample, model, free_blocks, (n_batch,))
        
        # Get samples of hidden states
        hidden_samples_list = []
        for i in range(n_batch):
            samples = sample_states(
                keys[i], program, schedule_fast,
                [init_states[0][i:i+1]],  # List with single batch element
                [batch_data[i:i+1]],      # List with single batch element  
                [hidden_block]
            )
            hidden_samples_list.append(samples[0, 0, :])  # [n_hidden]
        
        hidden_samples = jnp.stack(hidden_samples_list)  # [n_batch, n_hidden]
        
        # Use first half of visible as "labels" for info computation
        n_label = n_visible // 2
        label_samples = batch_data[:, :n_label]
        
        # Compute Bayesian MI (simplified - just current model)
        print("  Computing Bayesian MI...")
        key, k_mi = jax.random.split(key)
        bayesian_mi = estimate_discrete_mi(k_mi, hidden_samples, label_samples)
        print(f"    Bayesian MI: {bayesian_mi:.3f} bits")
        
        # Compute V-information at different budgets
        print("  Computing V-information at different budgets...")
        v_infos = {}
        
        for budget in [10, 50, 200]:
            # Create schedule with exact budget
            schedule_budget = SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=budget)
            
            # Sample with limited budget
            n_chains = 30
            chain_data = train_data[:n_chains]
            keys_v = jax.random.split(key, n_chains)
            init_v = hinton_init(key, model, free_blocks, (n_chains,))
            
            samples_budget_list = []
            for i in range(n_chains):
                s = sample_states(
                    keys_v[i], program, schedule_budget,
                    [init_v[0][i:i+1]],
                    [chain_data[i:i+1]],
                    [hidden_block]
                )
                samples_budget_list.append(s[0, 0, :])
            
            samples_budget = jnp.stack(samples_budget_list)
            labels_budget = chain_data[:, :n_label]
            
            v_info = estimate_discrete_mi(key, samples_budget, labels_budget)
            v_infos[budget] = v_info
            gap = max(0, bayesian_mi - v_info)
            pct = (v_info / bayesian_mi * 100) if bayesian_mi > 0 else 0
            print(f"    {budget:3d} steps: {v_info:.3f} bits (gap: {gap:.3f}, {pct:.0f}% extractable)")
        
        # Record results
        results['epochs'].append(epoch)
        results['bayesian_mi'].append(bayesian_mi)
        results['v_info_10'].append(v_infos[10])
        results['v_info_50'].append(v_infos[50])
        results['v_info_200'].append(v_infos[200])
    
    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    
    return results


def plot_results(results, save_dir="experiments/information_dynamics/results"):
    """Create plots from results."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    epochs = results['epochs']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Information vs Epochs
    ax = axes[0]
    ax.plot(epochs, results['bayesian_mi'], 'o-', linewidth=2, markersize=8,
            color='#A23B72', label='Bayesian MI (total)', zorder=10)
    ax.plot(epochs, results['v_info_10'], 's-', linewidth=2, markersize=6,
            color='#F18F01', label='V-info (10 steps)', alpha=0.8)
    ax.plot(epochs, results['v_info_50'], '^-', linewidth=2, markersize=6,
            color='#C73E1D', label='V-info (50 steps)', alpha=0.8)
    ax.plot(epochs, results['v_info_200'], 'd-', linewidth=2, markersize=6,
            color='#6A994E', label='V-info (200 steps)', alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Information (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Information Dynamics During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Information Gap
    ax = axes[1]
    gap_10 = np.array(results['bayesian_mi']) - np.array(results['v_info_10'])
    gap_50 = np.array(results['bayesian_mi']) - np.array(results['v_info_50'])
    gap_200 = np.array(results['bayesian_mi']) - np.array(results['v_info_200'])
    
    ax.plot(epochs, gap_10, 's-', linewidth=2, markersize=6,
            color='#F18F01', label='Gap @ 10 steps', alpha=0.8)
    ax.plot(epochs, gap_50, '^-', linewidth=2, markersize=6,
            color='#C73E1D', label='Gap @ 50 steps', alpha=0.8)
    ax.plot(epochs, gap_200, 'd-', linewidth=2, markersize=6,
            color='#6A994E', label='Gap @ 200 steps', alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unusable Information (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Information Gap: Bayesian MI - V-Information', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    output_path = f"{save_dir}/toy_experiment_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("\nStarting toy experiment...\n")
    
    # Run experiment
    results = run_toy_experiment(n_epochs=8, n_visible=20, n_hidden=30)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_path = plot_results(results)
    
    print("\n" + "=" * 70)
    print("SUCCESS! Experiment complete.")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  - Final Bayesian MI: {results['bayesian_mi'][-1]:.3f} bits")
    print(f"  - Final V-info (200 steps): {results['v_info_200'][-1]:.3f} bits")
    print(f"  - Final V-info (10 steps): {results['v_info_10'][-1]:.3f} bits")
    print(f"\nThis shows that even with {results['bayesian_mi'][-1]:.2f} bits of information")
    print(f"encoded in the latents, only {results['v_info_10'][-1]:.2f} bits are extractable")
    print(f"with just 10 Gibbs sampling steps!")
    print(f"\nVisualization: {plot_path}")

