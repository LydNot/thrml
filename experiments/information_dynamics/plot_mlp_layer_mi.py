"""
Plot just the mutual information between MLP hidden layers and digit labels.
Shows how information about the labels accumulates through the 5 hidden layers.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def plot_mlp_layer_mi(results_file):
    """Plot MI between each hidden layer and labels."""

    # Load results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print(f"Loaded results from: {results_file}")

    # Extract data
    epochs = np.array(results['epochs'])

    # Get layer MI data
    layer_keys = [k for k in results.keys() if k.startswith('layer_') and k.endswith('_mi')]
    layer_keys = sorted(layer_keys, key=lambda x: int(x.split('_')[1]))

    print(f"\nFound {len(layer_keys)} hidden layers")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color scheme - gradient from early to late layers
    layer_colors = ['#E63946', '#F77F00', '#FCBF49', '#06AED5', '#118AB2']

    # Plot each layer
    for i, layer_key in enumerate(layer_keys):
        layer_num = int(layer_key.split('_')[1])
        layer_mi = np.array(results[layer_key])

        ax.plot(epochs, layer_mi, 'o-',
                linewidth=2.5,
                markersize=8,
                color=layer_colors[i % len(layer_colors)],
                label=f'Layer {layer_num}',
                zorder=10-i)

        print(f"  Layer {layer_num}: {layer_mi[0]:.3f} → {layer_mi[-1]:.3f} bits")

    # Formatting
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mutual Information I(Layer; Labels) [bits]', fontsize=14, fontweight='bold')
    ax.set_title('Information About Labels in Each MLP Hidden Layer',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add subtle background
    ax.set_facecolor('#fafafa')

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/mlp_layer_mi_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"\n✅ Plot saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    # Use the most recent results file with 20 epochs
    results_file = "results/mnist_mlp_results_20251117_033314.pkl"

    print("=" * 70)
    print("PLOTTING MLP LAYER-WISE INFORMATION")
    print("=" * 70)

    output_path = plot_mlp_layer_mi(results_file)

    print("\n" + "=" * 70)
    print("✅ DONE!")
    print("=" * 70)
