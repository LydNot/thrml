#!/bin/bash
# Setup script for running experiments on Prime Intellect GPUs

echo "=========================================="
echo "Setting up THRML Information Dynamics"
echo "=========================================="
echo ""

# Check GPU
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Install THRML
echo "Installing THRML..."
cd /Users/mox/thrml
pip install -e .
echo ""

# Install additional dependencies
echo "Installing matplotlib and other deps..."
pip install matplotlib numpy
echo ""

# Verify JAX GPU
echo "Verifying JAX can see GPUs..."
python -c "import jax; print(f'✓ JAX found {jax.device_count()} GPU(s)'); print(f'  Devices: {jax.devices()}')"
echo ""

# Test imports
echo "Testing THRML imports..."
python -c "from thrml import SpinNode, IsingEBM; print('✓ THRML imports working')"
echo ""

# Check MNIST data
echo "Checking MNIST data..."
if [ -f "tests/mnist_test_data/train_data_filtered.npy" ]; then
    echo "✓ MNIST data found"
else
    echo "✗ MNIST data not found - please download"
fi
echo ""

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To run the experiment:"
echo "  cd experiments/information_dynamics"
echo "  python mnist_real_training.py"
echo ""
echo "Expected runtime: 1-2 hours on H100"
echo "Cost estimate: ~$2.50 for full run"


