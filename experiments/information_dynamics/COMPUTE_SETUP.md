# Setting Up Compute for Real THRML Experiments

## 🎯 Goal
Run actual EBM training with THRML on GPUs to get **real** Bayesian MI and V-information measurements (not simulated).

## 🖥️ Compute Access

### Option 1: Prime Intellect Platform (Recommended)
1. **Sign up**: https://app.primeintellect.ai
2. **Dashboard**: https://app.primeintellect.ai/dashboard
3. **Rent GPUs**: 
   - Single GPU: For development/debugging (~$1-2/hour)
   - Multi-GPU: For production experiments
   - Quote for larger clusters: https://app.primeintellect.ai/dashboard/quotes

### Option 2: Your Own Hardware
- **Minimum**: 1 NVIDIA GPU (RTX 3090/4090, A100, H100)
- **Recommended**: 2+ GPUs (1 for inference, 1+ for training)

## 📦 Setup Steps

### 1. Install Dependencies
```bash
cd /Users/mox/thrml

# THRML is already installed
pip list | grep thrml

# Install additional deps for experiments
pip install matplotlib optax
```

### 2. Verify GPU Access
```bash
# Check NVIDIA GPUs
nvidia-smi

# Check JAX can see GPUs
python -c "import jax; print(f'GPUs: {jax.devices()}')"
```

### 3. Test THRML on GPU
```bash
# Quick test of THRML sampling
python -c "
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

print('Testing THRML on GPU...')
nodes = [SpinNode() for _ in range(10)]
edges = [(nodes[i], nodes[i+1]) for i in range(9)]
model = IsingEBM(nodes, edges, jnp.zeros(10), jnp.ones(9)*0.5, jnp.array(1.0))
print('Model created successfully!')
print(f'Running on: {jax.devices()}')
"
```

## 🚀 Running Experiments

### Quick Test (5-10 minutes)
```bash
cd /Users/mox/thrml

# Run small-scale test
python experiments/information_dynamics/mnist_info_gpu_test.py
```

### Full Experiment (1-2 hours)
```bash
# Run complete information dynamics experiment
python experiments/information_dynamics/mnist_info_real.py \
  --n-epochs 10 \
  --n-visible 100 \
  --n-hidden 50 \
  --n-samples 1000
```

## 📊 Expected Results

### What You'll Get:
- ✅ Real EBM training on MNIST
- ✅ Actual Gibbs sampling (not simulated)
- ✅ True Bayesian MI measurements
- ✅ Real V-information at different budgets
- ✅ Publication-quality results

### Compute Requirements:
- **Small test**: 1 GPU, ~10 minutes, $0.20
- **Full experiment**: 1-2 GPUs, 1-2 hours, $2-4
- **Large scale**: 4-8 GPUs, 4-8 hours, $20-40

## 💰 Cost Estimation

### Prime Intellect Pricing (approximate):
- RTX 4090: ~$0.50/hour
- A100 (40GB): ~$1.50/hour
- A100 (80GB): ~$2.50/hour
- H100: ~$4.00/hour

### Our Experiments:
- Quick test: **~$0.20** (10 min on RTX 4090)
- Full paper results: **~$5-10** (2-4 hours on A100)
- Comprehensive ablations: **~$50** (full day on multiple GPUs)

## 🐛 Troubleshooting

### "Out of Memory"
```bash
# Reduce model size
--n-hidden 30  # instead of 50
--n-visible 50  # instead of 100
```

### "CUDA not available"
```bash
# Check JAX installation
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12]"
```

### "Slow sampling"
```bash
# Make sure you're using GPU
python -c "import jax; print(jax.default_backend())"
# Should print: 'gpu'
```

## 📝 Next Steps

1. **Get GPU access** via Prime Intellect or your hardware
2. **Run quick test** to validate setup
3. **Run full experiment** to get real results
4. **Update paper** with actual measurements
5. **Submit** to conference! 🎉

## 🎓 What This Achieves

Instead of simulated results, you'll have:
- **Real training dynamics** from actual EBM optimization
- **Real sampling convergence** from Gibbs chains
- **Real information measurements** from converged samples
- **Credible results** for publication

This is the difference between a **demo** and **real research**! 🚀

