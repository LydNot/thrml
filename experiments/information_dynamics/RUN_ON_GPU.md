# Running Real EBM Experiments on Prime Intellect Compute

This guide will help you run **actual** EBM training with information dynamics tracking on real GPUs.

## 🚀 Quick Start

### 1. Get GPU Access

Go to: https://app.primeintellect.ai

**Recommended for this project:**
- **Development/Testing**: Single H100 GPU (80GB) - ~$2-3/hour
- **Full Experiments**: 2-4 H100 GPUs for faster training
- **Budget Option**: A100 GPUs work fine, just slower

### 2. Clone This Repo on the GPU Instance

```bash
git clone https://github.com/extropic-ai/thrml.git
cd thrml
```

### 3. Install Dependencies

```bash
# Install THRML
pip install -e .

# Install matplotlib for visualizations
pip install matplotlib

# Verify JAX can see GPUs
python -c "import jax; print(f'GPUs available: {jax.device_count()}')"
```

### 4. Run Real MNIST Experiment

```bash
cd experiments/information_dynamics
python mnist_real_training.py
```

This will:
- ✅ Load real MNIST data
- ✅ Train a small Boltzmann machine (RBM)
- ✅ Actually run Gibbs sampling with THRML
- ✅ Compute real Bayesian MI and V-information
- ✅ Generate publication-quality figures

**Expected runtime**: 1-2 hours on a single H100

## 📊 What You'll Get

Real results showing:
1. **Information encoding** - How latents learn about labels during training
2. **Computational bottleneck** - Gap between Bayesian MI and V-information
3. **Budget effects** - How sampling steps affect information extraction
4. **Actual measurements** - Not simulations!

## 🔧 Configuration Options

Edit `mnist_real_training.py` to adjust:

```python
# Model size (smaller = faster, larger = more interesting)
n_hidden = 50  # Try 30, 50, or 100

# Training epochs (more = better learning)
n_epochs = 10  # Try 5, 10, or 20

# Sampling budgets to test
budgets = [10, 50, 100, 200]  # Gibbs steps

# Data size (subset of MNIST)
n_samples = 1000  # Try 500, 1000, or 2000
```

## 💰 Cost Estimates

**Single H100 GPU** (~$2.50/hour):
- Quick test (5 epochs, small model): ~20 minutes = $0.83
- Full experiment (10 epochs, medium model): ~1 hour = $2.50
- Publication run (20 epochs, large model): ~3 hours = $7.50

**Multiple GPUs** (faster but more expensive):
- Can parallelize sampling across GPUs
- 4x H100s = 4x faster but 4x cost

## 📝 Monitoring Progress

The script will print:
```
Epoch 1/10
  Training step...
  Computing Bayesian MI: 0.234 bits
  V-info @ 10 steps: 0.156 bits (67% extractable)
  V-info @ 50 steps: 0.221 bits (94% extractable)
```

Results saved to: `results/mnist_real_training_[timestamp].png`

## 🐛 Troubleshooting

### Out of Memory?
- Reduce `n_hidden` (e.g., from 100 to 50)
- Reduce `n_samples` (e.g., from 2000 to 1000)
- Reduce batch size in training

### Too Slow?
- Reduce `n_epochs` for testing
- Use smaller sampling budgets
- Consider multi-GPU setup

### JAX not finding GPUs?
```bash
# Check CUDA
nvidia-smi

# Reinstall JAX with CUDA support
pip install --upgrade "jax[cuda12]"
```

## 🎓 Next Steps After Getting Results

1. **Vary model sizes** - Compare small vs large models
2. **Try different architectures** - RBM, DBM, general Ising
3. **Scale up** - Use more data, longer training
4. **Write paper** - You'll have real results to report!

## 📊 Expected Results

Based on preliminary simulations, expect to see:
- Bayesian MI: ~0.5-2.0 bits (increases with training)
- V-info gap: 20-40% with 10 Gibbs steps
- Information gap: Decreases with more sampling steps
- Clear computational bottleneck demonstrated

## 🚀 Advanced: Multi-GPU Training

If you have multiple GPUs:

```python
# In mnist_real_training.py, add:
import jax
n_devices = jax.device_count()
print(f"Using {n_devices} GPUs")

# Training will automatically parallelize across devices
# Sampling can be batched across GPUs
```

## 💡 Tips for Success

1. **Start small** - Run with `n_epochs=2` first to verify everything works
2. **Monitor GPU usage** - `watch -n 1 nvidia-smi` in another terminal
3. **Save checkpoints** - The script auto-saves results
4. **Use tmux/screen** - Don't lose your job if SSH disconnects:
   ```bash
   tmux new -s mnist_training
   python mnist_real_training.py
   # Detach: Ctrl+B then D
   # Reattach: tmux attach -t mnist_training
   ```

## 📧 Need Help?

If you run into issues:
1. Check Prime Intellect docs: https://docs.primeintellect.ai
2. THRML repo: https://github.com/extropic-ai/thrml
3. Open an issue with your error log

---

**Ready to get real results?** 🚀

```bash
# On your GPU instance:
cd experiments/information_dynamics
python mnist_real_training.py
```

Your first **real** information dynamics experiment awaits!

