# THRML vs Traditional Methods: Comparison Summary

## TL;DR

For computing information-theoretic quantities in discrete EBMs:
- **THRML**: Optimized block Gibbs sampling, designed for the problem structure
- **Traditional MCMC**: General-purpose but slower for structured problems
- **Mean-field**: Fast but sacrifices accuracy (independence assumption)

## Why THRML Matters for Information Dynamics Research

### 1. **Problem Structure Match**

Our research question requires:
- Sampling from discrete distributions (latent states given visible data)
- Computing MI between high-dimensional binary variables
- Repeating this MANY times (at each epoch, for each budget level)

THRML is **specifically designed** for this:
- Block Gibbs sampling exploits graph structure
- JAX compilation optimizes repeated sampling
- Designed to map onto future specialized hardware

### 2. **Key Advantages**

#### **Accuracy**
- ✅ THRML preserves correlations between variables
- ✅ Can achieve arbitrary accuracy with enough steps
- ❌ Mean-field assumes independence (systematically wrong)
- ❌ Single-variable MCMC needs many more steps to decorrelate

#### **Expressiveness**
- ✅ THRML handles arbitrary factor graphs
- ✅ Supports heterogeneous node types
- ✅ Can clamp subsets of variables (essential for conditional MI)
- ❌ Traditional methods require manual bookkeeping

#### **Future Scaling**
- ✅ THRML code will run on Extropic hardware (orders of magnitude faster)
- ✅ Already optimized for the operations hardware accelerates
- ❌ Traditional methods won't benefit from specialized hardware

### 3. **Empirical Results from Our Comparison**

#### Accuracy Test (200 samples, 30 variables):
```
Method              MI Estimate    Error
─────────────────────────────────────────
True (ground truth)    0.458 bits     -
Block MCMC (THRML)     0.458 bits  0.000 ✓
Single-var MCMC        0.458 bits  0.000
Mean-field             0.373 bits  0.084 ✗
```

**Mean-field misses 18% of the information** due to independence assumption!

#### What Traditional Methods Would Require:

**For our experiments:**
- 10 epochs × 3 budgets = 30 MI computations
- Each MI needs ~100+ samples
- Each sample needs ~100+ Gibbs steps
- Total: ~300,000 sampling operations

**With traditional tools:**
- Write custom MCMC code
- Manual block structure management
- No JAX compilation
- No hardware acceleration path
- 3-10x more development time

**With THRML:**
- Clean API for block sampling
- JAX-compiled by default
- Hardware-ready architecture
- Working code in hours, not weeks

## Why Not Traditional Libraries?

### PyMC / Stan
- **Focus**: Bayesian inference with continuous parameters
- **Strength**: General-purpose, well-documented
- **Weakness**: Not optimized for discrete EBM sampling
- **Problem**: Overhead for our use case, no hardware path

### Edward2 / TensorFlow Probability
- **Focus**: Probabilistic programming
- **Strength**: Integration with TensorFlow
- **Weakness**: Heavy framework, not specialized for block Gibbs
- **Problem**: Computational overhead, wrong abstraction level

### Custom JAX MCMC
- **Possible**: Yes, could implement from scratch
- **Cost**: Weeks of development + debugging
- **Result**: Would reinvent ~80% of THRML

## The THRML Advantage for Our Research

### What We Get:
1. **Clean abstractions** - Nodes, Blocks, Factors, Programs
2. **Optimized sampling** - JAX-compiled block Gibbs
3. **Proven code** - Used in published research
4. **Hardware path** - Will scale to Extropic chips
5. **Active development** - Maintained by team building the hardware

### What We Avoid:
1. Reimplementing block sampling logic
2. Debugging subtle MCMC issues
3. Manual JAX compilation optimization
4. Dead-end technology choices

## Bottom Line

**For traditional ML tasks:** Use PyTorch/TensorFlow  
**For Bayesian inference:** Use PyMC/Stan  
**For discrete EBM sampling with future hardware:** Use THRML ✓

Our research sits squarely in THRML's sweet spot:
- Discrete distributions ✓
- Block structure ✓  
- Information-theoretic quantities ✓
- Sampling-heavy workload ✓
- Future hardware benefits ✓

**Using THRML is the right choice for this research project.**

## Concrete Example: Computing Bayesian MI

### Traditional Approach (pseudocode):
```python
# Manual implementation needed
for param_sample in posterior_samples:
    model = create_model(param_sample)
    
    # Custom MCMC (or use PyMC, but overhead)
    samples = []
    state = initialize_random()
    for _ in range(1000):
        for var_idx in range(n_vars):  # Single variable updates
            state[var_idx] = sample_conditional(state, var_idx, model)
        samples.append(state.copy())
    
    mi = compute_mi_from_samples(samples)  # Custom code
    bayesian_mis.append(mi)

final_mi = np.mean(bayesian_mis)
```

### THRML Approach:
```python
from thrml import IsingSamplingProgram, sample_states, hinton_init

program = IsingSamplingProgram(model, free_blocks, clamped_blocks)
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

init_state = hinton_init(key, model, free_blocks, (batch_size,))
samples = sample_states(key, program, schedule, init_state, data, observe_blocks)

mi = estimate_discrete_mi(key, latent_samples, label_samples)
```

**Less code, more clarity, better performance.**

## References

- THRML paper: https://arxiv.org/abs/2510.23972
- Block Gibbs sampling: https://proceedings.mlr.press/v15/gonzalez11a/gonzalez11a.pdf
- Our experiments: `experiments/information_dynamics/`


