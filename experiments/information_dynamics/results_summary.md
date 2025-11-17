# RBM vs MLP: Information Dynamics & Classification Performance

## MLP Results (3-layer: 100→50→25 hidden units)

### Training: 20 epochs (~8 seconds)

**Layer-wise Mutual Information (10 epochs):**
- Layer 1 (100 units): 1.16 bits (73% of max 1.585 bits)
- Layer 2 (50 units): 1.07 bits (67% of max)
- Layer 3 (25 units): 0.93 bits (59% of max)

**Classification Accuracy:**

| Epochs | Overall | Digit 0 | Digit 3 | Digit 4 |
|--------|---------|---------|---------|---------|
| 10     | 56.3%   | 87.5%   | 77.1%   | **4.1%** (catastrophic) |
| 20     | **75.3%** | 81.2% | 80.4%   | **64.3%** |

**Key Finding:** Digit 4 requires significantly more training. At 10 epochs, the model essentially ignores it (4.1% accuracy, barely better than always guessing wrong). By 20 epochs, it achieves 64.3%, though still lower than digits 0 and 3.

---

## RBM Results (100 hidden units)

### Training: 20 epochs (~2 hours 17 minutes)

**Mutual Information with Labels:**
- Final MI: 0.825 bits (52% of max 1.585 bits)
- Mean MI across epochs: 0.801 bits
- Range: [0.565, 1.151] bits (high volatility due to stochastic sampling)
- vs Random baseline: 0.03 bits → **27x improvement**

**Classification Accuracy:**
- **Not yet measured** (model weights not saved)
- Would require retraining with model saving enabled

---

## Comparison: Information vs Classification

### Information Captured (Mutual Information)
| Model | MI (bits) | % of max (1.585 bits) |
|-------|-----------|----------------------|
| RBM (trained) | 0.82 | 52% |
| MLP Layer 1 | 1.16 | 73% |
| MLP Layer 2 | 1.07 | 67% |
| MLP Layer 3 | 0.93 | 59% |
| Random | 0.03 | 2% |

**Winner: MLP** - Captures significantly more information (1.16 bits vs 0.82 bits)

### Classification Performance
| Model | Accuracy | Training Time |
|-------|----------|---------------|
| MLP (20 epochs) | **75.3%** | 8 seconds |
| RBM (20 epochs) | ??? | ~2h 17min |
| Random baseline | 33.3% | - |

**Winner: MLP** (at least in efficiency; RBM accuracy still unmeasured)

---

## Key Insights

1. **MLP is Much Faster**: 8 seconds vs 2+ hours for similar training
2. **MLP Captures More Information**: 1.16 bits vs 0.82 bits
3. **RBM is More Volatile**: MI fluctuates wildly (0.57-1.15 bits) due to stochastic Gibbs sampling
4. **MLP Shows Information Compression**: MI decreases in deeper layers (1.16→1.07→0.93), but accuracy still improves
5. **Digit 4 is Hardest**: Even at 20 epochs, MLP only achieves 64% on digit 4 vs 81% on digits 0 and 3

---

## Open Questions

1. **RBM Classification Accuracy**: Need to retrain with weight saving to measure
2. **Why is Digit 4 Harder?**: Requires investigation of visual features
3. **Can RBM Beat MLP?**: Given lower MI, seems unlikely, but worth measuring
4. **Training Time Trade-off**: Is 2+ hours worth it for RBM?
