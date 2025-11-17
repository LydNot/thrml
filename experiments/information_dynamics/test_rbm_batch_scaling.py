"""Estimate speedup from larger batches."""
import numpy as np

# Current config
current_batch_size = 10
total_samples = 100
n_batches_current = total_samples // current_batch_size

# Improved config  
improved_batch_size = 100
n_batches_improved = 1

# Time estimates
time_per_batch_sequential = 2.0  # seconds (mostly sequential Gibbs)
time_per_batch_parallel = 0.3    # seconds (overhead for parallelization)

current_time = n_batches_current * (time_per_batch_sequential + time_per_batch_parallel)
improved_time = n_batches_improved * (time_per_batch_sequential + time_per_batch_parallel * current_batch_size / improved_batch_size)

print("="*60)
print("BATCH SIZE OPTIMIZATION")
print("="*60)
print(f"\nCurrent approach:")
print(f"  Batch size: {current_batch_size}")
print(f"  Number of batches: {n_batches_current}")
print(f"  Total time: ~{current_time:.1f}s per training step")

print(f"\nImproved approach:")
print(f"  Batch size: {improved_batch_size}")
print(f"  Number of batches: {n_batches_improved}")
print(f"  Total time: ~{improved_time:.1f}s per training step")

speedup = current_time / improved_time
print(f"\n{'Potential speedup:':<25} {speedup:.1f}x")
print(f"{'Per epoch savings:':<25} ~{(current_time - improved_time) * 20:.0f}s")
print(f"{'Total experiment savings:':<25} ~{(current_time - improved_time) * 20 * 20 / 60:.0f} minutes")
