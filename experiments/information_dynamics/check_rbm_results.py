import pickle

with open('results/mnist_real_results_20251117_014743.pkl', 'rb') as f:
    results = pickle.load(f)

print("Keys in RBM results:")
for key in results.keys():
    val = results[key]
    if hasattr(val, 'shape'):
        print(f"  {key}: shape {val.shape}")
    elif isinstance(val, list):
        print(f"  {key}: list with {len(val)} items")
    else:
        print(f"  {key}: {type(val)}")
