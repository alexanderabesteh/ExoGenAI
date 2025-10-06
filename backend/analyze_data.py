"""
Analyze the dataset to understand what thresholds should filter
"""
import numpy as np
from pathlib import Path

# Load data
data = np.load('data/preprocessed_dataset.npz')
X_test = data['X_test']
y_test = data['y_test']

print("Analyzing dataset characteristics...")
print(f"Total samples: {len(X_test)}\n")

# Analyze by class
for class_idx in range(3):
    class_name = ['False Positive', 'Candidate', 'Exoplanet'][class_idx]
    mask = y_test == class_idx
    samples = X_test[mask]
    
    if len(samples) == 0:
        continue
    
    print(f"{class_name} (n={len(samples)}):")
    
    # Calculate statistics
    stds = [np.std(s) for s in samples]
    mins = [np.min(s) for s in samples]
    
    # Count dips at different thresholds
    dip_counts_1_5 = []
    dip_counts_2_0 = []
    
    for sample in samples:
        flux = sample.flatten()
        mean = np.mean(flux)
        std = np.std(flux)
        
        # Count dips at 1.5 sigma
        dips_1_5 = flux < (mean - 1.5 * std)
        count_1_5 = 0
        in_dip = False
        for is_dip in dips_1_5:
            if is_dip and not in_dip:
                count_1_5 += 1
                in_dip = True
            elif not is_dip:
                in_dip = False
        dip_counts_1_5.append(count_1_5)
        
        # Count dips at 2.0 sigma
        dips_2_0 = flux < (mean - 2.0 * std)
        count_2_0 = 0
        in_dip = False
        for is_dip in dips_2_0:
            if is_dip and not in_dip:
                count_2_0 += 1
                in_dip = True
            elif not is_dip:
                in_dip = False
        dip_counts_2_0.append(count_2_0)
    
    print(f"  Std dev: min={min(stds):.3f}, max={max(stds):.3f}, mean={np.mean(stds):.3f}")
    print(f"  Min flux: min={min(mins):.3f}, max={max(mins):.3f}, mean={np.mean(mins):.3f}")
    print(f"  Dips (1.5σ): min={min(dip_counts_1_5)}, max={max(dip_counts_1_5)}, mean={np.mean(dip_counts_1_5):.1f}")
    print(f"  Dips (2.0σ): min={min(dip_counts_2_0)}, max={max(dip_counts_2_0)}, mean={np.mean(dip_counts_2_0):.1f}")
    print()

print("\nRecommended thresholds to filter ~30% of false positives:")
print("Need to find characteristics where false positives differ from exoplanets")