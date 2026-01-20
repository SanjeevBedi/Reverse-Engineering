#!/usr/bin/env python3
"""
Verify that coordinates (first 4 columns) are the same in X1, X2, X3, and Y.
"""
import numpy as np
import sys

# Get training data file from command line
if len(sys.argv) > 1:
    training_file = sys.argv[1]
else:
    training_file = 'NN/training_data_20260119_211855.npz'

print(f"Loading training data from: {training_file}")
data = np.load(training_file)

X1 = data['X1']  # Top view
X2 = data['X2']  # Front view  
X3 = data['X3']  # Side view
Y = data['Y']    # Target (solid connectivity)

print(f"\nShapes:")
print(f"  X1 (top):   {X1.shape}")
print(f"  X2 (front): {X2.shape}")
print(f"  X3 (side):  {X3.shape}")
print(f"  Y (target): {Y.shape}")

# Check first sample
sample_idx = 0
print(f"\n{'='*80}")
print(f"Checking sample {sample_idx}:")
print(f"{'='*80}")

# Data is 3D: (num_samples, 100, 104)
x1_sample = X1[sample_idx, :, :]  # (100, 104)
x2_sample = X2[sample_idx, :, :]
x3_sample = X3[sample_idx, :, :]
y_sample = Y[sample_idx, :, :]

print(f"\nSample shapes (after removing batch dim):")
print(f"  x1_sample: {x1_sample.shape}")
print(f"  x2_sample: {x2_sample.shape}")
print(f"  x3_sample: {x3_sample.shape}")
print(f"  y_sample:  {y_sample.shape}")

# Extract coordinates (columns 0-3: index, x, y, z)
print(f"\n{'='*80}")
print("COORDINATE COMPARISON (first 10 vertices)")
print(f"{'='*80}")

print(f"\n{'Vertex':<8} {'X1 coords':<30} {'X2 coords':<30} {'X3 coords':<30} {'Y coords':<30} {'Match?'}")
print("-" * 140)

all_match = True
for i in range(min(10, x1_sample.shape[0])):
    x1_coords = x1_sample[i, 0:4]
    x2_coords = x2_sample[i, 0:4]
    x3_coords = x3_sample[i, 0:4]
    y_coords = y_sample[i, 0:4]
    
    # Check if all are equal
    coords_match = (np.allclose(x1_coords, x2_coords, atol=1e-6) and 
                    np.allclose(x1_coords, x3_coords, atol=1e-6) and 
                    np.allclose(x1_coords, y_coords, atol=1e-6))
    
    match_str = "✓" if coords_match else "✗"
    if not coords_match:
        all_match = False
    
    # Format coordinates
    x1_str = f"[{x1_coords[0]:.0f},{x1_coords[1]:.1f},{x1_coords[2]:.1f},{x1_coords[3]:.1f}]"
    x2_str = f"[{x2_coords[0]:.0f},{x2_coords[1]:.1f},{x2_coords[2]:.1f},{x2_coords[3]:.1f}]"
    x3_str = f"[{x3_coords[0]:.0f},{x3_coords[1]:.1f},{x3_coords[2]:.1f},{x3_coords[3]:.1f}]"
    y_str = f"[{y_coords[0]:.0f},{y_coords[1]:.1f},{y_coords[2]:.1f},{y_coords[3]:.1f}]"
    
    print(f"V{i:<7} {x1_str:<30} {x2_str:<30} {x3_str:<30} {y_str:<30} {match_str}")

print("\n" + "="*80)
if all_match:
    print("✓ ALL COORDINATES MATCH across X1, X2, X3, and Y!")
else:
    print("✗ COORDINATES DO NOT MATCH!")
print("="*80)

# Check connectivity values are different (they should be!)
print(f"\nConnectivity values (column 4+) comparison:")
print(f"  X1 connectivity range: [{x1_sample[:, 4:].min():.3f}, {x1_sample[:, 4:].max():.3f}]")
print(f"  X2 connectivity range: [{x2_sample[:, 4:].min():.3f}, {x2_sample[:, 4:].max():.3f}]")
print(f"  X3 connectivity range: [{x3_sample[:, 4:].min():.3f}, {x3_sample[:, 4:].max():.3f}]")
print(f"  Y  connectivity range: [{y_sample[:, 4:].min():.3f}, {y_sample[:, 4:].max():.3f}]")

# Check Y is symmetric
print(f"\nChecking if Y (target) is symmetric:")
symmetric = True
max_diff = 0
for i in range(min(20, y_sample.shape[0])):
    for j in range(i+1, min(20, y_sample.shape[0])):
        col_ij = 4 + j
        col_ji = 4 + i
        if col_ij < y_sample.shape[1] and col_ji < y_sample.shape[1]:
            val_ij = y_sample[i, col_ij]
            val_ji = y_sample[j, col_ji]
            diff = abs(val_ij - val_ji)
            max_diff = max(max_diff, diff)
            if diff > 0.001:
                symmetric = False
                print(f"  Asymmetry: Y[{i},{col_ij}]={val_ij:.3f} vs Y[{j},{col_ji}]={val_ji:.3f}, diff={diff:.3f}")

if symmetric:
    print(f"  ✓ Y is SYMMETRIC (max diff: {max_diff:.6f})")
else:
    print(f"  ✗ Y is NOT symmetric (max diff: {max_diff:.6f})")
