#!/usr/bin/env python3
"""
Trace specific vertices through the Build_Solid.py process
by directly loading the output files and checking unit scales.
"""

import sys
import os
import numpy as np

seed = 250

# Load connectivity matrices
matrix_file = f"Output/connectivity_matrices_seed_{seed}.npz"
data = np.load(matrix_file)

all_vertices = data['all_vertices']
print(f"Loaded {len(all_vertices)} vertices from {matrix_file}")

# Target vertices to trace (from reconstruction - these are in mm)
target_vertices_mm = {
    22: np.array([427.2, 333.6, 0.0]),
    23: np.array([427.2, 333.6, 186.4]),
    37: np.array([379.1, 381.6, 0.0]),
    38: np.array([379.1, 381.6, 186.4]),
    40: np.array([365.0, 333.6, 186.4]),
    41: np.array([365.0, 333.6, 304.2]),
    42: np.array([298.8, 301.2, 0.0]),
    43: np.array([298.8, 301.2, 186.4]),
    45: np.array([365.0, 235.0, 0.0]),
    46: np.array([365.0, 235.0, 186.4]),
}

# Convert to cm
target_vertices_cm = {k: v/10.0 for k, v in target_vertices_mm.items()}

print(f"\n{'='*70}")
print(f"UNIT ANALYSIS")
print(f"{'='*70}")
print("\nExtracted vertex coordinate ranges:")
print(f"  X: [{all_vertices[:,0].min():.3f}, {all_vertices[:,0].max():.3f}]")
print(f"  Y: [{all_vertices[:,1].min():.3f}, {all_vertices[:,1].max():.3f}]")
print(f"  Z: [{all_vertices[:,2].min():.3f}, {all_vertices[:,2].max():.3f}]")

print(f"\nTarget coordinate ranges (mm):")
target_arr_mm = np.array(list(target_vertices_mm.values()))
print(f"  X: [{target_arr_mm[:,0].min():.3f}, {target_arr_mm[:,0].max():.3f}]")
print(f"  Y: [{target_arr_mm[:,1].min():.3f}, {target_arr_mm[:,1].max():.3f}]")
print(f"  Z: [{target_arr_mm[:,2].min():.3f}, {target_arr_mm[:,2].max():.3f}]")

print(f"\nTarget coordinate ranges (cm):")
target_arr_cm = np.array(list(target_vertices_cm.values()))
print(f"  X: [{target_arr_cm[:,0].min():.3f}, {target_arr_cm[:,0].max():.3f}]")
print(f"  Y: [{target_arr_cm[:,1].min():.3f}, {target_arr_cm[:,1].max():.3f}]")
print(f"  Z: [{target_arr_cm[:,2].min():.3f}, {target_arr_cm[:,2].max():.3f}]")

# Determine scale
if all_vertices[:,0].max() > 100:
    print(f"\n→ Extracted vertices appear to be in MM scale (max coord > 100)")
    target_vertices = target_vertices_mm
    unit = "mm"
    tol = 0.5
else:
    print(f"\n→ Extracted vertices appear to be in CM scale (max coord < 100)")
    target_vertices = target_vertices_cm
    unit = "cm"
    tol = 0.05

print(f"\n{'='*70}")
print(f"CHECKING FOR TARGET VERTICES (using {unit} scale)")
print(f"{'='*70}")

def vertices_match(v1, v2, tol):
    """Check if two vertices match within tolerance"""
    return np.allclose(v1, v2, atol=tol)

# Check each target vertex
found_count = 0
for v_num in sorted(target_vertices.keys()):
    target = target_vertices[v_num]
    
    print(f"\n[V{v_num}] {target}")
    
    found = False
    for idx, vertex in enumerate(all_vertices):
        if vertices_match(vertex, target, tol):
            print(f"  ✓ FOUND as vertex {idx}: {vertex}")
            found = True
            found_count += 1
            break
    
    if not found:
        print(f"  ✗ NOT FOUND in extracted vertices")
        
        # Find closest match
        distances = [np.linalg.norm(v - target) for v in all_vertices]
        closest_idx = np.argmin(distances)
        closest_dist = distances[closest_idx]
        
        print(f"  Closest: V{closest_idx} at distance {closest_dist:.3f} {unit}")
        print(f"           coords: {all_vertices[closest_idx]}")
        
        if closest_dist < tol * 10:  # Within 10x tolerance
            print(f"  → CLOSE MATCH - might be rounding difference")

print(f"\n{'='*70}")
print(f"SUMMARY: Found {found_count}/10 target vertices in extracted geometry")
print(f"{'='*70}")
