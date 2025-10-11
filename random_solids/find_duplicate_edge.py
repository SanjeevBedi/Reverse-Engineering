"""
Find duplicate edges in the reconstructed edge list.
"""
import numpy as np

# Load unified summary
unified = np.load('unified_summary.npy')
n_vertices = len(unified)

# Extract adjacency matrix
adjacency = unified[:, 10:]

# Find all edges
edges = []
for i in range(n_vertices):
    for j in range(i+1, n_vertices):
        if adjacency[i, j] == 1:
            edges.append((i, j))

print(f"Total edges found: {len(edges)}")
print(f"Expected: 171")
print(f"Difference: {len(edges) - 171}")

# Check for any value > 1 in adjacency matrix (would indicate duplicate)
max_val = np.max(adjacency)
if max_val > 1:
    print(f"\nFound adjacency values > 1 (max = {max_val})")
    for i in range(n_vertices):
        for j in range(i+1, n_vertices):
            if adjacency[i, j] > 1:
                v1 = unified[i, 1:4]
                v2 = unified[j, 1:4]
                print(f"  V{i}--V{j}: count={adjacency[i,j]}, "
                      f"V{i}={v1}, V{j}={v2}")

# List all edges for inspection
print(f"\nAll {len(edges)} reconstructed edges:")
for i, (v1, v2) in enumerate(edges):
    if i < 20 or i >= len(edges) - 10:  # Show first 20 and last 10
        coords1 = unified[v1, 1:4]
        coords2 = unified[v2, 1:4]
        print(f"{i+1}. V{v1}--V{v2}: ({coords1[0]:.3f}, {coords1[1]:.3f}, "
              f"{coords1[2]:.3f}) → ({coords2[0]:.3f}, {coords2[1]:.3f}, "
              f"{coords2[2]:.3f})")
    elif i == 20:
        print("  ... (omitted) ...")
