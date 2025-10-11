"""
Verify the actual edge topology around the slot.
If there's a slot between V40 and V44, then V40-V44 should NOT be an edge.
"""
import numpy as np

unified = np.load('unified_summary.npy')
top_summary = np.load('top_view_summary.npy')
front_summary = np.load('front_view_summary.npy')

print("="*70)
print("VERIFYING EDGE TOPOLOGY AROUND THE SLOT")
print("="*70)

# Vertices on the line y=32.824, z=40.828
vertices_of_interest = [24, 40, 44, 114]

print("\nVertices on line y=32.824, z=40.828:")
for v_idx in vertices_of_interest:
    v = unified[v_idx, 1:4]
    print(f"  V{v_idx}: ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")

print("\nChecking which edges ACTUALLY exist in Top View:")
adj = unified[:, 10:]

edges_to_check = [
    (24, 40, "V24 → V40"),
    (40, 44, "V40 → V44 (across slot?)"),
    (44, 114, "V44 → V114"),
]

for v1, v2, label in edges_to_check:
    exists = adj[v1, v2] > 0 or adj[v2, v1] > 0
    print(f"  {label}: {'EXISTS ✓' if exists else 'MISSING ✗'}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("If V40→V44 is MISSING, there's a slot/gap between them.")
print("This means we should only expect 2 edges:")
print("  1. V24 → V40")
print("  2. V44 → V114")
print("\nThese 2 edges need to be detected in BOTH Top and Front views")
print("for the reconstruction algorithm to find them.")
