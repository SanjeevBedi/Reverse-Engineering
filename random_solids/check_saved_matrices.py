#!/usr/bin/env python3
"""
Check the final saved connectivity matrices for specific missing edges.
This verifies what Build_Solid.py actually saved to disk.
"""
import numpy as np

print("="*70)
print("CHECKING SAVED CONNECTIVITY MATRICES (Build_Solid output)")
print("="*70)

# Load the saved matrices
data = np.load('Output/connectivity_matrices_seed_250.npz', allow_pickle=True)
all_vertices = data['all_vertices']
top_matrix = data['top_view_matrix']
front_matrix = data['front_view_matrix']
side_matrix = data['side_view_matrix']

print(f"\nLoaded matrices:")
print(f"  all_vertices: {len(all_vertices)} vertices")
print(f"  top_matrix: {top_matrix.shape}")
print(f"  front_matrix: {front_matrix.shape}")
print(f"  side_matrix: {side_matrix.shape}")

# Target edges that should exist
target_edges = [
    ("V22-V23 (vertices 25-26)", 25, 26),
    ("V42-V43 (vertices 0-1)", 0, 1),
    ("V45-V46 (vertices 4-5)", 4, 5),
]

print("\n" + "="*70)
print("CHECKING TARGET EDGES IN SAVED MATRICES")
print("="*70)

for edge_name, v1_idx, v2_idx in target_edges:
    print(f"\n{edge_name}:")
    v1 = all_vertices[v1_idx]
    v2 = all_vertices[v2_idx]
    print(f"  V{v1_idx}: [{v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f}]")
    print(f"  V{v2_idx}: [{v2[0]:.2f}, {v2[1]:.2f}, {v2[2]:.2f}]")
    
    # Check TOP view (x, y projection)
    v1_top = (v1[0], v1[1])
    v2_top = (v2[0], v2[1])
    v1_top_row = None
    v2_top_row = None
    for i in range(top_matrix.shape[0]):
        if abs(top_matrix[i, 1] - v1_top[0]) < 0.1 and abs(top_matrix[i, 2] - v1_top[1]) < 0.1:
            v1_top_row = i
        if abs(top_matrix[i, 1] - v2_top[0]) < 0.1 and abs(top_matrix[i, 2] - v2_top[1]) < 0.1:
            v2_top_row = i
    
    top_conn = 0
    top_degenerate = (v1_top_row == v2_top_row) if v1_top_row is not None and v2_top_row is not None else False
    if v1_top_row is not None and v2_top_row is not None and not top_degenerate:
        if (3 + v2_top_row) < top_matrix.shape[1]:
            top_conn = top_matrix[v1_top_row, 3 + v2_top_row]
    
    # Check FRONT view (x, z projection)
    v1_front = (v1[0], v1[2])
    v2_front = (v2[0], v2[2])
    v1_front_row = None
    v2_front_row = None
    for i in range(front_matrix.shape[0]):
        if abs(front_matrix[i, 1] - v1_front[0]) < 0.1 and abs(front_matrix[i, 2] - v1_front[1]) < 0.1:
            v1_front_row = i
        if abs(front_matrix[i, 1] - v2_front[0]) < 0.1 and abs(front_matrix[i, 2] - v2_front[1]) < 0.1:
            v2_front_row = i
    
    front_conn = 0
    front_degenerate = (v1_front_row == v2_front_row) if v1_front_row is not None and v2_front_row is not None else False
    if v1_front_row is not None and v2_front_row is not None and not front_degenerate:
        if (3 + v2_front_row) < front_matrix.shape[1]:
            front_conn = front_matrix[v1_front_row, 3 + v2_front_row]
    
    # Check SIDE view (y, z projection)
    v1_side = (v1[1], v1[2])
    v2_side = (v2[1], v2[2])
    v1_side_row = None
    v2_side_row = None
    for i in range(side_matrix.shape[0]):
        if abs(side_matrix[i, 1] - v1_side[0]) < 0.1 and abs(side_matrix[i, 2] - v1_side[1]) < 0.1:
            v1_side_row = i
        if abs(side_matrix[i, 1] - v2_side[0]) < 0.1 and abs(side_matrix[i, 2] - v2_side[1]) < 0.1:
            v2_side_row = i
    
    side_conn = 0
    side_degenerate = (v1_side_row == v2_side_row) if v1_side_row is not None and v2_side_row is not None else False
    if v1_side_row is not None and v2_side_row is not None and not side_degenerate:
        if (3 + v2_side_row) < side_matrix.shape[1]:
            side_conn = side_matrix[v1_side_row, 3 + v2_side_row]
    
    print(f"\n  TOP view:")
    print(f"    Rows: {v1_top_row}, {v2_top_row}")
    print(f"    Degenerate: {top_degenerate}")
    print(f"    Connectivity: {top_conn}")
    
    print(f"  FRONT view:")
    print(f"    Rows: {v1_front_row}, {v2_front_row}")
    print(f"    Degenerate: {front_degenerate}")
    print(f"    Connectivity: {front_conn} {'❌ MISSING!' if not front_degenerate and front_conn == 0 else '✓'}")
    
    print(f"  SIDE view:")
    print(f"    Rows: {v1_side_row}, {v2_side_row}")
    print(f"    Degenerate: {side_degenerate}")
    print(f"    Connectivity: {side_conn} {'❌ MISSING!' if not side_degenerate and side_conn == 0 else '✓'}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nThese edges are MISSING from the saved connectivity matrices:")
print("  - V22-V23: Front view conn=0 (should be visible)")
print("  - V42-V43: Side view conn=0 (should be visible)")
print("  - V45-V46: Side view conn=0 (should be visible)")
print("\nConclusion: Build_Solid.py's create_view_connectivity_matrix function")
print("is NOT adding these edges to the matrices before saving.")
print("\nNext step: Add debug to create_view_connectivity_matrix to see WHY")
print("these edges are not being added during polygon edge processing.")
