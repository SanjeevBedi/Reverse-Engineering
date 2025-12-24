#!/usr/bin/env python3
"""
Check if edges between target vertices exist in the view connectivity matrices.
"""

import numpy as np

seed = 250
matrix_file = f"Output/connectivity_matrices_seed_{seed}.npz"
data = np.load(matrix_file)

all_vertices = data['all_vertices']
top_matrix = data['top_view_matrix']
front_matrix = data['front_view_matrix']
side_matrix = data['side_view_matrix']

print(f"Loaded matrices from {matrix_file}")
print(f"All vertices: {all_vertices.shape}")
print(f"Top matrix: {top_matrix.shape}")
print(f"Front matrix: {front_matrix.shape}")
print(f"Side matrix: {side_matrix.shape}")

# Vertex pairs (using indices found in trace script)
vertex_pairs = [
    (25, 26, "V22-V23"),  # [42.72, 33.36, 0] -> [42.72, 33.36, 18.64]
    (8, 9, "V37-V38"),    # [37.91, 38.16, 0] -> [37.91, 38.16, 18.64]
    (6, 7, "V40-V41"),    # [36.5, 33.36, 18.64] -> [36.5, 33.36, 30.42]
    (0, 1, "V42-V43"),    # [29.88, 30.12, 0] -> [29.88, 30.12, 18.64]
    (4, 5, "V45-V46"),    # [36.5, 23.5, 0] -> [36.5, 23.5, 18.64]
]

print(f"\n{'='*70}")
print(f"3D EDGE ANALYSIS")
print(f"{'='*70}")

for idx1, idx2, name in vertex_pairs:
    v1 = all_vertices[idx1]
    v2 = all_vertices[idx2]
    delta = v2 - v1
    
    print(f"\n[{name}] (vertex indices {idx1}↔{idx2})")
    print(f"  V{idx1}: {v1}")
    print(f"  V{idx2}: {v2}")
    print(f"  Δ: {delta}")
    
    # Classify edge orientation
    if abs(delta[0]) < 0.01 and abs(delta[1]) < 0.01:
        print(f"  → VERTICAL edge (Δx≈0, Δy≈0, Δz={delta[2]:.3f})")
    elif abs(delta[2]) < 0.01:
        print(f"  → HORIZONTAL edge (Δz≈0)")
    else:
        print(f"  → OBLIQUE edge")

# Helper function to find vertex in view matrix
def find_vertex_in_view(vertex_3d, view_matrix, view_name, projection):
    """
    Find a 3D vertex in a view matrix by projecting it and matching the 2D coords.
    view_matrix format: [index, proj_x, proj_y, connectivity...]
    projection: 'top', 'front', or 'side'
    """
    # Project vertex to 2D
    if projection == 'top':
        proj_2d = np.array([vertex_3d[0], vertex_3d[1]])  # Drop Z
    elif projection == 'front':
        proj_2d = np.array([vertex_3d[0], vertex_3d[2]])  # Drop Y
    elif projection == 'side':
        proj_2d = np.array([vertex_3d[1], vertex_3d[2]])  # Drop X
    else:
        return None
    
    # Search for matching 2D projection in view matrix
    for i in range(view_matrix.shape[0]):
        view_proj = view_matrix[i, 1:3]  # Columns 1,2 are proj_x, proj_y
        if np.allclose(proj_2d, view_proj, atol=0.01):
            return i
    
    return None

# Helper function to get connectivity value
def get_connectivity(view_matrix, idx1, idx2):
    """Get connectivity value between two vertices in a view matrix."""
    if idx1 is None or idx2 is None:
        return None
    if idx1 < 0 or idx1 >= view_matrix.shape[0]:
        return None
    if idx2 < 0 or idx2 >= view_matrix.shape[0]:
        return None
    
    # Connectivity starts at column 3
    conn_idx = 3 + idx2
    if conn_idx >= view_matrix.shape[1]:
        return None
    
    return view_matrix[idx1, conn_idx]

print(f"\n{'='*70}")
print(f"VIEW MATRIX CONNECTIVITY CHECK")
print(f"{'='*70}")

for idx1, idx2, name in vertex_pairs:
    v1 = all_vertices[idx1]
    v2 = all_vertices[idx2]
    
    print(f"\n[{name}] (3D vertices {idx1}↔{idx2})")
    print(f"  V1: {v1}")
    print(f"  V2: {v2}")
    
    # Check each view
    for view_matrix, view_name, projection in [
        (top_matrix, "Top", "top"),
        (front_matrix, "Front", "front"),
        (side_matrix, "Side", "side"),
    ]:
        # Find vertices in this view
        view_idx1 = find_vertex_in_view(v1, view_matrix, view_name, projection)
        view_idx2 = find_vertex_in_view(v2, view_matrix, view_name, projection)
        
        if view_idx1 is None or view_idx2 is None:
            print(f"  {view_name}: NOT FOUND (v1_idx={view_idx1}, v2_idx={view_idx2})")
            if view_idx1 is not None:
                print(f"         → V1 found but V2 missing")
            elif view_idx2 is not None:
                print(f"         → V2 found but V1 missing")
            else:
                print(f"         → Both vertices missing from view")
            continue
        
        if view_idx1 == view_idx2:
            print(f"  {view_name}: DEGENERATE (both project to view index {view_idx1})")
            continue
        
        # Get connectivity
        conn_12 = get_connectivity(view_matrix, view_idx1, view_idx2)
        conn_21 = get_connectivity(view_matrix, view_idx2, view_idx1)
        
        if conn_12 is None or conn_21 is None:
            print(f"  {view_name}: ERROR reading connectivity")
            continue
        
        conn_value = max(conn_12, conn_21)
        
        if conn_value == 0:
            status = "✗ NO EDGE"
        elif conn_value == 1:
            status = "✓ VISIBLE (solid)"
        elif conn_value == 2:
            status = "✓ HIDDEN (dashed)"
        else:
            status = f"? UNKNOWN ({conn_value})"
        
        print(f"  {view_name}: {status} (indices {view_idx1}↔{view_idx2}, conn={conn_value})")

print(f"\n{'='*70}")
