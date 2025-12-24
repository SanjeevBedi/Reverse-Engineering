#!/usr/bin/env python3
"""
Trace specific edges through the connectivity matrices to understand where they're lost.
"""
import numpy as np

# Load the saved connectivity matrices from Build_Solid
data = np.load('Output/connectivity_matrices_seed_250.npz', allow_pickle=True)
all_verts = data['all_vertices']
front_matrix = data['front_view_matrix']
side_matrix = data['side_view_matrix']

print("="*70)
print("TRACING VERTICAL EDGES IN SAVED CONNECTIVITY MATRICES")
print("="*70)

# Target vertical edges: V22-V23, V42-V43, V45-V46
target_edges = [
    ("V22-V23", 25, 26),
    ("V42-V43", 0, 1),
    ("V45-V46", 4, 5),
]

for name, v1_idx, v2_idx in target_edges:
    v1 = all_verts[v1_idx]
    v2 = all_verts[v2_idx]
    
    print(f"\n{'='*70}")
    print(f"EDGE: {name}")
    print(f"{'='*70}")
    print(f"Vertex {v1_idx}: [{v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f}]")
    print(f"Vertex {v2_idx}: [{v2[0]:.2f}, {v2[1]:.2f}, {v2[2]:.2f}]")
    print(f"Delta: Δx={abs(v2[0]-v1[0]):.2f}, Δy={abs(v2[1]-v1[1]):.2f}, Δz={abs(v2[2]-v1[2]):.2f}")
    
    # Project to Front view (x, z)
    v1_front = [v1[0], v1[2]]
    v2_front = [v2[0], v2[2]]
    
    # Project to Side view (y, z)
    v1_side = [v1[1], v1[2]]
    v2_side = [v2[1], v2[2]]
    
    print(f"\n--- FRONT VIEW (projects x,z) ---")
    print(f"V{v1_idx} front projection: [{v1_front[0]:.2f}, {v1_front[1]:.2f}]")
    print(f"V{v2_idx} front projection: [{v2_front[0]:.2f}, {v2_front[1]:.2f}]")
    
    # Find rows in front_matrix
    v1_front_row = None
    v2_front_row = None
    tol = 0.1
    
    for i in range(front_matrix.shape[0]):
        x, z = front_matrix[i, 1], front_matrix[i, 2]
        if abs(x - v1_front[0]) < tol and abs(z - v1_front[1]) < tol:
            v1_front_row = i
        if abs(x - v2_front[0]) < tol and abs(z - v2_front[1]) < tol:
            v2_front_row = i
    
    print(f"V{v1_idx} maps to front_matrix row: {v1_front_row}")
    print(f"V{v2_idx} maps to front_matrix row: {v2_front_row}")
    
    if v1_front_row is not None and v2_front_row is not None:
        # Check connectivity
        if v1_front_row < front_matrix.shape[0] and (3 + v2_front_row) < front_matrix.shape[1]:
            conn_12 = front_matrix[v1_front_row, 3 + v2_front_row]
        else:
            conn_12 = "OUT OF BOUNDS"
            
        if v2_front_row < front_matrix.shape[0] and (3 + v1_front_row) < front_matrix.shape[1]:
            conn_21 = front_matrix[v2_front_row, 3 + v1_front_row]
        else:
            conn_21 = "OUT OF BOUNDS"
        
        print(f"Connectivity in front_matrix:")
        print(f"  [{v1_front_row}, 3+{v2_front_row}] = [{v1_front_row}, {3+v2_front_row}] = {conn_12}")
        print(f"  [{v2_front_row}, 3+{v1_front_row}] = [{v2_front_row}, {3+v1_front_row}] = {conn_21}")
        
        # Show what each row connects to
        if v1_front_row < front_matrix.shape[0]:
            v1_connections = np.where(front_matrix[v1_front_row, 3:] > 0)[0]
            print(f"  Row {v1_front_row} connects to rows: {v1_connections}")
            
        if v2_front_row < front_matrix.shape[0]:
            v2_connections = np.where(front_matrix[v2_front_row, 3:] > 0)[0]
            print(f"  Row {v2_front_row} connects to rows: {v2_connections}")
    else:
        print(f"  ERROR: Could not find rows in front_matrix")
    
    print(f"\n--- SIDE VIEW (projects y,z) ---")
    print(f"V{v1_idx} side projection: [{v1_side[0]:.2f}, {v1_side[1]:.2f}]")
    print(f"V{v2_idx} side projection: [{v2_side[0]:.2f}, {v2_side[1]:.2f}]")
    
    # Find rows in side_matrix
    v1_side_row = None
    v2_side_row = None
    
    for i in range(side_matrix.shape[0]):
        y, z = side_matrix[i, 1], side_matrix[i, 2]
        if abs(y - v1_side[0]) < tol and abs(z - v1_side[1]) < tol:
            v1_side_row = i
        if abs(y - v2_side[0]) < tol and abs(z - v2_side[1]) < tol:
            v2_side_row = i
    
    print(f"V{v1_idx} maps to side_matrix row: {v1_side_row}")
    print(f"V{v2_idx} maps to side_matrix row: {v2_side_row}")
    
    if v1_side_row is not None and v2_side_row is not None:
        # Check connectivity
        if v1_side_row < side_matrix.shape[0] and (3 + v2_side_row) < side_matrix.shape[1]:
            conn_12 = side_matrix[v1_side_row, 3 + v2_side_row]
        else:
            conn_12 = "OUT OF BOUNDS"
            
        if v2_side_row < side_matrix.shape[0] and (3 + v1_side_row) < side_matrix.shape[1]:
            conn_21 = side_matrix[v2_side_row, 3 + v1_side_row]
        else:
            conn_21 = "OUT OF BOUNDS"
        
        print(f"Connectivity in side_matrix:")
        print(f"  [{v1_side_row}, 3+{v2_side_row}] = [{v1_side_row}, {3+v2_side_row}] = {conn_12}")
        print(f"  [{v2_side_row}, 3+{v1_side_row}] = [{v2_side_row}, {3+v1_side_row}] = {conn_21}")
        
        # Show what each row connects to
        if v1_side_row < side_matrix.shape[0]:
            v1_connections = np.where(side_matrix[v1_side_row, 3:] > 0)[0]
            print(f"  Row {v1_side_row} connects to rows: {v1_connections}")
            
        if v2_side_row < side_matrix.shape[0]:
            v2_connections = np.where(side_matrix[v2_side_row, 3:] > 0)[0]
            print(f"  Row {v2_side_row} connects to rows: {v2_connections}")
    else:
        print(f"  ERROR: Could not find rows in side_matrix")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Front matrix dimensions: {front_matrix.shape}")
print(f"Side matrix dimensions: {side_matrix.shape}")
print(f"Total vertices: {len(all_verts)}")
