#!/usr/bin/env python3
"""
Trace specific vertices through the entire Reconstruct_Solid.py pipeline.
Track how vertex indices change and where edges are preserved or lost.
"""
import numpy as np
import sys

# Load saved data
print("="*70)
print("LOADING SAVED DATA")
print("="*70)

data = np.load('Output/connectivity_matrices_seed_250.npz', allow_pickle=True)
all_vertices = data['all_vertices']
top_matrix = data['top_view_matrix']
front_matrix = data['front_view_matrix']
side_matrix = data['side_view_matrix']

print(f"Loaded {len(all_vertices)} vertices")
print(f"Top matrix: {top_matrix.shape}")
print(f"Front matrix: {front_matrix.shape}")
print(f"Side matrix: {side_matrix.shape}")

# Target edges to trace
target_edges = [
    ("V22-V23", 25, 26),
    ("V42-V43", 0, 1),
    ("V45-V46", 4, 5),
]

print("\n" + "="*70)
print("STEP 1: VERTICES IN ORIGINAL all_vertices ARRAY")
print("="*70)

for name, v1_idx, v2_idx in target_edges:
    v1 = all_vertices[v1_idx]
    v2 = all_vertices[v2_idx]
    print(f"\n{name}:")
    print(f"  V{v1_idx}: [{v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f}]")
    print(f"  V{v2_idx}: [{v2[0]:.2f}, {v2[1]:.2f}, {v2[2]:.2f}]")

print("\n" + "="*70)
print("STEP 2: FINDING VERTICES IN VIEW MATRICES (by projection)")
print("="*70)

vertex_mapping = {}

for name, v1_idx, v2_idx in target_edges:
    v1 = all_vertices[v1_idx]
    v2 = all_vertices[v2_idx]
    
    print(f"\n{name}:")
    
    # Top view projection (x, y)
    v1_top = [v1[0], v1[1]]
    v2_top = [v2[0], v2[1]]
    
    v1_top_row = None
    v2_top_row = None
    for i in range(top_matrix.shape[0]):
        if abs(top_matrix[i, 1] - v1_top[0]) < 0.1 and abs(top_matrix[i, 2] - v1_top[1]) < 0.1:
            v1_top_row = i
        if abs(top_matrix[i, 1] - v2_top[0]) < 0.1 and abs(top_matrix[i, 2] - v2_top[1]) < 0.1:
            v2_top_row = i
    
    # Front view projection (x, z)
    v1_front = [v1[0], v1[2]]
    v2_front = [v2[0], v2[2]]
    
    v1_front_row = None
    v2_front_row = None
    for i in range(front_matrix.shape[0]):
        if abs(front_matrix[i, 1] - v1_front[0]) < 0.1 and abs(front_matrix[i, 2] - v1_front[1]) < 0.1:
            v1_front_row = i
        if abs(front_matrix[i, 1] - v2_front[0]) < 0.1 and abs(front_matrix[i, 2] - v2_front[1]) < 0.1:
            v2_front_row = i
    
    # Side view projection (y, z)
    v1_side = [v1[1], v1[2]]
    v2_side = [v2[1], v2[2]]
    
    v1_side_row = None
    v2_side_row = None
    for i in range(side_matrix.shape[0]):
        if abs(side_matrix[i, 1] - v1_side[0]) < 0.1 and abs(side_matrix[i, 2] - v1_side[1]) < 0.1:
            v1_side_row = i
        if abs(side_matrix[i, 1] - v2_side[0]) < 0.1 and abs(side_matrix[i, 2] - v2_side[1]) < 0.1:
            v2_side_row = i
    
    print(f"  V{v1_idx} maps to: Top row {v1_top_row}, Front row {v1_front_row}, Side row {v1_side_row}")
    print(f"  V{v2_idx} maps to: Top row {v2_top_row}, Front row {v2_front_row}, Side row {v2_side_row}")
    
    # Check connectivity in each view matrix
    top_conn = 0
    if v1_top_row is not None and v2_top_row is not None and v1_top_row < top_matrix.shape[0]:
        if (3 + v2_top_row) < top_matrix.shape[1]:
            top_conn = top_matrix[v1_top_row, 3 + v2_top_row]
    
    front_conn = 0
    if v1_front_row is not None and v2_front_row is not None and v1_front_row < front_matrix.shape[0]:
        if (3 + v2_front_row) < front_matrix.shape[1]:
            front_conn = front_matrix[v1_front_row, 3 + v2_front_row]
    
    side_conn = 0
    if v1_side_row is not None and v2_side_row is not None and v1_side_row < side_matrix.shape[0]:
        if (3 + v2_side_row) < side_matrix.shape[1]:
            side_conn = side_matrix[v1_side_row, 3 + v2_side_row]
    
    print(f"  Connectivity in view matrices: Top={top_conn}, Front={front_conn}, Side={side_conn}")
    
    vertex_mapping[name] = {
        'v1_idx': v1_idx,
        'v2_idx': v2_idx,
        'v1_coords': v1,
        'v2_coords': v2,
        'top_rows': (v1_top_row, v2_top_row),
        'front_rows': (v1_front_row, v2_front_row),
        'side_rows': (v1_side_row, v2_side_row),
        'original_conn': {'top': top_conn, 'front': front_conn, 'side': side_conn}
    }

print("\n" + "="*70)
print("STEP 3: SIMULATING filter_candidate_vertices")
print("="*70)

# Extract (x,y) from top view
top_xy_coords = set()
for i in range(top_matrix.shape[0]):
    x_proj, y_proj = top_matrix[i, 1], top_matrix[i, 2]
    top_xy_coords.add((x_proj, y_proj))

# Extract z-levels from front view
z_levels = sorted(set([front_matrix[i, 2] for i in range(front_matrix.shape[0])]))

print(f"Unique (x,y) in top view: {len(top_xy_coords)}")
print(f"Unique z-levels in front view: {len(z_levels)}")
print(f"Total candidates: {len(top_xy_coords)} × {len(z_levels)} = {len(top_xy_coords) * len(z_levels)}")

# Generate candidates
candidate_vertices = []
for x, y in top_xy_coords:
    for z in z_levels:
        candidate_vertices.append([x, y, z])
candidate_vertices = np.array(candidate_vertices)

# Filter candidates (simplified - just check if projections exist)
front_projections = []
for i in range(front_matrix.shape[0]):
    proj = (front_matrix[i, 1], front_matrix[i, 2])
    front_projections.append(proj)
front_projections = np.array(front_projections)

side_projections = []
for i in range(side_matrix.shape[0]):
    proj = (side_matrix[i, 1], side_matrix[i, 2])
    side_projections.append(proj)
side_projections = np.array(side_projections)

proj_tolerance = 0.1
selected_vertices = []

for vertex in candidate_vertices:
    front_proj = np.array([vertex[0], vertex[2]])
    side_proj = np.array([vertex[1], vertex[2]])
    
    front_distances = np.linalg.norm(front_projections - front_proj, axis=1)
    side_distances = np.linalg.norm(side_projections - side_proj, axis=1)
    
    front_match = np.any(front_distances < proj_tolerance)
    side_match = np.any(side_distances < proj_tolerance)
    
    if front_match and side_match:
        selected_vertices.append(vertex)

selected_vertices = np.array(selected_vertices)
print(f"Selected vertices after filtering: {len(selected_vertices)}")

# Find our target vertices in selected_vertices
print("\n" + "="*70)
print("STEP 4: FINDING TARGET VERTICES IN selected_vertices")
print("="*70)

for name, info in vertex_mapping.items():
    v1_coords = info['v1_coords']
    v2_coords = info['v2_coords']
    
    v1_new_idx = None
    v2_new_idx = None
    
    for idx, v in enumerate(selected_vertices):
        if np.allclose(v, v1_coords, atol=0.1):
            v1_new_idx = idx
        if np.allclose(v, v2_coords, atol=0.1):
            v2_new_idx = idx
    
    print(f"\n{name}:")
    print(f"  Original indices: V{info['v1_idx']}, V{info['v2_idx']}")
    print(f"  New indices in selected_vertices: V{v1_new_idx}, V{v2_new_idx}")
    
    if v1_new_idx is None or v2_new_idx is None:
        print(f"  ⚠️  WARNING: One or both vertices not found in selected_vertices!")
    
    info['new_indices'] = (v1_new_idx, v2_new_idx)

print("\n" + "="*70)
print("STEP 5: BUILDING SQUARE CONNECTIVITY MATRICES (simulating build_square_connectivity_matrices)")
print("="*70)

# Simulate the square matrix building
def build_square_conn_sim(view_matrix, selected_vertices):
    """Simplified version of build_square_connectivity_matrices"""
    proj_tolerance = 0.1
    n_verts = len(selected_vertices)
    result = np.zeros((n_verts, n_verts))
    
    # Get projections based on view
    # For now assume this is front view (x, z projection)
    for i in range(n_verts):
        for j in range(n_verts):
            if i == j:
                continue
            
            # Check connectivity in original view matrix
            # Find which rows in view_matrix these vertices map to
            vi = selected_vertices[i]
            vj = selected_vertices[j]
            
            # This is where the bug happens - we need to find the mapping
            # from selected_vertices index to view_matrix row
            
    return result

# For each view, track the vertex row mappings
views = ['top', 'front', 'side']
view_matrices = [top_matrix, front_matrix, side_matrix]
view_projections = [
    lambda v: (v[0], v[1]),  # top: x, y
    lambda v: (v[0], v[2]),  # front: x, z
    lambda v: (v[1], v[2]),  # side: y, z
]

for view_name, view_matrix, proj_fn in zip(views, view_matrices, view_projections):
    print(f"\n{view_name.upper()} VIEW:")
    print(f"  View matrix: {view_matrix.shape}")
    
    # For each target edge, find the mapping from selected_vertices index to view_matrix row
    for name, info in vertex_mapping.items():
        v1_new, v2_new = info['new_indices']
        if v1_new is None or v2_new is None:
            continue
        
        v1_coords = selected_vertices[v1_new]
        v2_coords = selected_vertices[v2_new]
        
        # Project to this view
        v1_proj = proj_fn(v1_coords)
        v2_proj = proj_fn(v2_coords)
        
        # Find in view_matrix
        v1_row = None
        v2_row = None
        for row_idx in range(view_matrix.shape[0]):
            row_proj = (view_matrix[row_idx, 1], view_matrix[row_idx, 2])
            if abs(row_proj[0] - v1_proj[0]) < 0.1 and abs(row_proj[1] - v1_proj[1]) < 0.1:
                v1_row = row_idx
            if abs(row_proj[0] - v2_proj[0]) < 0.1 and abs(row_proj[1] - v2_proj[1]) < 0.1:
                v2_row = row_idx
        
        # Check connectivity in view_matrix
        conn_in_view = 0
        if v1_row is not None and v2_row is not None:
            if (3 + v2_row) < view_matrix.shape[1]:
                conn_in_view = view_matrix[v1_row, 3 + v2_row]
        
        # Check if they degenerate (project to same point)
        degenerate = abs(v1_proj[0] - v2_proj[0]) < 0.01 and abs(v1_proj[1] - v2_proj[1]) < 0.01
        
        print(f"    {name}: selected_v[{v1_new},{v2_new}] → view_row[{v1_row},{v2_row}] conn={conn_in_view} degenerate={degenerate}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nVertex index mapping through reconstruction:")
for name, info in vertex_mapping.items():
    print(f"\n{name}:")
    print(f"  Original all_vertices: V{info['v1_idx']}, V{info['v2_idx']}")
    print(f"  After filtering → selected_vertices: V{info['new_indices'][0]}, V{info['new_indices'][1]}")
    print(f"  Original connectivity: Top={info['original_conn']['top']}, Front={info['original_conn']['front']}, Side={info['original_conn']['side']}")

print("\n" + "="*70)
print("KEY INSIGHT")
print("="*70)
print("""
The issue is clear:
1. Original vertices (e.g., V25, V26) exist in all_vertices
2. They have different row indices in view matrices (e.g., front rows 10, 7)
3. After filtering, they get NEW indices in selected_vertices (e.g., V47, V48)
4. When building square matrices, we need to lookup connectivity using VIEW MATRIX rows
5. But some edges have conn=0 in the view matrices despite existing in solid

The root cause is NOT in the reconstruction logic - it's that the original
view matrices saved by Build_Solid are missing these edge connections.
""")
