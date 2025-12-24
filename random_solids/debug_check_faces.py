#!/usr/bin/env python3
"""
Check if vertical edges (22,23), (42,43), (45,46), (37,38), (40,41) 
are present in the extracted face polygons from Build_Solid.py
"""

import numpy as np
import os

# Load face polygons
seed = 250
face_file = f"Output/solid_faces_seed_{seed}.npy"
data = np.load(face_file, allow_pickle=True).item()

faces = data['faces']
print(f"Loaded {len(faces)} faces from {face_file}")

# Vertices we're looking for
target_vertices = {
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

# Edges we're looking for
target_edges = [
    (22, 23),
    (37, 38),
    (40, 41),
    (42, 43),
    (45, 46),
]

def vertices_match(v1, v2, tol=0.1):
    """Check if two vertices match within tolerance"""
    return np.allclose(v1, v2, atol=tol)

def find_vertex_in_polygon(target_v, polygon, tol=0.1):
    """Find if target vertex exists in polygon, return index or None"""
    for i, v in enumerate(polygon):
        if vertices_match(target_v, v, tol):
            return i
    return None

def check_edge_in_polygon(v1, v2, polygon, tol=0.1):
    """Check if edge (v1, v2) exists as consecutive vertices in polygon"""
    idx1 = find_vertex_in_polygon(v1, polygon, tol)
    if idx1 is None:
        return False, None
    
    # Check if v2 is the next or previous vertex
    n = len(polygon)
    next_idx = (idx1 + 1) % n
    prev_idx = (idx1 - 1) % n
    
    if vertices_match(polygon[next_idx], v2, tol):
        return True, (idx1, next_idx)
    if vertices_match(polygon[prev_idx], v2, tol):
        return True, (prev_idx, idx1)
    
    return False, None

# Check each face for our target edges
print("\n" + "="*70)
print("SEARCHING FOR VERTICAL EDGES IN FACE POLYGONS")
print("="*70)

for edge_nums in target_edges:
    v1_num, v2_num = edge_nums
    v1_coord = target_vertices[v1_num]
    v2_coord = target_vertices[v2_num]
    
    print(f"\n[Edge {v1_num}-{v2_num}]")
    print(f"  V{v1_num}: {v1_coord}")
    print(f"  V{v2_num}: {v2_coord}")
    
    found_in_faces = []
    
    for face_idx, face in enumerate(faces):
        outer = face['outer_boundary']
        
        # Check if edge is in outer boundary
        has_edge, indices = check_edge_in_polygon(v1_coord, v2_coord, outer)
        if has_edge:
            found_in_faces.append((face_idx, 'outer', indices))
        
        # Check holes
        for hole_idx, hole in enumerate(face.get('holes', [])):
            has_edge, indices = check_edge_in_polygon(v1_coord, v2_coord, hole)
            if has_edge:
                found_in_faces.append((face_idx, f'hole_{hole_idx}', indices))
    
    if found_in_faces:
        print(f"  ✓ FOUND in {len(found_in_faces)} face(s):")
        for face_idx, boundary_type, indices in found_in_faces:
            print(f"    - Face {face_idx}, {boundary_type}, indices {indices}")
    else:
        print(f"  ✗ NOT FOUND in any face polygon!")
        print(f"  → This edge will have conn=0 in original view matrices")
        
        # Check if individual vertices are present
        v1_faces = []
        v2_faces = []
        for face_idx, face in enumerate(faces):
            outer = face['outer_boundary']
            if find_vertex_in_polygon(v1_coord, outer) is not None:
                v1_faces.append(face_idx)
            if find_vertex_in_polygon(v2_coord, outer) is not None:
                v2_faces.append(face_idx)
        
        print(f"  V{v1_num} appears in faces: {v1_faces}")
        print(f"  V{v2_num} appears in faces: {v2_faces}")
        
        if v1_faces and v2_faces:
            common = set(v1_faces) & set(v2_faces)
            if common:
                print(f"  → Both vertices in same face(s) {list(common)} but not as consecutive edge!")
            else:
                print(f"  → Vertices in different faces - edge spans face boundary")

print("\n" + "="*70)
