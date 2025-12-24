#!/usr/bin/env python3
"""
Check which faces contain our target edges and whether those faces 
are being classified for each view.
"""
import numpy as np

# Load saved data
data = np.load('Output/connectivity_matrices_seed_250.npz', allow_pickle=True)
all_vertices = data['all_vertices']
face_polygons = data['face_polygons']

print("="*70)
print("FINDING WHICH FACES CONTAIN TARGET EDGES")
print("="*70)

target_edges = [
    ("V25-V26 (missing from FRONT view)", 25, 26),
    ("V0-V1 (missing from SIDE view)", 0, 1),
    ("V4-V5 (missing from SIDE view)", 4, 5),
]

for edge_name, v1_idx, v2_idx in target_edges:
    v1 = all_vertices[v1_idx]
    v2 = all_vertices[v2_idx]
    
    print(f"\n{edge_name}:")
    print(f"  V{v1_idx}: [{v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f}]")
    print(f"  V{v2_idx}: [{v2[0]:.2f}, {v2[1]:.2f}, {v2[2]:.2f}]")
    
    # Find which faces contain this edge
    faces_with_edge = []
    for face_idx, face_data in enumerate(face_polygons):
        vertices = face_data
        if not isinstance(vertices, (list, np.ndarray)):
            continue
        if len(vertices) < 3:
            continue
        
        # Check if edge (v1, v2) appears consecutively in this face
        for i in range(len(vertices)):
            vi = vertices[i]
            vj = vertices[(i+1) % len(vertices)]
            
            # Check if this edge matches our target
            if ((np.allclose(vi, v1, atol=0.01) and np.allclose(vj, v2, atol=0.01)) or
                (np.allclose(vi, v2, atol=0.01) and np.allclose(vj, v1, atol=0.01))):
                faces_with_edge.append(face_idx)
                break
    
    print(f"  Found in {len(faces_with_edge)} faces: {faces_with_edge}")
    
    # For each face, print its normal and vertices
    for face_idx in faces_with_edge:
        face_data = face_polygons[face_idx]
        vertices = face_data
        print(f"\n  Face {face_idx}: {len(vertices)} vertices")
        for i, v in enumerate(vertices):
            print(f"    V{i}: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]")
        
        # Calculate face normal
        if len(vertices) >= 3:
            v0 = np.array(vertices[0])
            v1 = np.array(vertices[1])
            v2 = np.array(vertices[2])
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            if np.linalg.norm(normal) > 1e-6:
                normal = normal / np.linalg.norm(normal)
                print(f"    Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                
                # Check dot product with view normals
                front_normal = np.array([0, -1, 0])
                side_normal = np.array([1, 0, 0])
                top_normal = np.array([0, 0, 1])
                
                dot_front = np.dot(normal, front_normal)
                dot_side = np.dot(normal, side_normal)
                dot_top = np.dot(normal, top_normal)
                
                print(f"    Dot with Front([0,-1,0]): {dot_front:.3f}")
                print(f"    Dot with Side([1,0,0]): {dot_side:.3f}")
                print(f"    Dot with Top([0,0,1]): {dot_top:.3f}")
