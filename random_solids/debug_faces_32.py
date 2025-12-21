#!/usr/bin/env python3
"""
Debug script to compare original faces from Build_Solid vs extracted faces from Reconstruct_Solid
"""

import numpy as np
import sys

def load_original_faces(seed):
    """Load original faces from Build_Solid output"""
    filename = f"Output/solid_faces_seed_{seed}.npy"
    loaded_data = np.load(filename, allow_pickle=True)
    
    if isinstance(loaded_data, np.ndarray) and loaded_data.shape == ():
        data_dict = loaded_data.item()
        face_data = data_dict['faces']
    else:
        face_data = loaded_data
    
    faces = []
    for face in face_data:
        outer = np.array(face['outer_boundary'])
        holes = [np.array(hole) for hole in face.get('holes', [])]
        faces.append({'outer': outer, 'holes': holes})
    
    return faces

def compute_face_normal(vertices):
    """Compute face normal from vertices using cross product"""
    if len(vertices) < 3:
        return None
    
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    
    if norm < 1e-6:
        return None
    
    return normal / norm

def compute_face_d(normal, vertex):
    """Compute d value from plane equation n·p + d = 0"""
    return -np.dot(normal, vertex)

def main():
    seed = 32
    
    print("="*70)
    print(f"FACE COMPARISON FOR SEED {seed}")
    print("="*70)
    
    # Load original faces
    original_faces = load_original_faces(seed)
    print(f"\nOriginal faces from Build_Solid: {len(original_faces)}")
    
    # Compute normals and d values for original faces
    print("\n" + "="*70)
    print("ORIGINAL FACE DETAILS")
    print("="*70)
    
    for idx, face in enumerate(original_faces):
        outer = face['outer']
        num_holes = len(face['holes'])
        
        normal = compute_face_normal(outer)
        if normal is not None:
            d = compute_face_d(normal, outer[0])
            print(f"Face {idx:2d}: {len(outer):2d} vertices, {num_holes} holes")
            print(f"         Normal: [{normal[0]:7.4f}, {normal[1]:7.4f}, {normal[2]:7.4f}]")
            print(f"         d = {d:10.4f}")
        else:
            print(f"Face {idx:2d}: {len(outer):2d} vertices, {num_holes} holes [DEGENERATE]")
    
    print("\n" + "="*70)
    print("Run Reconstruct_Solid.py to see which faces are extracted")
    print("="*70)

if __name__ == "__main__":
    main()
