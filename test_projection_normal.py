#!/usr/bin/env python3
"""
Test script to verify projection normal behavior and face selection
"""

import numpy as np
import sys
import os

# Add the current directory to Python path to import from main file
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')

def test_projection_normal_behavior():
    """Test projection normal normalization and face selection with [1,1,1] vs [-1,-1,-1]"""
    
    print("="*60)
    print("PROJECTION NORMAL TEST")
    print("="*60)
    
    # Test 1: Verify normalization
    print("\n1. Testing normalization:")
    
    test_vectors = [
        [1, 1, 1],
        [-1, -1, -1],
        [2, 2, 2],
        [-3, -3, -3],
        [1, 0, 0],
        [0.2, 1.0, 0.0]  # Default from code
    ]
    
    for vector in test_vectors:
        original = np.array(vector)
        normalized = original / np.linalg.norm(original)
        magnitude = np.linalg.norm(original)
        normalized_magnitude = np.linalg.norm(normalized)
        
        print(f"  Original: {original} (magnitude: {magnitude:.6f})")
        print(f"  Normalized: {normalized} (magnitude: {normalized_magnitude:.6f})")
        print()
    
    # Test 2: Face normal examples and dot products
    print("\n2. Testing dot products with sample face normals:")
    
    # Sample face normals for a cuboid (6 faces)
    face_normals = {
        'X_pos': np.array([1, 0, 0]),    # Right face
        'X_neg': np.array([-1, 0, 0]),   # Left face  
        'Y_pos': np.array([0, 1, 0]),    # Back face
        'Y_neg': np.array([0, -1, 0]),   # Front face
        'Z_pos': np.array([0, 0, 1]),    # Top face
        'Z_neg': np.array([0, 0, -1])    # Bottom face
    }
    
    projection_normals = {
        'positive': np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1]),
        'negative': np.array([-1, -1, -1]) / np.linalg.norm([-1, -1, -1])
    }
    
    for proj_name, proj_normal in projection_normals.items():
        print(f"\nProjection normal {proj_name}: {proj_normal}")
        positive_dots = 0
        negative_dots = 0
        
        for face_name, face_normal in face_normals.items():
            dot_product = np.dot(face_normal, proj_normal)
            print(f"  {face_name}: dot = {dot_product:.6f}")
            
            if dot_product > 0:
                positive_dots += 1
            elif dot_product < 0:
                negative_dots += 1
        
        print(f"  → Positive dots: {positive_dots}, Negative dots: {negative_dots}")
        print(f"  → Total faces with |dot| > 0: {positive_dots + negative_dots}")
    
    # Test 3: Expected behavior summary
    print("\n3. Expected Behavior Summary:")
    print("="*40)
    print("For a standard cuboid with 6 faces:")
    print("• Projection normal [1,1,1] should give specific dot product pattern")
    print("• Projection normal [-1,-1,-1] should give OPPOSITE dot product signs")
    print("• Total faces selected should be the same (6) regardless of projection direction")
    print("• Only the SIGN of dot products should change, not the count")
    print()
    
    # Verify that [1,1,1] and [-1,-1,-1] give opposite results
    proj_pos = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
    proj_neg = np.array([-1, -1, -1]) / np.linalg.norm([-1, -1, -1])
    
    print("4. Verification that [1,1,1] and [-1,-1,-1] are exact opposites:")
    print(f"  [1,1,1] normalized: {proj_pos}")
    print(f"  [-1,-1,-1] normalized: {proj_neg}")
    print(f"  Sum should be zero: {proj_pos + proj_neg}")
    print(f"  Dot product should be -1: {np.dot(proj_pos, proj_neg):.6f}")
    
    return proj_pos, proj_neg, face_normals

if __name__ == "__main__":
    test_projection_normal_behavior()
