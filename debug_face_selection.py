#!/usr/bin/env python3
"""
Debug script to analyze face selection behavior with default projection normal
"""

import numpy as np

def debug_face_grouping():
    """Debug the face grouping logic that's causing only 2 faces to be selected"""
    
    print("="*60)
    print("DEBUGGING FACE SELECTION LOGIC")
    print("="*60)
    
    # Default projection normal from the program
    projection_normal = np.array([0.2, 1.0, 0.0])
    unit_projection_normal = projection_normal / np.linalg.norm(projection_normal)
    print(f"Default projection normal: {projection_normal}")
    print(f"Unit projection normal: {unit_projection_normal}")
    print(f"This is a Y-dominant projection direction")
    
    # Simulate typical face normals from a boolean CUT operation
    # The boolean CUT removes material, so we might not have all 6 standard faces
    print(f"\nSimulating face normals from boolean CUT operation:")
    
    # Example face normals that might result from CUT operation
    simulated_faces = [
        # Standard cuboid faces that remain after CUT
        ("Face 1", np.array([1, 0, 0])),    # +X face (right)
        ("Face 2", np.array([-1, 0, 0])),   # -X face (left) 
        ("Face 3", np.array([0, 1, 0])),    # +Y face (back)
        ("Face 4", np.array([0, -1, 0])),   # -Y face (front)
        ("Face 5", np.array([0, 0, 1])),    # +Z face (top)
        ("Face 6", np.array([0, 0, -1])),   # -Z face (bottom)
        
        # Additional faces created by boolean intersection (non-axis-aligned)
        ("Face 7", np.array([0.5, 0.5, 0.0])),    # Diagonal face from cut
        ("Face 8", np.array([0.0, 0.3, 0.7])),    # Another intersection face
        ("Face 9", np.array([-0.6, 0.0, 0.8])),   # Complex intersection face
    ]
    
    # Group faces by direction like the main program does
    face_groups = {'X_pos': [], 'X_neg': [], 'Y_pos': [], 'Y_neg': [], 'Z_pos': [], 'Z_neg': []}
    
    for i, (face_name, face_normal) in enumerate(simulated_faces):
        unit_face_normal = face_normal / np.linalg.norm(face_normal)
        dot_product = np.dot(unit_face_normal, unit_projection_normal)
        
        print(f"{face_name}: normal={face_normal}, unit_normal={unit_face_normal}, dot={dot_product:.3f}")
        
        # Classify face by its primary normal direction (this is the key logic!)
        abs_normal = np.abs(unit_face_normal)
        max_component = np.max(abs_normal)
        
        # Find which component is dominant
        if abs_normal[0] == max_component:  # X-dominant normal
            if unit_face_normal[0] > 0:
                face_groups['X_pos'].append((i, face_name, dot_product))
                print(f"  → Classified as X_pos")
            else:
                face_groups['X_neg'].append((i, face_name, dot_product))
                print(f"  → Classified as X_neg")
        elif abs_normal[1] == max_component:  # Y-dominant normal
            if unit_face_normal[1] > 0:
                face_groups['Y_pos'].append((i, face_name, dot_product))
                print(f"  → Classified as Y_pos")
            else:
                face_groups['Y_neg'].append((i, face_name, dot_product))
                print(f"  → Classified as Y_neg")
        elif abs_normal[2] == max_component:  # Z-dominant normal
            if unit_face_normal[2] > 0:
                face_groups['Z_pos'].append((i, face_name, dot_product))
                print(f"  → Classified as Z_pos")
            else:
                face_groups['Z_neg'].append((i, face_name, dot_product))
                print(f"  → Classified as Z_neg")
        else:
            print(f"  → ERROR: Could not classify face!")
    
    print(f"\nFace groups after classification:")
    total_selected = 0
    for direction, faces_in_group in face_groups.items():
        print(f"  {direction}: {len(faces_in_group)} faces")
        if faces_in_group:
            for face_info in faces_in_group:
                print(f"    - {face_info[1]} (dot: {face_info[2]:.3f})")
            total_selected += 1  # One face will be selected from each non-empty group
    
    print(f"\nExpected selection: {total_selected} faces (one from each non-empty group)")
    print(f"Expected for proper HLR: 6 faces (one from each direction)")
    
    # Check if some directions are missing
    missing_directions = [direction for direction, faces in face_groups.items() if not faces]
    if missing_directions:
        print(f"\nMissing directions: {missing_directions}")
        print(f"This could happen if:")
        print(f"  1. Boolean CUT removed faces in those directions")
        print(f"  2. Face extraction failed for some faces")
        print(f"  3. Face normal calculation is incorrect")
    
    # Analyze why we might only get 2 faces
    print(f"\nAnalyzing possible causes for only 2 faces being selected:")
    
    # Check if face extraction is failing
    print(f"1. Face extraction failure:")
    print(f"   - If OpenCASCADE face extraction fails, fallback to simple cuboid")
    print(f"   - Simple cuboid should give 6 faces")
    print(f"   - If only 2 faces, there's a bug in face selection logic")
    
    # Check projection criteria
    print(f"2. Projection criteria:")
    print(f"   - Default normal [0.2, 1.0, 0.0] is Y-dominant")
    print(f"   - Should include faces from all directions")
    print(f"   - No filtering based on dot product sign (fixed recently)")
    
    # Check area/validity filtering
    print(f"3. Area/validity filtering:")
    print(f"   - Faces might be rejected due to small area")
    print(f"   - Projection might fail for complex faces")
    print(f"   - Invalid polygons might be filtered out")
    
    return face_groups

if __name__ == "__main__":
    debug_face_grouping()
