"""
Debug script to check why specific edges are missing from reconstruction.
"""
import numpy as np
import sys

def check_edge_in_view_summary(v1_3d, v2_3d, view_summary, view_name, match_indices):
    """
    Check if an edge exists in a view summary's connectivity matrix.
    
    Args:
        v1_3d: 3D coordinates of vertex 1
        v2_3d: 3D coordinates of vertex 2
        view_summary: View summary array
        view_name: Name of the view ('Top', 'Front', 'Side')
        match_indices: Indices to match [e.g., [0,1] for top view (x,y)]
    """
    tolerance = 1e-5
    
    print(f"\n  Checking {view_name} View:")
    print(f"    Match indices: {match_indices}")
    
    # Find matching vertices in view summary by projected coordinates
    v1_proj = v1_3d[match_indices]
    v2_proj = v2_3d[match_indices]
    
    print(f"    Looking for V1 projection: {v1_proj}")
    print(f"    Looking for V2 projection: {v2_proj}")
    
    # Find vertices with matching projections
    v1_candidates = []
    v2_candidates = []
    
    for i in range(len(view_summary)):
        vertex_3d = view_summary[i, :3]
        vertex_proj = vertex_3d[match_indices]
        
        if np.linalg.norm(vertex_proj - v1_proj) < tolerance:
            v1_candidates.append(i)
        if np.linalg.norm(vertex_proj - v2_proj) < tolerance:
            v2_candidates.append(i)
    
    print(f"    V1 candidates in view: {v1_candidates}")
    print(f"    V2 candidates in view: {v2_candidates}")
    
    # Check connectivity matrix for these candidates
    if not v1_candidates or not v2_candidates:
        print(f"    ✗ No matching vertices found in {view_name} view")
        return False
    
    # Check if any combination has connectivity
    connectivity_matrix = view_summary[:, 6:]
    found = False
    
    for i in v1_candidates:
        for j in v2_candidates:
            if i < len(connectivity_matrix) and j < connectivity_matrix.shape[1]:
                conn_value = connectivity_matrix[i, j]
                if conn_value > 0:
                    print(f"    ✓ Edge found: row {i} → col {j}, "
                          f"connectivity = {conn_value}")
                    found = True
    
    if not found:
        print(f"    ✗ No connectivity found in {view_name} view matrix")
    
    return found


def main():
    # Load unified summary
    summary = np.load('unified_summary.npy')
    
    # Check specific missing edges
    missing_edges = [
        (61, 113, "X-parallel"),
        (24, 114, "X-parallel"),
        (23, 17, "Diagonal"),
        (21, 23, "Y-parallel"),
    ]
    
    print("=" * 70)
    print("DEBUGGING MISSING EDGES")
    print("=" * 70)
    
    # We need to load the view summaries
    print("\nNote: This script checks the unified summary adjacency matrix.")
    print("To check view summaries, we need to re-run the main script.\n")
    
    for v1_idx, v2_idx, edge_type in missing_edges:
        if v1_idx >= len(summary) or v2_idx >= len(summary):
            print(f"\nSkipping V{v1_idx}--V{v2_idx}: "
                  f"vertex index out of range")
            continue
        
        v1_3d = summary[v1_idx, 1:4]
        v2_3d = summary[v2_idx, 1:4]
        
        same_x = abs(v1_3d[0] - v2_3d[0]) < 1e-6
        same_y = abs(v1_3d[1] - v2_3d[1]) < 1e-6
        same_z = abs(v1_3d[2] - v2_3d[2]) < 1e-6
        
        print("\n" + "=" * 70)
        print(f"Edge V{v1_idx} -- V{v2_idx} ({edge_type})")
        print("=" * 70)
        print(f"V{v1_idx}: ({v1_3d[0]:.6f}, {v1_3d[1]:.6f}, "
              f"{v1_3d[2]:.6f})")
        print(f"V{v2_idx}: ({v2_3d[0]:.6f}, {v2_3d[1]:.6f}, "
              f"{v2_3d[2]:.6f})")
        
        # Determine required views
        if same_x and same_y:
            required_views = ['front', 'side']
            print("Type: VERTICAL (z-parallel)")
        elif same_x and same_z:
            required_views = ['top', 'side']
            print("Type: Y-PARALLEL")
        elif same_y and same_z:
            required_views = ['top', 'front']
            print("Type: X-PARALLEL")
        else:
            required_views = ['top', 'front', 'side']
            print("Type: DIAGONAL")
        
        print(f"Required views: {required_views}")
        
        # Check adjacency matrix
        adjacency = summary[:, 10:]
        exists = adjacency[v1_idx, v2_idx] == 1
        print(f"Exists in reconstructed edges: {exists}")
        
        if not exists:
            print("\nPOSSIBLE REASONS FOR MISSING:")
            print("1. Edge doesn't exist in original solid geometry")
            print("2. Vertices not found in view summaries (projection issue)")
            print("3. Connectivity matrix in view summary is 0")
            print("4. Edge validation failed in required views")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("Run V6_current.py with additional debug output to check:")
    print("1. If these vertices appear in view summaries")
    print("2. If connectivity values exist in view matrices")
    print("3. Which phase should detect these edges")


if __name__ == "__main__":
    main()
