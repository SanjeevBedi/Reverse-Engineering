"""
Debug edge detection for missing edges.
Load the view summaries and check if edges exist in the connectivity matrices.
"""
import numpy as np


def check_missing_edges():
    """Check why specific edges are missing from reconstruction."""
    
    # Load unified summary (reconstructed edges)
    unified = np.load('unified_summary.npy')
    
    # Load view summaries (original connectivity from views)
    top_summary = np.load('top_view_summary.npy')
    front_summary = np.load('front_view_summary.npy')
    side_summary = np.load('side_view_summary.npy')
    
    print("="*70)
    print("DEBUG: MISSING EDGES ANALYSIS")
    print("="*70)
    print(f"Unified summary shape: {unified.shape}")
    print(f"Top summary shape: {top_summary.shape}")
    print(f"Front summary shape: {front_summary.shape}")
    print(f"Side summary shape: {side_summary.shape}")
    
    # Define missing edges
    missing_edges = [
        (61, 113, "X-parallel"),
        (24, 114, "X-parallel"),
        (21, 23, "Y-parallel"),
        (23, 17, "Diagonal"),
    ]
    
    tolerance = 1e-5
    
    for v1_idx, v2_idx, edge_type in missing_edges:
        print("\n" + "="*70)
        print(f"Checking Edge V{v1_idx} -- V{v2_idx} ({edge_type})")
        print("="*70)
        
        # Get 3D coordinates from unified summary
        v1_3d = unified[v1_idx, 1:4]  # columns 1-3 are x,y,z
        v2_3d = unified[v2_idx, 1:4]
        
        print(f"V{v1_idx}: ({v1_3d[0]:.6f}, {v1_3d[1]:.6f}, "
              f"{v1_3d[2]:.6f})")
        print(f"V{v2_idx}: ({v2_3d[0]:.6f}, {v2_3d[1]:.6f}, "
              f"{v2_3d[2]:.6f})")
        
        # Determine edge type
        dx = abs(v1_3d[0] - v2_3d[0])
        dy = abs(v1_3d[1] - v2_3d[1])
        dz = abs(v1_3d[2] - v2_3d[2])
        
        same_x = dx < tolerance
        same_y = dy < tolerance
        same_z = dz < tolerance
        
        print(f"Δx={dx:.6f}, Δy={dy:.6f}, Δz={dz:.6f}")
        print(f"same_x={same_x}, same_y={same_y}, same_z={same_z}")
        
        # Determine required views
        if same_y and same_z:
            required_views = ['Top', 'Front']
            edge_class = "X-PARALLEL"
        elif same_x and same_z:
            required_views = ['Top', 'Side']
            edge_class = "Y-PARALLEL"
        elif same_x and same_y:
            required_views = ['Front', 'Side']
            edge_class = "VERTICAL (Z-PARALLEL)"
        else:
            required_views = ['Top', 'Front', 'Side']
            edge_class = "DIAGONAL"
        
        print(f"Edge classification: {edge_class}")
        print(f"Required views: {required_views}")
        
        # Check each view
        view_configs = [
            ('Top', top_summary, [0, 1]),  # match x,y
            ('Front', front_summary, [0, 2]),  # match x,z
            ('Side', side_summary, [1, 2]),  # match y,z
        ]
        
        for view_name, view_summary, match_indices in view_configs:
            print(f"\n  {view_name} View (match indices {match_indices}):")
            
            # Find vertices matching projections
            v1_matches = []
            v2_matches = []
            
            v1_proj = [v1_3d[i] for i in match_indices]
            v2_proj = [v2_3d[i] for i in match_indices]
            
            print(f"    Looking for V{v1_idx} projection: {v1_proj}")
            print(f"    Looking for V{v2_idx} projection: {v2_proj}")
            
            for i in range(len(view_summary)):
                vertex_3d = view_summary[i, :3]
                vertex_proj = [vertex_3d[idx] for idx in match_indices]
                
                v1_dist = np.linalg.norm(
                    np.array(vertex_proj) - np.array(v1_proj))
                v2_dist = np.linalg.norm(
                    np.array(vertex_proj) - np.array(v2_proj))
                
                if v1_dist < tolerance:
                    v1_matches.append(i)
                if v2_dist < tolerance:
                    v2_matches.append(i)
            
            print(f"    V{v1_idx} matches (first 5): {v1_matches[:5]}")
            print(f"    V{v2_idx} matches (first 5): {v2_matches[:5]}")
            
            # Check connectivity matrix
            connectivity_start = 6  # connectivity matrix starts at col 6
            found_edge = False
            
            if not v1_matches:
                print(f"    ✗ V{v1_idx} NOT FOUND in {view_name} view!")
            elif not v2_matches:
                print(f"    ✗ V{v2_idx} NOT FOUND in {view_name} view!")
            else:
                for i in v1_matches[:3]:  # Check first few matches
                    for j in v2_matches[:3]:
                        if i == j:
                            continue
                        
                        # Check connectivity in both directions
                        if connectivity_start + j < view_summary.shape[1]:
                            conn_forward = view_summary[i, connectivity_start + j]
                            if conn_forward > 0:
                                print(f"    ✓ Edge found: "
                                      f"row {i} → col {j}, "
                                      f"connectivity = {conn_forward}")
                                found_edge = True
                        
                        if connectivity_start + i < view_summary.shape[1]:
                            conn_backward = view_summary[j, connectivity_start + i]
                            if conn_backward > 0 and not found_edge:
                                print(f"    ✓ Edge found: "
                                      f"row {j} → col {i}, "
                                      f"connectivity = {conn_backward}")
                                found_edge = True
                
                if not found_edge:
                    print(f"    ✗ No connectivity found in {view_name} view!")
            
            if view_name in required_views and not found_edge:
                print(f"    ⚠️  MISSING in REQUIRED view {view_name}!")
        
        # Check unified summary reconstruction
        adjacency = unified[:, 10:]
        reconstructed = adjacency[v1_idx, v2_idx] == 1
        print(f"\n  Reconstructed in unified summary: {reconstructed}")
        
        if not reconstructed:
            print("  ⚠️  EDGE NOT RECONSTRUCTED!")


if __name__ == "__main__":
    check_missing_edges()
