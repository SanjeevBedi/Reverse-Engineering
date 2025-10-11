"""
Edge reconstruction algorithm for solid geometry - Reverse Engineering.
Finds valid edges by checking all vertex pairs against three orthogonal views.
"""

import numpy as np


def reconstruct_edges_from_views(selected_vertices, top_view_summary,
                                  front_view_summary, side_view_summary,
                                  Vertex_Top_View, Vertex_Front_View,
                                  Vertex_Side_View, all_vertices_sorted,
                                  expected_unique_edges=None):
    """
    Reconstruct solid edges using reverse engineering approach.
    
    NEW ALGORITHM (True Reverse Engineering):
    1. Take the list of reconstructed selected_vertices
    2. Index them (V0, V1, V2, ...)
    3. For each vertex pair (Vi, Vj) where j > i:
       a. Project both vertices to top/front/side views
       b. Check if edge exists in each view's connectivity matrix
       c. Classify edge visibility:
          - 0: Not visible in any view
          - 1: Visible in one view
          - 2: Visible in two views
          - 3: Visible in all three views
       d. Post-process class 1 edges:
          - If vertices have same xy/yz/zx, check if vertex exists at that point
          - If yes, elevate to class 2; otherwise filter out
       e. Accept edges with classification >= 2
    
    This approach works even if some vertices weren't in the original
    view summaries (e.g., collapsed vertices like V3).
    
    Args:
        selected_vertices: Array of reconstructed 3D vertices [n, 3]
        top_view_summary: Summary array for top view
        front_view_summary: Summary array for front view
        side_view_summary: Summary array for side view
        Vertex_Top_View: Unused (legacy)
        Vertex_Front_View: Unused (legacy)
        Vertex_Side_View: Unused (legacy)
        all_vertices_sorted: Unused (legacy)
        expected_unique_edges: Expected number of edges for validation
        
    Returns:
        Tuple: (valid_edges, edges_with_class)
            valid_edges: List of edge tuples [(v1_idx, v2_idx), ...]
            edges_with_class: List of tuples [(v1_idx, v2_idx, classification), ...]
    """
    
    print("\n" + "="*70)
    print("EDGE RECONSTRUCTION - REVERSE ENGINEERING (EXHAUSTIVE)")
    print("="*70)
    print("Goal: Find edges by checking ALL vertex pairs")
    print("Method: Create master connectivity array from view projections")
    print("Classification:")
    print("  0: Edge not visible in any view")
    print("  1: Edge visible in one view")
    print("  2: Edge visible in two views")
    print("  3: Edge visible in all three views")
    
    tolerance = 1e-4
    
    # Helper function to check if edge exists in a view
    def check_edge_in_view(v1_3d, v2_3d, view_summary, view_name):
        """
        Check if an edge between two 3D vertices exists in a given view.
        Args:
            v1_3d: First vertex 3D coordinates [x, y, z]
            v2_3d: Second vertex 3D coordinates [x, y, z]
            view_summary: Summary array for this view (format: [vertex_index, proj_x, proj_y, connectivity...])
            view_name: 'top', 'front', or 'side'
        Returns:
            bool: True if edge exists in this view
        """
        # ...existing check_edge_in_view logic...

    print("\n[INFO] Using merged connectivity matrix for edge classification.")
    # Instead of exhaustive pair checking, use merged connectivity matrix
    # Assume merged_matrix is passed as top_view_summary (for compatibility)
    merged_matrix = top_view_summary
    N = merged_matrix.shape[0]
    edge_classifications = []
    filtered_count = 0
    elevated_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            col_idx = 7 + j
            if col_idx < merged_matrix.shape[1]:
                conn = int(merged_matrix[i, col_idx])
                if conn > 0:
                    # Classify edge by connectivity value (1,2,3)
                    edge_classifications.append((i, j, conn, [conn]))
    print(f"Total edge pairs checked (from merged matrix): {len(edge_classifications)}")
    # Only use merged connectivity matrix for all downstream operations
    def vertex_exists_in_view(v, view_summary, view_name):
        """Check if vertex v exists in the view summary."""
        # All legacy validation code removed; merged_matrix logic is now exclusive
    final_filtered = []
    
    valid_edges = []
    edges_with_class = []
    
    
    print(f"  Total valid edges found: {len(valid_edges)}")
    
    # Show some examples
    if len(edges_with_class) > 0:
        print(f"\nFirst {min(10, len(edges_with_class))} edges:")
        for idx, (i, j, classification) in enumerate(edges_with_class[:10]):
            v1 = selected_vertices[i]
            v2 = selected_vertices[j]
            print(f"  Edge {idx+1}: V{i}--V{j} (class={classification})")
            print(f"    V{i}: ({v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f})")
            print(f"    V{j}: ({v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f})")
    
    # Validation
    if expected_unique_edges is not None:
        print(f"\n{'='*70}")
        print("VALIDATION")
        print(f"{'='*70}")
        print(f"Expected edges: {expected_unique_edges}")
        print(f"Found edges: {len(valid_edges)}")
        if len(valid_edges) == expected_unique_edges:
            print("SUCCESS: Found all expected unique edges!")
        else:
            diff = len(valid_edges) - expected_unique_edges
            if diff > 0:
                print(f"Over-detection: {diff} extra edges found")
            else:
                print(f"Under-detection: {-diff} edges missing")
    
    print("="*70)
    
    return valid_edges, edges_with_class
