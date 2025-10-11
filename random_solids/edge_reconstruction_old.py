"""
Edge reconstruction algorithm for solid geometry.
Finds valid edges by checking if they appear in all three orthogonal views.
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
          - 1: Visible in one view (1-x, 2-y, 3-z axis)
          - 2: Visible in two views (1-xy, 2-yz, 3-xz planes)
          - 3: Visible in all three views
       d. Accept edges with classification ≥ required threshold
    
    This approach works even if some vertices weren't in the original
    view summaries (e.g., collapsed vertices like V3).
    
    Args:
        selected_vertices: Array of reconstructed 3D vertices [n, 3]
        top_view_summary: Summary array for top view
        front_view_summary: Summary array for front view  
        side_view_summary: Summary array for side view
        Vertex_Top_View: Vertex connectivity matrix for top view (unused)
        Vertex_Front_View: Vertex connectivity matrix for front view (unused)
        Vertex_Side_View: Vertex connectivity matrix for side view (unused)
        all_vertices_sorted: Original sorted vertices from solid (unused)
        
    Returns:
        List of valid edges as tuples: [(v1_idx, v2_idx), ...]
    """
    
    print("\n" + "="*70)
    print("EDGE RECONSTRUCTION ALGORITHM - REVERSE ENGINEERING")
    print("="*70)
    print("Goal: Find edges by checking all vertex pairs in reconstructed list")
    print("Method: Create master connectivity array from view projections")
    print("Classification:")
    print("  0: Edge not visible in any view")
    print("  1: Edge visible in one view")
    print("  2: Edge visible in two views")
    print("  3: Edge visible in all three views")
    
    valid_edges = []
    tolerance = 1e-4
    
    n_vertices = len(selected_vertices)
    print(f"\nProcessing {n_vertices} reconstructed vertices")
    print(f"Maximum possible edges: {n_vertices * (n_vertices - 1) // 2}")
    
    # Helper function to check if edge exists in a view by projection matching
    def check_edge_in_view(v1_3d, v2_3d, view_summary, match_coords, view_name):
        """
        Check if an edge between two 3D vertices exists in a view.
        
        Args:
            v1_3d: First vertex [x, y, z]
            v2_3d: Second vertex [x, y, z]
            view_summary: Summary array for the view
            match_coords: Which coordinates to match (e.g., [0,1] for x,y)
            view_name: Name for debugging
            
        Returns:
            bool: True if edge exists in this view
        """
        # Find vertices in view_summary that match v1 and v2 projections
        v1_matches = []
        v2_matches = []
        
        for i in range(view_summary.shape[0]):
            row_coords = view_summary[i, 0:3]
            v1_proj = [v1_3d[idx] for idx in match_coords]
            v2_proj = [v2_3d[idx] for idx in match_coords]
            row_proj = [row_coords[idx] for idx in match_coords]
            
            if np.allclose(row_proj, v1_proj, atol=tolerance):
                v1_matches.append(i)
            if np.allclose(row_proj, v2_proj, atol=tolerance):
                v2_matches.append(i)
        
        # Check connectivity between all matching pairs
        for i in v1_matches:
            for j in v2_matches:
                if i == j:
                    continue
                # Check both directions in connectivity matrix
                if 6 + j < view_summary.shape[1]:
                    if view_summary[i, 6 + j] > 0:
                        return True
                if 6 + i < view_summary.shape[1]:
                    if view_summary[j, 6 + i] > 0:
                        return True
        
        return False
    
    # Create master connectivity array
    # For each vertex pair, check visibility in each view
    print("\nCreating master connectivity array...")
    print("Checking all vertex pairs against three orthogonal views...")
    
    edge_classifications = []  # List of (v_i, v_j, classification, views_found)
    
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            v1 = selected_vertices[i]
            v2 = selected_vertices[j]
            
            # Check if edge exists in each view
            top_exists = check_edge_in_view(
                v1, v2, top_view_summary, [0, 1], "Top")
            front_exists = check_edge_in_view(
                v1, v2, front_view_summary, [0, 2], "Front")
            side_exists = check_edge_in_view(
                v1, v2, side_view_summary, [1, 2], "Side")
            
            views_found = []
            if top_exists:
                views_found.append('top')
            if front_exists:
                views_found.append('front')
            if side_exists:
                views_found.append('side')
            
            num_views = len(views_found)
            
            # Classification based on how many views show the edge
            if num_views == 0:
                classification = 0
            elif num_views == 1:
                classification = 1
            elif num_views == 2:
                classification = 2
            else:  # num_views == 3
                classification = 3
            
            edge_classifications.append((i, j, classification, views_found))
    
    # Filter edges: accept those with classification >= 1
    # (visible in at least one view)
    print(f"\nEdge classification summary:")
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, _, cls, _ in edge_classifications:
        class_counts[cls] += 1
    
    print(f"  Class 0 (not visible): {class_counts[0]} edges")
    print(f"  Class 1 (one view): {class_counts[1]} edges")
    print(f"  Class 2 (two views): {class_counts[2]} edges")
    print(f"  Class 3 (three views): {class_counts[3]} edges")
    
    # Accept edges with classification >= 1
    min_classification = 1
    print(f"\nAccepting edges with classification >= {min_classification}")
    
    for v_i, v_j, classification, views_found in edge_classifications:
        if classification >= min_classification:
            valid_edges.append((v_i, v_j))
            if len(valid_edges) <= 10:  # Show first 10
                v1 = selected_vertices[v_i]
                v2 = selected_vertices[v_j]
                print(f"  Edge V{v_i}--V{v_j}: class={classification}, "
                      f"views={views_found}")
                print(f"    V{v_i}: ({v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f})")
                print(f"    V{v_j}: ({v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f})")
    
    print(f"\nTotal valid edges found: {len(valid_edges)}")
    if expected_unique_edges:
        print(f"Expected edges: {expected_unique_edges}")
        print(f"Detection rate: "
              f"{100.0 * len(valid_edges) / expected_unique_edges:.1f}%")
    
    return valid_edges

            
        Returns:
            bool: True if edge exists in this view
        """
        # Determine which coordinates to match based on view
        if view_name == "Top":
            match_indices = [0, 1]  # Match x, y
        elif view_name == "Front":
            match_indices = [0, 2]  # Match x, z
        elif view_name == "Side":
            match_indices = [1, 2]  # Match y, z
        else:
            # Default to full 3D matching if view unknown
            match_indices = [0, 1, 2]
        
        # Find row indices in view_summary by matching projected coordinates
        v1_matches = []
        v2_matches = []
        
        debug_edge = (np.allclose(v1_world, [32.268, 38.420, 0.000], atol=tolerance) and
                     np.allclose(v2_world, [32.268, 63.742, 0.000], atol=tolerance))
        
        if debug_edge and view_name == "Side":
            print(f"\n          [DEBUG {view_name}] Checking V0→V10 edge")
            print(f"            Looking for v1: {v1_world}")
            print(f"            Looking for v2: {v2_world}")
            print(f"            Matching by indices {match_indices} (view={view_name})")
            print(f"            View summary has {view_summary.shape[0]} vertices")
        
        for i in range(view_summary.shape[0]):
            # Get 3D world coordinates from columns 0-2
            world_coords = view_summary[i, 0:3]
            
            # Match only the relevant coordinates for this view
            v1_proj = [v1_world[idx] for idx in match_indices]
            v2_proj = [v2_world[idx] for idx in match_indices]
            coord_proj = [world_coords[idx] for idx in match_indices]
            
            if np.allclose(coord_proj, v1_proj, atol=tolerance):
                v1_matches.append(i)
                if debug_edge and view_name == "Side" and len(v1_matches) <= 3:
                    print(f"            Found v1 match at row {i}: {world_coords} (projected: {coord_proj})")
            if np.allclose(coord_proj, v2_proj, atol=tolerance):
                v2_matches.append(i)
                if debug_edge and view_name == "Side" and len(v2_matches) <= 3:
                    print(f"            Found v2 match at row {i}: {world_coords} (projected: {coord_proj})")
        
        if debug_edge and view_name == "Side":
            print(f"            v1_matches: {v1_matches[:5]}, v2_matches: {v2_matches[:5]}")
        
        # Check all combinations of matches for edge connectivity
        for v1_idx in v1_matches:
            for v2_idx in v2_matches:
                if v1_idx == v2_idx:
                    continue
                # Check if edge exists in either direction
                if 6 + v2_idx < view_summary.shape[1]:
                    conn_val = view_summary[v1_idx, 6 + v2_idx]
                    if debug_edge and view_name == "Side" and conn_val > 0:
                        print(f"            Edge found: [{v1_idx}, 6+{v2_idx}] = {conn_val}")
                    if conn_val > 0:
                        return True
                if 6 + v1_idx < view_summary.shape[1]:
                    conn_val = view_summary[v2_idx, 6 + v1_idx]
                    if debug_edge and view_name == "Side" and conn_val > 0:
                        print(f"            Edge found: [{v2_idx}, 6+{v1_idx}] = {conn_val}")
                    if conn_val > 0:
                        return True
        
        if debug_edge and view_name == "Side":
            if len(v1_matches) == 0:
                print(f"            v1 NOT FOUND in {view_name} view!")
            if len(v2_matches) == 0:
                print(f"            v2 NOT FOUND in {view_name} view!")
        
        return False
    
    # Helper function to check if edge exists in a view using projection matching
    def edge_exists_in_view_by_projection(v1_world, v2_world, view_summary, vertex_matrix, projection_normal, view_name=""):
        """
        Check if an edge between two 3D vertices exists in a given view by checking
        if the projected edge matches any edge in the view's vertex matrix.
        
        Args:
            v1_world: First vertex 3D coordinates [x, y, z]
            v2_world: Second vertex 3D coordinates [x, y, z]
            view_summary: Summary array for this view
            vertex_matrix: Vertex connectivity matrix for this view
            projection_normal: Normal vector for this view's projection
            view_name: Name for debugging
            
        Returns:
            bool: True if edge exists in this view
        """
        # Project both vertices to the view's 2D coordinates
        def project_to_view(vertex_3d, normal):
            """Project 3D vertex to view coordinates"""
            normal = np.array(normal)
            normal = normal / np.linalg.norm(normal)
            
            # Create orthogonal basis
            if abs(normal[0]) < 0.9:
                temp = np.array([1.0, 0.0, 0.0])
            else:
                temp = np.array([0.0, 1.0, 0.0])
            
            u = temp - np.dot(temp, normal) * normal
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)
            v = v / np.linalg.norm(v)
            
            vertex = np.array(vertex_3d)
            proj_u = np.dot(vertex, u)
            proj_v = np.dot(vertex, v)
            return proj_u, proj_v
        
        v1_proj = project_to_view(v1_world, projection_normal)
        v2_proj = project_to_view(v2_world, projection_normal)
        
        # Find all vertices in view_summary that match the projected coordinates
        v1_candidates = []
        v2_candidates = []
        
        for i in range(view_summary.shape[0]):
            # Projected coordinates are in columns 3, 4 (or 4, 5 depending on view)
            # For consistency, use appropriate columns based on view
            view_proj_u = view_summary[i, 3]
            view_proj_v = view_summary[i, 4]
            
            if abs(view_proj_u - v1_proj[0]) < tolerance and abs(view_proj_v - v1_proj[1]) < tolerance:
                v1_candidates.append(i)
            if abs(view_proj_u - v2_proj[0]) < tolerance and abs(view_proj_v - v2_proj[1]) < tolerance:
                v2_candidates.append(i)
        
        # Check if any combination of candidates has an edge in the vertex matrix
        for v1_idx in v1_candidates:
            for v2_idx in v2_candidates:
                if (v1_idx < vertex_matrix.shape[0] and v2_idx < vertex_matrix.shape[1] and
                    vertex_matrix[v1_idx, v2_idx] > 0):
                    return True
                if (v2_idx < vertex_matrix.shape[0] and v1_idx < vertex_matrix.shape[1] and
                    vertex_matrix[v2_idx, v1_idx] > 0):
                    return True
        
        return False
    
    print(f"\nStep 1: Extracting edges from all three views...")
    print(f"Selected vertices: {len(selected_vertices)}")
    print(f"Strategy: Process Top → Front → Side views")
    print(f"  Mark found edges with value 11 to avoid duplicates")
    
    # Step 1: Process each vertex in top view
    edges_checked = 0
    edges_validated = 0
    
    print(f"\n{'='*70}")
    print(f"PHASE 1: TRAVERSING TOP VIEW CONNECTIVITY MATRIX")
    print(f"{'='*70}")
    print(f"Top view vertices: {top_view_summary.shape[0]}")

    
    for i in range(top_view_summary.shape[0]):
        # Get start vertex from top view
        start_world = top_view_summary[i, 0:3]  # World coordinates [x, y, z]
        start_x, start_y = start_world[0], start_world[1]
        
        # Check connectivity row for this vertex
        connectivity_row = top_view_summary[i, 6:]  # Extract connectivity from columns 6+
        connections = []
        for j in range(len(connectivity_row)):
            if connectivity_row[j] > 0:
                connections.append((j, connectivity_row[j]))
        
        if connections and i < 5:  # Show first 5 vertices with connections
            print(f"\n[Row {i}] Vertex at ({start_world[0]:.3f}, {start_world[1]:.3f}, {start_world[2]:.3f})")
            print(f"  Connectivity row (col 6+): {connectivity_row[:min(10, len(connectivity_row))]}...")
            print(f"  Connections found: {len(connections)}")
            for j, val in connections[:5]:  # Show first 5 connections
                end_world = top_view_summary[j, 0:3]
                print(f"    → Vertex {j} (value={val}): ({end_world[0]:.3f}, {end_world[1]:.3f}, {end_world[2]:.3f})")
        
        # Step 2: Find all edges from this vertex in top view
        for j in range(top_view_summary.shape[0]):
            if i == j:
                continue
                
            # Check if edge exists in top view using embedded connectivity matrix
            edge_value = 0
            if 6 + j < top_view_summary.shape[1]:
                edge_value = top_view_summary[i, 6 + j]
            
            if edge_value > 0:
                
                edges_checked += 1
                
                # Get end vertex from top view
                end_world = top_view_summary[j, 0:3]
                end_x, end_y = end_world[0], end_world[1]
                
                if edges_checked <= 3:  # Show first 3 edge checks in detail
                    print(f"\n  [Edge Check #{edges_checked}] Top view edge: Row {i} → Row {j} (value={edge_value})")
                    print(f"    Top view vertices (from summary array):")
                    print(f"      Start (row {i}): 3D = ({start_world[0]:.3f}, {start_world[1]:.3f}, {start_world[2]:.3f})")
                    print(f"      End   (row {j}): 3D = ({end_world[0]:.3f}, {end_world[1]:.3f}, {end_world[2]:.3f})")
                    print(f"      Edge in top view: ({start_x:.3f}, {start_y:.3f}) → ({end_x:.3f}, {end_y:.3f})")

                
                # Step 3a: Find vertices in selected_vertices with matching x,y for start
                start_candidates = find_vertices_by_xy(start_x, start_y, selected_vertices)
                
                # Step 3b: Find vertices in selected_vertices with matching x,y for end
                end_candidates = find_vertices_by_xy(end_x, end_y, selected_vertices)
                
                if edges_checked <= 3:
                    print(f"    Matching by x,y projection in selected_vertices:")
                    print(f"      Start ({start_x:.3f}, {start_y:.3f}) → {len(start_candidates)} candidates: {start_candidates[:5]}")
                    print(f"      End   ({end_x:.3f}, {end_y:.3f}) → {len(end_candidates)} candidates: {end_candidates[:5]}")
                
                # Step 3c: Try each combination
                combination_count = 0
                for start_idx in start_candidates:
                    start_3d = selected_vertices[start_idx]
                    
                    for end_idx in end_candidates:
                        end_3d = selected_vertices[end_idx]
                        combination_count += 1
                        
                        # Skip self-loops (vertex connecting to itself)
                        if start_idx == end_idx:
                            continue
                        
                        # Determine which views the edge should be visible in
                        # based on how many coordinates match
                        delta = np.abs(start_3d - end_3d)
                        
                        # Count which coordinates are the same
                        same_x = delta[0] < tolerance
                        same_y = delta[1] < tolerance
                        same_z = delta[2] < tolerance
                        
                        if edges_checked <= 3:
                            print(f"      Combination {combination_count}: Testing V{start_idx} → V{end_idx}")
                            print(f"        Start V{start_idx}: 3D = ({start_3d[0]:.3f}, {start_3d[1]:.3f}, {start_3d[2]:.3f})")
                            print(f"        End   V{end_idx}: 3D = ({end_3d[0]:.3f}, {end_3d[1]:.3f}, {end_3d[2]:.3f})")
                            print(f"        Δx={delta[0]:.4f}, Δy={delta[1]:.4f}, Δz={delta[2]:.4f}")
                            print(f"        Same: x={same_x}, y={same_y}, z={same_z}")
                        
                        # Determine required views based on edge orientation
                        if same_x and same_y:
                            # Vertical edge (only z changes)
                            # Visible in views that include z-axis: front & side
                            # Invisible in top view (projects to point)
                            required_views = ['front', 'side']
                        elif same_x and same_z:
                            # Edge parallel to y-axis (only y changes)
                            # Visible in views that include y-axis: top & side
                            # Invisible in front view (projects to point)
                            required_views = ['top', 'side']
                        elif same_y and same_z:
                            # Edge parallel to x-axis (only x changes)
                            # Visible in views that include x-axis: top & front
                            # Invisible in side view (projects to point)
                            required_views = ['top', 'front']
                        else:
                            # General edge, must exist in all three views
                            required_views = ['top', 'front', 'side']
                        
                        if edges_checked <= 3:
                            print(f"        Required views: {required_views}")
                        
                        # Check existence in required views
                        views_valid = True
                        
                        if 'front' in required_views:
                            front_exists = edge_exists_in_view(
                                start_3d, end_3d, front_view_summary,
                                [3, 5], "Front"
                            )
                            views_valid = views_valid and front_exists
                            if edges_checked <= 3:
                                print(f"        Front view check: {front_exists}")
                        
                        if 'side' in required_views:
                            side_exists = edge_exists_in_view(
                                start_3d, end_3d, side_view_summary,
                                [4, 5], "Side"
                            )
                            views_valid = views_valid and side_exists
                            if edges_checked <= 3:
                                print(f"        Side view check: {side_exists}")
                        
                        if edges_checked <= 3:
                            print(f"        Overall valid: {views_valid}")
                        
                        # Edge exists in top view (we're iterating from top view)
                        # If edge is valid in all required views, it's a valid edge
                        if views_valid:
                            # Avoid duplicates (check if reverse edge already added)
                            edge = tuple(sorted([start_idx, end_idx]))
                            if edge not in valid_edges:
                                valid_edges.append(edge)
                                edges_validated += 1
                                
                                if edges_validated <= 5:  # Show first 5 for debugging
                                    print(f"  Valid edge found: V{start_idx} -- V{end_idx}")
                                    print(f"    Start: ({start_3d[0]:.3f}, {start_3d[1]:.3f}, {start_3d[2]:.3f})")
                                    print(f"    End:   ({end_3d[0]:.3f}, {end_3d[1]:.3f}, {end_3d[2]:.3f})")
                                    print(f"    Required views: {required_views}")
    
    print(f"\nPhase 1 Complete - Top View Processing:")
    print(f"  Edges checked from top view: {edges_checked}")
    print(f"  Valid edges found: {len(valid_edges)}")
    
    # ==================================================================
    # PHASE 2: Process Front View (captures edges invisible in top view)
    # ==================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Processing Front View")
    print(f"{'='*70}")
    
    edges_checked_front = 0
    edges_found_front = 0
    
    # Process front view connectivity matrix
    for i in range(len(front_view_summary)):
        for j in range(len(front_view_summary)):
            if i == j:
                continue
            
            # Check connectivity in front view (column 6+j)
            connectivity = front_view_summary[i, 6 + j]
            
            if connectivity >= 1:  # Edge exists in front view
                edges_checked_front += 1
                
                # Get 3D coordinates from front view summary
                start_3d = front_view_summary[i, :3]
                end_3d = front_view_summary[j, :3]
                start_x, start_z = start_3d[0], start_3d[2]
                end_x, end_z = end_3d[0], end_3d[2]
                
                # Find matching vertices in selected_vertices by x,z
                start_candidates = find_vertices_by_xz(start_x, start_z,
                                                       selected_vertices)
                end_candidates = find_vertices_by_xz(end_x, end_z,
                                                     selected_vertices)
                
                if not start_candidates or not end_candidates:
                    continue
                
                # Try each combination
                for start_idx in start_candidates:
                    start_3d_full = selected_vertices[start_idx]
                    
                    for end_idx in end_candidates:
                        end_3d_full = selected_vertices[end_idx]
                        
                        # Skip self-loops (vertex connecting to itself)
                        if start_idx == end_idx:
                            continue
                        
                        # Check if this edge was already found
                        edge = tuple(sorted([start_idx, end_idx]))
                        if edge in valid_edges:
                            continue
                        
                        # Determine required views
                        same_x = abs(start_3d_full[0] - end_3d_full[0]) < 1e-6
                        same_y = abs(start_3d_full[1] - end_3d_full[1]) < 1e-6
                        same_z = abs(start_3d_full[2] - end_3d_full[2]) < 1e-6
                        
                        if same_x and same_y:
                            required_views = ['front', 'side']
                        elif same_x and same_z:
                            required_views = ['top', 'side']
                        elif same_y and same_z:
                            required_views = ['top', 'front']
                        else:
                            required_views = ['top', 'front', 'side']
                        
                        # Check existence in required views
                        views_valid = True
                        
                        if 'top' in required_views:
                            if not edge_exists_in_view(start_3d_full,
                                                       end_3d_full,
                                                       top_view_summary,
                                                       [0, 1], 'Top'):
                                views_valid = False
                        
                        if 'front' in required_views and views_valid:
                            if not edge_exists_in_view(start_3d_full,
                                                       end_3d_full,
                                                       front_view_summary,
                                                       [0, 2], 'Front'):
                                views_valid = False
                        
                        if 'side' in required_views and views_valid:
                            if not edge_exists_in_view(start_3d_full,
                                                       end_3d_full,
                                                       side_view_summary,
                                                       [1, 2], 'Side'):
                                views_valid = False
                        
                        if views_valid:
                            valid_edges.append(edge)
                            edges_found_front += 1
                            
                            if edges_found_front <= 10:
                                print(f"  [Front] New edge: V{start_idx} "
                                      f"-- V{end_idx}")
                                print(f"    Start: ({start_3d_full[0]:.3f}, "
                                      f"{start_3d_full[1]:.3f}, "
                                      f"{start_3d_full[2]:.3f})")
                                print(f"    End:   ({end_3d_full[0]:.3f}, "
                                      f"{end_3d_full[1]:.3f}, "
                                      f"{end_3d_full[2]:.3f})")
                                print(f"    Required views: {required_views}")
    
    print("\nPhase 2 Complete - Front View Processing:")
    print(f"  Edges checked from front view: {edges_checked_front}")
    print(f"  New edges found: {edges_found_front}")
    
    # ==================================================================
    # PHASE 3: Process Side View (final pass for complete coverage)
    # ==================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: Processing Side View")
    print(f"{'='*70}")
    
    edges_checked_side = 0
    edges_found_side = 0
    
    # Process side view connectivity matrix
    for i in range(len(side_view_summary)):
        for j in range(len(side_view_summary)):
            if i == j:
                continue
            
            # Check connectivity in side view (column 6+j)
            connectivity = side_view_summary[i, 6 + j]
            
            if connectivity >= 1:  # Edge exists in side view
                edges_checked_side += 1
                
                # Get 3D coordinates from side view summary
                start_3d = side_view_summary[i, :3]
                end_3d = side_view_summary[j, :3]
                start_y, start_z = start_3d[1], start_3d[2]
                end_y, end_z = end_3d[1], end_3d[2]
                
                # Find matching vertices in selected_vertices by y,z
                start_candidates = find_vertices_by_yz(start_y, start_z,
                                                       selected_vertices)
                end_candidates = find_vertices_by_yz(end_y, end_z,
                                                     selected_vertices)
                
                if not start_candidates or not end_candidates:
                    continue
                
                # Try each combination
                for start_idx in start_candidates:
                    start_3d_full = selected_vertices[start_idx]
                    
                    for end_idx in end_candidates:
                        end_3d_full = selected_vertices[end_idx]
                        
                        # Skip self-loops (vertex connecting to itself)
                        if start_idx == end_idx:
                            continue
                        
                        # Check if this edge was already found
                        edge = tuple(sorted([start_idx, end_idx]))
                        if edge in valid_edges:
                            continue
                        
                        # Determine required views
                        same_x = abs(start_3d_full[0] - end_3d_full[0]) < 1e-6
                        same_y = abs(start_3d_full[1] - end_3d_full[1]) < 1e-6
                        same_z = abs(start_3d_full[2] - end_3d_full[2]) < 1e-6
                        
                        if same_x and same_y:
                            required_views = ['front', 'side']
                        elif same_x and same_z:
                            required_views = ['top', 'side']
                        elif same_y and same_z:
                            required_views = ['top', 'front']
                        else:
                            required_views = ['top', 'front', 'side']
                        
                        # Check existence in required views
                        views_valid = True
                        
                        if 'top' in required_views:
                            if not edge_exists_in_view(start_3d_full,
                                                       end_3d_full,
                                                       top_view_summary,
                                                       [0, 1], 'Top'):
                                views_valid = False
                        
                        if 'front' in required_views and views_valid:
                            if not edge_exists_in_view(start_3d_full,
                                                       end_3d_full,
                                                       front_view_summary,
                                                       [0, 2], 'Front'):
                                views_valid = False
                        
                        if 'side' in required_views and views_valid:
                            if not edge_exists_in_view(start_3d_full,
                                                       end_3d_full,
                                                       side_view_summary,
                                                       [1, 2], 'Side'):
                                views_valid = False
                        
                        if views_valid:
                            valid_edges.append(edge)
                            edges_found_side += 1
                            
                            if edges_found_side <= 10:
                                print(f"  [Side] New edge: V{start_idx} "
                                      f"-- V{end_idx}")
                                print(f"    Start: ({start_3d_full[0]:.3f}, "
                                      f"{start_3d_full[1]:.3f}, "
                                      f"{start_3d_full[2]:.3f})")
                                print(f"    End:   ({end_3d_full[0]:.3f}, "
                                      f"{end_3d_full[1]:.3f}, "
                                      f"{end_3d_full[2]:.3f})")
                                print(f"    Required views: {required_views}")
    
    print("\nPhase 3 Complete - Side View Processing:")
    print(f"  Edges checked from side view: {edges_checked_side}")
    print(f"  New edges found: {edges_found_side}")
    
    print(f"\n{'='*70}")
    print("EDGE RECONSTRUCTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Phase 1 (Top View): {edges_validated} edges")
    print(f"  Phase 2 (Front View): {edges_found_front} new edges")
    print(f"  Phase 3 (Side View): {edges_found_side} new edges")
    print(f"  Total valid edges reconstructed: {len(valid_edges)}")
    if expected_unique_edges is not None:
        if len(valid_edges) == expected_unique_edges:
            print(f"  SUCCESS: Found all {expected_unique_edges} "
                  "expected unique edges!")
        else:
            diff = len(valid_edges) - expected_unique_edges
            if diff > 0:
                print(f"  WARNING: Found {diff} extra edge(s) - "
                      f"possible duplicates")
            else:
                print(f"  INCOMPLETE: Missing {-diff} edge(s) "
                      f"(expected {expected_unique_edges})")
    print("="*70)
    
    return valid_edges
