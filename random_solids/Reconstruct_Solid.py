#!/usr/bin/env python3
"""
Reconstruct_Solid.py - Reverse Engineering and Solid Reconstruction

This module handles:
1. Loading connectivity matrices and face polygons from saved files
2. Filtering candidate vertices using orthogonal view projections
3. Building merged connectivity matrix from three views
4. Extracting polygon faces from the connectivity matrix
5. Visualizing the reconstructed solid

Input files (named with seed value):
- connectivity_matrices_seed_XXXXX.npz: View connectivity matrices
- solid_faces_seed_XXXXX.npy: Original face polygons (for comparison)
"""

import os
import numpy as np
from shapely.geometry import Polygon
import argparse
import matplotlib.pyplot as plt

# Import necessary functions from V6_current
from V6_current import (
    visualize_3d_solid,
    extract_polygon_faces_from_connectivity,
    plot_extracted_polygon_faces,
    extract_wire_vertices_in_sequence
)


def load_connectivity_matrices(seed, input_dir="Output"):
    """Load connectivity matrices from file"""
    filename = os.path.join(input_dir, f"connectivity_matrices_seed_{seed}.npz")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Connectivity matrix file not found: {filename}\n"
            f"Please run Build_Solid.py first with --seed {seed}"
        )
    
    print(f"\n[LOAD] Loading connectivity matrices from: {filename}")
    data = np.load(filename, allow_pickle=True)
    
    all_vertices = data['all_vertices']
    top_view_matrix = data['top_view_matrix']
    front_view_matrix = data['front_view_matrix']
    side_view_matrix = data['side_view_matrix']
    
    print(f"       - Loaded {len(all_vertices)} vertices")
    print(f"       - Top view matrix: {top_view_matrix.shape}")
    print(f"       - Front view matrix: {front_view_matrix.shape}")
    print(f"       - Side view matrix: {side_view_matrix.shape}")
    
    return all_vertices, top_view_matrix, front_view_matrix, side_view_matrix


def load_face_polygons(seed, input_dir="Output"):
    """Load face polygons from file"""
    filename = os.path.join(input_dir, f"solid_faces_seed_{seed}.npy")
    
    if not os.path.exists(filename):
        print(f"[LOAD] Warning: Face polygon file not found: {filename}")
        return None
    
    print(f"[LOAD] Loading face polygons from: {filename}")
    face_data = np.load(filename, allow_pickle=True)
    print(f"       - Loaded {len(face_data)} faces")
    
    return face_data


def project_vertex_to_view(vertex, normal):
    """Project a 3D vertex to 2D view using coordinate dropping"""
    vertex = np.array(vertex)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    
    # Use coordinate dropping for standard orthogonal views
    if np.allclose(normal, [0, 0, 1], atol=1e-3):  # Top view
        return vertex[0], vertex[1]  # Drop Z, keep X,Y
    elif np.allclose(normal, [0, -1, 0], atol=1e-3):  # Front view
        return vertex[0], vertex[2]  # Drop Y, keep X,Z
    elif np.allclose(normal, [1, 0, 0], atol=1e-3):  # Side view
        return vertex[1], vertex[2]  # Drop X, keep Y,Z
    else:
        # For non-orthogonal views, use basis vector method
        if abs(normal[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        
        u = temp - np.dot(temp, normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        proj_u = np.dot(vertex, u)
        proj_v = np.dot(vertex, v)
        return proj_u, proj_v


def find_matching_row(proj_2d, view_matrix, tolerance=1e-5):
    """Find row in view matrix matching the 2D projection"""
    for i in range(view_matrix.shape[0]):
        matrix_proj = (view_matrix[i, 1], view_matrix[i, 2])
        if np.allclose(proj_2d, matrix_proj, atol=tolerance):
            return i
    return None


def filter_candidate_vertices(top_matrix, front_matrix, side_matrix):
    """
    Filter candidate vertices using reverse engineering approach.
    
    Step 1: Extract (x,y) from top view, z-levels from front view
    Step 2: Generate all combinations as candidates
    Step 3: Filter by checking projections against front and side views
    """
    print("\n" + "="*70)
    print("REVERSE ENGINEERING - VERTEX RECONSTRUCTION")
    print("="*70)
    
    # Step 1: Extract coordinates from views
    print("\nStep 1: Extracting coordinates from connectivity matrices...")
    # Check matrices are valid
    if (
        top_matrix is None
        or not hasattr(top_matrix, 'shape')
        or not isinstance(top_matrix, np.ndarray)
        or len(top_matrix.shape) == 0
        or top_matrix.shape[0] == 0
    ):
        print(f"[ERROR] Top view matrix is missing, empty, or malformed. Type: {type(top_matrix)}, shape: {getattr(top_matrix, 'shape', None)}")
        print(f"[ERROR] Value: {top_matrix}")
        return []
    if front_matrix is None or not hasattr(front_matrix, 'shape') or front_matrix.shape[0] == 0:
        print("[ERROR] Front view matrix is missing or empty. Cannot extract z-levels.")
        return []
    # Extract unique (x,y) from top view
    top_xy_coords = set()
    for i in range(top_matrix.shape[0]):
        x_proj, y_proj = top_matrix[i, 1], top_matrix[i, 2]
        top_xy_coords.add((x_proj, y_proj))
    # Extract unique z-levels from front view
    raw_z = [round(front_matrix[i, 2], 6) for i in range(front_matrix.shape[0])]
    z_levels = sorted(set(raw_z))
    print(f"  - Extracted (x,y) from top view: {len(top_xy_coords)} coordinates")
    print(f"  - Extracted z-levels from front view: {len(z_levels)} levels")
    
    # Step 2: Generate candidate vertices
    print("\nStep 2: Generating candidate vertices...")
    candidate_vertices = []
    for x, y in top_xy_coords:
        for z in z_levels:
            candidate_vertices.append([x, y, z])
    
    candidate_vertices = np.array(candidate_vertices)
    print(f"  - Total candidates: {len(candidate_vertices)}")
    print(f"  - Formula: {len(top_xy_coords)} × {len(z_levels)} = "
          f"{len(candidate_vertices)}")
    
    # Step 3: Filter by projection matching
    print("\nStep 3: Filtering candidates by projection matching...")
    print("  - Checking front and side view projections...")
    
    # Build sets of valid projections for each view
    front_projections = set()
    for i in range(front_matrix.shape[0]):
        proj = (round(front_matrix[i, 1], 6), round(front_matrix[i, 2], 6))
        front_projections.add(proj)
    
    side_projections = set()
    for i in range(side_matrix.shape[0]):
        proj = (round(side_matrix[i, 1], 6), round(side_matrix[i, 2], 6))
        side_projections.add(proj)
    
    # Filter candidates
    selected_vertices = []
    for vertex in candidate_vertices:
        # Check front view projection (x, z)
        front_proj = (round(vertex[0], 6), round(vertex[2], 6))
        # Check side view projection (y, z)
        side_proj = (round(vertex[1], 6), round(vertex[2], 6))
        
        if front_proj in front_projections and side_proj in side_projections:
            selected_vertices.append(vertex)
    
    selected_vertices = np.array(selected_vertices)
    print(f"\nStep 3 Complete: Filtered to {len(selected_vertices)} vertices")
    print(f"  - Reduction: {len(candidate_vertices)} → {len(selected_vertices)}")
    print(f"  - Filtered out: {len(candidate_vertices) - len(selected_vertices)} "
          "fake vertices")
    
    return selected_vertices


def build_square_connectivity_matrices(
        selected_vertices, top_matrix, front_matrix, side_matrix):
    """Build square connectivity matrices for the selected vertices"""
    print("\n" + "="*70)
    print("BUILDING CONNECTIVITY MATRICES")
    print("="*70)
    
    N = len(selected_vertices)
    top_conn = np.zeros((N, N), dtype=int)
    front_conn = np.zeros((N, N), dtype=int)
    side_conn = np.zeros((N, N), dtype=int)
    
    # Project all vertices to each view
    top_proj = np.zeros((N, 2))
    front_proj = np.zeros((N, 2))
    side_proj = np.zeros((N, 2))
    
    for idx, vertex in enumerate(selected_vertices):
        top_proj[idx] = project_vertex_to_view(vertex, [0, 0, 1])
        front_proj[idx] = project_vertex_to_view(vertex, [0, -1, 0])
        side_proj[idx] = project_vertex_to_view(vertex, [1, 0, 0])
    
    print(f"Projecting {N} vertices to each view...")
    
    # Build connectivity matrices
    for i in range(N):
        tp_i = top_proj[i]
        fp_i = front_proj[i]
        sp_i = side_proj[i]
        
        top_idx_i = find_matching_row(tp_i, top_matrix)
        front_idx_i = find_matching_row(fp_i, front_matrix)
        side_idx_i = find_matching_row(sp_i, side_matrix)
        
        for j in range(N):
            tp_j = top_proj[j]
            fp_j = front_proj[j]
            sp_j = side_proj[j]
            
            top_idx_j = find_matching_row(tp_j, top_matrix)
            front_idx_j = find_matching_row(fp_j, front_matrix)
            side_idx_j = find_matching_row(sp_j, side_matrix)
            
            # Top view connectivity
            if top_idx_i is not None and top_idx_j is not None:
                top_conn[i, j] = top_matrix[top_idx_i, 3 + top_idx_j]
            
            # Front view connectivity
            if front_idx_i is not None and front_idx_j is not None:
                front_conn[i, j] = front_matrix[front_idx_i, 3 + front_idx_j]
            
            # Side view connectivity
            if side_idx_i is not None and side_idx_j is not None:
                side_conn[i, j] = side_matrix[side_idx_i, 3 + side_idx_j]
            
            # Add connectivity for vertices at same 2D position in one view
            # if connected in other two views
            if (np.allclose(tp_i, tp_j, atol=1e-6) and 
                    front_conn[i, j] == 1 and side_conn[i, j] == 1):
                top_conn[i, j] = 1
            if (np.allclose(fp_i, fp_j, atol=1e-6) and 
                    top_conn[i, j] == 1 and side_conn[i, j] == 1):
                front_conn[i, j] = 1
            if (np.allclose(sp_i, sp_j, atol=1e-6) and 
                    front_conn[i, j] == 1 and top_conn[i, j] == 1):
                side_conn[i, j] = 1
    
    print(f"Built connectivity matrices:")
    print(f"  - Top view: {np.sum(top_conn > 0) // 2} edges")
    print(f"  - Front view: {np.sum(front_conn > 0) // 2} edges")
    print(f"  - Side view: {np.sum(side_conn > 0) // 2} edges")
    
    return top_conn, front_conn, side_conn, top_proj, front_proj, side_proj


def main():
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from Build_Solid import make_face_with_holes
    parser = argparse.ArgumentParser(
        description='Reconstruct 3D solid from engineering view connectivity matrices.'
    )
    parser.add_argument(
        '--no-occ-viewer', action='store_true',
        help='Skip OCC viewer launch for debugging (prints diagnostics only)'
    )
    parser.add_argument(
        '--seed', type=int, required=True,
        help='Random seed matching the Build_Solid.py run'
    )
    parser.add_argument(
        '--input-dir', type=str, default='Output',
        help='Directory containing saved connectivity matrices'
    )
    parser.add_argument(
        '--tolerance', type=float, default=1e-6,
        help='Tolerance for face coplanarity detection'
    )
    parser.add_argument(
        '--save-plot', action='store_true',
        help='Save polygon validity plot to file instead of showing interactively'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SOLID RECONSTRUCTION FROM ENGINEERING VIEWS")
    print("="*70)
    print(f"Seed: {args.seed}")
    
    # Load saved data
    print("\n[STEP 1] Loading saved connectivity matrices...")
    try:
        all_vertices, top_matrix, front_matrix, side_matrix = \
            load_connectivity_matrices(args.seed, args.input_dir)
        face_polygons = load_face_polygons(args.seed, args.input_dir)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return
    
    # Filter candidate vertices
    print("\n[STEP 2] Filtering candidate vertices...")
    selected_vertices = filter_candidate_vertices(
        top_matrix, front_matrix, side_matrix
    )
    print("\n[SELECTED VERTICES]")
    for idx, v in enumerate(selected_vertices):
        print(f"  {idx}: {v}")
    if len(selected_vertices) == 0:
        print("[ERROR] No vertices passed filtering!")
        return
    
    # Build square connectivity matrices
    print("\n[STEP 3] Building square connectivity matrices...")
    top_conn, front_conn, side_conn, top_proj, front_proj, side_proj = \
        build_square_connectivity_matrices(
            selected_vertices, top_matrix, front_matrix, side_matrix
        )
    
    # Build merged connectivity matrix
    print("\n[STEP 4] Building merged connectivity matrix...")
    merged_conn = top_conn + front_conn + side_conn
    
    total_edges = np.sum(merged_conn > 0) // 2
    edges_conn3 = np.sum(merged_conn == 3) // 2
    edges_conn2 = np.sum(merged_conn == 2) // 2
    edges_conn1 = np.sum(merged_conn == 1) // 2
    
    print(f"Merged connectivity matrix summary:")
    print(f"  - Total potential edges: {total_edges}")
    print(f"  - Edges with conn=3 (all views): {edges_conn3}")
    print(f"  - Edges with conn=2 (two views): {edges_conn2}")
    print(f"  - Edges with conn=1 (one view): {edges_conn1}")
    
    # Visualize reconstructed edges
    print("\n[STEP 5] Visualizing reconstructed solid with edges (SKIPPED - see Step 7 for polygon plot)...")
    # edges = []
    # for i in range(len(selected_vertices)):
    #     for j in range(i+1, len(selected_vertices)):
    #         if merged_conn[i, j] > 0:
    #             edges.append((i, j))
    # 
    # visualize_3d_solid(
    #     face_polygons if face_polygons is not None else [],
    #     selected_vertices=selected_vertices,
    #     edges=edges
    # )
    
    # Extract polygon faces from connectivity
    extracted_faces = extract_polygon_faces_from_connectivity(selected_vertices, merged_conn, tolerance=args.tolerance)
    print(f"\nExtracted {len(extracted_faces)} faces")
    
    # Validate that all polygons form closed cycles
    print("\n[STEP 6] Validating extracted polygons...")
    invalid_faces = []
    for face_idx, face in enumerate(extracted_faces):
        if not (isinstance(face, dict) and 'vertices' in face):
            continue
        
        face_num = face_idx + 1
        vertices = face['vertices']
        
        # Check if polygon forms a closed cycle
        if len(vertices) < 3:
            print(f"   WARNING: Face {face_num} has only {len(vertices)} "
                  f"vertices (need at least 3)")
            invalid_faces.append(face_num)
            continue
        
        # Check if vertices form a connected cycle
        # Build edge list from consecutive vertices
        edges_in_polygon = []
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            edges_in_polygon.append((v1, v2))
        
        # Check if each vertex appears exactly twice (once as start, once as end)
        vertex_count = {}
        for v1, v2 in edges_in_polygon:
            vertex_count[v1] = vertex_count.get(v1, 0) + 1
            vertex_count[v2] = vertex_count.get(v2, 0) + 1
        
        # In a closed cycle, each vertex should appear exactly twice
        non_cycle_vertices = [v for v, count in vertex_count.items() 
                             if count != 2]
        if len(non_cycle_vertices) > 0:
            print(f"   WARNING: Face {face_num} vertices don't form "
                  f"closed cycle")
            print(f"            Non-cycle vertices: {non_cycle_vertices}")
            invalid_faces.append(face_num)
        
        # Validate holes if present
        holes = face.get('holes', [])
        for hole_idx, hole in enumerate(holes):
            if len(hole) < 3:
                print(f"   WARNING: Face {face_num} hole {hole_idx + 1} "
                      f"has only {len(hole)} vertices")
                invalid_faces.append(face_num)
            else:
                # Check hole forms closed cycle
                hole_edges = []
                for i in range(len(hole)):
                    v1 = hole[i]
                    v2 = hole[(i + 1) % len(hole)]
                    hole_edges.append((v1, v2))
                
                hole_vertex_count = {}
                for v1, v2 in hole_edges:
                    hole_vertex_count[v1] = hole_vertex_count.get(v1, 0) + 1
                    hole_vertex_count[v2] = hole_vertex_count.get(v2, 0) + 1
                
                non_cycle_hole_verts = [v for v, count in 
                                       hole_vertex_count.items() if count != 2]
                if len(non_cycle_hole_verts) > 0:
                    print(f"   WARNING: Face {face_num} hole {hole_idx + 1} "
                          f"doesn't form closed cycle")
    
    if len(invalid_faces) > 0:
        unique_invalid = list(set(invalid_faces))
        print(f"   Found {len(unique_invalid)} face(s) with issues: "
              f"{sorted(unique_invalid)}")
    else:
        print(f"   All {len(extracted_faces)} faces validated successfully")
    
    # =========================================================================
    # STEP 6.5: Determine correct face orientations using ray-casting
    # =========================================================================
    print("\n[STEP 6.5] Determining face orientations (CCW relative to outward normal)...")
    
    # Find overall bounding box
    all_coords = np.array(selected_vertices)
    bbox_min = np.min(all_coords, axis=0)
    bbox_max = np.max(all_coords, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    
    print(f"   Bounding box: "
          f"X=[{bbox_min[0]:.2f}, {bbox_max[0]:.2f}], "
          f"Y=[{bbox_min[1]:.2f}, {bbox_max[1]:.2f}], "
          f"Z=[{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]")
    print(f"   Center: ({bbox_center[0]:.2f}, {bbox_center[1]:.2f}, "
          f"{bbox_center[2]:.2f})")
    
    faces_corrected = 0
    faces_ok = 0
    
    for face_idx, face in enumerate(extracted_faces):
        if not (isinstance(face, dict) and 'vertices' in face):
            continue
        
        face_num = face_idx + 1
        normal = np.array(face['normal'])
        
        # Get face vertices
        verts_indices = face['vertices']
        verts_3d = np.array([selected_vertices[v] for v in verts_indices])
        face_center = np.mean(verts_3d, axis=0)
        
        # Ray from bbox center to face center
        ray_dir = face_center - bbox_center
        ray_len = np.linalg.norm(ray_dir)
        if ray_len > 1e-6:
            ray_dir = ray_dir / ray_len
        else:
            # Face center coincides with bbox center - skip
            print(f"   WARNING: Face {face_num} center coincides with bbox center")
            continue
        
        # Check if normal points away from bbox center (outward)
        # Dot product > 0 means normal points in same direction as ray (outward)
        dot_product = np.dot(normal, ray_dir)
        
        # Compute signed area to determine winding order
        # Project face to 2D by dropping axis with largest normal component
        abs_normal = np.abs(normal)
        drop_axis = np.argmax(abs_normal)
        keep_axes = [i for i in range(3) if i != drop_axis]
        verts_2d = verts_3d[:, keep_axes]
        
        # Shoelace formula for signed area
        x = verts_2d[:, 0]
        y = verts_2d[:, 1]
        signed_area = 0.5 * (np.dot(x, np.roll(y, 1)) -
                            np.dot(y, np.roll(x, 1)))
        
        # Determine if vertices need to be reversed
        # We want: outward normal (dot > 0) + CCW winding (area > 0)
        # If normal points outward but area is negative (CW), reverse
        # If normal points inward but area is positive (CCW), reverse
        needs_reversal = (dot_product > 0) != (signed_area > 0)
        
        # Store orientation information
        face['is_outward_facing'] = dot_product > 0
        face['signed_area_2d'] = signed_area
        face['dot_product'] = dot_product
        
        # Apply reversal if needed
        if needs_reversal:
            face['vertices'] = list(reversed(face['vertices']))
            # Reverse the normal as well
            face['normal'] = tuple(-n for n in normal)
            faces_corrected += 1
            print(f"   Face {face_num}: CORRECTED (reversed vertices & normal) "
                  f"- dot={dot_product:.3f}, area={signed_area:.1f}")
        else:
            faces_ok += 1
    
    print(f"\n   Orientation summary: {faces_ok} faces correct, "
          f"{faces_corrected} faces corrected")
    
    # STEP 6: Display the first plot showing extracted faces with class colors
    print("\n[STEP 6] Displaying extracted polygon faces (original solid visualization)...")
    plot_extracted_polygon_faces(extracted_faces, selected_vertices, face_polygons)
    print("[STEP 6] First plot closed. Proceeding to Step 7...")
    
    # =========================================================================
    # STEP 7: Build topological edge-face data structure
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 7] BUILDING TOPOLOGICAL EDGE-FACE DATA STRUCTURE")
    print("="*70)
    
    # Step 7.1: Build vertex index array (already have selected_vertices)
    print(f"\n[STEP 7.1] Vertex array: {len(selected_vertices)} vertices")
    
    # Step 7.2: Build edge index array with face_1 and face_2 columns
    print(f"\n[STEP 7.2] Building edge index array with face columns...")
    print("   Collecting edges from extracted faces (not from connectivity matrix)...")
    
    # Collect all unique edges from the extracted faces
    edge_dict = {}  # key: (v1, v2) sorted tuple, value: edge index
    edge_list = []  # list of [v1_idx, v2_idx, face_1_idx, face_2_idx]
    
    # Scan through all extracted faces to find edges
    for face_idx, face in enumerate(extracted_faces):
        if not (isinstance(face, dict) and 'vertices' in face):
            continue
        
        # Get vertices for outer boundary
        polygon_vertices = face['vertices']
        
        # Add edges from outer boundary
        for i in range(len(polygon_vertices)):
            v1_idx = polygon_vertices[i]
            v2_idx = polygon_vertices[(i + 1) % len(polygon_vertices)]
            edge_tuple = tuple(sorted([v1_idx, v2_idx]))
            
            if edge_tuple not in edge_dict:
                edge_idx = len(edge_list)
                edge_dict[edge_tuple] = edge_idx
                edge_list.append([v1_idx, v2_idx, -1, -1])
        
        # Add edges from holes (if any)
        holes = face.get('holes', [])
        for hole in holes:
            for i in range(len(hole)):
                v1_idx = hole[i]
                v2_idx = hole[(i + 1) % len(hole)]
                edge_tuple = tuple(sorted([v1_idx, v2_idx]))
                
                if edge_tuple not in edge_dict:
                    edge_idx = len(edge_list)
                    edge_dict[edge_tuple] = edge_idx
                    edge_list.append([v1_idx, v2_idx, -1, -1])
    
    edge_array = np.array(edge_list, dtype=int)
    print(f"   Created edge array: {len(edge_array)} edges (from faces only)")
    print(f"   Columns: [vertex_1, vertex_2, face_1, face_2]")
    print(f"   Sample edges (first 5):")
    for i in range(min(5, len(edge_array))):
        print(f"     Edge {i}: v{edge_array[i,0]} <-> v{edge_array[i,1]}, faces: [{edge_array[i,2]}, {edge_array[i,3]}]")

    
    # Step 7.3: Build face-wire-edge data structure
    print(f"\n[STEP 7.3] Building face-wire-edge data structure...")
    
    # Create face array: each face has a list of wires (list of edge indices)
    face_array = []
    
    for face_idx, face in enumerate(extracted_faces):
        if not (isinstance(face, dict) and 'vertices' in face):
            continue
        
        # Use 1-based face index to match POLY FORM output
        face_number = face_idx + 1
        
        polygon_vertices = face['vertices']  # Outer boundary vertices
        holes = face.get('holes', [])  # Inner holes (list of vertex lists)
        
        # Build outer wire from polygon vertices
        outer_wire_edges = []
        for i in range(len(polygon_vertices)):
            v1_idx = polygon_vertices[i]
            v2_idx = polygon_vertices[(i + 1) % len(polygon_vertices)]
            edge_tuple = tuple(sorted([v1_idx, v2_idx]))
            
            if edge_tuple in edge_dict:
                edge_idx = edge_dict[edge_tuple]
                outer_wire_edges.append(edge_idx)
            else:
                print(f"   WARNING: Edge {edge_tuple} not found in edge_dict for face {face_number}")
        
        # Build inner wires (holes)
        hole_wires = []
        for hole_idx, hole_vertices in enumerate(holes):
            hole_wire_edges = []
            for i in range(len(hole_vertices)):
                v1_idx = hole_vertices[i]
                v2_idx = hole_vertices[(i + 1) % len(hole_vertices)]
                edge_tuple = tuple(sorted([v1_idx, v2_idx]))
                
                if edge_tuple in edge_dict:
                    edge_idx = edge_dict[edge_tuple]
                    hole_wire_edges.append(edge_idx)
                else:
                    print(f"   WARNING: Edge {edge_tuple} not found in edge_dict for face {face_number} hole {hole_idx + 1}")
            
            if len(hole_wire_edges) > 0:
                hole_wires.append(hole_wire_edges)
        
        # Combine outer wire and hole wires
        all_wires = [outer_wire_edges] + hole_wires
        
        face_array.append({
            'face_idx': face_number,  # 1-based index
            'wires': all_wires,  # First wire is outer, rest are holes
            'polygon': polygon_vertices,
            'holes': holes,
            'normal': face.get('normal', None)
        })
    
    print(f"   Created face array: {len(face_array)} faces")
    print(f"   Sample faces (first 3):")
    for i in range(min(3, len(face_array))):
        face_data = face_array[i]
        num_holes = len(face_data['wires']) - 1
        print(f"     Face {face_data['face_idx']}: {len(face_data['wires'][0])} edges in outer wire, {num_holes} hole(s)")
        print(f"       Outer boundary vertices: {face_data['polygon']}")
        print(f"       Outer wire edges: {face_data['wires'][0]}")
        for hole_idx in range(num_holes):
            print(f"       Hole {hole_idx + 1} vertices: {face_data['holes'][hole_idx]}")
            print(f"       Hole {hole_idx + 1} wire edges: {face_data['wires'][hole_idx + 1]}")
            
            # Show detailed edge-vertex mapping for this hole
            hole_wire = face_data['wires'][hole_idx + 1]
            hole_verts = face_data['holes'][hole_idx]
            print(f"       Hole {hole_idx + 1} wire construction:")
            for j, edge_idx in enumerate(hole_wire):
                v1, v2 = edge_array[edge_idx, 0], edge_array[edge_idx, 1]
                expected_v1 = hole_verts[j]
                expected_v2 = hole_verts[(j + 1) % len(hole_verts)]
                p1 = selected_vertices[v1]
                p2 = selected_vertices[v2]
                print(f"         Edge {edge_idx}: v{v1}→v{v2} (expected v{expected_v1}→v{expected_v2}), coords: ({p1[0]:.0f},{p1[1]:.0f},{p1[2]:.0f}) → ({p2[0]:.0f},{p2[1]:.0f},{p2[2]:.0f})")
    
    # Step 7.4: Fill face_1 and face_2 columns in edge array
    print(f"\n[STEP 7.4] Filling face_1 and face_2 columns in edge array...")
    print("   Processing both outer wire edges AND hole edges.")
    print("   Hole edges also separate faces (the face containing the hole from adjacent faces).")
    
    exception_count = 0
    same_face_error_count = 0
    
    for face in face_array:
        face_idx = face['face_idx']
        # Process ALL wires (outer wire AND hole wires)
        for wire_idx, wire in enumerate(face['wires']):
            wire_type = "outer" if wire_idx == 0 else f"hole {wire_idx}"
            for edge_idx in wire:
                # Each edge is associated with two faces
                # Fill the first empty slot (face_1 or face_2)
                if edge_array[edge_idx, 2] == -1:
                    edge_array[edge_idx, 2] = face_idx
                elif edge_array[edge_idx, 3] == -1:
                    # Check that we're not assigning the same face twice
                    if edge_array[edge_idx, 2] == face_idx:
                        same_face_error_count += 1
                        v1, v2 = edge_array[edge_idx, 0], edge_array[edge_idx, 1]
                        print(f"   ERROR: Edge {edge_idx} (v{v1}<->v{v2}) - trying to assign same face {face_idx} to both face_1 and face_2")
                        print(f"     This indicates the face has a self-touching boundary")
                    else:
                        edge_array[edge_idx, 3] = face_idx
                else:
                    # Both slots are filled - this is an over-constrained edge
                    exception_count += 1
                    v1, v2 = edge_array[edge_idx, 0], edge_array[edge_idx, 1]
                    existing_faces = [edge_array[edge_idx, 2], edge_array[edge_idx, 3]]
                    print(f"   EXCEPTION: Edge {edge_idx} (v{v1}<->v{v2}) already has two faces {existing_faces}, cannot add face {face_idx}")
                    print(f"     Vertices: {selected_vertices[v1]} <-> {selected_vertices[v2]}")
    
    if same_face_error_count > 0:
        print(f"   ERROR: {same_face_error_count} edge(s) with same face on both sides (self-touching)")
    
    if exception_count > 0:
        print(f"   WARNING: {exception_count} over-constrained edge(s) detected")
    else:
        print(f"   All edges successfully associated with faces")
    
    print(f"\n   Edge array after face association (first 10 edges):")
    for i in range(min(10, len(edge_array))):
        v1, v2, f1, f2 = edge_array[i]
        print(f"     Edge {i}: v{v1} <-> v{v2}, faces: [f{f1}, f{f2}]")
    
    # Validate edge-face associations
    # Check: face_1 == face_2 (and not both -1), OR face_1 != -1, OR face_2 != -1
    print(f"\n   Validating edge-face associations...")
    invalid_edges = []
    for edge_idx in range(len(edge_array)):
        f1, f2 = edge_array[edge_idx, 2], edge_array[edge_idx, 3]
        
        # Check if face_1 == face_2 (but not both -1)
        if f1 != -1 and f1 == f2:
            invalid_edges.append((edge_idx, 'same_face', f1, f2))
        
        # Note: We expect some edges to have only one face (boundary edges)
        # and some to have two faces (interior edges)
        # Both are valid, so we don't flag f1 != -1 or f2 != -1 as errors
    
    if len(invalid_edges) > 0:
        print(f"   ERROR: {len(invalid_edges)} edge(s) with invalid face assignments:")
        for edge_info in invalid_edges[:5]:  # Show first 5
            edge_idx, error_type, f1, f2 = edge_info
            v1, v2 = edge_array[edge_idx, 0], edge_array[edge_idx, 1]
            if error_type == 'same_face':
                print(f"     Edge {edge_idx}: v{v1}<->v{v2}, face_1 == face_2 = f{f1}")
    else:
        print(f"   All edges validated: no invalid face assignments")
    
    # Additional statistics
    edges_with_two_faces = sum(1 for e in edge_array if e[2] != -1 and e[3] != -1)
    edges_with_one_face = sum(1 for e in edge_array if (e[2] != -1 and e[3] == -1) or (e[2] == -1 and e[3] != -1))
    edges_with_no_faces = sum(1 for e in edge_array if e[2] == -1 and e[3] == -1)
    
    print(f"   Edge statistics:")
    print(f"     - Edges with 2 faces (interior): {edges_with_two_faces}")
    print(f"     - Edges with 1 face (boundary): {edges_with_one_face}")
    print(f"     - Edges with 0 faces (unused): {edges_with_no_faces}")
    
    # Step 7.4.5: Remove invalid faces/holes based on edge validation
    print(f"\n[STEP 7.4.5] Checking for invalid faces with self-touching boundaries...")
    print("   Rule: An edge cannot have the same face on both sides (self-touching)")
    print("   Note: Edges with only one face are VALID (they are boundary edges on the solid's exterior)")
    
    # Find edges where same face is on both sides (self-touching)
    invalid_edge_set = set()
    for edge_idx in range(len(edge_array)):
        f1, f2 = edge_array[edge_idx, 2], edge_array[edge_idx, 3]
        # Edge is invalid if same face appears on both sides
        if f1 != -1 and f1 == f2:
            invalid_edge_set.add(edge_idx)
    
    print(f"   Found {len(invalid_edge_set)} edge(s) with self-touching (same face on both sides)")
    
    # Show details of boundary edges (informational only, not removed)
    boundary_edges = []
    for edge_idx in range(len(edge_array)):
        f1, f2 = edge_array[edge_idx, 2], edge_array[edge_idx, 3]
        if (f1 == -1 and f2 != -1) or (f1 != -1 and f2 == -1):
            boundary_edges.append(edge_idx)
    
    print(f"   Found {len(boundary_edges)} boundary edge(s) with only one face (these are VALID)")
    
    # Show details of invalid edges (self-touching)
    if len(invalid_edge_set) > 0:
        print(f"\n   Details of invalid edges (self-touching):")
        for edge_idx in sorted(invalid_edge_set):
            v1, v2, f1, f2 = edge_array[edge_idx]
            p1 = selected_vertices[v1]
            p2 = selected_vertices[v2]
            print(f"      Edge {edge_idx}: v{v1}-v{v2} (({p1[0]:.2f},{p1[1]:.2f},{p1[2]:.2f}) to ({p2[0]:.2f},{p2[1]:.2f},{p2[2]:.2f})), face_1={f1}, face_2={f2} (SELF-TOUCHING)")
    
    # Process each face to check for invalid edges
    faces_to_remove = []
    modified_faces = []
    
    for face_idx, face in enumerate(face_array):
        face_num = face['face_idx']
        all_wires = face['wires']
        
        # Check all wires for invalid edges
        wire_validity = []  # List of (wire_idx, is_valid, invalid_edges)
        for wire_idx, wire in enumerate(all_wires):
            invalid_edges_in_wire = [e for e in wire if e in invalid_edge_set]
            is_valid = len(invalid_edges_in_wire) == 0
            wire_type = "outer" if wire_idx == 0 else f"hole {wire_idx}"
            wire_validity.append({
                'wire_idx': wire_idx,
                'wire': wire,
                'is_valid': is_valid,
                'invalid_edges': invalid_edges_in_wire,
                'wire_type': wire_type
            })
            
            if not is_valid:
                print(f"\n   Face {face_num} - {wire_type.capitalize()} wire has {len(invalid_edges_in_wire)} invalid edge(s): {invalid_edges_in_wire}")
        
        # Check if outer wire (wire 0) is valid
        outer_wire_valid = wire_validity[0]['is_valid']
        
        if not outer_wire_valid:
            # Outer wire is invalid, try to promote a valid hole to outer boundary
            print(f"   Face {face_num}: Outer boundary is invalid, searching for valid wire to promote...")
            
            # Count total wires and invalid wires
            total_wires = len(wire_validity)
            invalid_wires = [wv for wv in wire_validity if not wv['is_valid']]
            valid_wires = [wv for wv in wire_validity[1:] if wv['is_valid']]  # Exclude invalid outer
            
            print(f"   Face {face_num}: Total wires = {total_wires}, Invalid wires = {len(invalid_wires)}, Valid holes = {len(valid_wires)}")
            print(f"   Face {face_num}: Deleting {len(invalid_wires)} invalid wire(s): {[wv['wire_type'] for wv in invalid_wires]}")
            
            # Additional validation: Check if valid hole wires form simple polygons suitable for promotion
            # A hole polygon that is not a simple outer boundary should not be promoted
            promotable_wires = []
            for wv in valid_wires:
                hole_poly_idx = wv['wire_idx'] - 1  # Convert wire index to hole index (0-based)
                if hole_poly_idx < len(face['holes']):
                    hole_vertices = face['holes'][hole_poly_idx]
                    hole_verts_3d = [selected_vertices[v] for v in hole_vertices]
                    
                    # Check if polygon is simple (no self-intersections)
                    # Use Shapely to validate
                    try:
                        from shapely.geometry import Polygon
                        # Project to 2D for validation
                        hole_2d = [(v[0], v[1]) for v in hole_verts_3d]
                        poly = Polygon(hole_2d)
                        
                        # Check if polygon is valid and simple
                        if poly.is_valid and poly.is_simple and not poly.is_empty:
                            promotable_wires.append(wv)
                            print(f"   Face {face_num}: {wv['wire_type']} is a valid simple polygon (promotable)")
                        else:
                            print(f"   Face {face_num}: {wv['wire_type']} is NOT a simple polygon (not promotable): is_valid={poly.is_valid}, is_simple={poly.is_simple}")
                    except Exception as e:
                        print(f"   Face {face_num}: Error validating {wv['wire_type']}: {e}")
            
            valid_wires = promotable_wires  # Only use promotable wires
            
            if len(valid_wires) == 0:
                # No valid wires found - remove the entire face
                faces_to_remove.append(face_idx)
                print(f"   Face {face_num}: EXCEPTION - REMOVING face (no valid wires found)")
            else:
                # Found at least one valid wire - promote the first one as outer boundary
                # Use only the remaining VALID wires as holes (invalid wires are discarded)
                promoted_wire = valid_wires[0]
                remaining_valid_wires = valid_wires[1:]
                
                print(f"   Face {face_num}: Promoting {promoted_wire['wire_type']} to outer boundary")
                print(f"   Face {face_num}: New structure: 1 outer wire + {len(remaining_valid_wires)} hole(s)")
                
                # Build new polygon from promoted wire
                new_outer_wire = promoted_wire['wire']
                new_polygon = []
                for edge_idx in new_outer_wire:
                    v_idx = edge_array[edge_idx, 0]
                    if v_idx not in new_polygon:
                        new_polygon.append(v_idx)
                
                # Build list of remaining hole wires and hole polygons (ONLY VALID ONES)
                new_hole_wires = [wv['wire'] for wv in remaining_valid_wires]
                new_holes = []
                for hole_wire in new_hole_wires:
                    hole_poly = []
                    for edge_idx in hole_wire:
                        v_idx = edge_array[edge_idx, 0]
                        if v_idx not in hole_poly:
                            hole_poly.append(v_idx)
                    new_holes.append(hole_poly)
                
                modified_faces.append({
                    'original_idx': face_idx,
                    'face_idx': face_num,
                    'wires': [new_outer_wire] + new_hole_wires,  # Only valid wires
                    'polygon': new_polygon,
                    'holes': new_holes,
                    'normal': face['normal']
                })
        else:
            # Outer boundary is valid - keep it and only include valid holes
            valid_hole_wires = [wv['wire'] for wv in wire_validity[1:] if wv['is_valid']]
            invalid_hole_count = sum(1 for wv in wire_validity[1:] if not wv['is_valid'])
            
            if invalid_hole_count > 0:
                print(f"   Face {face_num}: Keeping valid outer boundary, removing {invalid_hole_count} invalid hole(s)")
                
                # Rebuild holes list with only valid holes
                valid_holes = []
                for wv in wire_validity[1:]:
                    if wv['is_valid']:
                        hole_poly = []
                        for edge_idx in wv['wire']:
                            v_idx = edge_array[edge_idx, 0]
                            if v_idx not in hole_poly:
                                hole_poly.append(v_idx)
                        valid_holes.append(hole_poly)
                
                modified_faces.append({
                    'original_idx': face_idx,
                    'face_idx': face_num,
                    'wires': [wire_validity[0]['wire']] + valid_hole_wires,
                    'polygon': face['polygon'],
                    'holes': valid_holes,
                    'normal': face['normal']
                })
            else:
                # Face is completely valid - no changes needed
                modified_faces.append(face)
    
    # Update face_array
    face_array = [f for i, f in enumerate(modified_faces) if i not in faces_to_remove]
    
    print(f"   Removed {len(faces_to_remove)} invalid face(s)")
    print(f"   Remaining faces: {len(face_array)}")

    
    # Step 7.5: Identify dummy polygons (edges with only one face)
    print(f"\n[STEP 7.5] Identifying dummy polygons...")
    
    # Find all edges with only one associated face
    boundary_edges = []
    for edge_idx, edge in enumerate(edge_array):
        f1, f2 = edge[2], edge[3]
        if (f1 == -1 and f2 != -1) or (f1 != -1 and f2 == -1):
            boundary_edges.append(edge_idx)
    
    print(f"   Found {len(boundary_edges)} boundary edge(s) (edges with only one face)")
    
    # Find faces that contain boundary edges - these may be dummy faces
    # Build a mapping of face_idx to actual face in face_array
    face_idx_to_face = {face['face_idx']: face for face in face_array}
    
    dummy_face_nums = set()
    for edge_idx in boundary_edges:
        f1, f2 = edge_array[edge_idx, 2], edge_array[edge_idx, 3]
        if f1 != -1:
            dummy_face_nums.add(f1)
        if f2 != -1:
            dummy_face_nums.add(f2)
    
    print(f"   Potential dummy faces: {sorted(dummy_face_nums)}")
    
    # Determine which faces are dummy by checking if ALL their edges are boundary edges
    confirmed_dummy_faces = []
    for face_num in dummy_face_nums:
        if face_num not in face_idx_to_face:
            # Face was removed in Step 7.4.5
            print(f"   Face {face_num} was already removed (invalid edges)")
            continue
        
        face = face_idx_to_face[face_num]
        all_boundary = True
        for wire in face['wires']:
            for edge_idx in wire:
                if edge_idx not in boundary_edges:
                    all_boundary = False
                    break
            if not all_boundary:
                break
        
        if all_boundary:
            confirmed_dummy_faces.append(face_num)
    
    print(f"   Confirmed dummy faces (all edges boundary): "
          f"{confirmed_dummy_faces}")
    
    # Step 7.6: Plot wires with color coding
    print(f"\n[STEP 7.6] Plotting wires with color coding...")
    print("   Color scheme:")
    print("     - Light blue: Outer wire (boundary of valid faces)")
    print("     - Yellow: Inner wire (holes in faces)")
    print("     - Red: Dummy polygon (not needed for topology)")
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.widgets import CheckButtons
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Store text objects for toggling
    vertex_texts = []
    face_texts = []
    
    # Plot each face's wires
    for face in face_array:
        face_idx = face['face_idx']
        is_dummy = face_idx in confirmed_dummy_faces
        
        for wire_idx, wire in enumerate(face['wires']):
            # Determine color
            if is_dummy:
                color = 'red'
                label = f'Dummy F{face_idx}'
            elif wire_idx == 0:
                color = '#ADD8E6'  # Light blue for outer wire
                label = f'Outer F{face_idx}'
            else:
                color = 'yellow'
                label = f'Hole F{face_idx}'
            
            # Build polygon vertices from wire edges
            wire_vertices = []
            polygon = face['polygon']
            for v_idx in polygon:
                wire_vertices.append(selected_vertices[v_idx])
            
            wire_vertices = np.array(wire_vertices)
            if len(wire_vertices) > 2:
                # Close the wire for plotting
                wire_closed = np.vstack([wire_vertices, wire_vertices[0]])
                ax.plot(wire_closed[:, 0], wire_closed[:, 1], wire_closed[:, 2], 
                       color=color, linewidth=2,
                       label=label if wire_idx == 0 else None)
                
                # Plot vertices
                ax.scatter(wire_vertices[:, 0], wire_vertices[:, 1],
                          wire_vertices[:, 2], color=color, s=40)
                
                # Add vertex numbers
                for i, v_idx in enumerate(polygon):
                    v = selected_vertices[v_idx]
                    txt = ax.text(v[0], v[1], v[2], f'V{v_idx}', 
                           color='black', fontsize=8, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', 
                                   alpha=0.7, edgecolor='none'))
                    vertex_texts.append(txt)
                
                # Add face number at centroid
                if wire_idx == 0:  # Only add face number once per face
                    centroid = np.mean(wire_vertices, axis=0)
                    face_color = 'darkred' if is_dummy else 'darkblue'
                    txt = ax.text(centroid[0], centroid[1], centroid[2],
                           f'F{face_idx}',
                           color=face_color, fontsize=12, ha='center',
                           va='center', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor='yellow',
                                   alpha=0.8, edgecolor='black',
                                   linewidth=1.5))
                    face_texts.append(txt)
    
    ax.set_title('Wire Classification: Light Blue=Outer, Yellow=Hole, Red=Dummy', 
                fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Enable mouse rotation
    ax.mouse_init()
    
    # Add legend with unique labels only
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right',
             fontsize=8)
    
    # Add toggle buttons for labels
    checkbox_ax = plt.axes([0.02, 0.7, 0.15, 0.15])
    labels_check = ['Vertex Labels', 'Face Labels']
    visibility = [True, True]
    check = CheckButtons(checkbox_ax, labels_check, visibility)
    
    def toggle_labels(label):
        if label == 'Vertex Labels':
            for txt in vertex_texts:
                txt.set_visible(not txt.get_visible())
        elif label == 'Face Labels':
            for txt in face_texts:
                txt.set_visible(not txt.get_visible())
        fig.canvas.draw_idle()
    
    check.on_clicked(toggle_labels)
    
    plt.tight_layout()
    print("[STEP 7.6] Displaying wire classification plot...")
    print("           (Use mouse to rotate the 3D view)")
    print("           (Use checkboxes to toggle labels)")
    plt.show()
    print("[STEP 7.6] Wire classification plot closed.")
    
    # =========================================================================
    # Helper function for computing signed area in 3D
    # =========================================================================
    def compute_signed_area_3d(vertices_3d, normal):
        """
        Compute signed area of a polygon in 3D space.
        Projects vertices onto plane defined by normal, then computes
        2D signed area.
        Positive area = CCW winding, Negative = CW winding.
        """
        if len(vertices_3d) < 3:
            return 0.0
        
        # Convert to numpy array
        verts = np.array(vertices_3d)
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)
        
        # Create coordinate system in the plane
        # Choose arbitrary perpendicular vector
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1, 0, 0])
        else:
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # Project vertices to 2D
        verts_2d = []
        for vert in verts:
            x = np.dot(vert, u)
            y = np.dot(vert, v)
            verts_2d.append([x, y])
        
        # Compute signed area using shoelace formula
        area = 0.0
        for i in range(len(verts_2d)):
            x1, y1 = verts_2d[i]
            x2, y2 = verts_2d[(i + 1) % len(verts_2d)]
            area += x1 * y2 - x2 * y1
        
        return area / 2.0
    
    # =========================================================================
    # STEP 8: Build solid from face-wire topology using OCC stitching
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 8] BUILDING SOLID FROM WIRE-BASED FACES")
    print("="*70)
    
    print(f"\n[STEP 8.1] Creating OCC faces from wires (excluding dummy faces)...")
    
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing
    )
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties
    
    occ_faces_list = []
    
    for face in face_array:
        face_idx = face['face_idx']
        
        # Skip dummy faces
        if face_idx in confirmed_dummy_faces:
            print(f"   Skipping dummy face {face_idx}")
            continue
        
        # Get vertices for outer wire
        polygon = face['polygon']
        outer_verts_3d = [selected_vertices[v_idx] for v_idx in polygon]
        
        # Build OCC wire from vertices
        try:
            # Create edges for the wire
            wire_edges = []
            for i in range(len(outer_verts_3d)):
                p1 = gp_Pnt(*outer_verts_3d[i])
                p2 = gp_Pnt(*outer_verts_3d[(i + 1) % len(outer_verts_3d)])
                edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
                wire_edges.append(edge)
            
            # Build wire from edges
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge in wire_edges:
                wire_builder.Add(edge)
            
            if not wire_builder.IsDone():
                print(f"   ERROR: Failed to build wire for face {face_idx}")
                continue
            
            outer_wire = wire_builder.Wire()
            
            # Build face from outer wire
            face_builder = BRepBuilderAPI_MakeFace(outer_wire)
            
            if not face_builder.IsDone():
                print(f"   ERROR: Failed to build face from wire for face {face_idx}")
                continue
            
            # Add hole wires if present
            # Holes must be traversed in opposite direction to outer boundary
            if len(face['holes']) > 0:
                print(f"   Face {face_idx}: Processing {len(face['holes'])} hole(s)")
                
                # Compute signed area of outer boundary to determine its winding
                outer_normal = face['normal']
                outer_signed_area = compute_signed_area_3d(
                    outer_verts_3d, outer_normal)
                print(f"   Face {face_idx}: Outer boundary signed area = "
                      f"{outer_signed_area:.6f}")
                
                for hole_idx, hole_vertex_indices in enumerate(face['holes']):
                    try:
                        hole_verts_3d = [selected_vertices[v_idx]
                                        for v_idx in hole_vertex_indices]
                        
                        # Check hole winding direction
                        hole_signed_area = compute_signed_area_3d(
                            hole_verts_3d, outer_normal)
                        print(f"   Face {face_idx}: Hole {hole_idx + 1} "
                              f"has {len(hole_verts_3d)} vertices, "
                              f"signed area = {hole_signed_area:.6f}")
                        
                        # If same sign, hole needs to be reversed
                        if (outer_signed_area > 0 and hole_signed_area > 0) or \
                           (outer_signed_area < 0 and hole_signed_area < 0):
                            print(f"   Face {face_idx}: Reversing hole {hole_idx + 1} "
                                  f"(same winding as outer)")
                            hole_verts_3d = list(reversed(hole_verts_3d))
                        
                        # Create edges for the hole wire
                        hole_edges = []
                        for i in range(len(hole_verts_3d)):
                            p1 = gp_Pnt(*hole_verts_3d[i])
                            p2_idx = (i + 1) % len(hole_verts_3d)
                            p2 = gp_Pnt(*hole_verts_3d[p2_idx])
                            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
                            hole_edges.append(edge)
                        
                        # Build hole wire from edges
                        hole_wire_builder = BRepBuilderAPI_MakeWire()
                        for edge in hole_edges:
                            hole_wire_builder.Add(edge)
                        
                        if hole_wire_builder.IsDone():
                            hole_wire = hole_wire_builder.Wire()
                            face_builder.Add(hole_wire)
                            print(f"   Face {face_idx}: Successfully added "
                                  f"hole {hole_idx + 1}")
                        else:
                            print(f"   Face {face_idx}: Failed to build "
                                  f"hole wire {hole_idx + 1}")
                    except Exception as e:
                        print(f"   Face {face_idx}: Exception adding "
                              f"hole {hole_idx + 1}: {e}")
            
            occ_face = face_builder.Face()
            
            # Validate face
            analyzer = BRepCheck_Analyzer(occ_face)
            is_valid = analyzer.IsValid()
            
            if is_valid:
                occ_faces_list.append(occ_face)
                print(f"   Face {face_idx}: Valid OCC face created")
            else:
                print(f"   Face {face_idx}: Invalid OCC face (skipped)")
                
        except Exception as e:
            print(f"   ERROR: Exception creating face {face_idx}: {e}")
            continue
    
    print(f"\n[STEP 8.1] Created {len(occ_faces_list)} valid OCC faces")
    
    # Step 8.2: Stitch faces into a solid using BRepBuilderAPI_Sewing
    print(f"\n[STEP 8.2] Stitching faces into solid...")
    
    solid_is_valid = False  # Track solid validity for Step 8.3
    
    if len(occ_faces_list) == 0:
        print("   ERROR: No valid faces to stitch")
        occ_solid = None
    else:
        sewing = BRepBuilderAPI_Sewing()
        sewing.SetTolerance(1e-6)
        
        for occ_face in occ_faces_list:
            sewing.Add(occ_face)
        
        print(f"   Sewing {len(occ_faces_list)} faces...")
        sewing.Perform()
        
        sewn_shape = sewing.SewedShape()
        
        # Diagnostic: Check sewing results
        print(f"   Sewing diagnostics:")
        print(f"     - Number of free edges: {sewing.NbFreeEdges()}")
        print(f"     - Number of multiple edges: {sewing.NbMultipleEdges()}")
        print(f"     - Number of contigous edges: {sewing.NbContigousEdges()}")
        print(f"     - Number of degenerated shapes: {sewing.NbDegeneratedShapes()}")
        print(f"     - Number of deleted faces: {sewing.NbDeletedFaces()}")
        
        # Try larger tolerance if there are free edges
        if sewing.NbFreeEdges() > 0:
            print(f"   WARNING: {sewing.NbFreeEdges()} free edges detected")
            
            # Identify which edges are free
            from OCC.Core.TopAbs import TopAbs_EDGE
            from OCC.Core.TopoDS import topods_Edge
            from OCC.Core.TopExp import TopExp_Explorer
            print(f"   Identifying free edges...")
            edge_explorer = TopExp_Explorer(sewn_shape, TopAbs_EDGE)
            free_edge_count = 0
            while edge_explorer.More() and free_edge_count < 10:  # Limit to first 10
                edge = topods_Edge(edge_explorer.Current())
                # Check if edge is free (not shared by 2 faces)
                from OCC.Core.TopExp import topexp
                from OCC.Core.TopAbs import TopAbs_FACE
                face_iter = TopExp_Explorer(sewn_shape, TopAbs_FACE)
                edge_face_count = 0
                while face_iter.More():
                    face = face_iter.Current()
                    edge_in_face = TopExp_Explorer(face, TopAbs_EDGE)
                    while edge_in_face.More():
                        if edge.IsSame(edge_in_face.Current()):
                            edge_face_count += 1
                            break
                        edge_in_face.Next()
                    face_iter.Next()
                
                if edge_face_count < 2:
                    from OCC.Core.BRep import BRep_Tool
                    first, last = topexp.FirstVertex(edge), topexp.LastVertex(edge)
                    p1 = BRep_Tool.Pnt(first)
                    p2 = BRep_Tool.Pnt(last)
                    print(f"     Free edge {free_edge_count+1}: ({p1.X():.1f},{p1.Y():.1f},{p1.Z():.1f}) <-> ({p2.X():.1f},{p2.Y():.1f},{p2.Z():.1f})")
                    free_edge_count += 1
                
                edge_explorer.Next()
            
            print(f"   Trying larger tolerance (1e-4)...")
            sewing2 = BRepBuilderAPI_Sewing()
            sewing2.SetTolerance(1e-4)
            for occ_face in occ_faces_list:
                sewing2.Add(occ_face)
            sewing2.Perform()
            
            if sewing2.NbFreeEdges() < sewing.NbFreeEdges():
                print(f"   Improved: {sewing2.NbFreeEdges()} free edges with larger tolerance")
                sewn_shape = sewing2.SewedShape()
                sewing = sewing2
            else:
                print(f"   No improvement with larger tolerance")
        
        # Try to convert to solid
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SHELL
        from OCC.Core.TopoDS import topods_Shell
        from OCC.Core.ShapeFix import ShapeFix_Shell, ShapeFix_Solid
        
        # Extract shell from sewn shape
        shell_explorer = TopExp_Explorer(sewn_shape, TopAbs_SHELL)
        if shell_explorer.More():
            shell = topods_Shell(shell_explorer.Current())
            
            # Fix shell orientation
            shell_fixer = ShapeFix_Shell()
            shell_fixer.Init(shell)
            shell_fixer.Perform()
            fixed_shell = shell_fixer.Shell()
            
            # Build solid from shell
            try:
                solid_builder = BRepBuilderAPI_MakeSolid(fixed_shell)
                if solid_builder.IsDone():
                    occ_solid = solid_builder.Solid()
                    
                    # Fix solid
                    solid_fixer = ShapeFix_Solid()
                    solid_fixer.Init(occ_solid)
                    solid_fixer.Perform()
                    occ_solid = solid_fixer.Solid()
                    
                    # Validate
                    analyzer = BRepCheck_Analyzer(occ_solid)
                    solid_is_valid = analyzer.IsValid()
                    
                    print(f"   Solid created: Valid={solid_is_valid}")
                    
                    # Compute volume
                    props = GProp_GProps()
                    brepgprop_VolumeProperties(occ_solid, props)
                    volume = props.Mass()
                    print(f"   Solid volume: {volume}")
                    
                    if not solid_is_valid:
                        print("   WARNING: Solid is invalid (topology errors)")
                    else:
                        print("   SUCCESS: Valid solid created!")
                else:
                    print("   ERROR: Failed to build solid from shell")
                    occ_solid = None
            except Exception as e:
                print(f"   ERROR: Exception building solid: {e}")
                occ_solid = None
        else:
            print("   ERROR: No shell found in sewn shape")
            occ_solid = None
    
    # Step 8.3: Extract faces from OCC solid and plot using matplotlib
    if occ_solid is not None:
        print(f"\n[STEP 8.3] Extracting faces from OCC solid and plotting...")
        
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
        from OCC.Core.TopoDS import topods_Face, topods_Wire
        
        # Extract all faces from the solid
        face_explorer = TopExp_Explorer(occ_solid, TopAbs_FACE)
        extracted_occ_faces = []
        
        face_count = 0
        while face_explorer.More():
            face = topods_Face(face_explorer.Current())
            face_count += 1
            
            print(f"   Processing Face {face_count}...")
            
            # Extract wires from face
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            wires = []
            
            wire_idx = 0
            while wire_explorer.More():
                wire = topods_Wire(wire_explorer.Current())
                wire_idx += 1
                
                # Use the proper vertex sequencing function from V6_current
                wire_id = f"{face_count}.{wire_idx}"
                wire_vertices = extract_wire_vertices_in_sequence(wire, wire_id)
                
                if len(wire_vertices) > 0:
                    wires.append(np.array(wire_vertices))
                
                wire_explorer.Next()
            
            if len(wires) > 0:
                extracted_occ_faces.append({
                    'outer_boundary': wires[0],  # First wire is outer
                    'holes': wires[1:] if len(wires) > 1 else []  # Rest are holes
                })
            
            face_explorer.Next()
        
        print(f"   Extracted {face_count} faces from OCC solid")
        print(f"   Faces with holes: {sum(1 for f in extracted_occ_faces if len(f['holes']) > 0)}")
        
        # Plot the extracted faces
        print(f"\n[STEP 8.3] Plotting reconstructed solid faces...")
        
        from matplotlib.widgets import CheckButtons
        
        # Helper function to find vertex index in selected_vertices
        def find_vertex_index(coord_3d, selected_verts, tolerance=1e-6):
            """Find the index of a 3D coordinate in selected_vertices array."""
            for idx, v in enumerate(selected_verts):
                if np.linalg.norm(coord_3d - v) < tolerance:
                    return idx
            return None  # Not found
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Store text objects for toggling
        vertex_texts = []
        face_texts = []
        
        for face_idx, face_data in enumerate(extracted_occ_faces):
            outer = face_data['outer_boundary']
            holes = face_data['holes']
            
            # Plot outer boundary in blue
            if len(outer) > 2:
                outer_closed = np.vstack([outer, outer[0]])
                ax.plot(outer_closed[:, 0], outer_closed[:, 1],
                       outer_closed[:, 2],
                       color='blue', linewidth=2,
                       label='Outer' if face_idx == 0 else None)
                ax.scatter(outer[:, 0], outer[:, 1], outer[:, 2],
                          color='blue', s=40)
                
                # Add vertex labels with actual vertex indices
                for v_idx, v in enumerate(outer):
                    # Find the actual vertex index in selected_vertices
                    actual_idx = find_vertex_index(v, selected_vertices)
                    if actual_idx is not None:
                        label = f'v{actual_idx}'
                    else:
                        label = f'V{v_idx}'  # Fallback to local index
                    
                    txt = ax.text(v[0], v[1], v[2], label,
                           color='black', fontsize=8, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white',
                                   alpha=0.7, edgecolor='none'))
                    vertex_texts.append(txt)
                
                # Add face number at centroid
                centroid = np.mean(outer, axis=0)
                txt = ax.text(centroid[0], centroid[1], centroid[2],
                       f'F{face_idx}',
                       color='darkblue', fontsize=12, ha='center', va='center',
                       weight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                               alpha=0.8, edgecolor='black', linewidth=1.5))
                face_texts.append(txt)
            
            # Plot holes in red
            for hole_idx, hole in enumerate(holes):
                if len(hole) > 2:
                    hole_closed = np.vstack([hole, hole[0]])
                    ax.plot(hole_closed[:, 0], hole_closed[:, 1],
                           hole_closed[:, 2],
                           color='red', linewidth=2, linestyle='--',
                           label='Hole' if face_idx == 0 and hole_idx == 0 else None)
                    ax.scatter(hole[:, 0], hole[:, 1], hole[:, 2],
                              color='red', s=30)
                    
                    # Add vertex labels for holes with actual indices
                    for v_idx, v in enumerate(hole):
                        # Find the actual vertex index in selected_vertices
                        actual_idx = find_vertex_index(v, selected_vertices)
                        if actual_idx is not None:
                            label = f'v{actual_idx}'
                        else:
                            label = f'H{v_idx}'  # Fallback to local index
                        
                        txt = ax.text(v[0], v[1], v[2], label,
                               color='red', fontsize=7, ha='center', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='white',
                                       alpha=0.7, edgecolor='none'))
                        vertex_texts.append(txt)
        
        ax.set_title(f'Reconstructed Solid: {face_count} Faces (Blue=Outer, Red=Holes)',
                    fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Enable mouse rotation
        ax.mouse_init()
        
        if face_count > 0:
            ax.legend(loc='upper right', fontsize=10)
        
        # Add toggle buttons for labels
        checkbox_ax = plt.axes([0.02, 0.7, 0.15, 0.15])
        labels_check = ['Vertex Labels', 'Face Labels']
        visibility = [True, True]
        check = CheckButtons(checkbox_ax, labels_check, visibility)
        
        def toggle_labels(label):
            if label == 'Vertex Labels':
                for txt in vertex_texts:
                    txt.set_visible(not txt.get_visible())
            elif label == 'Face Labels':
                for txt in face_texts:
                    txt.set_visible(not txt.get_visible())
            fig.canvas.draw_idle()
        
        check.on_clicked(toggle_labels)
        
        plt.tight_layout()
        print("[STEP 8.3] Displaying reconstructed solid plot...")
        print("           (Use mouse to rotate the 3D view)")
        print("           (Use checkboxes to toggle labels)")
        plt.show()
        print("[STEP 8.3] Reconstructed solid plot closed.")
    else:
        print(f"\n[STEP 8.3] No solid to display (construction failed)")
    
    print("\n" + "="*70)
    print("[COMPLETED] Reconstruction process finished.")
    print("="*70)


if __name__ == "__main__":
    main()
