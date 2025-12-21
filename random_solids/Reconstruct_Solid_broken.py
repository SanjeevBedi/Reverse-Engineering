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
import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.ioff()  # Turn off interactive mode

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
    
    # Load unit conversion parameters (with defaults for backward compatibility)
    # Handle both regular strings and numpy strings
    try:
        units = str(data['units']) if 'units' in data else 'cm'
    except:
        units = 'cm'
    
    try:
        drawing_scale_real = float(data['drawing_scale_real']) if 'drawing_scale_real' in data else 1.0
    except:
        drawing_scale_real = 1.0
    
    try:
        drawing_scale_drawing = float(data['drawing_scale_drawing']) if 'drawing_scale_drawing' in data else 1.0
    except:
        drawing_scale_drawing = 1.0
    
    # Unit conversion factors to mm
    unit_to_mm = {
        'mm': 1.0,
        'cm': 10.0,
        'inches': 25.4,
        'in': 25.4,
        'feet': 304.8,
        'ft': 304.8,
        'm': 1000.0
    }
    
    # Calculate conversion factor
    # The solid is built in the specified units
    # We need to convert TO mm for internal calculations
    # scale_factor = (units to mm) * (D_real/D_drawing)
    conversion = unit_to_mm.get(units.lower(), 1.0)
    scale_factor = conversion * (drawing_scale_real / drawing_scale_drawing)
    
    print(f"       - Drawing units: {units}")
    print(f"       - Drawing scale: {drawing_scale_real}:{drawing_scale_drawing}")
    print(f"       - Conversion to mm: {conversion} mm per {units}")
    print(f"       - Scale factor applied: {scale_factor}")
    
    # Apply unit conversion to all coordinates
    all_vertices = all_vertices * scale_factor
    
    # CRITICAL: View matrices have structure [connectivity, x, y]
    # Column 0 is connectivity (0 or 1), columns 1-2 are coordinates
    # Only scale the coordinate columns, NOT the connectivity column
    top_view_matrix[:, 1:3] = top_view_matrix[:, 1:3] * scale_factor
    front_view_matrix[:, 1:3] = front_view_matrix[:, 1:3] * scale_factor
    side_view_matrix[:, 1:3] = side_view_matrix[:, 1:3] * scale_factor
    
    print(f"       - Loaded {len(all_vertices)} vertices")
    print(f"       - Top view matrix: {top_view_matrix.shape}")
    print(f"       - Front view matrix: {front_view_matrix.shape}")
    print(f"       - Side view matrix: {side_view_matrix.shape}")
    
    # Debug: Show sample vertices with rounding to 0.1 mm for display
    if len(all_vertices) > 0:
        print(f"\n       - Sample vertices (rounded to 0.1 mm for display):")
        for i in range(min(3, len(all_vertices))):
            v = all_vertices[i]
            v_rounded = np.round(v, 1)
            print(f"         Vertex {i}: [{v_rounded[0]:.1f}, "
                  f"{v_rounded[1]:.1f}, {v_rounded[2]:.1f}] mm")
    
    return (all_vertices, top_view_matrix, front_view_matrix, side_view_matrix,
            units, drawing_scale_real, drawing_scale_drawing)


def process_boundary_edges(extracted_faces, selected_vertices, merged_conn,
                           tolerance=1e-6):
    """
    Process boundary edges to find missing polygons/holes.
    
    From boundary edges, identify closed/unclosed polygons using connectivity.
    - Closed polygons: check if they belong to existing face (hole or separate)
    - Unclosed polygons: find pairs, determine plane, add missing edges
    
    Args:
        extracted_faces: List of face dicts with 'vertices', 'normal', etc.
        selected_vertices: Numpy array of 3D vertex coordinates
        merged_conn: Merged connectivity matrix
        tolerance: Tolerance for geometric comparisons
        
    Returns:
        Updated list of faces with processed boundary edges
    """
    print("\n[BOUNDARY] Starting boundary edge processing...")
    
    # Track edges from rejected interior polygons that should be removed
    rejected_polygon_edges = []
    
    # Step 1: Build edge-to-face mapping from current extracted faces
    edge_face_map = {}
    for face_idx, face in enumerate(extracted_faces):
        if not isinstance(face, dict) or 'vertices' not in face:
            continue
        
        verts = face['vertices']
        # Add main boundary edges
        for i in range(len(verts)):
            v1_idx = verts[i]
            v2_idx = verts[(i + 1) % len(verts)]
            edge = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            if edge not in edge_face_map:
                edge_face_map[edge] = []
            edge_face_map[edge].append(('boundary', face_idx))
        
        # Add hole edges if present
        if 'holes' in face:
            for hole_idx, hole in enumerate(face['holes']):
                for i in range(len(hole)):
                    v1_idx = hole[i]
                    v2_idx = hole[(i + 1) % len(hole)]
                    edge = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
                    if edge not in edge_face_map:
                        edge_face_map[edge] = []
                    edge_face_map[edge].append(('hole', face_idx))
    
    # Step 2: Identify boundary edges (edges appearing in only 1 face)
    boundary_edges = []
    for edge, face_list in edge_face_map.items():
        if len(face_list) == 1:
            boundary_edges.append(edge)
    
    print(f"[BOUNDARY] Found {len(boundary_edges)} boundary edges")
    if len(boundary_edges) > 0:
        print("[BOUNDARY] Boundary edge details:")
        for i, edge in enumerate(boundary_edges, 1):
            v1, v2 = edge
            coord1 = selected_vertices[v1]
            coord2 = selected_vertices[v2]
            print(f"[BOUNDARY]   Edge {i}: v{v1}-v{v2} "
                  f"({coord1[0]:.1f},{coord1[1]:.1f},{coord1[2]:.1f}) <-> "
                  f"({coord2[0]:.1f},{coord2[1]:.1f},{coord2[2]:.1f})")
    
    if len(boundary_edges) == 0:
        print("[BOUNDARY] No boundary edges to process")
        return extracted_faces
    
    # Step 3: Build connectivity graph from boundary edges
    boundary_graph = {}
    for edge in boundary_edges:
        v1, v2 = edge
        if v1 not in boundary_graph:
            boundary_graph[v1] = []
        if v2 not in boundary_graph:
            boundary_graph[v2] = []
        boundary_graph[v1].append(v2)
        boundary_graph[v2].append(v1)
    
    # Step 4: Find closed polygons from boundary edges
    visited_edges = set()
    closed_polygons = []
    unclosed_chains = []
    
    def trace_polygon(start_v, current_v, path, visited):
        """Trace a polygon/chain from boundary edges"""
        if current_v not in boundary_graph:
            return None
        
        neighbors = boundary_graph[current_v]
        for next_v in neighbors:
            edge = (min(current_v, next_v), max(current_v, next_v))
            if edge in visited:
                continue
            
            visited.add(edge)
            path.append(next_v)
            
            # Check if we closed the loop
            if next_v == start_v and len(path) >= 3:
                return 'closed'
            
            # Continue tracing
            result = trace_polygon(start_v, next_v, path, visited)
            if result == 'closed':
                return 'closed'
        
        return 'unclosed'
    
    # Trace all polygons/chains from boundary edges
    for start_edge in boundary_edges:
        if start_edge in visited_edges:
            continue
        
        v1, v2 = start_edge
        path = [v1, v2]
        local_visited = {start_edge}
        
        result = trace_polygon(v1, v2, path, local_visited)
        
        visited_edges.update(local_visited)
        
        if result == 'closed':
            closed_polygons.append(path[:-1])  # Remove duplicate last vertex
            print(f"[BOUNDARY] Found closed polygon: {len(path)-1} vertices")
        else:
            unclosed_chains.append(path)
            print(f"[BOUNDARY] Found unclosed chain: {len(path)} vertices")
    
    print(f"[BOUNDARY] Total: {len(closed_polygons)} closed, " +
          f"{len(unclosed_chains)} unclosed")
    
    # Step 5: Process closed polygons
    for poly_verts in closed_polygons:
        rejected_edges = process_closed_polygon(poly_verts, extracted_faces,
                                               selected_vertices, tolerance)
        if rejected_edges:
            rejected_polygon_edges.extend(rejected_edges)
    
    # Step 6: Process unclosed chains (find pairs and complete them)
    if len(unclosed_chains) > 0:
        process_unclosed_chains(unclosed_chains, extracted_faces,
                                selected_vertices, merged_conn, tolerance)
    
    # Step 7: Remove rejected polygon edges from all faces
    # Note: Rejected edges represent interior faces that were correctly not created
    # We store them to filter out from "free edge" warnings during sewing
    if rejected_polygon_edges:
        print(f"\n[BOUNDARY] Storing {len(rejected_polygon_edges)} edges "
              f"from rejected interior polygons for filtering...")
        # Store in a format accessible later during sewing
        # Add as a return value or store globally
    
    # Return both the faces and the rejected edges
    return extracted_faces, rejected_polygon_edges


def process_closed_polygon(poly_verts, extracted_faces, selected_vertices,
                           tolerance=1e-6):
    """
    Process a closed polygon found in boundary edges.
    
    Returns:
        List of rejected edges if polygon is interior, empty list otherwise
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    
    print(f"\n[BOUNDARY] Processing closed polygon with " +
          f"{len(poly_verts)} vertices")
    
    # Determine which face plane this polygon belongs to
    poly_coords = np.array([selected_vertices[v] for v in poly_verts])
    
    # Fit plane to polygon vertices
    poly_normal, poly_d = fit_plane_to_points(poly_coords, tolerance)
    
    if poly_normal is None:
        print("[BOUNDARY] Could not fit plane to polygon")
        return []
    
    # Find matching face by comparing normals and d values
    best_match_idx = None
    best_match_dist = float('inf')
    
    for face_idx, face in enumerate(extracted_faces):
        if not isinstance(face, dict) or 'normal' not in face:
            continue
        
        face_normal = np.array(face['normal'])
        face_d = face.get('d', 0)
        
        # Check if normals are parallel (or anti-parallel)
        dot = abs(np.dot(poly_normal, face_normal))
        if dot < 0.999:  # ~2.5 degree tolerance
            continue
        
        # Check if d values are close
        d_diff = abs(poly_d - face_d)
        if d_diff < best_match_dist:
            best_match_dist = d_diff
            best_match_idx = face_idx
    
    if best_match_idx is not None and best_match_dist < tolerance * 10:
        # Polygon belongs to an existing face
        print(f"[BOUNDARY] Polygon matches face {best_match_idx+1}")
        process_polygon_on_existing_face(poly_verts, best_match_idx,
                                        extracted_faces, selected_vertices,
                                        tolerance)
        return []
    else:
        # Create new face for this polygon
        print("[BOUNDARY] Creating new face for polygon")
        rejected_edges = create_new_face_from_polygon(
            poly_verts, poly_normal, poly_d,
            extracted_faces, selected_vertices)
        return rejected_edges if rejected_edges else []


def process_polygon_on_existing_face(poly_verts, face_idx, extracted_faces,
                                     selected_vertices, tolerance=1e-6):
    """Determine if polygon is a hole, duplicate, or outside existing face"""
    from shapely.geometry import Polygon as ShapelyPolygon
    
    face = extracted_faces[face_idx]
    
    # Convert face boundary to 2D for comparison
    face_verts_3d = [selected_vertices[v] for v in face['vertices']]
    face_normal = np.array(face['normal'])
    
    # Project to 2D
    face_verts_2d = project_to_2d(face_verts_3d, face_normal)
    poly_verts_3d = [selected_vertices[v] for v in poly_verts]
    poly_verts_2d = project_to_2d(poly_verts_3d, face_normal)
    
    face_poly = ShapelyPolygon(face_verts_2d)
    new_poly = ShapelyPolygon(poly_verts_2d)
    
    # Check if duplicate
    if face_poly.equals(new_poly):
        print("[BOUNDARY] Polygon is duplicate of face boundary - deleting")
        return
    
    # Check if polygon is inside face (hole)
    if face_poly.contains(new_poly):
        # Check if it touches the boundary
        if face_poly.boundary.intersects(new_poly.boundary):
            print("[BOUNDARY] Polygon touches boundary - exception raised")
            return
        
        print("[BOUNDARY] Polygon is inside face - making it a hole")
        if 'holes' not in face:
            face['holes'] = []
        face['holes'].append(poly_verts)
        return
    
    # Check if polygon is outside face
    if not face_poly.intersects(new_poly):
        print("[BOUNDARY] Polygon is outside face - creating separate face")
        poly_normal = np.array(face['normal'])
        poly_d = face.get('d', 0)
        create_new_face_from_polygon(poly_verts, poly_normal, poly_d,
                                     extracted_faces, selected_vertices)
        return
    
    # Polygons intersect but don't match above cases
    print("[BOUNDARY] Polygon intersects face boundary - exception raised")


def process_unclosed_chains(unclosed_chains, extracted_faces,
                            selected_vertices, merged_conn, tolerance=1e-6):
    """Process unclosed polygon chains by finding pairs and completing them"""
    print(f"\n[BOUNDARY] Processing {len(unclosed_chains)} unclosed chains")
    
    if len(unclosed_chains) < 2:
        print("[BOUNDARY] Need at least 2 chains to form pairs")
        return
    
    # Group chains by the plane they lie on
    chain_planes = []
    for chain in unclosed_chains:
        chain_coords = np.array([selected_vertices[v] for v in chain])
        normal, d = fit_plane_to_points(chain_coords, tolerance)
        if normal is not None:
            chain_planes.append((chain, normal, d))
    
    # Find pairs of chains on the same plane
    processed = set()
    for i in range(len(chain_planes)):
        if i in processed:
            continue
        
        chain1, normal1, d1 = chain_planes[i]
        
        for j in range(i+1, len(chain_planes)):
            if j in processed:
                continue
            
            chain2, normal2, d2 = chain_planes[j]
            
            # Check if on same plane
            dot = abs(np.dot(normal1, normal2))
            d_diff = abs(d1 - d2)
            
            if dot > 0.999 and d_diff < tolerance * 10:
                # Chains are on same plane - try to complete
                print(f"[BOUNDARY] Found chain pair on same plane")
                complete_polygon_from_chains(chain1, chain2, normal1, d1,
                                            extracted_faces, selected_vertices,
                                            merged_conn, tolerance)
                processed.add(i)
                processed.add(j)
                break


def complete_polygon_from_chains(chain1, chain2, normal, d,
                                 extracted_faces, selected_vertices,
                                 merged_conn, tolerance=1e-6):
    """Complete a polygon by connecting two chains with missing edges"""
    print(f"[BOUNDARY] Attempting to complete polygon from 2 chains")
    print(f"[BOUNDARY] Chain 1: {len(chain1)} vertices")
    print(f"[BOUNDARY] Chain 2: {len(chain2)} vertices")
    
    # Find endpoint vertices that should connect
    # Try different connection combinations
    connections = [
        (chain1[0], chain2[0]),   # Start to start
        (chain1[0], chain2[-1]),  # Start to end
        (chain1[-1], chain2[0]),  # End to start
        (chain1[-1], chain2[-1])  # End to end
    ]
    
    # Find which connection creates a valid closed polygon
    for v1, v2 in connections:
        # Check if these vertices lie on two different faces
        v1_faces = find_faces_containing_vertex(v1, extracted_faces)
        v2_faces = find_faces_containing_vertex(v2, extracted_faces)
        
        common_faces = set(v1_faces) & set(v2_faces)
        
        if len(common_faces) >= 2:
            # These vertices lie on at least 2 common faces
            # Add edge between them
            print(f"[BOUNDARY] Adding edge between vertices {v1} and {v2}")
            
            # Build complete polygon
            if v1 == chain1[0] and v2 == chain2[0]:
                poly_verts = list(reversed(chain1)) + chain2
            elif v1 == chain1[0] and v2 == chain2[-1]:
                poly_verts = list(reversed(chain1)) + list(reversed(chain2))
            elif v1 == chain1[-1] and v2 == chain2[0]:
                poly_verts = chain1 + chain2
            else:  # v1 == chain1[-1] and v2 == chain2[-1]
                poly_verts = chain1 + list(reversed(chain2))
            
            # Create new face from completed polygon
            create_new_face_from_polygon(poly_verts, normal, d,
                                        extracted_faces, selected_vertices)
            return
    
    print("[BOUNDARY] Could not find valid connection for chains")


def fit_plane_to_points(points, tolerance=1e-6):
    """Fit a plane to a set of 3D points using SVD"""
    if len(points) < 3:
        return None, None
    
    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # SVD to find normal vector
    try:
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]  # Last row of V^T is normal with smallest variance
        normal = normal / np.linalg.norm(normal)
        
        # Calculate d: n · p + d = 0 => d = -n · centroid
        d = -np.dot(normal, centroid)
        
        return normal, d
    except:
        return None, None


def project_to_2d(points_3d, normal):
    """Project 3D points to 2D plane perpendicular to normal"""
    # Find which axis to drop (one with largest normal component)
    abs_normal = np.abs(normal)
    drop_axis = np.argmax(abs_normal)
    keep_axes = [i for i in range(3) if i != drop_axis]
    
    points_2d = []
    for p in points_3d:
        points_2d.append([p[keep_axes[0]], p[keep_axes[1]]])
    
    return np.array(points_2d)


def ray_intersects_polygon(ray_origin, ray_direction, polygon_coords,
                           polygon_holes=None, epsilon=1e-8,
                           return_t=False):
    """
    Check if a ray intersects a polygon using plane intersection +
    point-in-polygon test.
    
    Args:
        ray_origin: 3D point where ray starts
        ray_direction: 3D normalized direction vector
        polygon_coords: Nx3 array of polygon vertex coordinates
        polygon_holes: List of hole polygons (each is Mx3 array), or None
        epsilon: Tolerance for geometric tests
        return_t: If True, return (bool, t_value) instead of just bool
        
    Returns:
        bool: True if ray intersects polygon (inside boundary, outside holes)
        OR (bool, t_value) if return_t=True
    """
    from shapely.geometry import Polygon as ShapelyPolygon, Point
    
    if len(polygon_coords) < 3:
        return (False, None) if return_t else False
    
    # Step 1: Compute plane of polygon
    v0 = polygon_coords[0]
    v1 = polygon_coords[1]
    v2 = polygon_coords[2]
    
    # Polygon normal
    edge1 = v1 - v0
    edge2 = v2 - v0
    poly_normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(poly_normal)
    if norm < epsilon:
        return (False, None) if return_t else False
    poly_normal = poly_normal / norm
    
    # Plane equation: n·p + d = 0
    poly_d = -np.dot(poly_normal, v0)
    
    # Step 2: Find intersection of ray with plane
    denom = np.dot(poly_normal, ray_direction)
    if abs(denom) < epsilon:
        # Ray parallel to plane
        return (False, None) if return_t else False
    
    t = -(np.dot(poly_normal, ray_origin) + poly_d) / denom
    
    # Only consider intersections in positive ray direction
    if t <= epsilon:
        return (False, None) if return_t else False
    
    # Intersection point
    intersection_point = ray_origin + t * ray_direction
    
    # Step 3: Check if intersection point is inside polygon boundary
    # Project polygon and intersection point to 2D
    poly_2d = project_to_2d(polygon_coords, poly_normal)
    
    # Project intersection point to 2D
    # Find dominant axis
    abs_normal = np.abs(poly_normal)
    dominant_idx = np.argmax(abs_normal)
    
    if dominant_idx == 0:  # x is dominant, use y-z plane
        int_2d = Point(intersection_point[1], intersection_point[2])
    elif dominant_idx == 1:  # y is dominant, use x-z plane
        int_2d = Point(intersection_point[0], intersection_point[2])
    else:  # z is dominant, use x-y plane
        int_2d = Point(intersection_point[0], intersection_point[1])
    
    # Create polygon boundary
    boundary_poly = ShapelyPolygon(poly_2d)
    
    # Check if point is inside boundary
    if not boundary_poly.contains(int_2d):
        return (False, None) if return_t else False
    
    # Step 4: Check if point is outside all holes
    if polygon_holes is not None and len(polygon_holes) > 0:
        for hole_coords in polygon_holes:
            hole_2d = project_to_2d(hole_coords, poly_normal)
            hole_poly = ShapelyPolygon(hole_2d)
            if hole_poly.contains(int_2d):
                # Point is inside a hole
                return (False, None) if return_t else False
    
    return (True, t) if return_t else True


def create_new_face_from_polygon(poly_verts, normal, d, extracted_faces,
                                 selected_vertices):
    """
    Create a new face from a polygon with ray-casting validation.
    
    Returns:
        List of rejected edges if polygon is interior, empty list otherwise
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    
    # Step 1: Validate face using ray-casting
    print(f"[BOUNDARY] Validating new face with {len(poly_verts)} vertices "
          f"using ray-casting...")
    
    # Get 3D coordinates
    poly_coords_3d = np.array([selected_vertices[v] for v in poly_verts])
    
    # Project to 2D to find interior point
    poly_coords_2d = project_to_2d(poly_coords_3d, normal)
    poly_shapely = ShapelyPolygon(poly_coords_2d)
    
    if not poly_shapely.is_valid:
        print("[BOUNDARY] ✗ Invalid polygon geometry - skipping")
        return []
    
    # Get interior point in 2D
    interior_2d = poly_shapely.representative_point()
    
    # Project interior point back to 3D on the face plane
    # Solve for the missing coordinate using plane equation
    poly_normal = np.array(normal)
    
    # Find which component of normal is dominant
    abs_normal = np.abs(poly_normal)
    dominant_idx = np.argmax(abs_normal)
    
    if dominant_idx == 2:  # z is dominant
        # Solve for z: n·p + d = 0 => nz*z = -(nx*x + ny*y + d)
        interior_3d = np.array([interior_2d.x, interior_2d.y, 0.0])
        interior_3d[2] = (-(poly_normal[0] * interior_2d.x +
                            poly_normal[1] * interior_2d.y + d) /
                          poly_normal[2])
    elif dominant_idx == 1:  # y is dominant
        # Solve for y
        interior_3d = np.array([interior_2d.x, 0.0, interior_2d.y])
        interior_3d[1] = (-(poly_normal[0] * interior_2d.x +
                            poly_normal[2] * interior_2d.y + d) /
                          poly_normal[1])
    else:  # x is dominant
        # Solve for x
        interior_3d = np.array([0.0, interior_2d.x, interior_2d.y])
        interior_3d[0] = (-(poly_normal[1] * interior_2d.x +
                            poly_normal[2] * interior_2d.y + d) /
                          poly_normal[0])
    
    # Step 2: Cast ray from interior point in normal direction
    ray_origin = interior_3d
    ray_direction = poly_normal / np.linalg.norm(poly_normal)
    
    intersection_count = 0
    intersected_faces = []
    
    # Test against all existing faces
    for face_idx, face in enumerate(extracted_faces):
        if not isinstance(face, dict) or 'vertices' not in face:
            continue
        
        face_verts = face['vertices']
        if len(face_verts) < 3:
            continue
        
        # Get face vertices
        face_coords = np.array([selected_vertices[v] for v in face_verts])
        
        # Get face holes if any
        face_holes = None
        if 'holes' in face and len(face['holes']) > 0:
            face_holes = [
                np.array([selected_vertices[v] for v in hole])
                for hole in face['holes']
            ]
        
        # Check if ray intersects this face
        # This properly checks: intersection inside boundary AND outside holes
        result, t_val = ray_intersects_polygon(ray_origin, ray_direction,
                                               face_coords, face_holes,
                                               return_t=True)
        if result:
            intersection_count += 1
            intersected_faces.append((face_idx + 1, t_val))
    
    print(f"[BOUNDARY]   Ray-casting result: {intersection_count} "
          f"intersection(s)")
    if len(intersected_faces) > 0:
        face_ids = [f[0] for f in intersected_faces]
        print(f"[BOUNDARY]   Intersected faces: {face_ids}")
        t_values = [f'{t:.4f}' for _, t in intersected_faces]
        print(f"[BOUNDARY]   Distance t values: {t_values}")
    
    # Step 3: Decide whether to add face
    if intersection_count % 2 == 1:
        # Odd intersections = interior point is inside solid
        print("[BOUNDARY]   ✗ Face is interior (odd parity) - "
              "NOT creating face")
        
        # Return edges from this rejected polygon so they can be removed
        rejected_edges = []
        for i in range(len(poly_verts)):
            v1 = poly_verts[i]
            v2 = poly_verts[(i + 1) % len(poly_verts)]
            edge = tuple(sorted([v1, v2]))
            rejected_edges.append(edge)
        
        print(f"[BOUNDARY]   Marking {len(rejected_edges)} edges for removal")
        return rejected_edges
    else:
        # Even intersections = interior point is outside solid
        print("[BOUNDARY]   ✓ Face is exterior (even parity) - "
              "creating face")
    
    # Step 4: Create the face
    new_face = {
        'vertices': poly_verts,
        'normal': tuple(normal),
        'd': d,
        'holes': []
    }
    
    extracted_faces.append(new_face)
    print(f"[BOUNDARY] Created new face {len(extracted_faces)} with " +
          f"{len(poly_verts)} vertices")
    
    return []  # No rejected edges


def find_faces_containing_vertex(v_idx, extracted_faces):
    """Find all faces that contain a given vertex"""
    containing_faces = []
    
    for face_idx, face in enumerate(extracted_faces):
        if not isinstance(face, dict) or 'vertices' not in face:
            continue
        
        if v_idx in face['vertices']:
            containing_faces.append(face_idx)
            continue
        
        # Check holes
        if 'holes' in face:
            for hole in face['holes']:
                if v_idx in hole:
                    containing_faces.append(face_idx)
                    break
    
    return containing_faces


def load_face_polygons(seed, input_dir="Output", 
                       units="cm", drawing_scale_real=1.0, 
                       drawing_scale_drawing=1.0):
    """Load face polygons from file and apply unit conversion"""
    filename = os.path.join(input_dir, f"solid_faces_seed_{seed}.npy")
    
    if not os.path.exists(filename):
        print(f"[LOAD] Warning: Face polygon file not found: {filename}")
        return None, None
    
    print(f"[LOAD] Loading face polygons from: {filename}")
    loaded_data = np.load(filename, allow_pickle=True)
    
    # Handle both old format (just array) and new format (dict with metadata)
    original_volume_raw = None
    if isinstance(loaded_data, np.ndarray):
        # Check if it's a 0-d array containing a dict (new format)
        if loaded_data.shape == () and isinstance(loaded_data.item(), dict):
            # New format: dictionary with 'faces' and 'volume'
            data_dict = loaded_data.item()
            face_data = data_dict.get('faces', [])
            original_volume_raw = data_dict.get('volume', None)
            print(f"       - Loaded {len(face_data)} faces (new format)")
            if original_volume_raw is not None:
                print(f"       - Original volume: {original_volume_raw:.6f}")
        else:
            # Old format: just face array
            face_data = loaded_data
            original_volume_raw = None
            print(f"       - Loaded {len(face_data)} faces (old format)")
    else:
        # Fallback
        face_data = loaded_data
        original_volume_raw = None
        print(f"       - Loaded data (unknown format)")
    
    # Apply same unit conversion as vertices
    unit_to_mm = {
        'mm': 1.0,
        'cm': 10.0,
        'inches': 25.4,
        'in': 25.4,
        'feet': 304.8,
        'ft': 304.8,
        'm': 1000.0
    }
    
    conversion = unit_to_mm.get(units.lower(), 1.0)
    scale_factor = conversion * (drawing_scale_real / drawing_scale_drawing)
    volume_scale = scale_factor ** 3
    
    # Scale volume if available
    original_volume_scaled = None
    if original_volume_raw is not None:
        original_volume_scaled = original_volume_raw * volume_scale
        print(f"       - Scaled volume: {original_volume_scaled:.6f} mm³")
    
    # Scale all vertices in all faces
    scaled_face_data = []
    for face in face_data:
        if isinstance(face, dict):
            scaled_face = face.copy()
            
            # Handle 'outer_boundary' key (from Build_Solid.py)
            if 'outer_boundary' in face:
                scaled_face['outer_boundary'] = (
                    np.array(face['outer_boundary']) * scale_factor
                )
            
            # Handle 'vertices' key (alternative format)
            if 'vertices' in face:
                scaled_face['vertices'] = (
                    np.array(face['vertices']) * scale_factor
                )
            
            # Handle 'holes' key
            if 'holes' in face:
                scaled_holes = []
                for hole in face['holes']:
                    scaled_holes.append(np.array(hole) * scale_factor)
                scaled_face['holes'] = scaled_holes
            
            scaled_face_data.append(scaled_face)
        else:
            # Handle other face formats if any
            scaled_face_data.append(face)
    
    print(f"       - Applied scale factor: {scale_factor}")
    
    # Debug: Print structure of first face
    if len(scaled_face_data) > 0:
        first_face = scaled_face_data[0]
        if isinstance(first_face, dict):
            print(f"       - First face keys: {list(first_face.keys())}")
            if 'outer_boundary' in first_face:
                print(f"       - First face has 'outer_boundary' with "
                      f"{len(first_face['outer_boundary'])} vertices")
    
    return np.array(scaled_face_data, dtype=object), original_volume_scaled


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
    raw_z = [front_matrix[i, 2] for i in range(front_matrix.shape[0])]
    z_levels = sorted(set(raw_z))
    print(f"  - Extracted (x,y) from top view: {len(top_xy_coords)} coordinates")
    print(f"  - Extracted z-levels from front view: {len(z_levels)} levels")
    
    # Debug: Also check z-levels in side view
    side_z = sorted(set([side_matrix[i, 2] for i in range(side_matrix.shape[0])]))
    print(f"  - Z-levels in side view: {len(side_z)} levels")
    
    # Check for discrepancies between front and side z-levels
    front_z_set = set(z_levels)
    side_z_set = set(side_z)
    if front_z_set != side_z_set:
        only_in_front = front_z_set - side_z_set
        only_in_side = side_z_set - front_z_set
        print(f"  [WARNING] Z-level mismatch between views!")
        if only_in_front:
            print(f"    Only in front: {len(only_in_front)} levels")
            for z in sorted(only_in_front)[:5]:
                print(f"      z = {z:.6f}")
        if only_in_side:
            print(f"    Only in side: {len(only_in_side)} levels")
            for z in sorted(only_in_side)[:5]:
                print(f"      z = {z:.6f}")
    
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
    
    # Build arrays of valid projections for each view (for tolerance matching)
    # Use original precision from connectivity matrices
    front_projections = []
    for i in range(front_matrix.shape[0]):
        proj = (front_matrix[i, 1], front_matrix[i, 2])
        front_projections.append(proj)
    front_projections = np.array(front_projections)
    
    side_projections = []
    for i in range(side_matrix.shape[0]):
        proj = (side_matrix[i, 1], side_matrix[i, 2])
        side_projections.append(proj)
    side_projections = np.array(side_projections)
    
    # Tolerance for projection matching (to handle slight coordinate variations)
    # Increased to 0.1mm to handle rounding and scale differences
    proj_tolerance = 0.1  # 0.1 mm tolerance
    
    # Filter candidates
    selected_vertices = []
    rejected_vertices = []  # Track rejected vertices for debugging
    
    for vertex in candidate_vertices:
        # Check front view projection (x, z)
        front_proj = np.array([vertex[0], vertex[2]])
        # Check side view projection (y, z)
        side_proj = np.array([vertex[1], vertex[2]])
        
        # Check if projections exist within tolerance
        front_distances = np.linalg.norm(front_projections - front_proj, axis=1)
        side_distances = np.linalg.norm(side_projections - side_proj, axis=1)
        
        front_match = np.any(front_distances < proj_tolerance)
        side_match = np.any(side_distances < proj_tolerance)
        
        if front_match and side_match:
            selected_vertices.append(vertex)
        else:
            # Store rejection info
            min_front_dist = np.min(front_distances)
            min_side_dist = np.min(side_distances)
            rejected_vertices.append({
                'vertex': vertex,
                'front_match': front_match,
                'side_match': side_match,
                'min_front_dist': min_front_dist,
                'min_side_dist': min_side_dist
            })
    
    # Debug output for rejected vertices
    if len(rejected_vertices) > 0:
        print(f"\n  [DEBUG] Rejected {len(rejected_vertices)} candidate vertices:")
        # Filter to show only rejections where BOTH distances are < 1.0
        close_rejections = [r for r in rejected_vertices
                            if r['min_front_dist'] < 1.0
                            and r['min_side_dist'] < 1.0]
        if close_rejections:
            print(f"  [DEBUG] Showing rejections with BOTH min_dists < 1.0 "
                  f"({len(close_rejections)} found):")
            for i, rej in enumerate(close_rejections[:20]):
                v = rej['vertex']
                print(f"    Candidate ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}):")
                print(f"      Front (x,z) match: {rej['front_match']} "
                      f"(min_dist={rej['min_front_dist']:.6f}, "
                      f"tolerance={proj_tolerance})")
                print(f"      Side (y,z) match: {rej['side_match']} "
                      f"(min_dist={rej['min_side_dist']:.6f}, "
                      f"tolerance={proj_tolerance})")
                
                # Show closest projections for analysis
                front_proj = np.array([v[0], v[2]])
                side_proj = np.array([v[1], v[2]])
                front_dists = np.linalg.norm(front_projections - front_proj,
                                             axis=1)
                side_dists = np.linalg.norm(side_projections - side_proj,
                                            axis=1)
                closest_front_idx = np.argmin(front_dists)
                closest_side_idx = np.argmin(side_dists)
                
                print(f"      Candidate front proj: ({v[0]:.6f}, {v[2]:.6f})")
                print(f"      Closest front proj: "
                      f"({front_projections[closest_front_idx][0]:.6f}, "
                      f"{front_projections[closest_front_idx][1]:.6f})")
                print(f"      Candidate side proj: ({v[1]:.6f}, {v[2]:.6f})")
                print(f"      Closest side proj: "
                      f"({side_projections[closest_side_idx][0]:.6f}, "
                      f"{side_projections[closest_side_idx][1]:.6f})")
                
                if not rej['front_match']:
                    print(f"      ✗ REJECTED: Front view projection not found")
                if not rej['side_match']:
                    print(f"      ✗ REJECTED: Side view projection not found")
        else:
            print(f"  [DEBUG] No rejections with BOTH min_dists < 1.0 found")
    
    selected_vertices = np.array(selected_vertices)
    print(f"\nStep 3 Complete: Filtered to {len(selected_vertices)} vertices")
    print(f"  - Reduction: {len(candidate_vertices)} → {len(selected_vertices)}")
    print(f"  - Filtered out: {len(candidate_vertices) - len(selected_vertices)} "
          "fake vertices")
    
    # Merge vertices within 0.01 units of each other - DISABLED
    # merge_tolerance = 0.01
    # merged_vertices = []
    # vertex_mapping = {}  # Maps old index to new index
    # 
    # for i in range(len(selected_vertices)):
    #     if i in vertex_mapping:
    #         continue  # Already merged
    #     
    #     # Start a new merged vertex
    #     merged_idx = len(merged_vertices)
    #     vertex_mapping[i] = merged_idx
    #     merged_vertices.append(selected_vertices[i].copy())
    #     
    #     # Find all vertices within tolerance and merge them
    #     for j in range(i + 1, len(selected_vertices)):
    #         if j in vertex_mapping:
    #             continue  # Already merged
    #         
    #         dist = np.linalg.norm(
    #             selected_vertices[i] - selected_vertices[j])
    #         if dist < merge_tolerance:
    #             vertex_mapping[j] = merged_idx
    # 
    # merged_vertices = np.array(merged_vertices)
    # num_merged = len(selected_vertices) - len(merged_vertices)
    # 
    # if num_merged > 0:
    #     selected_vertices = merged_vertices
    
    # Debug: Show which (x,y) pairs exist at which z-levels
    # print("\n[DEBUG] Analyzing selected vertices by (x,y) pairs:")
    # xy_to_z = {}
    # for v in selected_vertices:
    #     xy_key = (round(v[0], 6), round(v[1], 6))
    #     z_val = round(v[2], 6)
    #     if xy_key not in xy_to_z:
    #         xy_to_z[xy_key] = []
    #     xy_to_z[xy_key].append(z_val)
    # 
    # # Sort by x, then y for readability
    # for xy_key in sorted(xy_to_z.keys()):
    #     z_levels = sorted(xy_to_z[xy_key])
    #     print(f"  (x={xy_key[0]:.2f}, y={xy_key[1]:.2f}): "
    #           f"z-levels = {z_levels}")
    
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
    parser.add_argument(
        '--no-graphics', action='store_true',
        help='Save graphics to PDF instead of displaying interactively'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SOLID RECONSTRUCTION FROM ENGINEERING VIEWS")
    print("="*70)
    print(f"Seed: {args.seed}")
    
    # Load saved data
    print("\n[STEP 1] Loading saved connectivity matrices...")
    try:
        (all_vertices, top_matrix, front_matrix, side_matrix,
         units, drawing_scale_real, drawing_scale_drawing) = \
            load_connectivity_matrices(args.seed, args.input_dir)
        face_polygons, original_volume_from_file = load_face_polygons(
            args.seed, args.input_dir,
            units, drawing_scale_real, 
            drawing_scale_drawing)
        
        # Ensure face_polygons is a list, not None
        if face_polygons is None:
            print("[LOAD] Warning: No original face polygons available for comparison")
            face_polygons = []
            original_volume_from_file = None
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return
    
    # Filter candidate vertices
    print("\n[STEP 2] Filtering candidate vertices...")
    selected_vertices = filter_candidate_vertices(
        top_matrix, front_matrix, side_matrix
    )
    
    # Display vertices rounded to 0.1 mm precision
    # NOTE: We do NOT round the actual coordinates - they must stay precise
    # for geometric algorithms (connectivity, face extraction) to work correctly.
    # Rounding is only for display purposes.
    print("\n[SELECTED VERTICES] (rounded to 0.1 mm for display)")
    for idx, v in enumerate(selected_vertices):
        v_rounded = np.round(v, 1)
        print(f"  {idx}: [{v_rounded[0]:.1f}, {v_rounded[1]:.1f}, "
              f"{v_rounded[2]:.1f}] mm")
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
    
    print(f"\n[DEBUG] merged_conn has {np.sum(merged_conn == 3)} edges with conn=3 BEFORE rounding")
    print(f"[DEBUG] Sample merged_conn values (rows 0-5, cols 0-5):")
    print(merged_conn[:5, :5])
    
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
    
    # =========================================================================
    # STEP 5.5: Round vertex coordinates to 0.1 mm precision
    # =========================================================================
    print("\n[STEP 5.5] Rounding vertex coordinates to 0.1 mm precision...")
    print("   This is done AFTER building connectivity but BEFORE polygon extraction")
    
    # Store original vertices before rounding
    original_vertices = selected_vertices.copy()
    
    # Round the vertex array
    selected_vertices = np.round(selected_vertices, 1)
    
    print(f"   Rounded {len(selected_vertices)} vertices to 1 decimal place")
    
    # Check for duplicate vertices after rounding
    print("\n   Checking for duplicate vertices after rounding...")
    duplicates_found = []
    for i in range(len(selected_vertices)):
        for j in range(i+1, len(selected_vertices)):
            if np.allclose(selected_vertices[i], selected_vertices[j], atol=1e-6):
                duplicates_found.append((i, j, selected_vertices[i]))
    
    if duplicates_found:
        print(f"   WARNING: Found {len(duplicates_found)} duplicate vertex pairs:")
        for i, j, v in duplicates_found[:10]:  # Show first 10
            print(f"      Vertices {i} and {j} are both at {v}")
        if len(duplicates_found) > 10:
            print(f"      ... and {len(duplicates_found) - 10} more duplicates")
        print(f"   Note: Deduplication will be performed in Step 5.7 after connectivity rebuild")
    else:
        print(f"   ✓ All {len(selected_vertices)} vertices are unique")
    
    # Display sample vertices to confirm rounding
    print("\n   Sample vertices (first 5, after rounding):")
    for idx in range(min(5, len(selected_vertices))):
        v = selected_vertices[idx]
        print(f"      Vertex {idx}: [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}] mm")
    
    # =========================================================================
    # STEP 5.6: Rebuild connectivity matrices for deduplicated vertices
    # =========================================================================
    print("\n[STEP 5.6] Rebuilding connectivity matrices for deduplicated vertices...")
    print("   Mapping old vertices to deduplicated rounded vertices (tolerance ±0.05mm)")
    
    # Helper function to remap connectivity matrix with tolerance
    def remap_connectivity_matrix(old_matrix, old_vertices, new_vertices, 
                                   view_name, x_idx, y_idx):
        """
        Remap connectivity from old matrix to new deduplicated vertices.
        Uses ±0.05mm tolerance when matching old vertices to new vertices.
        """
        N_old = old_matrix.shape[0]
        N_new = len(new_vertices)
        
        print(f"   {view_name}: Remapping {N_old} vertices to {N_new} vertices")
        
        # Create new connectivity matrix [N_new x (N_new + 3)]
        new_matrix = np.zeros((N_new, N_new + 3), dtype=float)
        
        # Fill vertex indices and 2D coordinates
        for i in range(N_new):
            new_matrix[i, 0] = i
            new_matrix[i, 1] = new_vertices[i, x_idx]
            new_matrix[i, 2] = new_vertices[i, y_idx]
        
        # Create mapping: old_vertex_index -> list of new_vertex_indices
        # Use ±0.05mm tolerance for matching 2D coordinates
        old_to_new_vertex_map = {}
        for old_idx in range(N_old):
            old_coord_2d = old_matrix[old_idx, 1:3]
            matches = []
            for new_idx in range(N_new):
                new_coord_2d = np.array([new_vertices[new_idx, x_idx],
                                          new_vertices[new_idx, y_idx]])
                # Use ±0.05mm tolerance (but compare with rounded coordinates)
                # The old matrix coords may not be rounded, so we need tolerance
                if np.allclose(old_coord_2d, new_coord_2d, atol=0.1):
                    matches.append(new_idx)
            if matches:
                old_to_new_vertex_map[old_idx] = matches
        
        print(f"      Mapped {len(old_to_new_vertex_map)} old vertices to new vertices")
        
        # Debug: Check for unmapped vertices
        unmapped = N_old - len(old_to_new_vertex_map)
        if unmapped > 0:
            print(f"      WARNING: {unmapped} old vertices had no match in new vertices!")
        
        # Map connectivity values
        # Only process upper triangle (old_col_idx > old_row_idx) to avoid double counting
        # since connectivity matrices are symmetric
        edges_mapped = 0
        for old_row_idx in range(N_old):
            new_row_indices = old_to_new_vertex_map.get(old_row_idx, [])
            if not new_row_indices:
                continue
            
            # Only process upper triangle: old_col_idx > old_row_idx
            for old_col_idx in range(old_row_idx + 1, N_old):
                conn_col = 3 + old_col_idx
                if conn_col >= old_matrix.shape[1]:
                    continue
                
                conn_value = old_matrix[old_row_idx, conn_col]
                if conn_value == 1:
                    new_col_indices = old_to_new_vertex_map.get(old_col_idx, [])
                    
                    # Update ALL combinations of mapped vertices
                    for new_row in new_row_indices:
                        for new_col in new_col_indices:
                            if new_row != new_col:
                                # Update both (new_row, new_col) and (new_col, new_row) for symmetry
                                dest_col_forward = 3 + new_col
                                dest_col_reverse = 3 + new_row
                                
                                if dest_col_forward < new_matrix.shape[1]:
                                    new_matrix[new_row, dest_col_forward] = 1
                                    edges_mapped += 1
                                
                                if dest_col_reverse < new_matrix.shape[1]:
                                    new_matrix[new_col, dest_col_reverse] = 1
        
        print(f"      Mapped {edges_mapped} edge connections (symmetric pairs)")
        return new_matrix
    
    # Rebuild each view's connectivity matrix
    print("\n   Remapping top view matrix...")
    new_top_matrix = remap_connectivity_matrix(
        top_matrix, original_vertices, selected_vertices,
        "Top view", x_idx=0, y_idx=1
    )
    
    print("\n   Remapping front view matrix...")
    new_front_matrix = remap_connectivity_matrix(
        front_matrix, original_vertices, selected_vertices,
        "Front view", x_idx=0, y_idx=2
    )
    
    print("\n   Remapping side view matrix...")
    new_side_matrix = remap_connectivity_matrix(
        side_matrix, original_vertices, selected_vertices,
        "Side view", x_idx=1, y_idx=2
    )
    
    # Replace old matrices
    top_matrix = new_top_matrix
    front_matrix = new_front_matrix
    side_matrix = new_side_matrix
    
    # Rebuild merged connectivity matrix
    print("\n   Rebuilding merged connectivity matrix...")
    N = len(selected_vertices)
    merged_conn = np.zeros((N, N), dtype=int)
    
    # Merge connectivity from all three views
    for i in range(N):
        for j in range(N):
            col_idx = 3 + j
            top_conn = int(top_matrix[i, col_idx]) if col_idx < top_matrix.shape[1] else 0
            front_conn = int(front_matrix[i, col_idx]) if col_idx < front_matrix.shape[1] else 0
            side_conn = int(side_matrix[i, col_idx]) if col_idx < side_matrix.shape[1] else 0
            merged_conn[i, j] = top_conn + front_conn + side_conn
    
    # Upgrade conn=2 to conn=3 for edges perpendicular to one view
    # (edges where vertices share two coincident coordinates)
    print("\n   Upgrading conn=2 edges where vertices share 2 coordinates...")
    upgraded_count = 0
    for i in range(N):
        for j in range(i+1, N):  # Only process upper triangle
            if merged_conn[i, j] == 2:
                v_i = selected_vertices[i]
                v_j = selected_vertices[j]
                
                # Count how many coordinates are coincident (within tolerance)
                coincident_coords = 0
                if np.abs(v_i[0] - v_j[0]) < 0.15:  # X coordinate
                    coincident_coords += 1
                if np.abs(v_i[1] - v_j[1]) < 0.15:  # Y coordinate
                    coincident_coords += 1
                if np.abs(v_i[2] - v_j[2]) < 0.15:  # Z coordinate
                    coincident_coords += 1
                
                # If 2 coordinates are coincident, the edge is perpendicular to one view
                if coincident_coords == 2:
                    merged_conn[i, j] = 3
                    merged_conn[j, i] = 3  # Update symmetric cell
                    upgraded_count += 1
    
    print(f"   Upgraded {upgraded_count} edges from conn=2 to conn=3")
    
    # Report final connectivity stats
    edges_conn3 = np.sum(merged_conn == 3) // 2
    edges_conn2 = np.sum(merged_conn == 2) // 2
    edges_conn1 = np.sum(merged_conn == 1) // 2
    print(f"   Final connectivity: {edges_conn3} edges with conn=3, {edges_conn2} with conn=2, {edges_conn1} with conn=1")
    
    # =========================================================================
    # STEP 5.7: Deduplicate vertices after remapping
    # =========================================================================
    print("\n[STEP 5.7] Deduplicating vertices (merging vertices closer than 0.05mm)...")
    
    unique_vertices = []
    vertex_mapping = {}  # Maps old index -> new index
    
    for i in range(len(selected_vertices)):
        v = selected_vertices[i]
        # Check if this vertex is close to any existing unique vertex
        found_match = False
        for j, uv in enumerate(unique_vertices):
            if np.allclose(v, uv, atol=0.05):  # Within 0.05mm
                vertex_mapping[i] = j
                found_match = True
                break
        
        if not found_match:
            vertex_mapping[i] = len(unique_vertices)
            unique_vertices.append(v)
    
    print(f"   Reduced from {len(selected_vertices)} to {len(unique_vertices)} unique vertices")
    
    # Update vertices and merge connectivity
    if len(unique_vertices) < len(selected_vertices):
        print(f"   Merging connectivity for deduplicated vertices...")
        
        # Create new merged connectivity matrix
        N_unique = len(unique_vertices)
        new_merged_conn = np.zeros((N_unique, N_unique), dtype=int)
        
        # Map old connectivity to new connectivity
        for i in range(len(selected_vertices)):
            for j in range(len(selected_vertices)):
                if merged_conn[i, j] > 0:
                    new_i = vertex_mapping[i]
                    new_j = vertex_mapping[j]
                    if new_i != new_j:  # No self-edges
                        new_merged_conn[new_i, new_j] = max(
                            new_merged_conn[new_i, new_j],
                            merged_conn[i, j]
                        )
        
        # Update arrays
        selected_vertices = np.array(unique_vertices)
        merged_conn = new_merged_conn
        N = len(selected_vertices)  # Update N after deduplication
        
        edges_conn3_after = np.sum(merged_conn == 3) // 2
        edges_conn2_after = np.sum(merged_conn == 2) // 2
        edges_conn1_after = np.sum(merged_conn == 1) // 2
        print(f"   After deduplication: {edges_conn3_after} edges with conn=3, {edges_conn2_after} with conn=2, {edges_conn1_after} with conn=1")
    
    # Display final selected vertices (after deduplication)
    print(f"\n{'='*70}")
    print(f"FINAL SELECTED VERTICES (After Step 3)")
    print(f"{'='*70}")
    print(f"Total vertices: {len(selected_vertices)}")
    print(f"\nVertex list (index: [x, y, z] in mm):")
    for idx, v in enumerate(selected_vertices):
        print(f"  {idx:3d}: [{v[0]:7.1f}, {v[1]:7.1f}, {v[2]:7.1f}]")
    
    # =========================================================================
    # STEP 6: EXTRACTING POLYGON FACES FROM CONNECTIVITY
    # =========================================================================
    
    # Extract polygon faces from connectivity
    print("\n[DEBUG] About to call extract_polygon_faces_from_connectivity...")
    print(f"[DEBUG] selected_vertices shape: {selected_vertices.shape}")
    print(f"[DEBUG] merged_conn shape: {merged_conn.shape}")
    print(f"[DEBUG] tolerance: {args.tolerance}")
    print(f"[DEBUG] Sample vertices (first 5):")
    for i in range(min(5, len(selected_vertices))):
        print(f"  Vertex {i}: {selected_vertices[i]}")
    print(f"[DEBUG] Sample connectivity (first 5x5):")
    print(merged_conn[:5, :5])
    
    extracted_faces = extract_polygon_faces_from_connectivity(
        selected_vertices, merged_conn, tolerance=args.tolerance)
    
    print(f"\n[CRITICAL] Extracted {len(extracted_faces)} faces")
    
    # =========================================================================
    # STEP 6.1: Check for and split non-planar polygons
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 6.1] CHECKING FOR NON-PLANAR POLYGONS")
    print("="*70)
    
    split_faces = []
    faces_to_remove = []
    
    for face_idx, face in enumerate(extracted_faces):
        if not isinstance(face, dict) or 'vertices' not in face:
            continue
        
        polygon = face['vertices']
        if len(polygon) < 3:
            continue
        
        outer_verts_3d = [selected_vertices[v_idx] for v_idx in polygon]
        
        # Fit plane to polygon vertices using SVD
        centroid = np.mean(outer_verts_3d, axis=0)
        centered = np.array(outer_verts_3d) - centroid
        _, _, Vt = np.linalg.svd(centered)
        computed_normal = Vt[-1]
        computed_d = -np.dot(computed_normal, centroid)
        
        # Check planarity - max distance from best-fit plane
        max_dist = 0.0
        for v in outer_verts_3d:
            dist = abs(np.dot(computed_normal, v) + computed_d)
            max_dist = max(max_dist, dist)
        
        # Check if polygon is non-planar beyond tolerance (0.1mm)
        if max_dist > 0.1:
            print(f"\n   Face {face_idx + 1}: NON-PLANAR polygon detected, " +
                  f"max dist = {max_dist:.6f}mm")
            print(f"      Original polygon: {polygon}")
            
            # Find alternate edges from connectivity matrix
            alternate_edges = []
            
            for i, v_idx in enumerate(polygon):
                prev_idx = polygon[(i - 1) % len(polygon)]
                next_idx = polygon[(i + 1) % len(polygon)]
                
                for j, v_other in enumerate(polygon):
                    if v_other == v_idx or v_other == prev_idx or v_other == next_idx:
                        continue
                    
                    if merged_conn[v_idx, v_other] > 0:
                        edge = tuple(sorted([v_idx, v_other]))
                        if edge not in alternate_edges:
                            alternate_edges.append(edge)
            
            if len(alternate_edges) > 0:
                print(f"      Found {len(alternate_edges)} alternate edge(s): " +
                      f"{alternate_edges}")
                
                # Test each alternate edge for planarity improvement
                best_edge = None
                best_planarity_score = float('inf')
                
                for edge in alternate_edges:
                    v1, v2 = edge
                    pos1 = polygon.index(v1)
                    pos2 = polygon.index(v2)
                    
                    if pos1 > pos2:
                        pos1, pos2 = pos2, pos1
                        v1, v2 = v2, v1
                    
                    poly1_verts = polygon[pos1:pos2+1]
                    poly2_verts = polygon[pos2:] + polygon[:pos1+1]
                    
                    if len(poly1_verts) >= 3 and len(poly2_verts) >= 3:
                        poly1_coords = np.array([selected_vertices[v]
                                                 for v in poly1_verts])
                        poly2_coords = np.array([selected_vertices[v]
                                                 for v in poly2_verts])
                        
                        # Fit planes
                        c1 = np.mean(poly1_coords, axis=0)
                        _, _, V1 = np.linalg.svd(poly1_coords - c1)
                        n1 = V1[-1]
                        d1 = -np.dot(n1, c1)
                        
                        c2 = np.mean(poly2_coords, axis=0)
                        _, _, V2 = np.linalg.svd(poly2_coords - c2)
                        n2 = V2[-1]
                        d2 = -np.dot(n2, c2)
                        
                        max_dev1 = max([abs(np.dot(n1, v) + d1)
                                       for v in poly1_coords])
                        max_dev2 = max([abs(np.dot(n2, v) + d2)
                                       for v in poly2_coords])
                        
                        planarity_score = max(max_dev1, max_dev2)
                        
                        if planarity_score < best_planarity_score:
                            best_planarity_score = planarity_score
                            best_edge = (v1, v2, pos1, pos2,
                                        poly1_verts, poly2_verts,
                                        n1, d1, n2, d2)
                
                if best_edge is not None and best_planarity_score < max_dist:
                    v1, v2, pos1, pos2, poly1_verts, poly2_verts, n1, d1, n2, d2 = best_edge
                    
                    print(f"      Best edge: {v1}-{v2} " +
                          f"(planarity score: {best_planarity_score:.3f}mm)")
                    print(f"        Face {face_idx + 1}a: {poly1_verts}")
                    print(f"        Face {face_idx + 1}b: {poly2_verts}")
                    
                    # Mark original face for removal
                    faces_to_remove.append(face_idx)
                    
                    # Create two new faces
                    new_face1 = {
                        'vertices': poly1_verts,
                        'normal': tuple(n1),
                        'd': d1,
                        'holes': []
                    }
                    new_face2 = {
                        'vertices': poly2_verts,
                        'normal': tuple(n2),
                        'd': d2,
                        'holes': []
                    }
                    split_faces.append(new_face1)
                    split_faces.append(new_face2)
    
    # Remove non-planar faces and add split faces
    if len(faces_to_remove) > 0:
        print(f"\n   Removing {len(faces_to_remove)} non-planar face(s)")
        print(f"   Adding {len(split_faces)} split face(s)")
        
        # Remove faces in reverse order to maintain indices
        for face_idx in sorted(faces_to_remove, reverse=True):
            del extracted_faces[face_idx]
        
        # Add split faces
        extracted_faces.extend(split_faces)
        
        print(f"\n[CRITICAL] After non-planar splitting: " +
              f"{len(extracted_faces)} faces")
    else:
        print(f"\n   No non-planar polygons found (all within 0.1mm tolerance)")
    
    # =========================================================================
    # STEP 6.2: Process boundary edges to find missing polygons/holes
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 6.2] PROCESSING BOUNDARY EDGES")
    print("="*70)
    
    extracted_faces, rejected_interior_edges = process_boundary_edges(
        extracted_faces, selected_vertices, merged_conn,
        tolerance=args.tolerance)
    
    print(f"\n[CRITICAL] After boundary processing: " +
          f"{len(extracted_faces)} faces")
    
    if rejected_interior_edges:
        print(f"[CRITICAL] {len(rejected_interior_edges)} edges from " +
              f"rejected interior polygons will be filtered from free edge warnings")
    
    # ADDED: Analyze which vertices appear in fewer than 3 faces
    print(f"\n[POLY FORM] Analyzing vertex usage across faces...")
    vertex_face_count = {}
    for v_idx in range(len(selected_vertices)):
        vertex_face_count[v_idx] = 0
    
    for face_idx, face in enumerate(extracted_faces):
        if isinstance(face, dict) and 'vertices' in face:
            for v_idx in face['vertices']:
                vertex_face_count[v_idx] += 1
    
    vertices_less_than_3 = [v_idx for v_idx, count in vertex_face_count.items() if count < 3]
    print(f"[POLY FORM] Step 4 Analysis: {len(vertices_less_than_3)} vertices have < 3 faces")
    if len(vertices_less_than_3) > 0:
        print(f"[POLY FORM] Vertices with < 3 faces:")
        for v_idx in sorted(vertices_less_than_3):
            v = selected_vertices[v_idx]
            count = vertex_face_count[v_idx]
            print(f"  Vertex {v_idx}: [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}] mm - appears in {count} face(s)")
    
    # Debug: Check if faces have vertices
    if len(extracted_faces) > 0:
        print(f"[DEBUG] First face type: {type(extracted_faces[0])}")
        print(f"[DEBUG] First face has {len(extracted_faces[0].get('vertices', []))} vertices")
        print(f"[DEBUG] Face keys: {list(extracted_faces[0].keys())}")
        if 'vertices' in extracted_faces[0]:
            print(f"[DEBUG] First face vertices: {extracted_faces[0]['vertices']}")
    else:
        print("[CRITICAL WARNING] No faces were extracted!")
        print("[DEBUG] This means extract_polygon_faces_from_connectivity returned empty list")
    
    # Store original face count for later reference
    original_face_count = len(extracted_faces)
    
    # =========================================================================
    # STEP 6.5: Determine correct face orientations using ray-casting
    # =========================================================================
    print("\n[STEP 6.5] Determining face orientations (CCW relative to outward normal)...")
    
    # Find overall bounding box
    all_points = selected_vertices
    bbox_min = np.min(all_points, axis=0)
    bbox_max = np.max(all_points, axis=0)
    
    # IMPROVED: Use midpoint on bottom face (Z=min) as reference point
    # This is more reliable than bbox center for concave shapes
    # Find all vertices on the bottom face (Z = bbox_min[2])
    z_min = bbox_min[2]
    bottom_vertices = [v for v in all_points if abs(v[2] - z_min) < 1e-6]
    
    if len(bottom_vertices) > 0:
        # Use centroid of bottom face vertices as reference point
        reference_point = np.mean(bottom_vertices, axis=0)
        print(f"   Reference point: Bottom face centroid at "
              f"({reference_point[0]:.2f}, {reference_point[1]:.2f}, "
              f"{reference_point[2]:.2f})")
        print(f"   ({len(bottom_vertices)} vertices on bottom face)")
    else:
        # Fallback to bbox center if no bottom face found
        reference_point = (bbox_min + bbox_max) / 2.0
        print(f"   Reference point: Bbox center (fallback) at "
              f"({reference_point[0]:.2f}, {reference_point[1]:.2f}, "
              f"{reference_point[2]:.2f})")
    
    print(f"   Bounding box: "
          f"X=[{bbox_min[0]:.2f}, {bbox_max[0]:.2f}], "
          f"Y=[{bbox_min[1]:.2f}, {bbox_max[1]:.2f}], "
          f"Z=[{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]")
    
    # =========================================================================
    # STEP 8: Build solid from face-wire topology using OCC stitching
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 8] BUILDING SOLID FROM WIRE-BASED FACES")
    print("="*70)
    
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
        
        # Ray from reference point to face center
        ray_dir = face_center - reference_point
        ray_len = np.linalg.norm(ray_dir)
        if ray_len > 1e-6:
            ray_dir = ray_dir / ray_len
        else:
            # Face center coincides with reference point - skip
            print(f"   WARNING: Face {face_num} center coincides "
                  f"with reference point")
            continue
        
        # Check if normal points away from reference point (outward)
        # Dot > 0: normal points same direction as ray (outward)
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
    
    # Debug: Check what we're passing to the plot function
    print(f"[DEBUG] Passing {len(extracted_faces)} extracted faces to plot")
    print(f"[DEBUG] Passing {len(face_polygons)} original faces to plot")
    
    # Debug: Check structure of face_polygons
    if len(face_polygons) > 0:
        first_face = face_polygons[0]
        if isinstance(first_face, dict):
            print(f"[DEBUG] Original face structure has keys: {list(first_face.keys())}")
    
    plot_extracted_polygon_faces(extracted_faces, selected_vertices, 
                                 face_polygons)
    print("[STEP 6] Plot closed.")
    
    # No dummy faces to remove
    confirmed_dummy_faces = []
    
    # =========================================================================
    # STEP 8: Build solid from face-wire topology using OCC stitching
    # =========================================================================
    print("\n" + "="*70)
    print("[STEP 8] BUILDING SOLID FROM WIRE-BASED FACES")
    print("="*70)
    
    print("\n[STEP 8.1] Creating OCC faces from wires...")
    
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    )
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties
    
    occ_faces_list = []
    occ_hole_faces_list = []  # Store hole faces separately for second sewing stage
    occ_hole_adjacent_faces = []  # Faces that share edges with holes
    faces_with_holes = set()  # Track which face indices have holes
    
    # Collect split faces separately to avoid infinite loop
    split_faces = []
    
    # Helper function to compute signed area of 3D polygon
    def compute_signed_area_3d(vertices_3d, normal):
        """
        Compute signed area of a 3D polygon projected onto a plane.
        
        Args:
            vertices_3d: List of 3D coordinates [(x,y,z), ...]
            normal: Normal vector of the plane [nx, ny, nz]
        
        Returns:
            Signed area (positive or negative based on winding)
        """
        if len(vertices_3d) < 3:
            return 0.0
        
        # Convert to numpy array
        vertices_3d = np.array(vertices_3d)
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)
        
        # Choose projection axes based on dominant normal component
        abs_normal = np.abs(normal)
        drop_axis = np.argmax(abs_normal)
        keep_axes = [i for i in range(3) if i != drop_axis]
        
        # Project to 2D
        vertices_2d = vertices_3d[:, keep_axes]
        
        # Shoelace formula for signed area
        x = vertices_2d[:, 0]
        y = vertices_2d[:, 1]
        signed_area = 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        # Adjust sign based on which axis was dropped
        if drop_axis == 1:  # Dropped Y axis (front view)
            signed_area = -signed_area
        
        return signed_area
    
    # Pass merged_conn for connectivity checking during non-planar detection
    
    # =========================================================================
    # WINDING DIRECTION CONVENTION (from Make_Wire_Box.py)
    # =========================================================================
    # All wires (outer boundaries AND holes) must wind in the SAME direction
    # when viewed from outside the solid (CCW for +Z faces, etc.)
    # 
    # - Outer wire: CCW when viewed from outside
    # - Hole wire: CCW when viewed from outside (SAME as outer, not opposite!)
    # 
    # This creates voids (empty space) for holes. The key insight from
    # Make_Wire_Box.py is that holes are added as INTERNAL WIRES to faces,
    # not as separate face objects.
    # =========================================================================
    
    for face_idx, face in enumerate(extracted_faces):
        if not (isinstance(face, dict) and 'vertices' in face):
            continue
        
        # Skip dummy faces
        if face_idx in confirmed_dummy_faces:
            print(f"   Skipping dummy face {face_idx + 1}")
            continue
        
        # Get vertices for outer wire
        polygon = face['vertices']
        outer_verts_3d = [selected_vertices[v_idx] for v_idx in polygon]
        
        # Get holes if any
        holes = face.get('holes', [])
        
        # Get alternates if any (these will be treated as separate faces)
        alternates = face.get('alternates', [])
        
        # ALWAYS recompute plane equation from actual vertex coordinates
        # This ensures the plane equation accurately represents the vertices
        if len(outer_verts_3d) >= 3:
            centroid = np.mean(outer_verts_3d, axis=0)
            centered = np.array(outer_verts_3d) - centroid
            
            # SVD to find best-fit plane
            _, _, Vt = np.linalg.svd(centered)
            computed_normal = Vt[-1]
            
            # Preserve orientation if we had a stored normal
            stored_normal = face.get('normal', None)
            if stored_normal is not None:
                if np.dot(computed_normal, stored_normal) < 0:
                    computed_normal = -computed_normal
            
            # Calculate d from plane equation: n·p + d = 0
            computed_d = -np.dot(computed_normal, centroid)
            
            # Verify fit quality - check each vertex distance to plane
            vertex_distances = []
            max_dist = 0.0
            for v_idx, v in zip(polygon, outer_verts_3d):
                dist = abs(np.dot(computed_normal, v) + computed_d)
                vertex_distances.append((v_idx, dist))
                max_dist = max(max_dist, dist)
            
            # Update face with accurate plane equation
            face['normal'] = computed_normal
            face['d'] = computed_d
            
            # Check planarity for warning purposes only
            if max_dist > 0.01:
                # Small deviation - note it but proceed
                print(f"   Face {face_idx + 1}: Small planarity deviation "
                      f"{max_dist:.6f}mm (acceptable)")
        
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
                print(f"   ERROR: Failed to build wire for face " +
                      f"{face_idx + 1}")
                continue
            
            outer_wire = wire_builder.Wire()
            
            # Build face from outer wire using computed plane equation
            face_normal = face.get('normal', None)
            face_d = face.get('d', None)
            if face_normal is not None and face_d is not None:
                # Create a plane from the face equation
                from OCC.Core.gp import gp_Pln, gp_Dir, gp_Pnt
                
                # Plane point: use centroid of vertices
                centroid = np.mean(outer_verts_3d, axis=0)
                plane_pnt = gp_Pnt(centroid[0], centroid[1], centroid[2])
                plane_dir = gp_Dir(face_normal[0], face_normal[1],
                                   face_normal[2])
                plane = gp_Pln(plane_pnt, plane_dir)
                
                # Build face on the plane with the wire
                face_builder = BRepBuilderAPI_MakeFace(plane, outer_wire)
            else:
                # No plane info, let OCC figure it out
                face_builder = BRepBuilderAPI_MakeFace(outer_wire)
            
            if not face_builder.IsDone():
                print(f"   ERROR: Failed to build face from wire for " +
                      f"face {face_idx + 1}")
                continue
            
            # Add holes as internal wires to the face
            # CRITICAL: Holes must wind SAME direction as outer (both CCW)
            # This creates voids, not filled areas
            if len(holes) > 0:
                faces_with_holes.add(face_idx)
                print(f"   Face {face_idx + 1}: Adding {len(holes)} hole(s) as internal wires")
                
                # Determine outer boundary winding
                outer_normal = face.get('normal', np.array([0, 0, 1]))
                outer_signed_area = compute_signed_area_3d(
                    outer_verts_3d, outer_normal)
                
                for hole_idx, hole_vertex_indices in enumerate(holes):
                    try:
                        hole_verts_3d = [selected_vertices[v_idx]
                                         for v_idx in hole_vertex_indices]
                        
                        # Check hole winding - must be SAME as outer
                        hole_signed_area = compute_signed_area_3d(
                            hole_verts_3d, outer_normal)
                        
                        # If opposite sign to outer, reverse the hole to match
                        if (outer_signed_area > 0 and hole_signed_area < 0) or \
                           (outer_signed_area < 0 and hole_signed_area > 0):
                            print(f"   Hole {hole_idx + 1}: Reversing to match outer winding (both CCW)")
                            hole_verts_3d = list(reversed(hole_verts_3d))
                            hole_vertex_indices = list(reversed(hole_vertex_indices))
                        else:
                            print(f"   Hole {hole_idx + 1}: Winding matches outer (both CCW)")
                        
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
                        
                        if not hole_wire_builder.IsDone():
                            print(f"   WARNING: Failed to build wire for hole {hole_idx + 1}")
                            continue
                        
                        hole_wire = hole_wire_builder.Wire()
                        
                        # Add hole wire to face as internal boundary
                        face_builder.Add(hole_wire)
                        print(f"   Added hole {hole_idx + 1} as internal wire ({len(hole_verts_3d)} vertices)")
                        
                        # Store hole info for creating wall faces later
                        occ_hole_faces_list.append({
                            'parent_idx': face_idx,
                            'hole_idx': hole_idx,
                            'vertex_indices': hole_vertex_indices,
                            'vertices_3d': hole_verts_3d
                        })
                        
                    except Exception as e:
                        print(f"   ERROR adding hole {hole_idx + 1}: {e}")
            
            # Get the built face (with holes as internal wires)
            occ_face = face_builder.Face()
            occ_faces_list.append(occ_face)
            print(f"   Successfully created face {face_idx + 1}")
            
        except Exception as e:
            print(f"   ERROR building face {face_idx + 1}: {e}")
        
        # Process alternates as separate faces (they are NOT holes)
        if alternates:
            print(f"   Face {face_idx + 1}: Processing {len(alternates)} alternate polygon(s) as separate faces")
            for alt_idx, alternate in enumerate(alternates):
                try:
                    # Get alternate vertices
                    if isinstance(alternate, dict) and 'vertices' in alternate:
                        alt_polygon = alternate['vertices']
                    else:
                        alt_polygon = alternate
                    
                    alt_verts_3d = [selected_vertices[v_idx] for v_idx in alt_polygon]
                    
                    # Build OCC wire from alternate vertices
                    alt_wire_edges = []
                    for i in range(len(alt_verts_3d)):
                        p1 = gp_Pnt(*alt_verts_3d[i])
                        p2 = gp_Pnt(*alt_verts_3d[(i + 1) % len(alt_verts_3d)])
                        edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
                        alt_wire_edges.append(edge)
                    
                    # Build wire from edges
                    alt_wire_builder = BRepBuilderAPI_MakeWire()
                    for edge in alt_wire_edges:
                        alt_wire_builder.Add(edge)
                    
                    if not alt_wire_builder.IsDone():
                        print(f"   WARNING: Failed to build wire for alternate {alt_idx + 1} of face {face_idx + 1}")
                        continue
                    
                    alt_wire = alt_wire_builder.Wire()
                    
                    # Build face from alternate wire
                    alt_face_builder = BRepBuilderAPI_MakeFace(alt_wire)
                    
                    if not alt_face_builder.IsDone():
                        print(f"   WARNING: Failed to build face for alternate {alt_idx + 1} of face {face_idx + 1}")
                        continue
                    
                    # Get the built alternate face
                    alt_occ_face = alt_face_builder.Face()
                    occ_faces_list.append(alt_occ_face)
                    print(f"   Successfully created alternate face {face_idx + 1}.{alt_idx + 1} ({len(alt_polygon)} vertices)")
                    
                except Exception as e:
                    print(f"   ERROR building alternate {alt_idx + 1} for face {face_idx + 1}: {e}")
    
    # Process any split faces that were created from non-planar polygons
    if split_faces:
        print(f"\n[STEP 8.1.5] Processing {len(split_faces)} split faces from non-planar polygons...")
        # Note: Don't add these back to extracted_faces to avoid infinite loop
        # Just note them for informational purposes
        print(f"   Split faces were already incorporated into the face list")
    
    print(f"\n   Created {len(occ_faces_list)} outer faces from extracted faces")
    print(f"   Created {len(occ_hole_faces_list)} hole faces")
    
    # Step 8.2: Create hole wall faces to connect holes on parallel faces
    # TODO: Implement hole wall face creation (similar to Make_Wire_Box.py)
    # For now, we just have faces with holes as internal wires
    print(f"\n[STEP 8.2] Creating hole wall faces...")
    print(f"   Note: Hole wall face creation not yet implemented")
    print(f"   Holes are currently internal wires only (no connecting walls)")
    
    # Step 8.3: Single-stage sewing - sew all outer faces together
    print(f"\n[STEP 8.3] Stitching all faces into solid (single-stage)...")
    print(f"   Face summary:")
    
    # Print summary of all faces created (with vertex information)
    face_counter = 0
    for face_idx, face in enumerate(extracted_faces):
        if not (isinstance(face, dict) and 'vertices' in face):
            continue
        
        if face_idx in confirmed_dummy_faces:
            continue
        
        face_counter += 1
        vertices = face['vertices']
        holes = face.get('holes', [])
        alternates = face.get('alternates', [])
        
        # Print main face
        print(f"     Face {face_counter}: {len(vertices)} vertices: {vertices}")
        
        # Print holes if any
        if holes:
            for hole_idx, hole in enumerate(holes):
                print(f"       Hole {hole_idx + 1}: {len(hole)} vertices: {hole}")
    
    print(f"   Total outer faces: {len(occ_faces_list)}")
    print(f"   Hole information stored: {len(occ_hole_faces_list)} holes")
    print(f"     (Holes are internal wires, not separate faces)")
    print()
    
    solid_is_valid = False  # Track solid validity for Step 8.3
    free_edge_polygons_to_delete = []  # Will be populated after first sewing
    faces_were_merged = False  # Track if we need to re-sew after merging
    
    if len(occ_faces_list) == 0:
        print("   ERROR: No valid faces to stitch")
        occ_solid = None
    else:
        # Single-stage sewing: Add ONLY outer faces (holes are already internal wires)
        print(f"\n   Sewing {len(occ_faces_list)} outer faces...")
        print(f"   (Note: {len(occ_hole_faces_list)} holes are internal wires, not sewn separately)")
        sewing = BRepBuilderAPI_Sewing()
        sewing.SetTolerance(0.1005)
        
        # Add all outer faces (which already have holes as internal wires)
        for occ_face in occ_faces_list:
            sewing.Add(occ_face)
        
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
            free_edges_list = []
            
            while edge_explorer.More() and free_edge_count < 20:  # Limit to first 20
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
                    
                    # Find matching vertex indices
                    coord1 = np.array([p1.X(), p1.Y(), p1.Z()])
                    coord2 = np.array([p2.X(), p2.Y(), p2.Z()])
                    
                    v1_idx = None
                    v2_idx = None
                    for v_idx, v_coord in enumerate(selected_vertices):
                        dist1 = np.linalg.norm(v_coord - coord1)
                        dist2 = np.linalg.norm(v_coord - coord2)
                        if dist1 < 1e-6 and v1_idx is None:
                            v1_idx = v_idx
                        if dist2 < 1e-6 and v2_idx is None:
                            v2_idx = v_idx
                    
                    v1_str = f"v{v1_idx}" if v1_idx is not None else "?"
                    v2_str = f"v{v2_idx}" if v2_idx is not None else "?"
                    print(f"     Free edge {free_edge_count+1}: {v1_str}-{v2_str} "
                          f"({p1.X():.2f},{p1.Y():.2f},{p1.Z():.2f}) <-> "
                          f"({p2.X():.2f},{p2.Y():.2f},{p2.Z():.2f})")
                    
                    if v1_idx is not None and v2_idx is not None:
                        free_edges_list.append((v1_idx, v2_idx))
                    
                    free_edge_count += 1
                
                edge_explorer.Next()
            
            # Print summary
            if len(free_edges_list) > 0:
                # Filter out rejected interior edges (from boundary processing)
                if 'rejected_interior_edges' in locals() and rejected_interior_edges:
                    rejected_edge_set = set(rejected_interior_edges)
                    filtered_free_edges = []
                    rejected_count = 0
                    
                    for v1, v2 in free_edges_list:
                        edge = tuple(sorted([v1, v2]))
                        if edge in rejected_edge_set:
                            rejected_count += 1
                        else:
                            filtered_free_edges.append((v1, v2))
                    
                    if rejected_count > 0:
                        print(f"\n   Filtered out {rejected_count} free edge(s) " +
                              f"from rejected interior polygons")
                        print(f"   (These edges correctly represent interior faces)")
                    
                    free_edges_list = filtered_free_edges
                
                if len(free_edges_list) > 0:
                    print(f"\n   Summary: {len(free_edges_list)} free edges found:")
                    print(f"   Edges: {free_edges_list}")
                    
                    # Get unique vertices involved
                    unique_verts = set()
                    for v1, v2 in free_edges_list:
                        unique_verts.add(v1)
                        unique_verts.add(v2)
                    print(f"   Involves {len(unique_verts)} unique vertices: "
                          f"{sorted(unique_verts)}")
                    
                    # Check if free edges form closed polygons (faces to delete)
                    print(f"\n   Analyzing free edges for closed polygons...")
                    
                    # Build adjacency graph from free edges
                    from collections import defaultdict
                    edge_graph = defaultdict(list)
                    for v1, v2 in free_edges_list:
                        edge_graph[v1].append(v2)
                        edge_graph[v2].append(v1)
                    
                    # Find closed loops (polygons)
                    visited_edges = set()
                    closed_polygons = []
                    
                    for start_v in unique_verts:
                        if start_v in edge_graph and len(edge_graph[start_v]) > 0:
                            # Try to trace a path from this vertex
                            path = [start_v]
                            current = start_v
                            
                            while True:
                                # Find next unvisited neighbor
                                next_v = None
                                for neighbor in edge_graph[current]:
                                    edge_key = (min(current, neighbor),
                                               max(current, neighbor))
                                    if edge_key not in visited_edges:
                                        next_v = neighbor
                                        visited_edges.add(edge_key)
                                        break
                                
                                if next_v is None:
                                    break  # No more unvisited edges
                                
                                if next_v == start_v:
                                    # Closed loop found!
                                    if len(path) >= 3:
                                        closed_polygons.append(path[:])
                                    break
                                
                                path.append(next_v)
                                current = next_v
                                
                                # Safety check
                                if len(path) > len(unique_verts):
                                    break
                    
                    if closed_polygons:
                        print(f"   Found {len(closed_polygons)} closed polygon(s) "
                              f"formed by free edges:")
                        
                        # Check each closed polygon for planarity
                        for poly_idx, poly in enumerate(closed_polygons):
                            print(f"     Polygon {poly_idx+1}: {len(poly)} "
                                  f"vertices: {poly}")
                            
                            # Check planarity of this polygon
                            if len(poly) >= 3:
                                outer_verts_3d = [selected_vertices[v_idx] for v_idx in poly]
                            
                            # Fit plane using SVD
                            centroid = np.mean(outer_verts_3d, axis=0)
                            centered = np.array(outer_verts_3d) - centroid
                            _, _, Vt = np.linalg.svd(centered)
                            computed_normal = Vt[-1]
                            computed_d = -np.dot(computed_normal, centroid)
                            
                            # Check max distance from fitted plane
                            max_dist = 0.0
                            for v in outer_verts_3d:
                                dist = abs(np.dot(computed_normal, v) + computed_d)
                                max_dist = max(max_dist, dist)
                            
                            if max_dist > 0.1:
                                print(f"       ⚠️  NON-PLANAR polygon detected! "
                                      f"max deviation = {max_dist:.6f}mm")
                                print(f"       → This polygon needs to be split into "
                                      f"planar sub-faces")
                                
                                # Find alternate edges using connectivity matrix
                                alternate_edges = []
                                for i, v_idx in enumerate(poly):
                                    prev_idx = poly[(i - 1) % len(poly)]
                                    next_idx = poly[(i + 1) % len(poly)]
                                    
                                    for j, v_other in enumerate(poly):
                                        if v_other == v_idx or v_other == prev_idx or v_other == next_idx:
                                            continue
                                        
                                        if merged_conn[v_idx, v_other] > 0:
                                            edge = tuple(sorted([v_idx, v_other]))
                                            if edge not in alternate_edges:
                                                alternate_edges.append(edge)
                                
                                if len(alternate_edges) > 0:
                                    print(f"       Found {len(alternate_edges)} alternate edge(s)")
                                    
                                    # Test each alternate edge for best split
                                    best_edge = None
                                    best_planarity_score = float('inf')
                                    
                                    for edge in alternate_edges:
                                        v1, v2 = edge
                                        pos1 = poly.index(v1)
                                        pos2 = poly.index(v2)
                                        
                                        if pos1 > pos2:
                                            pos1, pos2 = pos2, pos1
                                            v1, v2 = v2, v1
                                        
                                        poly1_verts = poly[pos1:pos2+1]
                                        poly2_verts = poly[pos2:] + poly[:pos1+1]
                                        
                                        if len(poly1_verts) >= 3 and len(poly2_verts) >= 3:
                                            poly1_coords = np.array([selected_vertices[v]
                                                                     for v in poly1_verts])
                                            poly2_coords = np.array([selected_vertices[v]
                                                                     for v in poly2_verts])
                                            
                                            # Fit planes
                                            c1 = np.mean(poly1_coords, axis=0)
                                            _, _, V1 = np.linalg.svd(poly1_coords - c1)
                                            n1 = V1[-1]
                                            d1 = -np.dot(n1, c1)
                                            
                                            c2 = np.mean(poly2_coords, axis=0)
                                            _, _, V2 = np.linalg.svd(poly2_coords - c2)
                                            n2 = V2[-1]
                                            d2 = -np.dot(n2, c2)
                                            
                                            max_dev1 = max([abs(np.dot(n1, v) + d1)
                                                           for v in poly1_coords])
                                            max_dev2 = max([abs(np.dot(n2, v) + d2)
                                                           for v in poly2_coords])
                                            
                                            planarity_score = max(max_dev1, max_dev2)
                                            
                                            if planarity_score < best_planarity_score:
                                                best_planarity_score = planarity_score
                                                best_edge = (v1, v2, poly1_verts, poly2_verts,
                                                            n1, d1, n2, d2)
                                    
                                    if best_edge is not None and best_planarity_score < max_dist:
                                        v1, v2, poly1_verts, poly2_verts, n1, d1, n2, d2 = best_edge
                                        
                                        print(f"       ✓ Best split edge: v{v1}-v{v2} "
                                              f"(planarity: {best_planarity_score:.6f}mm)")
                                        print(f"         Sub-polygon 1: {poly1_verts}")
                                        print(f"         Sub-polygon 2: {poly2_verts}")
                                        
                                        # Use ray-casting to check if these faces are exterior
                                        # (should be added) or interior (should be ignored)
                                        print(f"       → Ray-casting test to determine if faces are exterior...")
                                        
                                        faces_to_add = []
                                        
                                        for face_idx, (poly_verts, normal, d) in enumerate([
                                            (poly1_verts, n1, d1),
                                            (poly2_verts, n2, d2)
                                        ], 1):
                                            # Get interior point of polygon
                                            poly_coords_2d = []
                                            # Project to dominant plane for 2D representation
                                            abs_normal = np.abs(normal)
                                            if abs_normal[2] > abs_normal[0] and abs_normal[2] > abs_normal[1]:
                                                # XY plane
                                                poly_coords_2d = [(selected_vertices[v][0], 
                                                                  selected_vertices[v][1]) 
                                                                 for v in poly_verts]
                                            elif abs_normal[1] > abs_normal[0]:
                                                # XZ plane
                                                poly_coords_2d = [(selected_vertices[v][0], 
                                                                  selected_vertices[v][2]) 
                                                                 for v in poly_verts]
                                            else:
                                                # YZ plane
                                                poly_coords_2d = [(selected_vertices[v][1], 
                                                                  selected_vertices[v][2]) 
                                                                 for v in poly_verts]
                                            
                                            try:
                                                from shapely.geometry import Polygon as ShapelyPoly
                                                poly_2d = ShapelyPoly(poly_coords_2d)
                                                interior_pt = poly_2d.representative_point()
                                                
                                                # Cast ray from interior point through solid
                                                ray_direction = normal / np.linalg.norm(normal)
                                                
                                                # Reconstruct 3D point from 2D interior point
                                                if abs_normal[2] > abs_normal[0] and abs_normal[2] > abs_normal[1]:
                                                    # Solve for z: n·p + d = 0
                                                    z = -(normal[0]*interior_pt.x + normal[1]*interior_pt.y + d) / normal[2]
                                                    ray_origin = np.array([interior_pt.x, interior_pt.y, z])
                                                elif abs_normal[1] > abs_normal[0]:
                                                    # Solve for y
                                                    y = -(normal[0]*interior_pt.x + normal[2]*interior_pt.y + d) / normal[1]
                                                    ray_origin = np.array([interior_pt.x, y, interior_pt.y])
                                                else:
                                                    # Solve for x
                                                    x = -(normal[1]*interior_pt.x + normal[2]*interior_pt.y + d) / normal[0]
                                                    ray_origin = np.array([x, interior_pt.x, interior_pt.y])
                                                
                                                # Count intersections with all existing faces
                                                intersection_count = 0
                                                for existing_face in extracted_faces[:-2]:  # Exclude the two we just added
                                                    face_verts = existing_face['vertices']
                                                    if len(face_verts) < 3:
                                                        continue
                                                    
                                                    face_coords = [selected_vertices[v] for v in face_verts]
                                                    v0, v1, v2 = face_coords[0], face_coords[1], face_coords[2]
                                                    face_normal = np.cross(np.array(v1) - np.array(v0), 
                                                                          np.array(v2) - np.array(v0))
                                                    norm = np.linalg.norm(face_normal)
                                                    if norm < 1e-9:
                                                        continue
                                                    face_normal = face_normal / norm
                                                    
                                                    # Ray-plane intersection
                                                    denom = np.dot(ray_direction, face_normal)
                                                    if abs(denom) < 1e-6:
                                                        continue
                                                    
                                                    t = np.dot((np.array(v0) - ray_origin), face_normal) / denom
                                                    if t < 1e-6:
                                                        continue
                                                    
                                                    int_pt = ray_origin + t * ray_direction
                                                    
                                                    # Check if intersection is inside face polygon
                                                    # Project to same dominant plane
                                                    face_abs_normal = np.abs(face_normal)
                                                    if face_abs_normal[2] > face_abs_normal[0] and face_abs_normal[2] > face_abs_normal[1]:
                                                        face_2d_pts = [(v[0], v[1]) for v in face_coords]
                                                        int_2d = (int_pt[0], int_pt[1])
                                                    elif face_abs_normal[1] > face_abs_normal[0]:
                                                        face_2d_pts = [(v[0], v[2]) for v in face_coords]
                                                        int_2d = (int_pt[0], int_pt[2])
                                                    else:
                                                        face_2d_pts = [(v[1], v[2]) for v in face_coords]
                                                        int_2d = (int_pt[1], int_pt[2])
                                                    
                                                    try:
                                                        from shapely.geometry import Point
                                                        face_poly_2d = ShapelyPoly(face_2d_pts)
                                                        pt = Point(int_2d)
                                                        if face_poly_2d.contains(pt) or face_poly_2d.intersects(pt):
                                                            if face_poly_2d.boundary.distance(pt) > 1e-6:
                                                                intersection_count += 1
                                                    except:
                                                        pass
                                                
                                                parity = "odd" if intersection_count % 2 == 1 else "even"
                                                print(f"         Sub-polygon {face_idx}: {intersection_count} intersections ({parity})")
                                                
                                                if intersection_count % 2 == 1:
                                                    # Odd = exterior face, should be added
                                                    print(f"         → Exterior face - ADDING")
                                                    faces_to_add.append({
                                                        'vertices': poly_verts,
                                                        'normal': tuple(normal),
                                                        'd': d,
                                                        'holes': []
                                                    })
                                                else:
                                                    # Even = interior/dangling, should NOT be added
                                                    print(f"         → Interior/dangling face - SKIPPING")
                                                    
                                            except Exception as e:
                                                print(f"         ✗ Ray-casting failed: {e}")
                                                # If ray-casting fails, default to adding the face
                                                faces_to_add.append({
                                                    'vertices': poly_verts,
                                                    'normal': tuple(normal),
                                                    'd': d,
                                                    'holes': []
                                                })
                                        
                                        # Remove the two faces we tentatively added
                                        extracted_faces = extracted_faces[:-2]
                                        
                                        # Add only the exterior faces
                                        for face in faces_to_add:
                                            extracted_faces.append(face)
                                        
                                        print(f"       → Added {len(faces_to_add)} exterior face(s) from split")
                                        
                                        # Create two new planar faces to add
                                        # (REMOVED - replaced with ray-casting logic above)
                                        
                                        # Don't mark for deletion since we're adding replacement faces
                                        continue
                                    else:
                                        print(f"       ✗ Could not find good split "
                                              f"(best score: {best_planarity_score:.6f}mm)")
                                else:
                                    print(f"       ✗ No alternate edges found for splitting")
                            else:
                                print(f"       ✓ Planar polygon (deviation: {max_dist:.6f}mm)")
                                
                                # Calculate normal and d for this polygon
                                outer_verts_3d = [selected_vertices[v_idx] for v_idx in poly]
                                centroid = np.mean(outer_verts_3d, axis=0)
                                centered = np.array(outer_verts_3d) - centroid
                                _, _, Vt = np.linalg.svd(centered)
                                computed_normal = Vt[-1]
                                computed_d = -np.dot(computed_normal, centroid)
                                
                                # Check if this polygon is coplanar with any existing face
                                # (faces sharing edges with same plane equation)
                                coplanar_face_idx = None
                                poly_edges = set()
                                for i in range(len(poly)):
                                    edge = tuple(sorted([poly[i], poly[(i+1) % len(poly)]]))
                                    poly_edges.add(edge)
                                
                                for ext_face_idx, ext_face in enumerate(extracted_faces):
                                    if not isinstance(ext_face, dict):
                                        continue
                                    
                                    # Check if faces share edges
                                    ext_verts = ext_face.get('vertices', [])
                                    ext_edges = set()
                                    for i in range(len(ext_verts)):
                                        edge = tuple(sorted([ext_verts[i], ext_verts[(i+1) % len(ext_verts)]]))
                                        ext_edges.add(edge)
                                    
                                    shared_edges = poly_edges & ext_edges
                                    if len(shared_edges) == 0:
                                        continue
                                    
                                    # Check if coplanar (same plane equation)
                                    ext_normal = np.array(ext_face.get('normal', [0, 0, 0]))
                                    ext_d = ext_face.get('d', 0)
                                    
                                    # Normalize normals for comparison
                                    ext_normal_norm = ext_normal / (np.linalg.norm(ext_normal) + 1e-10)
                                    comp_normal_norm = computed_normal / (np.linalg.norm(computed_normal) + 1e-10)
                                    
                                    # Check if normals are parallel (same or opposite)
                                    dot_product = np.dot(ext_normal_norm, comp_normal_norm)
                                    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                                    
                                    # Allow for opposite normals
                                    if angle > np.pi / 2:
                                        angle = np.pi - angle
                                    
                                    # Check distance difference
                                    dist_diff = abs(ext_d - computed_d)
                                    
                                    # Tolerance: 0.01 radians (~0.57°) and 0.05mm
                                    if angle < 0.01 and dist_diff < 0.05:
                                        coplanar_face_idx = ext_face_idx
                                        print(f"       → Found coplanar adjacent face: Face {ext_face_idx+1}")
                                        print(f"         (shares {len(shared_edges)} edge(s), angle={angle:.6f}rad, dist_diff={dist_diff:.6f}mm)")
                                        break
                                
                                if coplanar_face_idx is not None:
                                    # Merge the two faces by removing shared edges
                                    print(f"       → Merging with Face {coplanar_face_idx+1}")
                                    
                                    existing_face = extracted_faces[coplanar_face_idx]
                                    existing_verts = existing_face.get('vertices', [])
                                    existing_holes = existing_face.get('holes', [])
                                    
                                    # Get edges from both polygons
                                    ext_edges = []
                                    for i in range(len(existing_verts)):
                                        ext_edges.append((existing_verts[i], existing_verts[(i+1) % len(existing_verts)]))
                                    
                                    poly_edges_ordered = []
                                    for i in range(len(poly)):
                                        poly_edges_ordered.append((poly[i], poly[(i+1) % len(poly)]))
                                    
                                    # Find shared edges (considering both directions)
                                    shared_edges = set()
                                    for e1 in ext_edges:
                                        for e2 in poly_edges_ordered:
                                            if (e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0]):
                                                shared_edges.add(tuple(sorted(e1)))
                                                break
                                    
                                    # Remove shared edges from both polygons
                                    remaining_ext_edges = [e for e in ext_edges 
                                                          if tuple(sorted(e)) not in shared_edges]
                                    remaining_poly_edges = [e for e in poly_edges_ordered 
                                                           if tuple(sorted(e)) not in shared_edges]
                                    
                                    # Combine remaining edges
                                    all_remaining_edges = remaining_ext_edges + remaining_poly_edges
                                    
                                    print(f"         Shared edges: {len(shared_edges)}, Remaining: {len(all_remaining_edges)}")
                                    print(f"         Existing face edges: {ext_edges}")
                                    print(f"         Boundary polygon edges: {poly_edges_ordered}")
                                    print(f"         Shared: {sorted(list(shared_edges))}")
                                    print(f"         Remaining edges: {all_remaining_edges}")
                                    
                                    # Build adjacency graph to find merged boundary
                                    from collections import defaultdict
                                    edge_graph = defaultdict(list)
                                    for v1, v2 in all_remaining_edges:
                                        edge_graph[v1].append(v2)
                                        edge_graph[v2].append(v1)
                                    
                                    # Trace the merged boundary
                                    if len(all_remaining_edges) > 0:
                                        # Start from any vertex
                                        start_v = all_remaining_edges[0][0]
                                        merged_verts = [start_v]
                                        current = start_v
                                        visited_edges = set()
                                        
                                        while True:
                                            # Find next vertex
                                            next_v = None
                                            for neighbor in edge_graph[current]:
                                                edge_key = tuple(sorted([current, neighbor]))
                                                if edge_key not in visited_edges:
                                                    next_v = neighbor
                                                    visited_edges.add(edge_key)
                                                    break
                                            
                                            if next_v is None or next_v == start_v:
                                                break
                                            
                                            merged_verts.append(next_v)
                                            current = next_v
                                            
                                            # Safety check
                                            if len(merged_verts) > len(all_remaining_edges) + 2:
                                                break
                                        
                                        if len(merged_verts) >= 3:
                                            # Update existing face with merged boundary
                                            existing_face['vertices'] = merged_verts
                                            
                                            print(f"       ✓ Merged into single face with {len(merged_verts)} vertices")
                                            print(f"         Original: {len(existing_verts)} verts, New polygon: {len(poly)} verts")
                                            print(f"         Merged vertices: {merged_verts}")
                                            
                                            # Rebuild the OCC face with merged vertices
                                            print(f"       → Rebuilding OCC face {coplanar_face_idx+1} with merged geometry")
                                            try:
                                                # Get 3D coordinates of merged vertices
                                                merged_coords_3d = [selected_vertices[v] for v in merged_verts]
                                                
                                                # Create OCC face from merged vertices
                                                from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
                                                from OCC.Core.gp import gp_Pnt
                                                
                                                poly_maker = BRepBuilderAPI_MakePolygon()
                                                for coord in merged_coords_3d:
                                                    poly_maker.Add(gp_Pnt(coord[0], coord[1], coord[2]))
                                                poly_maker.Close()
                                                
                                                if poly_maker.IsDone():
                                                    wire = poly_maker.Wire()
                                                    face_maker = BRepBuilderAPI_MakeFace(wire)
                                                    
                                                    if face_maker.IsDone():
                                                        new_occ_face = face_maker.Face()
                                                        # Replace the old face in occ_faces_list
                                                        occ_faces_list[coplanar_face_idx] = new_occ_face
                                                        print(f"       ✓ Rebuilt OCC face successfully")
                                                        faces_were_merged = True  # Set flag to trigger re-sewing
                                                    else:
                                                        print(f"       ✗ Failed to create OCC face from wire")
                                                else:
                                                    print(f"       ✗ Failed to create wire from vertices")
                                            except Exception as e:
                                                print(f"       ✗ Error rebuilding OCC face: {e}")
                                            
                                            # Keep existing holes as-is
                                            if len(existing_holes) > 0:
                                                print(f"         Preserving {len(existing_holes)} hole(s)")
                                        else:
                                            print(f"       ✗ Merge failed: invalid merged boundary ({len(merged_verts)} verts)")
                                            new_face = {
                                                'vertices': poly,
                                                'normal': tuple(computed_normal),
                                                'd': computed_d,
                                                'holes': []
                                            }
                                            extracted_faces.append(new_face)
                                            print(f"       ✓ Added planar face with {len(poly)} vertices")
                                    else:
                                        print(f"       ✗ Merge failed: no remaining edges")
                                        new_face = {
                                            'vertices': poly,
                                            'normal': tuple(computed_normal),
                                            'd': computed_d,
                                            'holes': []
                                        }
                                        extracted_faces.append(new_face)
                                        print(f"       ✓ Added planar face with {len(poly)} vertices")
                                else:
                                    # No coplanar face found, add as new face
                                    print(f"       → Adding as new face to close free edges")
                                    new_face = {
                                        'vertices': poly,
                                        'normal': tuple(computed_normal),
                                        'd': computed_d,
                                        'holes': []
                                    }
                                    extracted_faces.append(new_face)
                                    print(f"       ✓ Added planar face with {len(poly)} vertices")
                        
                        # Try to find which face in extracted_faces matches
                        # this polygon (for deletion if it's a degenerate face)
                        poly_set = set(poly)
                        for ext_face_idx, ext_face in enumerate(extracted_faces):
                            if not isinstance(ext_face, dict):
                                continue
                            face_verts = set(ext_face.get('vertices', []))
                            if poly_set == face_verts:
                                print(f"       → Matches extracted Face "
                                      f"{ext_face_idx+1}")
                                free_edge_polygons_to_delete.append(
                                    ext_face_idx)
                                break
                else:
                    print(f"   No closed polygons found from free edges")
            
        # If faces were merged, rebuild OCC faces and re-sew
        if faces_were_merged:
            print(f"\n   ━━━ FACES WERE MERGED - REBUILDING AND RE-SEWING ━━━")
            print(f"   Deleting old sewn shape and rebuilding from updated faces...")
            
            # Clear the old sewn shape
            del sewn_shape
            
            # Rebuild ALL OCC faces from extracted_faces
            # (Some faces were updated with merged vertices)
            print(f"   Rebuilding {len(extracted_faces)} OCC faces from updated data...")
            occ_faces_list = []
            
            for face_idx, face_data in enumerate(extracted_faces):
                if not isinstance(face_data, dict):
                    continue
                
                outer_boundary = face_data.get('vertices', [])
                holes = face_data.get('holes', [])
                
                if len(outer_boundary) < 3:
                    continue
                
                # Get 3D coordinates
                outer_coords_3d = [selected_vertices[v] for v in outer_boundary]
                
                # Create wire for outer boundary
                from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
                from OCC.Core.gp import gp_Pnt
                
                poly_maker = BRepBuilderAPI_MakePolygon()
                for coord in outer_coords_3d:
                    poly_maker.Add(gp_Pnt(coord[0], coord[1], coord[2]))
                poly_maker.Close()
                
                if not poly_maker.IsDone():
                    print(f"     WARNING: Failed to create wire for face {face_idx+1}")
                    continue
                
                outer_wire = poly_maker.Wire()
                
                # Create face (with holes if present)
                if len(holes) > 0:
                    # Face has holes - need to add them as internal wires
                    face_maker = BRepBuilderAPI_MakeFace(outer_wire)
                    
                    for hole in holes:
                        if len(hole) < 3:
                            continue
                        hole_coords_3d = [selected_vertices[v] for v in hole]
                        hole_poly_maker = BRepBuilderAPI_MakePolygon()
                        for coord in hole_coords_3d:
                            hole_poly_maker.Add(gp_Pnt(coord[0], coord[1], coord[2]))
                        hole_poly_maker.Close()
                        
                        if hole_poly_maker.IsDone():
                            hole_wire = hole_poly_maker.Wire()
                            face_maker.Add(hole_wire)
                    
                    if face_maker.IsDone():
                        occ_faces_list.append(face_maker.Face())
                else:
                    # Simple face without holes
                    face_maker = BRepBuilderAPI_MakeFace(outer_wire)
                    if face_maker.IsDone():
                        occ_faces_list.append(face_maker.Face())
            
            print(f"   Rebuilt {len(occ_faces_list)} OCC faces")
            
            # Re-sew the faces
            print(f"\n   Re-sewing {len(occ_faces_list)} faces...")
            sewing = BRepBuilderAPI_Sewing()
            sewing.SetTolerance(0.1005)
            
            for occ_face in occ_faces_list:
                sewing.Add(occ_face)
            
            sewing.Perform()
            sewn_shape = sewing.SewedShape()
            
            print(f"   Re-sewing diagnostics:")
            print(f"     - Number of free edges: {sewing.NbFreeEdges()}")
            print(f"     - Number of multiple edges: {sewing.NbMultipleEdges()}")
            print(f"     - Number of contigous edges: {sewing.NbContigousEdges()}")
            
            if sewing.NbFreeEdges() > 0:
                print(f"   WARNING: Still {sewing.NbFreeEdges()} free edges after re-sewing")
            else:
                print(f"   ✓ No free edges - solid is closed!")
        
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_FACE
        from OCC.Core.TopoDS import topods_Shell, topods_Face
        from OCC.Core.ShapeFix import ShapeFix_Shell, ShapeFix_Solid, ShapeFix_Shape
        
        # Count faces in sewn shape BEFORE extracting shell
        print(f"\n   Counting faces in sewn shape...")
        sewn_face_explorer = TopExp_Explorer(sewn_shape, TopAbs_FACE)
        sewn_face_count = 0
        while sewn_face_explorer.More():
            sewn_face_count += 1
            sewn_face_explorer.Next()
        print(f"   Sewn shape contains {sewn_face_count} faces")
        
        # Try to fix the sewn shape first with ShapeFix_Shape
        print(f"   Fixing sewn shape with ShapeFix_Shape...")
        shape_fixer = ShapeFix_Shape()
        shape_fixer.Init(sewn_shape)
        
        try:
            shape_fixer.Perform()
            fixed_sewn_shape = shape_fixer.Shape()
            
            # Count faces after fixing
            fixed_sewn_face_explorer = TopExp_Explorer(fixed_sewn_shape, TopAbs_FACE)
            fixed_sewn_face_count = 0
            while fixed_sewn_face_explorer.More():
                fixed_sewn_face_count += 1
                fixed_sewn_face_explorer.Next()
            print(f"   Fixed sewn shape contains {fixed_sewn_face_count} faces")
        except RuntimeError as e:
            print(f"   WARNING: ShapeFix_Shape failed: {str(e)}")
            print(f"   Continuing with unfixed sewn shape (may have degenerate geometry)")
            fixed_sewn_shape = sewn_shape
            fixed_sewn_face_count = sewn_face_count
        
        # Count shells in the fixed sewn shape
        shell_explorer = TopExp_Explorer(fixed_sewn_shape, TopAbs_SHELL)
        shell_count = 0
        all_shells = []
        while shell_explorer.More():
            shell_count += 1
            shell = topods_Shell(shell_explorer.Current())
            all_shells.append(shell)
            
            # Count faces in this shell
            shell_face_explorer = TopExp_Explorer(shell, TopAbs_FACE)
            shell_face_count = 0
            while shell_face_explorer.More():
                shell_face_count += 1
                shell_face_explorer.Next()
            print(f"   Shell {shell_count} contains {shell_face_count} faces")
            
            shell_explorer.Next()
        
        print(f"   Total shells found: {shell_count}")
        
        # Separate shells into two groups for visualization
        # Group 1: Shell with most faces (main/outer shell)
        # Group 2: All other shells
        main_shell = None
        other_shells = []
        
        if shell_count > 0:
            # Find shell with most faces
            best_shell = None
            best_shell_face_count = 0
            best_shell_idx = -1
            
            for shell_idx, shell in enumerate(all_shells):
                shell_face_explorer = TopExp_Explorer(shell, TopAbs_FACE)
                shell_face_count = 0
                while shell_face_explorer.More():
                    shell_face_count += 1
                    shell_face_explorer.Next()
                
                if shell_face_count > best_shell_face_count:
                    best_shell_face_count = shell_face_count
                    best_shell = shell
                    best_shell_idx = shell_idx
            
            print(f"   Using shell {best_shell_idx + 1} with {best_shell_face_count} faces as main shell")
            main_shell = best_shell
            
            # Collect other shells
            for shell_idx, shell in enumerate(all_shells):
                if shell_idx != best_shell_idx:
                    shell_face_explorer = TopExp_Explorer(shell, TopAbs_FACE)
                    shell_face_count = 0
                    while shell_face_explorer.More():
                        shell_face_count += 1
                        shell_face_explorer.Next()
                    other_shells.append({
                        'shell': shell,
                        'index': shell_idx + 1,
                        'face_count': shell_face_count
                    })
            
            if len(other_shells) > 0:
                print(f"   Other shells ({len(other_shells)}):")
                for other in other_shells:
                    print(f"     Shell {other['index']}: "
                          f"{other['face_count']} faces")
                
                # Print detailed information about other shells
                print(f"\n   Detailed shell analysis:")
                for other in other_shells:
                    print(f"\n   Shell {other['index']} details:")
                    shell_obj = other['shell']
                    
                    # Extract and print each face in this shell
                    face_explorer = TopExp_Explorer(shell_obj, TopAbs_FACE)
                    shell_face_num = 0
                    while face_explorer.More():
                        shell_face_num += 1
                        face = topods_Face(face_explorer.Current())
                        
                        # Extract vertices from this face
                        from OCC.Core.BRep import BRep_Tool
                        from OCC.Core.TopExp import TopExp_Explorer
                        from OCC.Core.TopAbs import TopAbs_VERTEX
                        from OCC.Core.TopoDS import topods_Vertex
                        
                        vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
                        face_vertices = []
                        while vertex_explorer.More():
                            vertex = topods_Vertex(vertex_explorer.Current())
                            pnt = BRep_Tool.Pnt(vertex)
                            face_vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
                            vertex_explorer.Next()
                        
                        print(f"     Face {shell_face_num}: "
                              f"{len(face_vertices)} vertices")
                        for v_idx, v in enumerate(face_vertices):
                            print(f"       v{v_idx}: ({v[0]:.1f}, "
                                  f"{v[1]:.1f}, {v[2]:.1f})")
                        
                        face_explorer.Next()
                
                # Check for small shells (likely degenerate faces to delete)
                print(f"\n   Analyzing small shells for degenerate faces...")
                for other in other_shells:
                    if other['face_count'] <= 2:
                        print(f"     Shell {other['index']} has only "
                              f"{other['face_count']} face(s) - "
                              f"analyzing for deletion")
                        
                        # Extract vertices from faces in this small shell
                        shell_obj = other['shell']
                        face_explorer_small = TopExp_Explorer(
                            shell_obj, TopAbs_FACE)
                        
                        while face_explorer_small.More():
                            shell_face = topods_Face(
                                face_explorer_small.Current())
                            
                            # Extract vertices from this face
                            vertex_explorer = TopExp_Explorer(
                                shell_face, TopAbs_VERTEX)
                            face_vertex_coords = []
                            while vertex_explorer.More():
                                vertex = topods_Vertex(
                                    vertex_explorer.Current())
                                pnt = BRep_Tool.Pnt(vertex)
                                face_vertex_coords.append(
                                    (round(pnt.X(), 1),
                                     round(pnt.Y(), 1),
                                     round(pnt.Z(), 1)))
                                vertex_explorer.Next()
                            
                            # Get unique vertex coordinates
                            unique_coords = list(set(face_vertex_coords))
                            
                            # Try to match to extracted_faces
                            # Match ALL faces with same coordinates (not just first)
                            matched_count = 0
                            for ext_face_idx, ext_face in \
                                    enumerate(extracted_faces):
                                if not isinstance(ext_face, dict):
                                    continue
                                ext_verts = ext_face.get('vertices', [])
                                if len(ext_verts) == 0:
                                    continue
                                
                                # Get coordinates of extracted face
                                ext_coords = set()
                                for v_idx in ext_verts:
                                    v = selected_vertices[v_idx]
                                    ext_coords.add(
                                        (round(v[0], 1),
                                         round(v[1], 1),
                                         round(v[2], 1)))
                                
                                # Check if coordinates match
                                if ext_coords == set(unique_coords):
                                    print(f"       → Matches extracted "
                                          f"Face {ext_face_idx+1} "
                                          f"(vertices: {ext_verts})")
                                    if ext_face_idx not in \
                                       free_edge_polygons_to_delete:
                                        free_edge_polygons_to_delete.append(
                                            ext_face_idx)
                                    matched_count += 1
                                    # Don't break - continue to find all matches
                            
                            if matched_count == 0:
                                print(f"       → No match found in "
                                      f"extracted faces")
                            else:
                                print(f"       → Total matches: {matched_count}")
                            
                            face_explorer_small.Next()
                
                # If we found faces to delete, rebuild solid without them
                if len(free_edge_polygons_to_delete) > 0:
                    print(f"\n   Found {len(free_edge_polygons_to_delete)} "
                          f"problematic face(s) to delete:")
                    for face_idx in sorted(free_edge_polygons_to_delete):
                        if face_idx < len(extracted_faces):
                            face = extracted_faces[face_idx]
                            if isinstance(face, dict):
                                verts = face.get('vertices', [])
                                print(f"     Face {face_idx+1}: {verts}")
                    
                    print(f"\n   Rebuilding solid without these faces...")
                    
                    # Remove problematic faces from extracted_faces
                    # (in reverse order to maintain indices)
                    for face_idx in sorted(free_edge_polygons_to_delete,
                                          reverse=True):
                        if face_idx < len(extracted_faces):
                            del extracted_faces[face_idx]
                    
                    print(f"   Remaining faces: {len(extracted_faces)}")
                    
                    # Rebuild OCC faces from updated extracted_faces
                    print(f"\n   Rebuilding OCC faces...")
                    occ_faces_list = []
                    occ_hole_faces_list = []
                    faces_with_holes = set()
                    
                    for face_idx, face in enumerate(extracted_faces):
                        if not (isinstance(face, dict) and
                                'vertices' in face):
                            continue
                        
                        if face_idx in confirmed_dummy_faces:
                            continue
                        
                        polygon = face['vertices']
                        outer_verts_3d = [selected_vertices[v_idx]
                                         for v_idx in polygon]
                        holes = face.get('holes', [])
                        
                        if len(outer_verts_3d) >= 3:
                            centroid = np.mean(outer_verts_3d, axis=0)
                            centered = np.array(outer_verts_3d) - centroid
                            _, _, Vt = np.linalg.svd(centered)
                            computed_normal = Vt[-1]
                            computed_d = -np.dot(computed_normal, centroid)
                            
                            face['normal'] = tuple(computed_normal)
                            face['d'] = computed_d
                        
                        try:
                            # Build wire
                            edges = []
                            for i in range(len(outer_verts_3d)):
                                p1 = gp_Pnt(*outer_verts_3d[i])
                                p2_idx = (i + 1) % len(outer_verts_3d)
                                p2 = gp_Pnt(*outer_verts_3d[p2_idx])
                                edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
                                edges.append(edge)
                            
                            wire_builder = BRepBuilderAPI_MakeWire()
                            for edge in edges:
                                wire_builder.Add(edge)
                            
                            if not wire_builder.IsDone():
                                continue
                            
                            outer_wire = wire_builder.Wire()
                            
                            # Build face
                            face_normal = face.get('normal', None)
                            face_d = face.get('d', None)
                            if face_normal is not None and face_d is not None:
                                from OCC.Core.gp import gp_Pln, gp_Dir, gp_Pnt
                                plane_pnt = gp_Pnt(centroid[0], centroid[1],
                                                  centroid[2])
                                plane_dir = gp_Dir(face_normal[0],
                                                  face_normal[1],
                                                  face_normal[2])
                                plane = gp_Pln(plane_pnt, plane_dir)
                                face_builder = BRepBuilderAPI_MakeFace(
                                    plane, outer_wire)
                            else:
                                face_builder = BRepBuilderAPI_MakeFace(
                                    outer_wire)
                            
                            if not face_builder.IsDone():
                                continue
                            
                            # Add holes as internal wires
                            if len(holes) > 0:
                                faces_with_holes.add(face_idx)
                                outer_normal = np.array(face.get('normal',
                                                        np.array([0, 0, 1])))
                                outer_signed_area = compute_signed_area_3d(
                                    outer_verts_3d, outer_normal)
                                
                                for hole_idx, hole_vertex_indices in \
                                        enumerate(holes):
                                    try:
                                        hole_verts_3d = [
                                            selected_vertices[v_idx]
                                            for v_idx in hole_vertex_indices]
                                        
                                        hole_signed_area = \
                                            compute_signed_area_3d(
                                                hole_verts_3d, outer_normal)
                                        
                                        if ((outer_signed_area > 0 and
                                             hole_signed_area < 0) or
                                            (outer_signed_area < 0 and
                                             hole_signed_area > 0)):
                                            hole_verts_3d = list(
                                                reversed(hole_verts_3d))
                                        
                                        hole_edges = []
                                        for i in range(len(hole_verts_3d)):
                                            p1 = gp_Pnt(*hole_verts_3d[i])
                                            p2_idx = (i + 1) % \
                                                len(hole_verts_3d)
                                            p2 = gp_Pnt(*hole_verts_3d[p2_idx])
                                            edge = BRepBuilderAPI_MakeEdge(
                                                p1, p2).Edge()
                                            hole_edges.append(edge)
                                        
                                        hole_wire_builder = \
                                            BRepBuilderAPI_MakeWire()
                                        for edge in hole_edges:
                                            hole_wire_builder.Add(edge)
                                        
                                        if not hole_wire_builder.IsDone():
                                            continue
                                        
                                        hole_wire = hole_wire_builder.Wire()
                                        face_builder.Add(hole_wire)
                                    except Exception as e:
                                        pass
                            
                            occ_face = face_builder.Face()
                            occ_faces_list.append(occ_face)
                        
                        except Exception as e:
                            pass
                    
                    print(f"   Rebuilt {len(occ_faces_list)} OCC faces")
                    
                    # Re-sew with the updated faces
                    print(f"\n   Re-sewing {len(occ_faces_list)} faces...")
                    sewing = BRepBuilderAPI_Sewing()
                    sewing.SetTolerance(0.1005)
                    
                    for occ_face in occ_faces_list:
                        sewing.Add(occ_face)
                    
                    sewing.Perform()
                    sewn_shape = sewing.SewedShape()
                    
                    print(f"   Re-sewing diagnostics:")
                    print(f"     - Number of free edges: "
                          f"{sewing.NbFreeEdges()}")
                    print(f"     - Number of contigous edges: "
                          f"{sewing.NbContigousEdges()}")
                    
                    # Re-analyze shells
                    shape_fixer = ShapeFix_Shape()
                    shape_fixer.Init(sewn_shape)
                    shape_fixer.Perform()
                    fixed_sewn_shape = shape_fixer.Shape()
                    
                    shell_explorer = TopExp_Explorer(
                        fixed_sewn_shape, TopAbs_SHELL)
                    shell_count = 0
                    all_shells = []
                    while shell_explorer.More():
                        shell_count += 1
                        shell = topods_Shell(shell_explorer.Current())
                        all_shells.append(shell)
                        shell_explorer.Next()
                    
                    print(f"   After rebuild: {shell_count} shell(s) found")
                    
                    # Find main shell and rebuild other_shells list
                    other_shells = []
                    if shell_count > 0:
                        best_shell = None
                        best_shell_face_count = 0
                        best_shell_idx = -1
                        
                        for shell_idx, shell in enumerate(all_shells):
                            shell_face_explorer = TopExp_Explorer(
                                shell, TopAbs_FACE)
                            shell_face_count = 0
                            while shell_face_explorer.More():
                                shell_face_count += 1
                                shell_face_explorer.Next()
                            
                            print(f"     Shell {shell_idx+1}: "
                                  f"{shell_face_count} faces")
                            
                            if shell_face_count > best_shell_face_count:
                                best_shell_face_count = shell_face_count
                                best_shell = shell
                                best_shell_idx = shell_idx
                        
                        # Collect other shells (after rebuild)
                        for shell_idx, shell in enumerate(all_shells):
                            if shell_idx != best_shell_idx:
                                shell_face_explorer = TopExp_Explorer(
                                    shell, TopAbs_FACE)
                                shell_face_count = 0
                                while shell_face_explorer.More():
                                    shell_face_count += 1
                                    shell_face_explorer.Next()
                                other_shells.append({
                                    'shell': shell,
                                    'index': shell_idx + 1,
                                    'face_count': shell_face_count
                                })
                        
                        main_shell = best_shell
                        print(f"   Using shell with "
                              f"{best_shell_face_count} faces")
            
            # Use main shell for solid creation
            shell = main_shell
            
            # Fix shell orientation
            shell_fixer = ShapeFix_Shell()
            shell_fixer.Init(shell)
            shell_fixer.Perform()
            fixed_shell = shell_fixer.Shell()
            
            # Count faces after fixing shell
            fixed_face_explorer = TopExp_Explorer(fixed_shell, TopAbs_FACE)
            fixed_face_count = 0
            while fixed_face_explorer.More():
                fixed_face_count += 1
                fixed_face_explorer.Next()
            print(f"   Fixed main shell contains {fixed_face_count} faces")
            
            # Build solid from shell(s) - add main shell and other shells
            # Following OCC example pattern: solid_maker.Add(shell) for each shell
            try:
                solid_builder = BRepBuilderAPI_MakeSolid()
                
                # Add main (outer) shell
                solid_builder.Add(fixed_shell)
                print(f"   Added main shell to solid")
                
                # Add other shells (holes/voids) if any
                if len(other_shells) > 0:
                    print(f"   Adding {len(other_shells)} inner shells (holes)...")
                    for other in other_shells:
                        # Fix orientation of inner shell
                        inner_shell_fixer = ShapeFix_Shell()
                        inner_shell_fixer.Init(other['shell'])
                        inner_shell_fixer.Perform()
                        fixed_inner_shell = inner_shell_fixer.Shell()
                        
                        solid_builder.Add(fixed_inner_shell)
                        print(f"     Added Shell {other['index']} " +
                              f"({other['face_count']} faces) as inner void")
                
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
                    
                    # Compute volume first to check orientation
                    props = GProp_GProps()
                    brepgprop_VolumeProperties(occ_solid, props)
                    volume = props.Mass()
                    print(f"   Solid volume: {volume}")
                    
                    # FIX: If volume is negative, reverse the shell orientation
                    if volume < 0:
                        print(f"   FIXING: Negative volume detected - "
                              f"reversing shell orientation...")
                        try:
                            # Get the shell from the solid
                            from OCC.Core.TopExp import TopExp_Explorer
                            from OCC.Core.TopAbs import TopAbs_SHELL
                            from OCC.Core.TopoDS import topods_Shell
                            
                            shell_exp = TopExp_Explorer(occ_solid, TopAbs_SHELL)
                            if shell_exp.More():
                                shell = topods_Shell(shell_exp.Current())
                                
                                # Reverse the shell
                                shell.Reverse()
                                
                                # Rebuild solid from reversed shell
                                solid_maker = BRepBuilderAPI_MakeSolid(shell)
                                if solid_maker.IsDone():
                                    occ_solid = solid_maker.Solid()
                                    
                                    # Recompute volume
                                    props = GProp_GProps()
                                    brepgprop_VolumeProperties(occ_solid, props)
                                    volume = props.Mass()
                                    print(f"   After reversal: volume = {volume}")
                                    
                                    # Revalidate
                                    analyzer = BRepCheck_Analyzer(occ_solid)
                                    solid_is_valid = analyzer.IsValid()
                                    print(f"   After reversal: Valid={solid_is_valid}")
                                else:
                                    print(f"   ERROR: Failed to rebuild solid "
                                          f"from reversed shell")
                        except Exception as e:
                            print(f"   ERROR: Shell reversal failed: {e}")
                    
                    if not solid_is_valid:
                        print("   WARNING: Solid is invalid - checking details...")
                        
                        # Check free edges
                        num_free_edges = sewing.NbFreeEdges()
                        if num_free_edges > 0:
                            print(f"      Issue 1: {num_free_edges} free edge(s) "
                                  f"detected (shell not closed)")
                        else:
                            print(f"      ✓ No free edges - shell is closed")
                        
                        # Check volume sign (negative = inside-out shell)
                        if volume < 0:
                            print(f"      Issue 2: NEGATIVE VOLUME detected!")
                            print(f"         Volume = {volume:.2f}")
                            print(f"         This indicates shell orientation is REVERSED")
                            print(f"         Shell normals are pointing INWARD instead of OUTWARD")
                            print(f"         FIX: Reverse shell orientation or fix face normals")
                        else:
                            print(f"      ✓ Volume is positive ({volume:.2f})")
                        
                        # Try to get detailed error info from analyzer
                        try:
                            from OCC.Core.BRepCheck import BRepCheck_Status
                            # Note: Getting detailed status is complex in OCC
                            print(f"      Checking individual components...")
                            
                            # Check shell
                            from OCC.Core.TopExp import TopExp_Explorer
                            from OCC.Core.TopAbs import TopAbs_SHELL
                            from OCC.Core.TopoDS import topods_Shell
                            
                            shell_exp = TopExp_Explorer(occ_solid, TopAbs_SHELL)
                            if shell_exp.More():
                                shell_to_check = topods_Shell(shell_exp.Current())
                                shell_analyzer = BRepCheck_Analyzer(shell_to_check)
                                if not shell_analyzer.IsValid():
                                    print(f"      Issue 3: Shell itself is invalid")
                                else:
                                    print(f"      ✓ Shell is valid")
                        except Exception as e:
                            print(f"      Could not get detailed status: {e}")
                        
                        print("   Invalid solid - see issues above")
                    
                    if solid_is_valid:
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
    
    # Step 8.3: Extract faces from OCC solid and compare with original
    if occ_solid is not None:
        print(f"\n[STEP 8.3] Extracting faces from OCC solid and comparing with original...")
        
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
        from OCC.Core.TopoDS import topods_Face, topods_Wire
        
        # Extract all faces from the reconstructed solid
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
        
        print(f"   Extracted {face_count} faces from reconstructed solid")
        print(f"   Faces with holes: {sum(1 for f in extracted_occ_faces if len(f['holes']) > 0)}")
        
        # Compare with original faces
        print(f"\n[STEP 8.3.1] Comparing reconstructed faces with original faces...")
        
        def compute_face_plane(vertices):
            """Compute plane equation (normal, distance) from face vertices"""
            if len(vertices) < 3:
                return None, None
            
            # Use first 3 non-collinear vertices to compute plane
            v0 = vertices[0]
            v1 = vertices[1]
            v2 = vertices[2]
            
            # Two edge vectors
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            # Normal is cross product
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            
            if norm_len < 1e-10:
                return None, None
            
            normal = normal / norm_len  # Normalize
            
            # Distance from origin: d = -normal · v0
            distance = -np.dot(normal, v0)
            
            return normal, distance
        
        def faces_coplanar(face1, face2, angle_tol=0.01, dist_tol=0.05):
            """Check if two faces are on the same plane
            angle_tol: tolerance for normal angle difference (radians)
            dist_tol: tolerance for distance difference (mm)
            """
            n1, d1 = compute_face_plane(face1['outer_boundary'])
            n2, d2 = compute_face_plane(face2['outer_boundary'])
            
            if n1 is None or n2 is None:
                return False
            
            # Check if normals are parallel (same or opposite direction)
            dot = np.dot(n1, n2)
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            
            # Allow for opposite normals (flipped faces)
            if angle > np.pi / 2:
                angle = np.pi - angle
                d2 = -d2  # Flip distance for opposite normal
            
            if angle > angle_tol:
                return False
            
            # Check distance
            if abs(d1 - d2) > dist_tol:
                return False
            
            return True
        
        def faces_overlap(face1, face2):
            """Check if two coplanar faces overlap using Shapely 2D projection"""
            try:
                from shapely.geometry import Polygon
                from shapely.ops import unary_union
                
                # Project both faces onto a 2D plane for comparison
                # Use the plane's coordinate system
                n1, d1 = compute_face_plane(face1['outer_boundary'])
                if n1 is None:
                    return False, 0.0
                
                # Create coordinate system on the plane
                # Choose u axis perpendicular to normal
                if abs(n1[0]) < 0.9:
                    u = np.array([1, 0, 0])
                else:
                    u = np.array([0, 1, 0])
                
                u = u - np.dot(u, n1) * n1
                u = u / np.linalg.norm(u)
                
                v = np.cross(n1, u)
                
                # Project vertices to 2D
                def project_to_2d(vertices):
                    coords = []
                    for vertex in vertices:
                        x = np.dot(vertex, u)
                        y = np.dot(vertex, v)
                        coords.append((x, y))
                    return coords
                
                # Create Shapely polygons
                outer1_2d = project_to_2d(face1['outer_boundary'])
                outer2_2d = project_to_2d(face2['outer_boundary'])
                
                poly1 = Polygon(outer1_2d)
                poly2 = Polygon(outer2_2d)
                
                if not poly1.is_valid or not poly2.is_valid:
                    return False, 0.0
                
                # Check intersection
                intersection = poly1.intersection(poly2)
                overlap_area = intersection.area
                
                # Calculate overlap percentage relative to smaller polygon
                min_area = min(poly1.area, poly2.area)
                if min_area < 1e-10:
                    return False, 0.0
                
                overlap_ratio = overlap_area / min_area
                
                # Consider faces overlapping if >90% of smaller face overlaps
                return overlap_ratio > 0.9, overlap_ratio
                
            except Exception as e:
                print(f"      Warning: Shapely overlap check failed: {e}")
                return False, 0.0
        
        def faces_match(face1, face2, debug=False):
            """Check if two faces match using plane equation and overlap"""
            # First check if coplanar
            if not faces_coplanar(face1, face2):
                if debug:
                    print(f"      Faces not coplanar")
                return False
            
            # Check geometric overlap
            overlaps, ratio = faces_overlap(face1, face2)
            if debug:
                print(f"      Coplanar: Yes, Overlap ratio: {ratio:.2%}")
            
            return overlaps
        
        # Match reconstructed faces to original faces
        face_matches = []  # List of (recon_idx, orig_idx, matched)
        matched_original = []
        
        if face_polygons is not None and len(face_polygons) > 0:
            matched_original = [False] * len(face_polygons)
        
        # Debug: Print first face from each set
        if len(extracted_occ_faces) > 0:
            print(f"\n   Debug: First reconstructed face has {len(extracted_occ_faces[0]['outer_boundary'])} vertices")
            print(f"          Sample vertex: {extracted_occ_faces[0]['outer_boundary'][0]}")
        
        if face_polygons is not None and len(face_polygons) > 0:
            orig_first = face_polygons[0]
            outer_key = 'outer_boundary' if 'outer_boundary' in orig_first else 'vertices'
            if outer_key in orig_first:
                print(f"   Debug: First original face has {len(orig_first[outer_key])} vertices")
                print(f"          Sample vertex: {orig_first[outer_key][0]}")
        
        for recon_idx, recon_face in enumerate(extracted_occ_faces):
            matched = False
            matched_orig_idx = -1
            
            if face_polygons is not None and len(face_polygons) > 0:
                for orig_idx, orig_face in enumerate(face_polygons):
                    if not matched_original[orig_idx]:
                        # Show debug for first comparison only
                        debug = (recon_idx == 0 and orig_idx == 0)
                        if faces_match(recon_face, orig_face, debug=debug):
                            matched = True
                            matched_orig_idx = orig_idx
                            matched_original[orig_idx] = True
                            break
            
            face_matches.append({
                'recon_idx': recon_idx,
                'orig_idx': matched_orig_idx,
                'matched': matched
            })
        
        # Count matches
        num_matched = sum(1 for m in face_matches if m['matched'])
        num_unmatched_recon = len(extracted_occ_faces) - num_matched
        num_unmatched_orig = sum(1 for m in matched_original if not m) if len(matched_original) > 0 else 0
        
        print(f"\n   Face Matching Results:")
        orig_face_count = len(face_polygons) if face_polygons is not None else 0
        print(f"   - Original faces: {orig_face_count}")
        print(f"   - Reconstructed faces: {len(extracted_occ_faces)}")
        print(f"   - Matched faces: {num_matched}")
        print(f"   - Unmatched reconstructed faces: {num_unmatched_recon}")
        print(f"   - Missing original faces: {num_unmatched_orig}")
        
        if num_unmatched_recon > 0:
            print(f"\n   Unmatched Reconstructed Faces:")
            for match in face_matches:
                if not match['matched']:
                    recon_face = extracted_occ_faces[match['recon_idx']]
                    vertices = recon_face['outer_boundary']
                    vertex_list = ', '.join([f"[{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]" 
                                             for v in vertices[:5]])
                    if len(vertices) > 5:
                        vertex_list += f" ... ({len(vertices)} total)"
                    print(f"     - Reconstructed face {match['recon_idx'] + 1}: {vertex_list}")
        
        if num_unmatched_orig > 0:
            print(f"\n   Missing Original Faces:")
            for orig_idx, is_matched in enumerate(matched_original):
                if not is_matched:
                    orig_face = face_polygons[orig_idx]
                    outer_key = 'outer_boundary' if 'outer_boundary' in orig_face else 'vertices'
                    if outer_key in orig_face:
                        vertices = orig_face[outer_key]
                        vertex_list = ', '.join([f"[{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]" 
                                                 for v in vertices[:5]])
                        if len(vertices) > 5:
                            vertex_list += f" ... ({len(vertices)} total)"
                        print(f"     - Original face {orig_idx + 1}: {vertex_list}")
                    else:
                        print(f"     - Original face {orig_idx + 1}")
        
        # Plot the extracted faces with color coding
        print(f"\n[STEP 8.3.2] Plotting reconstructed solid faces with match highlighting...")
        
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
        
        # Import for 3D polygon collection
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.widgets import CheckButtons
        
        # Store text objects and polygon collections for toggling
        vertex_texts = []
        face_texts = []
        matched_collections = []
        unmatched_collections = []
        matched_lines = []
        unmatched_lines = []
        
        for face_idx, face_data in enumerate(extracted_occ_faces):
            outer = face_data['outer_boundary']
            holes = face_data['holes']
            
            # Determine if this face is matched or unmatched
            is_matched = face_matches[face_idx]['matched']
            
            # Color coding: Green for matched, Red for unmatched
            if is_matched:
                face_color = 'green'
                edge_color = 'darkgreen'
                label_color = 'darkgreen'
                label_bg = 'lightgreen'
            else:
                face_color = 'red'
                edge_color = 'darkred'
                label_color = 'darkred'
                label_bg = 'lightcoral'
            
            # Plot outer boundary with colored shading
            if len(outer) > 2:
                # Create filled polygon with transparency
                poly = Poly3DCollection([outer], alpha=0.3, 
                                       facecolors=face_color,
                                       edgecolors=edge_color, linewidths=2)
                ax.add_collection3d(poly)
                if is_matched:
                    matched_collections.append(poly)
                else:
                    unmatched_collections.append(poly)
                
                # Plot outer boundary edges
                outer_closed = np.vstack([outer, outer[0]])
                line = ax.plot(outer_closed[:, 0], outer_closed[:, 1],
                               outer_closed[:, 2],
                               color=edge_color, linewidth=2,
                               label=('Matched' if is_matched else 'Unmatched')
                                     if face_idx == 0 or (face_idx == 1 and not face_matches[0]['matched'])
                                     else None)
                if is_matched:
                    matched_lines.extend(line)
                else:
                    unmatched_lines.extend(line)
                    
                scatter = ax.scatter(outer[:, 0], outer[:, 1], outer[:, 2],
                                     color=edge_color, s=40)
                if is_matched:
                    matched_collections.append(scatter)
                else:
                    unmatched_collections.append(scatter)
                
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
                
                # Add face number at centroid with match status
                centroid = np.mean(outer, axis=0)
                face_label = f'F{face_idx+1}'
                if not is_matched:
                    face_label += ' ✗'
                else:
                    face_label += ' ✓'
                    
                txt = ax.text(centroid[0], centroid[1], centroid[2],
                       face_label,
                       color=label_color, fontsize=12, ha='center', va='center',
                       weight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor=label_bg,
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
        
        ax.set_title(f'Reconstructed Solid: {face_count} Faces (Colored Shading)',
                    fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Enable mouse rotation
        ax.mouse_init()
        
        if face_count > 0:
            ax.legend(loc='upper right', fontsize=10)
        
        # Add toggle buttons for labels and shading
        checkbox_ax = plt.axes([0.02, 0.65, 0.17, 0.25])
        labels_check = ['Vertex Labels', 'Face Labels', 'Polygon Shading',
                       'Matched Faces', 'Unmatched Faces']
        visibility = [True, True, True, True, True]
        check = CheckButtons(checkbox_ax, labels_check, visibility)
        
        def toggle_labels(label):
            if label == 'Vertex Labels':
                for txt in vertex_texts:
                    txt.set_visible(not txt.get_visible())
            elif label == 'Face Labels':
                for txt in face_texts:
                    txt.set_visible(not txt.get_visible())
            elif label == 'Polygon Shading':
                for poly in matched_collections + unmatched_collections:
                    if isinstance(poly, Poly3DCollection):
                        poly.set_visible(not poly.get_visible())
            elif label == 'Matched Faces':
                vis = not matched_collections[0].get_visible() if matched_collections else True
                for item in matched_collections:
                    item.set_visible(vis)
                for line in matched_lines:
                    line.set_visible(vis)
            elif label == 'Unmatched Faces':
                vis = not unmatched_collections[0].get_visible() if unmatched_collections else True
                for item in unmatched_collections:
                    item.set_visible(vis)
                for line in unmatched_lines:
                    line.set_visible(vis)
            fig.canvas.draw_idle()
        
        check.on_clicked(toggle_labels)
        
        plt.tight_layout()
        print("[STEP 8.3] Reconstructed solid plot created.")
        
        # Create second plot for shell visualization
        if shell_count > 1:
            print(f"[STEP 8.4] Creating shell visualization ({shell_count} shells)...")
            
            fig2 = plt.figure(figsize=(14, 10))
            ax2 = fig2.add_subplot(111, projection='3d')
            
            # Function to extract and plot shell faces
            def plot_shell(ax, shell, color, label, alpha=0.5, reverse_normals=False):
                from OCC.Core.BRep import BRep_Tool
                from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
                from OCC.Core.TopExp import TopExp_Explorer
                from OCC.Core.TopoDS import topods_Face, topods_Wire, topods_Edge
                from OCC.Core.BRepTools import BRepTools_WireExplorer
                
                face_exp = TopExp_Explorer(shell, TopAbs_FACE)
                face_polys = []
                
                while face_exp.More():
                    face = topods_Face(face_exp.Current())
                    
                    # Get outer wire
                    wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
                    if wire_exp.More():
                        wire = topods_Wire(wire_exp.Current())
                        
                        # Use BRepTools_WireExplorer for proper ordered traversal
                        wire_explorer = BRepTools_WireExplorer(wire)
                        vertices = []
                        
                        while wire_explorer.More():
                            # Get current vertex from wire explorer
                            vertex = wire_explorer.CurrentVertex()
                            pnt = BRep_Tool.Pnt(vertex)
                            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
                            wire_explorer.Next()
                        
                        if len(vertices) > 2:
                            # Reverse vertex order to flip normals if needed
                            if reverse_normals:
                                vertices = list(reversed(vertices))
                            # Close the polygon by adding first vertex at the end
                            vertices.append(vertices[0])
                            face_polys.append(np.array(vertices))
                    
                    face_exp.Next()
                
                # Plot all faces
                poly_collection = Poly3DCollection(face_polys, alpha=alpha,
                                                   facecolors=color,
                                                   edgecolors='black',
                                                   linewidths=1)
                ax.add_collection3d(poly_collection)
                return poly_collection, face_polys
            
            # Plot main shell initially
            main_shell_polys = []
            other_shell_polys = []
            
            if main_shell is not None:
                main_poly, main_polys = plot_shell(ax2, main_shell, 'cyan',
                                                   'Main Shell', alpha=0.6)
                main_shell_polys.append(main_poly)
            
            # Plot other shells (initially hidden) with reversed normals
            for other in other_shells:
                other_poly, other_polys = plot_shell(ax2, other['shell'], 'orange',
                                                     f"Shell {other['index']}",
                                                     alpha=0.6, reverse_normals=True)
                other_shell_polys.append(other_poly)
                other_poly.set_visible(False)  # Initially hidden
            
            ax2.set_title(f'Shell Visualization: {shell_count} Shells Found',
                         fontsize=14)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.mouse_init()
            
            # Add toggle buttons for shells
            checkbox_ax2 = plt.axes([0.02, 0.7, 0.15, 0.2])
            shell_labels = ['Main Shell (cyan)', 'Other Shells (orange)']
            shell_visibility = [True, False]
            check2 = CheckButtons(checkbox_ax2, shell_labels, shell_visibility)
            
            def toggle_shells(label):
                if label == 'Main Shell (cyan)':
                    for poly in main_shell_polys:
                        poly.set_visible(not poly.get_visible())
                elif label == 'Other Shells (orange)':
                    for poly in other_shell_polys:
                        poly.set_visible(not poly.get_visible())
                fig2.canvas.draw_idle()
            
            check2.on_clicked(toggle_shells)
            
            plt.tight_layout()
            print("[STEP 8.4] Shell visualization created.")
        
        # Handle graphics display or save to PDF
        if args.no_graphics:
            # Create PDFfiles directory if it doesn't exist
            pdf_output_dir = "PDFfiles"
            os.makedirs(pdf_output_dir, exist_ok=True)
            pdf_filename = os.path.join(pdf_output_dir,
                                       f"reconstruct_solid_seed_{args.seed}.pdf")
            print(f"[STEP 8.5] Saving all graphics to {pdf_filename}...")
            try:
                with PdfPages(pdf_filename) as pdf:
                    for fig_num in plt.get_fignums():
                        pdf.savefig(plt.figure(fig_num))
                print(f"[STEP 8.5] Graphics saved to PDF: {pdf_filename}")
                plt.close('all')
            except Exception as e:
                print(f"[ERROR] Exception during PDF save: {e}")
        else:
            print("[DEBUG] About to call plt.show(block=True)...")
            plt.show(block=True)
            print("[DEBUG] Returned from plt.show(block=True)")
            plt.close(fig)
            print("[STEP 8.3] Plot closed after user interaction.")
    else:
        print(f"\n[STEP 8.3] No solid to display (construction failed)")
    
    # Generate summary statistics
    print("\n" + "="*70)
    print("RECONSTRUCTION SUMMARY STATISTICS")
    print("="*70)
    
    summary_stats = {}
    summary_stats['seed'] = args.seed
    
    # Original solid statistics
    original_num_faces = len(face_polygons) if face_polygons is not None else 0
    summary_stats['original_num_faces'] = original_num_faces
    print("Original Solid:")
    print(f"  - Number of faces: {original_num_faces}")
    
    # Use volume from saved file (already scaled to mm³)
    if original_volume_from_file is not None:
        original_volume = original_volume_from_file
        summary_stats['original_volume'] = original_volume
        print(f"  - Volume: {original_volume:.6f} mm³")
    else:
        print("  - Volume: Not available (old file format or missing data)")
        summary_stats['original_volume'] = None
        original_volume = None
    
    # Reconstructed solid statistics
    if occ_solid is not None:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_FACE
        from OCC.Core.TopoDS import topods_Shell
        from OCC.Core.BRepGProp import brepgprop_VolumeProperties
        from OCC.Core.GProp import GProp_GProps
        
        # Count shells
        shell_exp = TopExp_Explorer(occ_solid, TopAbs_SHELL)
        num_shells = 0
        shell_face_counts = []
        
        while shell_exp.More():
            num_shells += 1
            shell = topods_Shell(shell_exp.Current())
            
            # Count faces in this shell
            face_exp = TopExp_Explorer(shell, TopAbs_FACE)
            shell_faces = 0
            while face_exp.More():
                shell_faces += 1
                face_exp.Next()
            shell_face_counts.append(shell_faces)
            
            shell_exp.Next()
        
        summary_stats['reconstructed_num_shells'] = num_shells
        summary_stats['reconstructed_shell_face_counts'] = shell_face_counts
        summary_stats['reconstructed_total_faces'] = sum(shell_face_counts)
        
        # Compute volume
        props = GProp_GProps()
        brepgprop_VolumeProperties(occ_solid, props)
        recon_volume = props.Mass()
        summary_stats['reconstructed_volume'] = recon_volume
        
        # Add face matching statistics if available
        if 'face_matches' in locals():
            num_matched = sum(1 for m in face_matches if m['matched'])
            num_unmatched_recon = len(face_matches) - num_matched
            matched_orig_count = sum(1 for m in matched_original if m) if 'matched_original' in locals() else 0
            num_missing_orig = original_num_faces - matched_orig_count
            
            summary_stats['matched_faces'] = num_matched
            summary_stats['unmatched_reconstructed_faces'] = num_unmatched_recon
            summary_stats['missing_original_faces'] = num_missing_orig
        
        print("\nReconstructed Solid:")
        print(f"  - Number of shells: {num_shells}")
        for i, fc in enumerate(shell_face_counts):
            print(f"    - Shell {i+1}: {fc} faces")
        print(f"  - Total faces: {sum(shell_face_counts)}")
        print(f"  - Volume: {recon_volume:.6f} mm³")
        
        if 'face_matches' in locals():
            print("\nFace Matching:")
            print(f"  - Matched faces: {num_matched}/{original_num_faces}")
            print(f"  - Unmatched reconstructed faces: {num_unmatched_recon}")
            print(f"  - Missing original faces: {num_missing_orig}")
    else:
        summary_stats['reconstructed_num_shells'] = 0
        summary_stats['reconstructed_shell_face_counts'] = []
        summary_stats['reconstructed_total_faces'] = 0
        summary_stats['reconstructed_volume'] = None
        summary_stats['matched_faces'] = 0
        summary_stats['unmatched_reconstructed_faces'] = 0
        summary_stats['missing_original_faces'] = original_num_faces
        
        print("\nReconstructed Solid:")
        print("  - Construction failed")
    
    # Save summary to file
    summary_dir = "Summary"
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(
        summary_dir, f"reconstruction_summary_seed_{args.seed}.txt")
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RECONSTRUCTION SUMMARY STATISTICS\n")
        f.write("="*70 + "\n")
        f.write(f"Seed: {args.seed}\n\n")
        
        f.write("Original Solid:\n")
        f.write(f"  - Number of faces: {original_num_faces}\n")
        if original_volume is not None:
            f.write(f"  - Volume: {original_volume:.6f} mm³\n\n")
        else:
            f.write("  - Volume: Not available\n\n")
        
        f.write("Reconstructed Solid:\n")
        if summary_stats['reconstructed_num_shells'] > 0:
            f.write(f"  - Number of shells: "
                   f"{summary_stats['reconstructed_num_shells']}\n")
            for i, fc in enumerate(
                    summary_stats['reconstructed_shell_face_counts']):
                f.write(f"    - Shell {i+1}: {fc} faces\n")
            f.write(f"  - Total faces: "
                   f"{summary_stats['reconstructed_total_faces']}\n")
            if summary_stats['reconstructed_volume'] is not None:
                f.write(f"  - Volume: "
                       f"{summary_stats['reconstructed_volume']:.6f} mm³\n")
            
            # Add face matching information
            if 'matched_faces' in summary_stats:
                f.write("\nFace Matching:\n")
                f.write(f"  - Matched faces: {summary_stats['matched_faces']}"
                       f"/{summary_stats['original_num_faces']}\n")
                f.write(f"  - Unmatched reconstructed faces: "
                       f"{summary_stats['unmatched_reconstructed_faces']}\n")
                f.write(f"  - Missing original faces: "
                       f"{summary_stats['missing_original_faces']}\n")
                
                # List unmatched reconstructed faces with vertices
                if 'face_matches' in locals() and num_unmatched_recon > 0:
                    f.write("\nUnmatched Reconstructed Faces (with vertices):\n")
                    for match in face_matches:
                        if not match['matched']:
                            recon_face = extracted_occ_faces[match['recon_idx']]
                            vertices = recon_face['outer_boundary']
                            f.write(f"  Face {match['recon_idx'] + 1}:\n")
                            for i, v in enumerate(vertices):
                                f.write(f"    v{i}: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]\n")
                
                # List missing original faces with vertices
                if 'matched_original' in locals() and num_unmatched_orig > 0:
                    f.write("\nMissing Original Faces (with vertices):\n")
                    for orig_idx, is_matched in enumerate(matched_original):
                        if not is_matched:
                            orig_face = face_polygons[orig_idx]
                            outer_key = 'outer_boundary' if 'outer_boundary' in orig_face else 'vertices'
                            if outer_key in orig_face:
                                vertices = orig_face[outer_key]
                                f.write(f"  Original Face {orig_idx + 1}:\n")
                                for v_idx, vertex in enumerate(vertices):
                                    f.write(f"    v{v_idx}: [{vertex[0]:.2f}, {vertex[1]:.2f}, {vertex[2]:.2f}]\n")
        else:
            f.write("  - Construction failed\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "="*70)
    print("[COMPLETED] Reconstruction process finished.")
    print("="*70)
    
    # Close all matplotlib resources to prevent hanging
    if not args.no_graphics:
        print("[DEBUG] About to call plt.close('all')...")
        plt.close('all')
        print("[DEBUG] Returned from plt.close('all')")
    print("[DEBUG] Exiting main() function...")


if __name__ == "__main__":
    main()
    print("[DEBUG] Returned from main(), program should exit now...")
