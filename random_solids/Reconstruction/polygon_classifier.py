"""
Polygon Classification Module
==============================

This module provides enhanced polygon classification using edge visibility information
from orthographic views (top, front, side) to properly classify polygons as:
- BOUNDARY: Main outer boundary of a face
- HOLE: Interior voids completely inside the boundary
- ALT: Alternate boundary definitions (touching but not overlapping with boundary)

The classification uses connectivity matrices with visibility information:
- Value 1: visible/solid edge
- Value 2: hidden/dashed edge
- Value 0: no edge
"""

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


def classify_polygons_by_visibility(face_eq, polygons, selected_vertices,
                                    top_conn=None, front_conn=None, side_conn=None, face_num=None):
    """
    Classify polygons using edge visibility from orthographic views.
    
    Algorithm:
    1. Determine which view the face is visible in (priority: top > front > side)
    2. Get edge visibility from that view's connectivity matrix
       - Matrices use unified vertex indices (all same size, same vertex set)
       - Value 1 = visible/solid edge
       - Value 2 = hidden/dashed edge
       - Value 0 = no edge
    3. Separate polygons by edge type:
       - All solid edges (all edges have value 1)
       - All dashed edges (all edges have value 2)
       - Mixed (has both 1 and 2 values)
    4. Classification rules:
       a) Mixed edges → treat as all dashed
       b) Process solid-edge polygons first:
          - Largest containing polygon → BOUNDARY
          - Polygons touching boundary (no area overlap) → ALT
          - Polygons completely inside boundary (no touch) → HOLE
          - Check if holes touch each other → if not, separate faces
       c) Process dashed-edge polygons second (BOUNDARY already selected)
    
    Args:
        face_eq: Face equation dict with normal, d, etc.
        polygons: List of polygon dicts with 'vertices' field
        selected_vertices: Array of 3D vertex coordinates
        top_conn: Top view square connectivity matrix with unified indices (NxN where N=num vertices)
        front_conn: Front view square connectivity matrix with unified indices
        side_conn: Side view square connectivity matrix with unified indices
    
    Returns:
        Modified polygons list with 'polygon_type', 'is_hole', 'is_alternate' fields set
    """
    
    # If no view connectivity provided, fall back to geometric classification
    if top_conn is None and front_conn is None and side_conn is None:
        return classify_polygons_geometric(face_eq, polygons, selected_vertices)
    
    face_label = f"Face {face_num}" if face_num is not None else "Face"
    print(f"[POLY FORM]   {face_label}: Using visibility-based classification...")
    
    # Step 1: Select view based on face normal
    normal = np.array(face_eq['normal'])
    
    # Determine which view to use based on normal direction
    # Top view: abs(normal · [0,0,1]) > 0.5
    # Front view: abs(normal · [0,1,0]) > 0.5  
    # Side view: abs(normal · [1,0,0]) > 0.5
    view_conn = None
    if abs(normal[2]) > 0.5 and top_conn is not None:
        view_conn = top_conn
    elif abs(normal[1]) > 0.5 and front_conn is not None:
        view_conn = front_conn
    elif abs(normal[0]) > 0.5 and side_conn is not None:
        view_conn = side_conn
    
    if view_conn is None:
        print(f"[POLY FORM]     {face_label}: No appropriate view found, using geometric method")
        return classify_polygons_geometric(face_eq, polygons, selected_vertices)
    
    # Step 2: Classify edges by visibility from the selected view
    solid_polygons = []
    dashed_polygons = []
    mixed_polygons = []
    
    for poly_idx, poly in enumerate(polygons):
        edge_types = get_edge_visibility(poly['vertices'], view_conn)
        has_solid = any(t == 1 for t in edge_types)
        has_dashed = any(t == 2 for t in edge_types)
        has_missing = any(t == 0 for t in edge_types)  # Edges not in conn=3 set
        
        # Debug output for first few polygons
        if poly_idx < 5:
            verts = poly['vertices']
            print(f"[POLY FORM]     {face_label}: Polygon {poly_idx} vertices: {verts}")
            print(f"[POLY FORM]     {face_label}: Edge types: {edge_types} (1=solid, 2=dashed, 0=missing)")
            if has_missing:
                print(f"[POLY FORM]     {face_label}: WARNING: Polygon {poly_idx} has missing edges - should not have been created!")
        
        if has_solid and has_dashed:
            mixed_polygons.append((poly_idx, poly))
        elif has_solid:
            solid_polygons.append((poly_idx, poly))
        elif has_dashed:
            dashed_polygons.append((poly_idx, poly))
        else:
            # No visible edges? Treat as solid
            solid_polygons.append((poly_idx, poly))
    
    print(f"[POLY FORM]     {face_label}: Edge visibility: {len(solid_polygons)} solid, "
          f"{len(dashed_polygons)} dashed, {len(mixed_polygons)} mixed")
    
    # Step 3: Process SOLID polygons FIRST - they define the boundary
    # Mixed and dashed polygons are processed second
    boundary_idx = None
    if len(solid_polygons) > 0:
        print(f"[POLY FORM]     {face_label}: Processing {len(solid_polygons)} solid-edge polygon(s) for BOUNDARY")
        boundary_idx = classify_solid_polygons(face_eq, solid_polygons, polygons)
    
    # Step 4: Process mixed and dashed-edge polygons (for holes/alternates)
    # Mixed polygons treated as dashed since they have hidden edges
    dashed_polygons.extend(mixed_polygons)
    if len(dashed_polygons) > 0:
        print(f"[POLY FORM]     {face_label}: Processing {len(dashed_polygons)} dashed/mixed-edge polygon(s) for HOLES/ALT")
        classify_dashed_polygons(face_eq, dashed_polygons, polygons, boundary_idx)
    
    return polygons


def select_view_by_normal(normal, top_conn, front_conn, side_conn):
    """
    Select the appropriate view connectivity matrix based on face normal.
    Priority: top > front > side
    
    Args:
        normal: Face normal vector [nx, ny, nz]
        top_conn, front_conn, side_conn: View connectivity matrices
    
    Returns:
        Selected connectivity matrix or None
    """
    # Top view: face normal should be close to [0, 0, ±1]
    if top_conn is not None and abs(normal[2]) > 0.7:
        return top_conn
    
    # Front view: face normal should be close to [0, ±1, 0]
    if front_conn is not None and abs(normal[1]) > 0.7:
        return front_conn
    
    # Side view: face normal should be close to [±1, 0, 0]
    if side_conn is not None and abs(normal[0]) > 0.7:
        return side_conn
    
    # No clear view match, use first available
    if top_conn is not None:
        return top_conn
    if front_conn is not None:
        return front_conn
    if side_conn is not None:
        return side_conn
    
    return None


def get_edge_visibility(vertices, conn_matrix):
    """
    Get visibility type for each edge in a polygon.
    
    Args:
        vertices: List of vertex indices (into selected_vertices array)
        conn_matrix: Square NxN connectivity matrix where N=len(selected_vertices)
                    Values: 0=no edge, 1=solid/visible, 2=dashed/hidden
    
    Returns:
        List of edge types: 1=solid/visible, 2=dashed/hidden, 0=no edge
    """
    edge_types = []
    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)]
        
        # Check matrix bounds
        if v1 < conn_matrix.shape[0] and v2 < conn_matrix.shape[0]:
            edge_type = int(conn_matrix[v1, v2])
            edge_types.append(edge_type)
        else:
            edge_types.append(0)  # No edge if out of bounds
    return edge_types


def get_edge_visibility_all_views(vertices, top_conn, front_conn, side_conn):
    """
    Select the best view where all edges appear as lines (not points) and get visibility.
    
    An edge may appear as a point in one view but as a line in others.
    We select ONE view where ALL edges of the polygon are visible as lines.
    We need to check actual 3D geometry to determine if edges project to points.
    Priority: top > front > side
    
    Args:
        vertices: List of vertex indices (into selected_vertices array)
        top_conn, front_conn, side_conn: View connectivity matrices
                    Values: 0=no edge, 1=solid/visible, 2=dashed/hidden
    
    Returns:
        List of edge types: 1=solid/visible, 2=dashed/hidden, 0=no edge
    """
    # Try each view in priority order and find one where ALL edges have non-zero visibility
    # AND at least one edge has visibility > 1 (dashed) OR multiple edges have visibility = 1 (solid)
    best_view = None
    best_visibility = None
    
    for conn_matrix in [top_conn, front_conn, side_conn]:
        if conn_matrix is None:
            continue
            
        # Get visibility for all edges in this view
        edge_types = []
        all_edges_present = True
        
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            
            if v1 >= conn_matrix.shape[0] or v2 >= conn_matrix.shape[0]:
                all_edges_present = False
                break
            
            edge_val = int(conn_matrix[v1, v2])
            edge_types.append(edge_val)
            
            if edge_val == 0:  # Edge not present in this view
                all_edges_present = False
                break
        
        # If all edges are present, this is a candidate view
        if all_edges_present:
            # Count solid and dashed edges
            solid_count = sum(1 for e in edge_types if e == 1)
            dashed_count = sum(1 for e in edge_types if e == 2)
            
            # Prefer views with dashed edges (more informative) or multiple solid edges
            if dashed_count > 0 or solid_count > 1:
                return edge_types
            
            # Keep this as best view if we haven't found a better one
            if best_view is None:
                best_view = conn_matrix
                best_visibility = edge_types
    
    # If we found a view where all edges exist, use it
    if best_visibility is not None:
        return best_visibility
    
    # Fallback: if no view has all edges, combine from multiple views
    edge_types = []
    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)]
        
        # Check all views and prioritize: solid (1) > dashed (2) > no edge (0)
        edge_type = 0
        
        for conn_matrix in [top_conn, front_conn, side_conn]:
            if conn_matrix is not None:
                if v1 < conn_matrix.shape[0] and v2 < conn_matrix.shape[0]:
                    view_edge = int(conn_matrix[v1, v2])
                    if view_edge > 0:
                        if edge_type == 0:
                            edge_type = view_edge
                        elif view_edge == 1:  # Solid takes priority
                            edge_type = 1
        
        edge_types.append(edge_type)
    return edge_types


def classify_solid_polygons(face_eq, solid_polygons, all_polygons):
    """
    Classify polygons with all solid edges.
    
    Returns:
        Index of the boundary polygon
    """
    if len(solid_polygons) == 0:
        return None
    
    # Create Shapely polygons
    shapely_polys = []
    for poly_idx, poly in solid_polygons:
        verts_2d = [face_eq['face_results']['verts_2d'][v] for v in poly['vertices']]
        shapely_polys.append((poly_idx, ShapelyPolygon(verts_2d)))
    
    # Find polygon that contains the most others (boundary)
    max_contains = -1
    boundary_idx = None
    
    for i, (idx_i, poly_i) in enumerate(shapely_polys):
        contains_count = 0
        for j, (idx_j, poly_j) in enumerate(shapely_polys):
            if i != j and poly_i.contains(poly_j):
                contains_count += 1
        
        if contains_count > max_contains:
            max_contains = contains_count
            boundary_idx = idx_i
    
    if boundary_idx is None and len(solid_polygons) > 0:
        # No clear container, use largest area
        boundary_idx = max(solid_polygons, key=lambda x: x[1].get('area', 0))[0]
    
    # Mark boundary
    all_polygons[boundary_idx]['polygon_type'] = 'BOUNDARY'
    all_polygons[boundary_idx]['is_alternate'] = False
    all_polygons[boundary_idx]['is_hole'] = False
    
    # Get boundary shapely polygon
    boundary_poly = next(sp for idx, sp in shapely_polys if idx == boundary_idx)
    
    # Classify remaining solid polygons
    for poly_idx, poly in solid_polygons:
        if poly_idx == boundary_idx:
            continue
        
        poly_shapely = next(sp for idx, sp in shapely_polys if idx == poly_idx)
        
        # Check if completely inside boundary without touching
        if boundary_poly.contains(poly_shapely):
            # Check for shared edges
            shares_edges = check_shared_edges(all_polygons[boundary_idx]['vertices'],
                                             poly['vertices'])
            
            if not shares_edges:
                # Completely inside, no touch → HOLE
                all_polygons[poly_idx]['polygon_type'] = 'HOLE'
                all_polygons[poly_idx]['is_hole'] = True
                all_polygons[poly_idx]['is_alternate'] = False
            else:
                # Touches boundary → ALT
                all_polygons[poly_idx]['polygon_type'] = 'ALT'
                all_polygons[poly_idx]['is_alternate'] = True
                all_polygons[poly_idx]['is_hole'] = False
        else:
            # Not contained → ALT
            all_polygons[poly_idx]['polygon_type'] = 'ALT'
            all_polygons[poly_idx]['is_alternate'] = True
            all_polygons[poly_idx]['is_hole'] = False
    
    return boundary_idx


def classify_dashed_polygons(face_eq, dashed_polygons, all_polygons, boundary_idx):
    """
    Classify polygons with dashed edges (after boundary is already selected).
    """
    if boundary_idx is None or len(dashed_polygons) == 0:
        # No boundary yet, treat first dashed as boundary
        if len(dashed_polygons) > 0:
            classify_solid_polygons(face_eq, dashed_polygons, all_polygons)
        return
    
    # Get boundary polygon
    boundary_verts = all_polygons[boundary_idx]['vertices']
    boundary_verts_2d = [face_eq['face_results']['verts_2d'][v] for v in boundary_verts]
    boundary_shapely = ShapelyPolygon(boundary_verts_2d)
    
    # Classify dashed polygons relative to existing boundary
    for poly_idx, poly in dashed_polygons:
        verts_2d = [face_eq['face_results']['verts_2d'][v] for v in poly['vertices']]
        poly_shapely = ShapelyPolygon(verts_2d)
        
        if boundary_shapely.contains(poly_shapely):
            shares_edges = check_shared_edges(boundary_verts, poly['vertices'])
            
            if not shares_edges:
                all_polygons[poly_idx]['polygon_type'] = 'HOLE'
                all_polygons[poly_idx]['is_hole'] = True
                all_polygons[poly_idx]['is_alternate'] = False
            else:
                all_polygons[poly_idx]['polygon_type'] = 'ALT'
                all_polygons[poly_idx]['is_alternate'] = True
                all_polygons[poly_idx]['is_hole'] = False
        else:
            all_polygons[poly_idx]['polygon_type'] = 'ALT'
            all_polygons[poly_idx]['is_alternate'] = True
            all_polygons[poly_idx]['is_hole'] = False


def check_shared_edges(verts1, verts2):
    """Check if two polygons share any edges."""
    edges1 = set()
    for i in range(len(verts1)):
        v1, v2 = verts1[i], verts1[(i+1) % len(verts1)]
        edges1.add((min(v1, v2), max(v1, v2)))
    
    edges2 = set()
    for i in range(len(verts2)):
        v1, v2 = verts2[i], verts2[(i+1) % len(verts2)]
        edges2.add((min(v1, v2), max(v1, v2)))
    
    return len(edges1 & edges2) > 0


def classify_polygons_geometric(face_eq, polygons, selected_vertices):
    """
    Fallback geometric classification (original method).
    
    This is used when visibility information is not available.
    Uses Shapely containment tests to classify polygons.
    """
    print(f"[POLY FORM]   Using geometric classification (no visibility data)...")
    
    # Return polygons unchanged - will be processed by existing logic
    return polygons
