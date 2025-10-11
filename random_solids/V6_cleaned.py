def calculate_depth_along_normal(point_3d, projection_normal):
    """Calculate depth of a 3D point along the projection normal."""
    if point_3d is None:
        return 0
    try:
        return np.dot(point_3d, projection_normal)
    except Exception:
        return 0
def intersect_line_with_face(point_2d, projection_normal, face_vertices_3d):
    """Intersect a line with a 3D face to find depth."""
    try:
        if face_vertices_3d is None or len(face_vertices_3d) < 3:
            return None

        # 1. Create orthogonal basis vectors for the projection plane
        normal = np.array(projection_normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        if abs(normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        u = temp - np.dot(temp, normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        # 2. Convert the 2D intersection point to a 3D point on the projection plane
        # Accept both tuple/list and shapely Point
        if hasattr(point_2d, 'x') and hasattr(point_2d, 'y'):
            px, py = point_2d.x, point_2d.y
        elif isinstance(point_2d, (tuple, list, np.ndarray)) and len(point_2d) == 2:
            px, py = point_2d[0], point_2d[1]
        else:
            return None
        plane_origin = np.array([0, 0, 0])  # Simplification: origin
        point_3d_on_plane = plane_origin + px * u + py * v

        # 3. Define the face plane using the first three vertices
        v0, v1, v2 = np.array(face_vertices_3d[0]), np.array(face_vertices_3d[1]), np.array(face_vertices_3d[2])

        # 4. Compute the face normal using the robust function
        face_normal = np.cross(v1 - v0, v2 - v0)
        if np.linalg.norm(face_normal) < 1e-8:
            return None
        face_normal = face_normal / np.linalg.norm(face_normal)

        # 5. Ray-plane intersection
        denominator = np.dot(normal, face_normal)
        if abs(denominator) > 1e-6:
            t = np.dot((v0 - point_3d_on_plane), face_normal) / denominator
            intersection_3d = point_3d_on_plane + t * normal
            return intersection_3d
    except Exception as e:
        print(f"Error in line-face intersection: {e}")
        return None
def get_face_normal_from_opencascade(face):
    """
    Extract the correct face normal from OpenCASCADE face using robust surface derivatives and orientation.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.gp import gp_Pnt
    try:
        face_orientation = face.Orientation()
        face_name = getattr(face, 'Name', None)
        print(f"[DEBUG] Face: {face_name}, orientation value: {face_orientation}")
        surface = BRepAdaptor_Surface(face)
        u_min = surface.FirstUParameter()
        u_max = surface.LastUParameter()
        v_min = surface.FirstVParameter()
        v_max = surface.LastVParameter()
        u_mid = (u_min + u_max) / 2.0
        v_mid = (v_min + v_max) / 2.0
        point = surface.Value(u_mid, v_mid)
        d1u = surface.DN(u_mid, v_mid, 1, 0)
        d1v = surface.DN(u_mid, v_mid, 0, 1)
        normal_vec = d1u.Crossed(d1v)
        if normal_vec.Magnitude() > 1e-10:
            normal_vec.Normalize()
            orientation_multiplier = 1.0
            if face_orientation == TopAbs_REVERSED:
                print(f"[DEBUG] Face orientation is REVERSED (TopAbs_REVERSED). Flipping normal.")
                orientation_multiplier = -1.0
            face_normal = np.array([
                normal_vec.X() * orientation_multiplier,
                normal_vec.Y() * orientation_multiplier,
                normal_vec.Z() * orientation_multiplier
            ])
            print(f"[DEBUG] Face: {face_name}, computed normal: [{face_normal[0]:.4f}, {face_normal[1]:.4f}, {face_normal[2]:.4f}]")
            return face_normal
    except Exception as e:
        print(f"Error in get_face_normal_from_opencascade: {e}")
        return None

def extract_ordered_wire_vertices(wire):
    """
    Given an OpenCASCADE wire, extract the ordered list of 3D vertices forming the polygon boundary.
    Returns a list of (x, y, z) tuples in order.
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    # Step 1: Collect all edges and their vertices
    edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    edges = []
    vertices_map = {}
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        vtx_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        vtx_coords = []
        while vtx_explorer.More():
            vtx = topods.Vertex(vtx_explorer.Current())
            pnt = BRep_Tool.Pnt(vtx)
            vtx_coords.append((pnt.X(), pnt.Y(), pnt.Z()))
            vtx_explorer.Next()
        if len(vtx_coords) == 2:
            edges.append((vtx_coords[0], vtx_coords[1]))
        edge_explorer.Next()
    if not edges:
        return []
    used_edges = set()
    # Start with first edge
    v_start, v_next = edges[0]
    ordered = [v_start, v_next]
    used_edges.add(0)
    while len(used_edges) < len(edges):
        last_v = ordered[-1]
        found = False
        for idx, (a, b) in enumerate(edges):
            if idx in used_edges:
                continue
            if all(abs(x - y) < 1e-8 for x, y in zip(last_v, a)):
                ordered.append(b)
                used_edges.add(idx)
                found = True
                break
            elif all(abs(x - y) < 1e-8 for x, y in zip(last_v, b)):
                ordered.append(a)
                used_edges.add(idx)
                found = True
                break
        if not found:
            break  # No connecting edge found
    # Remove consecutive duplicates
    result = [ordered[0]]
    for v in ordered[1:]:
        if not all(abs(a - b) < 1e-8 for a, b in zip(v, result[-1])):
            result.append(v)
    # Ensure closure
    if result and not all(abs(a - b) < 1e-8 for a, b in zip(result[0], result[-1])):
        result.append(result[0])
    return result
import numpy as np
from shapely.geometry import Polygon
from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE, TopAbs_FACE
from OCC.Core.BRep import BRep_Tool

def find_interior_point(polygon, debug=False):
    """
    Return a guaranteed interior point using Shapely's representative_point().
    """
    if polygon.is_empty:
        if debug:
            print("[DEBUG] Polygon is empty, no interior point.")
        return None, 'none'
    pt = polygon.representative_point()
    if debug:
        print(f"[DEBUG] Shapely representative_point: ({pt.x}, {pt.y})")
    return np.array([pt.x, pt.y]), 'representative_point'

  
def create_polygon_from_projection(projected_vertices, allow_invalid=False):
    if len(projected_vertices) == 0:
        return Polygon()
    projected_vertices = np.array(projected_vertices)
    # original_vertex_count = len(projected_vertices)  # Unused
    if len(projected_vertices) > 0:
        # Remove duplicate last vertex if present
        if np.allclose(projected_vertices[0], projected_vertices[-1]):
            projected_vertices = projected_vertices[:-1]
    try:
        poly = Polygon(projected_vertices)
        if not poly.is_valid and allow_invalid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return Polygon()
# Project 3D face vertices to a 2D plane for engineering drawing display
  
def project_face_to_projection_plane(face_vertices, projection_normal):
    """
    Project 3D face vertices to a 2D plane for engineering drawing display.
    """
    if face_vertices is None or len(face_vertices) == 0 or projection_normal is None:
        return np.zeros((0, 2))
    face_vertices = np.array(face_vertices)
    projection_normal = np.array(projection_normal)
    normal = projection_normal / np.linalg.norm(projection_normal)
    # Create two orthogonal vectors in the projection plane
    if abs(normal[0]) < 0.9:
        temp = np.array([1.0, 0.0, 0.0])
    else:
        temp = np.array([0.0, 1.0, 0.0])
    u = temp - np.dot(temp, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    projected = []
    for vertex in face_vertices:
        proj_u = np.dot(vertex, u)
        proj_v = np.dot(vertex, v)
        projected.append([proj_u, proj_v])
    return np.array(projected)
# Robust normal extraction copied from V6_current.py
  
def get_face_normal_from_opencascade(face):
    """
    Extract the correct face normal from OpenCASCADE face using multiple robust methods.
    Tries several approaches to get the correct outward-pointing normal:
      1. GeomLProp_SLProps with orientation
      2. Surface derivatives with orientation
      3. BRepGProp_Face method
      4. Geometric analysis fallback
    """
    try:
    # from OCC.Core.BRep import BRep_Tool  # Unused
        surf = BRep_Tool.Surface(face)
        # Try to get UV bounds
        try:
            from OCC.Core.BRepTools import breptools_UVBounds
            umin, umax, vmin, vmax = breptools_UVBounds(face)
            u = (umin + umax) / 2.0
            v = (vmin + vmax) / 2.0
        except Exception:
            u, v = 0.5, 0.5
        try:
            gp_normal = surf.Normal(u, v)
            normal = np.array([gp_normal.X(), gp_normal.Y(), gp_normal.Z()])
            if np.linalg.norm(normal) > 1e-8:
                return normal
        except Exception:
            pass
        # Fallback: use BRepGProp_Face if available
        try:
            from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
            from OCC.Core.GProp import GProp_GProps
            props = GProp_GProps()
            brepgprop_SurfaceProperties(face, props)
            center = props.CentreOfMass()
            # Try normal at center
            gp_normal = surf.Normal(center.X(), center.Y())
            normal = np.array([gp_normal.X(), gp_normal.Y(), gp_normal.Z()])
            if np.linalg.norm(normal) > 1e-8:
                return normal
        except Exception:
            pass
        # Last fallback: geometric analysis (difference of first two vertices)
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_VERTEX
            from OCC.Core.TopoDS import topods
            vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
            pts = []
            while vertex_explorer.More():
                v = topods.Vertex(vertex_explorer.Current())
                p = BRep_Tool.Pnt(v)
                pts.append(np.array([p.X(), p.Y(), p.Z()]))
                vertex_explorer.Next()
            if len(pts) >= 3:
                v1, v2, v3 = pts[:3]
                normal = np.cross(v2 - v1, v3 - v1)
                if np.linalg.norm(normal) > 1e-8:
                    return normal / np.linalg.norm(normal)
        except Exception:
            pass
    except Exception:
        pass
    return None


def classify_faces_by_projection(face_polygons, unit_projection_normal):
    array_B = []
    # ...existing code for initial classification and array_B population...
    # ...existing code for initial classification and array_B population...
    # ...existing code for initial classification and array_B population...

    # After all main processing, explicitly check and print intersection for all pairs in Array_B
    print("\n[DEBUG] Explicit intersection checks for all pairs in Array_B:")
    for i in range(len(array_B)):
        for j in range(i + 1, len(array_B)):
            Pi_data = array_B[i]
            Pj_data = array_B[j]
            Pi = Pi_data.get('polygon', None)
            Pj = Pj_data.get('polygon', None)
            Pi_name = Pi_data.get('name', f"Face_{i + 1}")
            Pj_name = Pj_data.get('name', f"Face_{j + 1}")
            if Pi is None or Pj is None:
                print(f"[DEBUG] Skipping intersection: {Pi_name} or {Pj_name} is None")
                continue
            intersection = Pi.intersection(Pj)
            area_val = getattr(intersection, 'area', None)
            print(f"[DEBUG] Intersection check: {Pi_name} ∩ {Pj_name}, "
                  f"area={area_val}")
            if (not intersection.is_empty and hasattr(intersection, 'area') and
                area_val is not None and area_val > 1e-6):
                # Find interior point for depth analysis
                result = find_interior_point(intersection, debug=False)
                if isinstance(result, tuple):
                    interior_point, method_used = result
                else:
                    interior_point = result
                Pi_intersection_3d = intersect_line_with_face(
                    interior_point, unit_projection_normal,
                    Pi_data.get('parent_face'))
                Pj_intersection_3d = intersect_line_with_face(
                    interior_point, unit_projection_normal,
                    Pj_data.get('parent_face'))
                Pi_depth = calculate_depth_along_normal(
                    Pi_intersection_3d, unit_projection_normal)
                Pj_depth = calculate_depth_along_normal(
                    Pj_intersection_3d, unit_projection_normal)
                print(f"[DEBUG] Depths at intersection point: {Pi_name}={Pi_depth:.4f}, "
                      f"{Pj_name}={Pj_depth:.4f}")
    # ...existing code...
    # After all classification and intersection logic, print debug info for Array_B polygons
    # Move this block to just before the return statement
    """
    Enhanced face classification with historic polygon classification algorithm (intersection/depth-based).
    """
    print("\n" + "="*60)
    print("ENHANCED FACE CLASSIFICATION WITH HISTORIC ALGORITHM")
    print("="*60)
    array_A_initial = []  # Initial classification for processing
    array_B = []  # Depth-processed polygons (visible)
    array_C = []  # Hidden faces + intersections
    print(f"Unit projection normal: [{unit_projection_normal[0]:.6f}, {unit_projection_normal[1]:.6f}, {unit_projection_normal[2]:.6f}]")
    print("\nStep 1: Initial classification and polygon projection...")
    valid_polygons = []
    for i, polygon_data in enumerate(face_polygons):
        face_id = polygon_data.get('face_id', i+1)
        face_normal = polygon_data.get('normal')
        outer_boundary = polygon_data.get('outer_boundary', [])
        if face_normal is None or len(outer_boundary) < 3:
            print(f"[DEBUG] Face F{face_id}: Invalid data - skipping (normal={face_normal}, vertices={len(outer_boundary)})")
            continue

        unit_face_normal = face_normal / np.linalg.norm(face_normal)
        dot_product = np.dot(unit_face_normal, unit_projection_normal)
        print(f"[SBDEBUG] Face F{face_id}: dot_product={dot_product:.3f} unit_face_normal={unit_face_normal:.3f}, unit_projection_normal={unit_projection_normal:.3f}")
        try:
            projected_outer = project_face_to_projection_plane(outer_boundary, unit_projection_normal)
            cutouts = polygon_data.get('cutouts', [])
            projected_holes = []
            for cutout in cutouts:
                if cutout and len(cutout) >= 3:
                    projected_cutout = project_face_to_projection_plane(cutout, unit_projection_normal)
                    projected_holes.append(projected_cutout)
            if projected_holes:
                def signed_area(coords):
                    # Shoelace formula for signed area (positive: CCW, negative: CW)
                    x = [c[0] for c in coords]
                    y = [c[1] for c in coords]
                    return 0.5 * sum(x[i] * y[(i+1)%len(coords)] - x[(i+1)%len(coords)] * y[i] for i in range(len(coords)))

                print(f"[DEBUG] Face {face_id}: Constructing polygon with {len(projected_holes)} hole(s)")
                for h_idx, hole in enumerate(projected_holes):
                    area = signed_area(hole)
                    orientation = 'CCW' if area > 0 else 'CW'
                    print(f"[DEBUG]   Hole {h_idx+1} winding: {orientation}, signed_area={area:.2f}")
                    print(f"[DEBUG]   Hole {h_idx+1} coords: {[tuple(map(lambda x: round(x,2), c)) for c in hole]}")
                polygon = Polygon(projected_outer, holes=projected_holes)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
            else:
                polygon = create_polygon_from_projection(projected_outer, allow_invalid=True)
            if polygon.area > 1e-6:
                polygon_data_enhanced = {
                    'polygon': polygon,
                    'name': f"Face_{face_id}",
                    'normal': unit_face_normal,
                    'parent_face': np.array(outer_boundary),  # 3D vertices
                    'original_index': i,
                    'dot_product': dot_product,
                    'has_holes': len(projected_holes) > 0
                }
                valid_polygons.append(polygon_data_enhanced)
                array_A_initial.append(polygon_data_enhanced)
                hole_info = f" with {len(projected_holes)} holes" if projected_holes else ""
                print(f"  → Added Face_{face_id} (area: {polygon.area:.2f}){hole_info}")
                print(f"      [DEBUG] Face_{face_id} dot_product: {dot_product:.6f} {'(visible)' if dot_product > 0 else '(hidden)'}")
        except Exception as e:
            print(f"Face F{face_id}: Projection error - {e}")

    print(f"\nStep 2: Starting historic polygon classification algorithm...")
    print(f"Initial array_A: {len(array_A_initial)} polygons")
    print(f"[DEBUG] Initial faces in array_A before historic algorithm:")
    for i, poly_data in enumerate(array_A_initial):
        name = poly_data['name']
        dot_product = poly_data['dot_product']
        area = poly_data['polygon'].area
        print(f"  [DEBUG] Array_A[{i}] = {name} (dot={dot_product:.3f}, area={area:.2f})")

    # Print summary of hidden polygons for top view
    if np.allclose(unit_projection_normal, [0, 0, 1], atol=1e-3):
        print("\nSUMMARY: Hidden polygons (Array_C) for Top View:")
        for poly_data in array_C:
            name = poly_data.get('name', 'Unknown')
            dot_product = poly_data.get('dot_product', 0)
            print(f"  - {name} (dot_product={dot_product:.3f})")

    # Display initial array_A contents before sorting
    if array_A_initial:
        print(f"\n" + "="*60)
        print("ARRAY A - INITIAL FACE CLASSIFICATION (BEFORE SORTING)")
        print("="*60)
        for i, poly_data in enumerate(array_A_initial):
            polygon = poly_data['polygon']
            name = poly_data['name']
            normal = poly_data['normal']
            dot_product = poly_data['dot_product']
            # Check for invalid polygons in Array_A
            if not polygon.is_valid:
                from shapely.validation import explain_validity
                reason = explain_validity(polygon)
                print(f"  [WARNING] {name} is invalid in Array_A: {reason}")
                print(f"    [INVESTIGATE] Polygon WKT: {polygon.wkt}")
            # Handle both Polygon and MultiPolygon cases
            if hasattr(polygon, 'exterior'):
                vertex_count = len(polygon.exterior.coords) - 1
                coords = list(polygon.exterior.coords[:-1])
            elif hasattr(polygon, 'geoms') and len(polygon.geoms) > 0:
                largest_poly = max(polygon.geoms, key=lambda p: p.area)
                vertex_count = len(largest_poly.exterior.coords) - 1
                coords = list(largest_poly.exterior.coords[:-1])
            else:
                vertex_count = 0
                coords = []
            print(f"  Face A{i+1} ({name}):")
            print(f"    • Area: {polygon.area:.2f}")
            print(f"    • Vertices: {vertex_count}")
            print(f"    • Dot product: {dot_product:.6f}")
            if hasattr(normal, '__len__') and len(normal) >= 3 and not isinstance(normal, str):
                try:
                    print(f"    • Face normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                except (TypeError, IndexError):
                    print(f"    • Face normal: {normal}")
            else:
                print(f"    • Face normal: {normal}")
            if coords:
                coords_str = " → ".join([f"({c[0]:.1f},{c[1]:.1f})" for c in coords])
                print(f"    • 2D Polygon: {coords_str}")
            else:
                print(f"    • 2D Polygon: [No coordinates available]")
        print("="*60)
        print("ARRAY A DISPLAY COMPLETE - NOW STARTING HISTORIC ALGORITHM")
        print("="*60)

    if len(array_A_initial) >= 1:
        # Step 2.1: Move first polygon to array_B as seed
        first_polygon = array_A_initial.pop(0)
        array_B.append(first_polygon)
        print(f"[DEBUG] Moved {first_polygon['name']} from array_A[0] to array_B as seed")
        # Step 2.2: Process remaining polygons with depth-based classification
        idx = 0
        while array_A_initial:
            Pi_data = array_A_initial.pop(0)
            print(f"[DEBUG] Popped {Pi_data['name']} from array_A[{idx}] for processing")
            Pi = Pi_data.get('polygon', None)
            Pi_name = Pi_data.get('name', None)
            Pi_parent_face = Pi_data.get('parent_face', None)
            idx += 1
            for j in reversed(range(len(array_B))):
                Pj_data = array_B[j]
                Pj = Pj_data.get('polygon', None)
                Pj_name = Pj_data.get('name', None)
                Pj_parent_face = Pj_data.get('parent_face', None)
                if Pi is None or Pj is None:
                    print(f"[DEBUG] Skipping intersection: Pi or Pj is None ({Pi_name}, {Pj_name})")
                    Pi_data = None
                    continue
                try:
                    intersection = Pi.intersection(Pj)
                    print(f"[DEBUG] Checking intersection: {Pi_name} vs {Pj_name}, area={getattr(intersection, 'area', None)}")
                    if intersection.is_empty:
                        print(f"[DEBUG] Intersection is empty: {Pi_name} vs {Pj_name}")
                    elif not hasattr(intersection, 'area') or intersection.area <= 1e-6:
                        print(f"[DEBUG] Intersection too small: {Pi_name} vs {Pj_name}, area={getattr(intersection, 'area', None)}")
                    else:
                        result = find_interior_point(intersection, debug=False)
                        if isinstance(result, tuple):
                            interior_point, method_used = result
                        else:
                            interior_point = result
                        if interior_point is None:
                            print(f"[DEBUG] No interior point found for intersection: {Pi_name} vs {Pj_name}")
                            continue
                        try:
                            Pi_intersection_3d = intersect_line_with_face(
                                interior_point, unit_projection_normal, Pi_parent_face)
                            Pj_intersection_3d = intersect_line_with_face(
                                interior_point, unit_projection_normal, Pj_parent_face)
                            Pi_depth = calculate_depth_along_normal(Pi_intersection_3d, unit_projection_normal)
                            Pj_depth = calculate_depth_along_normal(Pj_intersection_3d, unit_projection_normal)
                            print(f"[DEBUG] Depths at intersection point: Pi_depth={Pi_depth:.4f}, Pj_depth={Pj_depth:.4f}")
                            distance = abs(Pi_depth - Pj_depth)
                            print(f"[DEBUG] Calculated distance between faces: {distance:.4f}")
                            intersection_name = f"Intersection_{Pi_name}_{Pj_name}"
                            intersection_data = {
                                'polygon': intersection,
                                'name': intersection_name,
                                'normal': 'intersection',
                                'parent_face': Pi_parent_face if Pi_depth > Pj_depth else Pj_parent_face,
                                'associated_face': Pi_name if Pi_depth > Pj_depth else Pj_name,
                                'original_index': -1,
                                'dot_product': 0
                            }
                            array_C.append(intersection_data)
                            print(f"[DEBUG] Added intersection to Array_C: {intersection_name}, area={intersection.area}")
                            if Pi_depth > Pj_depth:
                                new_Pj = Pj.difference(Pi)
                                if not new_Pj.is_empty and new_Pj.area > 1e-6:
                                    array_B[j]['polygon'] = new_Pj
                                    array_B[j]['name'] = f"Modified_{Pj_name}"
                                else:
                                    array_B.pop(j)
                            else:
                                new_Pi = Pi.difference(Pj)
                                if not new_Pi.is_empty and new_Pi.area > 1e-6:
                                    Pi = new_Pi
                                    Pi_data['polygon'] = new_Pi
                                    Pi_data['name'] = f"Modified_{Pi_name}"
                                else:
                                    Pi_data['polygon'] = new_Pi
                                    break
                        except Exception as e:
                            print(f"[DEBUG] Depth calculation failed for {Pi_name} vs {Pj_name}: {e}")
                            continue
                except Exception as e:
                    print(f"[DEBUG] Exception in intersection loop: {Pi_name} vs {Pj_name}: {e}")
            if Pi_data is not None and Pi_data.get('polygon', None) is not None and hasattr(Pi_data['polygon'], 'area') and Pi_data['polygon'].area > 1e-6:
                array_B.append(Pi_data)
                print(f"[DEBUG] Added {Pi_data['name']} to Array_B after intersection processing, area={Pi_data['polygon'].area}")
        faces_to_move = []
        for i, poly_data in enumerate(array_B):
            if poly_data['dot_product'] <= 0:
                faces_to_move.append(i)
        for i in reversed(faces_to_move):
            moved_face = array_B.pop(i)
            array_C.append(moved_face)

    print("\n===== FINAL ARRAY_B =====")
    for poly_data in array_B:
        print(f"  {poly_data['name']}: area={poly_data['polygon'].area:.2f}, dot={poly_data.get('dot_product', 'N/A')}")
    print("\n===== FINAL ARRAY_C =====")
    for poly_data in array_C:
        print(f"  {poly_data['name']}: area={poly_data['polygon'].area:.2f}, dot={poly_data.get('dot_product', 'N/A')}")
    return [], array_B, array_C
    dists = np.linalg.norm(projected_2d - interior_point_2d, axis=1)
    min_idx = np.argmin(dists)
    closest_3d = np.array(parent_face_3d[min_idx])
    # The intersection point in 3D is along the projection normal passing through closest_3d
    # Since the projection is orthogonal, we can reconstruct the 3D point as:
    # closest_3d + t * projection_normal, where t is chosen so that the projection matches interior_point_2d
    # For simplicity, return closest_3d (approximate)
    add_pj_to_array_b = False
    pj_to_add = None
    pj_area_to_add = None
    for j in reversed(range(len(array_B))):
                Pj_data = array_B[j]
                Pj = Pj_data['polygon']
                Pj_name = Pj_data['name']
                Pj_parent_face = Pj_data['parent_face']
                try:
                    intersection = Pi.intersection(Pj)
                    print(f"[DEBUG] Checking intersection: {Pi_name} vs {Pj_name}, area={getattr(intersection, 'area', None)}")
                    if intersection.is_empty:
                        print(f"[DEBUG] Intersection is empty: {Pi_name} vs {Pj_name}")
                    elif not hasattr(intersection, 'area') or intersection.area <= 1e-6:
                        print(f"[DEBUG] Intersection too small: {Pi_name} vs {Pj_name}, area={getattr(intersection, 'area', None)}")
                    else:
                        # Find interior point for depth analysis
                        result = find_interior_point(intersection, debug=False)
                        if isinstance(result, tuple):
                            interior_point, method_used = result
                        else:
                            interior_point = result
                        if interior_point is None:
                            print(f"[DEBUG] No interior point found for intersection: {Pi_name} vs {Pj_name}")
                            continue
                        # Calculate 3D depths using line-face intersection
                        try:
                            Pi_intersection_3d = intersect_line_with_face(
                                interior_point, unit_projection_normal, Pi_parent_face)
                            Pj_intersection_3d = intersect_line_with_face(
                                interior_point, unit_projection_normal, Pj_parent_face)
                            Pi_depth = calculate_depth_along_normal(Pi_intersection_3d, unit_projection_normal)
                            Pj_depth = calculate_depth_along_normal(Pj_intersection_3d, unit_projection_normal)
                            print(f"[DEBUG] Depths at intersection point: Pi_depth={Pi_depth:.4f}, Pj_depth={Pj_depth:.4f}")
                            if Pi_depth > Pj_depth:
                                print(f"[DEBUG] {Pi_name} is above {Pj_name} at intersection (distance: {Pi_depth - Pj_depth:.4f})")
                            elif Pj_depth > Pi_depth:
                                print(f"[DEBUG] {Pj_name} is above {Pi_name} at intersection (distance: {Pj_depth - Pi_depth:.4f})")
                                if Pj is not None and hasattr(Pj, 'area') and Pj.area > 1e-6:
                                    add_pj_to_array_b = True
                                    pj_to_add = Pj_data
                                    pj_area_to_add = Pj.area
                            else:
                                print(f"[DEBUG] {Pi_name} and {Pj_name} are at the same depth at intersection.")
                        except Exception as e:
                            print(f"[DEBUG] Depth calculation failed for {Pi_name} vs {Pj_name}: {e}")
                            continue
                        # Add intersection to array_C
                        intersection_name = f"Intersection_{Pi_name}_{Pj_name}"
                        intersection_data = {
                            'polygon': intersection,
                            'name': intersection_name,
                            'normal': 'intersection',
                            'parent_face': Pi_parent_face if Pi_depth > Pj_depth else Pj_parent_face,
                            'associated_face': Pi_name if Pi_depth > Pj_depth else Pj_name,
                            'original_index': -1,
                            'dot_product': 0
                        }
                        array_C.append(intersection_data)
                        print(f"[DEBUG] Added intersection to Array_C: {intersection_name}, area={intersection.area}")
                        # Apply depth-based boolean operations
                        if Pj_depth > Pi_depth:
                            try:
                                print(f"[DEBUG] Pj before difference: area={Pj.area}, valid={Pj.is_valid}")
                                print(f"[DEBUG] Pi before difference: area={Pi.area}, valid={Pi.is_valid}")
                                if hasattr(Pj, 'exterior'):
                                    print(f"[DEBUG] Pj exterior coords: {[tuple(map(lambda x: round(x,2), c)) for c in Pj.exterior.coords]}")
                                if hasattr(Pi, 'exterior'):
                                    print(f"[DEBUG] Pi exterior coords: {[tuple(map(lambda x: round(x,2), c)) for c in Pi.exterior.coords]}")
                                new_Pj = Pj.difference(Pi)
                                print(f"[DEBUG] After Pj.difference(Pi): area={getattr(new_Pj, 'area', None)}, valid={getattr(new_Pj, 'is_valid', None)}, empty={getattr(new_Pj, 'is_empty', None)}")
                                if hasattr(new_Pj, 'exterior'):
                                    print(f"[DEBUG] new_Pj exterior coords: {[tuple(map(lambda x: round(x,2), c)) for c in new_Pj.exterior.coords]}")
                                if not new_Pj.is_empty and new_Pj.area > 1e-6:
                                    # If only a hole remains, set outer boundary to reversed hole
                                    if hasattr(new_Pj, 'interiors') and len(new_Pj.interiors) == 1 and (not hasattr(new_Pj, 'exterior') or len(new_Pj.exterior.coords) <= 2):
                                        hole_coords = list(new_Pj.interiors[0].coords)
                                        new_Pj = Polygon(hole_coords[::-1])
                                    array_B[j]['polygon'] = new_Pj
                                    array_B[j]['name'] = f"Modified_{Pj_name}"
                                else:
                                    array_B.pop(j)
                            except Exception as e:
                                print(f"[DEBUG] Exception during Pj.difference(Pi): {e}")
                        elif Pi_depth > Pj_depth:
                            try:
                                new_Pi = Pi.difference(Pj)
                                print(f"[DEBUG] After Pi.difference(Pj): Pi area={getattr(new_Pi, 'area', None)}, Pj area={getattr(Pj, 'area', None)}")
                                if not new_Pi.is_empty and new_Pi.area > 1e-6:
                                    # If only a hole remains, set outer boundary to reversed hole
                                    if hasattr(new_Pi, 'interiors') and len(new_Pi.interiors) == 1 and (not hasattr(new_Pi, 'exterior') or len(new_Pi.exterior.coords) <= 2):
                                        hole_coords = list(new_Pi.interiors[0].coords)
                                        new_Pi = Polygon(hole_coords[::-1])
                                    Pi = new_Pi
                                    Pi_data['polygon'] = new_Pi
                                    Pi_data['name'] = f"Modified_{Pi_name}"
                                else:
                                    Pi_data['polygon'] = new_Pi
                                    # Do not add Pi to Array_B if empty
                            except Exception as e:
                                print(f"[DEBUG] Exception during Pi.difference(Pj): {e}")
                except Exception as e:
                    print(f"[DEBUG] Exception in intersection loop: {Pi_name} vs {Pj_name}: {e}")
            # After all intersections, add Pi to array_B if it still has area
    if Pi_data['polygon'] is not None and hasattr(Pi_data['polygon'], 'area') and Pi_data['polygon'].area > 1e-6:
        array_B.append(Pi_data)
        print(f"[DEBUG] Added {Pi_data['name']} to Array_B after intersection processing, area={Pi_data['polygon'].area}")
    # If any Pj was above Pi and had area, add it to Array_B
    if add_pj_to_array_b and pj_to_add is not None:
        array_B.append(pj_to_add)
        print(f"[DEBUG] Added {pj_to_add['name']} to Array_B because it is above and has area={pj_area_to_add}")
def extract_faces_from_solid(solid):
    from OCC.Core.BRep import BRep_Tool
    faces = []
    shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
    shell_count = 0
    while shell_explorer.More():
        shell_count += 1
        shell = shell_explorer.Current()
        print(f"[DEBUG] Shell {shell_count}:")
        face_explorer = TopExp_Explorer(shell, TopAbs_FACE)
        face_idx = 0
        while face_explorer.More():
            face_idx += 1
            face = topods.Face(face_explorer.Current())
            outer_boundary = None
            cutouts = []
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            wire_count = 0
            wires = []
            while wire_explorer.More():
                wire = wire_explorer.Current()
                wires.append(wire)
                wire_explorer.Next()
            print(f"[DEBUG] Face {face_idx}: Found {len(wires)} wires")
            if wires:
                outer_boundary = extract_ordered_wire_vertices(wires[0])
                for i, wire in enumerate(wires[1:], 2):
                    cutout_vertices = extract_ordered_wire_vertices(wire)
                    if cutout_vertices:
                        cutouts.append(cutout_vertices)
            else:
                print(f"[DEBUG] Face {face_idx}: No wires found, using fallback vertex extraction")
                outer_boundary = []
                cutouts = []
            print(f"[DEBUG] Face {face_idx}: Outer boundary vertices: {len(outer_boundary) if outer_boundary else 0}")
            print(f"[DEBUG] Face {face_idx}: Number of cutouts: {len(cutouts)}")
            for c_idx, cutout in enumerate(cutouts):
                print(f"    [DEBUG] Cutout {c_idx+1} vertices: {len(cutout)}")
                if cutout:
                    coords_str = ' → '.join([f"({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})" for v in cutout])
                    print(f"        [DEBUG] Cutout {c_idx+1} coords: {coords_str}")
            face_normal = None
            cross_normal = None
            if outer_boundary and len(outer_boundary) >= 3:
                face_normal = get_face_normal_from_opencascade(face)
                print(f"[DEBUG] Face {face_idx}: OpenCASCADE normal: {face_normal}")
                
                # Get face and wire orientations for debugging
                try:
                    face_orientation = face.Orientation()
                    print(f"[DEBUG] Face {face_idx}: Face orientation: "
                          f"{face_orientation}")
                except Exception as e:
                    print(f"[DEBUG] Face {face_idx}: Cannot get face "
                          f"orientation: {e}")
                
                # Print wire orientations
                for w_idx, wire in enumerate(wires):
                    try:
                        wire_orientation = wire.Orientation()
                        print(f"[DEBUG] Face {face_idx} Wire {w_idx+1}: "
                              f"Wire orientation: {wire_orientation}")
                    except Exception as e:
                        print(f"[DEBUG] Face {face_idx} Wire {w_idx+1}: "
                              f"Cannot get wire orientation: {e}")
                
                if face_normal is None:
                    v0 = np.array(outer_boundary[0])
                    v1 = np.array(outer_boundary[1])
                    v2 = np.array(outer_boundary[2])
                    cross_normal = np.cross(v1 - v0, v2 - v0)
                    if np.linalg.norm(cross_normal) > 1e-8:
                        cross_normal = cross_normal / np.linalg.norm(
                            cross_normal)
                        print(f"[DEBUG] Face {face_idx}: Cross-product "
                              f"normal: {cross_normal}")
                        print(f"[DEBUG] Face {face_idx}: Using cross-product "
                              f"normal as fallback.")
                        face_normal = cross_normal
                    else:
                        cross_normal = None
                        print(f"[DEBUG] Face {face_idx}: Skipped normal "
                              f"extraction, including anyway.")
                
                # NORMAL DIRECTION VALIDATION using Shapely + OpenCASCADE
                if (face_normal is not None and outer_boundary and
                        len(outer_boundary) >= 3):
                    try:
                        # Create 2D polygon from outer boundary for Shapely
                        boundary_3d = np.array(outer_boundary)
                        
                        # Find best 2D projection plane based on normal
                        normal_abs = np.abs(face_normal)
                        max_component = np.argmax(normal_abs)
                        
                        if max_component == 0:  # Normal mostly along X
                            boundary_2d = boundary_3d[:, [1, 2]]
                            print(f"[DEBUG] Face {face_idx}: Projecting to "
                                  f"YZ plane for interior point")
                        elif max_component == 1:  # Normal mostly along Y
                            boundary_2d = boundary_3d[:, [0, 2]]
                            print(f"[DEBUG] Face {face_idx}: Projecting to "
                                  f"XZ plane for interior point")
                        else:  # Normal mostly along Z
                            boundary_2d = boundary_3d[:, [0, 1]]
                            print(f"[DEBUG] Face {face_idx}: Projecting to "
                                  f"XY plane for interior point")
                        
                        # Create Shapely polygon and find interior point
                        try:
                            from shapely.geometry import Polygon
                            poly_2d = Polygon(boundary_2d)
                            if poly_2d.is_valid and not poly_2d.is_empty:
                                interior_point_2d = (
                                    poly_2d.representative_point())
                                print(f"[DEBUG] Face {face_idx}: Shapely "
                                      f"interior point 2D: "
                                      f"({interior_point_2d.x:.6f}, "
                                      f"{interior_point_2d.y:.6f})")
                                
                                # Use face centroid as 3D interior point
                                centroid_3d = np.mean(boundary_3d, axis=0)
                                print(f"[DEBUG] Face {face_idx}: Using face "
                                      f"centroid as 3D interior point: "
                                      f"({centroid_3d[0]:.6f}, "
                                      f"{centroid_3d[1]:.6f}, "
                                      f"{centroid_3d[2]:.6f})")
                                
                                # Test normal direction by moving along normal
                                test_distance = 1e-3  # Small distance
                                test_point_pos = (centroid_3d +
                                                  test_distance * face_normal)
                                test_point_neg = (centroid_3d -
                                                  test_distance * face_normal)
                                
                                print(f"[DEBUG] Face {face_idx}: Test point "
                                      f"(+normal): ({test_point_pos[0]:.6f}, "
                                      f"{test_point_pos[1]:.6f}, "
                                      f"{test_point_pos[2]:.6f})")
                                print(f"[DEBUG] Face {face_idx}: Test point "
                                      f"(-normal): ({test_point_neg[0]:.6f}, "
                                      f"{test_point_neg[1]:.6f}, "
                                      f"{test_point_neg[2]:.6f})")
                                
                                # Use OpenCASCADE solid classifier
                                from OCC.Core.BRepClass3d import (
                                    BRepClass3d_SolidClassifier)
                                from OCC.Core.gp import gp_Pnt
                                classifier = BRepClass3d_SolidClassifier()
                                
                                # Test positive normal direction
                                test_pnt_pos = gp_Pnt(test_point_pos[0],
                                                      test_point_pos[1],
                                                      test_point_pos[2])
                                classifier.Perform(solid, test_pnt_pos)
                                state_pos = classifier.State()
                                print(f"[DEBUG] Face {face_idx}: Classifier "
                                      f"state (+normal): {state_pos} "
                                      f"(0=OUT, 1=IN, 2=ON)")
                                
                                # Test negative normal direction
                                test_pnt_neg = gp_Pnt(test_point_neg[0],
                                                      test_point_neg[1],
                                                      test_point_neg[2])
                                classifier.Perform(solid, test_pnt_neg)
                                state_neg = classifier.State()
                                print(f"[DEBUG] Face {face_idx}: Classifier "
                                      f"state (-normal): {state_neg} "
                                      f"(0=OUT, 1=IN, 2=ON)")
                                
                                # Determine if normal should be flipped
                                normal_corrected = face_normal.copy()
                                if state_pos == 1:  # Point along +normal is inside
                                    print(f"[DEBUG] Face {face_idx}: +Normal "
                                          f"points INWARD, flipping normal "
                                          f"direction")
                                    normal_corrected = -face_normal
                                elif state_neg == 1:  # Point along -normal inside
                                    print(f"[DEBUG] Face {face_idx}: +Normal "
                                          f"points OUTWARD, keeping normal "
                                          f"direction")
                                else:
                                    print(f"[DEBUG] Face {face_idx}: Ambiguous "
                                          f"classifier result, keeping original "
                                          f"normal")
                                
                                print(f"[DEBUG] Face {face_idx}: Original "
                                      f"normal: {face_normal}")
                                print(f"[DEBUG] Face {face_idx}: Corrected "
                                      f"normal: {normal_corrected}")
                                
                                # Update face_normal with corrected version
                                face_normal = normal_corrected
                                
                            else:
                                print(f"[DEBUG] Face {face_idx}: Invalid "
                                      f"Shapely polygon, skipping normal "
                                      f"validation")
                        except Exception as e:
                            print(f"[DEBUG] Face {face_idx}: Shapely polygon "
                                  f"creation failed: {e}")
                    except Exception as e:
                        print(f"[DEBUG] Face {face_idx}: Normal direction "
                              f"validation failed: {e}")
                if face_normal is not None and outer_boundary and len(outer_boundary) >= 3:
                    try:
                        # Create 2D polygon from outer boundary for Shapely (project to best plane)
                        boundary_3d = np.array(outer_boundary)
                        
                        # Find the best 2D projection plane based on normal direction
                        normal_abs = np.abs(face_normal)
                        max_component = np.argmax(normal_abs)
                        
                        if max_component == 0:  # Normal mostly along X, project to YZ plane
                            boundary_2d = boundary_3d[:, [1, 2]]
                            print(f"[DEBUG] Face {face_idx}: Projecting to YZ plane for interior point")
                        elif max_component == 1:  # Normal mostly along Y, project to XZ plane
                            boundary_2d = boundary_3d[:, [0, 2]]
                            print(f"[DEBUG] Face {face_idx}: Projecting to XZ plane for interior point")
                        else:  # Normal mostly along Z, project to XY plane
                            boundary_2d = boundary_3d[:, [0, 1]]
                            print(f"[DEBUG] Face {face_idx}: Projecting to XY plane for interior point")
                        
                        # Create Shapely polygon and find interior point
                        try:
                            poly_2d = Polygon(boundary_2d)
                            if poly_2d.is_valid and not poly_2d.is_empty:
                                interior_point_2d = poly_2d.representative_point()
                                print(f"[DEBUG] Face {face_idx}: Shapely interior point 2D: ({interior_point_2d.x:.6f}, {interior_point_2d.y:.6f})")
                                
                                # Map 2D interior point back to 3D
                                # Find the 3D point on the face plane that corresponds to this 2D point
                                centroid_3d = np.mean(boundary_3d, axis=0)
                                print(f"[DEBUG] Face {face_idx}: Using face centroid as 3D interior point: ({centroid_3d[0]:.6f}, {centroid_3d[1]:.6f}, {centroid_3d[2]:.6f})")
                                
                                # Test normal direction by moving along normal and checking if inside solid
                                test_distance = 1e-3  # Small distance along normal
                                test_point_pos = centroid_3d + test_distance * face_normal
                                test_point_neg = centroid_3d - test_distance * face_normal
                                
                                print(f"[DEBUG] Face {face_idx}: Test point (+normal): ({test_point_pos[0]:.6f}, {test_point_pos[1]:.6f}, {test_point_pos[2]:.6f})")
                                print(f"[DEBUG] Face {face_idx}: Test point (-normal): ({test_point_neg[0]:.6f}, {test_point_neg[1]:.6f}, {test_point_neg[2]:.6f})")
                                
                                # Use OpenCASCADE solid classifier to check if points are inside
                                classifier = BRepClass3d_SolidClassifier()
                                
                                # Test positive normal direction
                                classifier.Perform(solid, gp_Pnt(test_point_pos[0], test_point_pos[1], test_point_pos[2]), 1e-6)
                                state_pos = classifier.State()
                                print(f"[DEBUG] Face {face_idx}: Classifier state (+normal): {state_pos} (0=OUT, 1=IN, 2=ON)")
                                
                                # Test negative normal direction
                                classifier.Perform(solid, gp_Pnt(test_point_neg[0], test_point_neg[1], test_point_neg[2]), 1e-6)
                                state_neg = classifier.State()
                                print(f"[DEBUG] Face {face_idx}: Classifier state (-normal): {state_neg} (0=OUT, 1=IN, 2=ON)")
                                
                                # Determine if normal should be flipped
                                normal_corrected = face_normal.copy()
                                if state_pos == 1:  # Point along +normal is inside solid
                                    print(f"[DEBUG] Face {face_idx}: +Normal points INWARD, flipping normal direction")
                                    normal_corrected = -face_normal
                                elif state_neg == 1:  # Point along -normal is inside solid
                                    print(f"[DEBUG] Face {face_idx}: +Normal points OUTWARD, keeping normal direction")
                                else:
                                    print(f"[DEBUG] Face {face_idx}: Ambiguous classifier result, keeping original normal")
                                
                                print(f"[DEBUG] Face {face_idx}: Original normal: {face_normal}")
                                print(f"[DEBUG] Face {face_idx}: Corrected normal: {normal_corrected}")
                                
                                # Update face_normal with corrected version
                                face_normal = normal_corrected
                                
                            else:
                                print(f"[DEBUG] Face {face_idx}: Invalid "
                                      f"Shapely polygon, skipping normal "
                                      f"validation")
                        except Exception as e:
                            print(f"[DEBUG] Face {face_idx}: Shapely polygon "
                                  f"creation failed: {e}")
                    except Exception as e:
                        print(f"[DEBUG] Face {face_idx}: Normal direction "
                              f"validation failed: {e}")
                
                face_data = {
                    'face_obj': face,
                    'outer_boundary': (outer_boundary if outer_boundary
                                       is not None else []),
                    'cutouts': cutouts,
                    'normal': face_normal,
                    'cross_normal': cross_normal,
                    'face_id': face_idx
                }
                faces.append(face_data)
            face_explorer.Next()
        shell_explorer.Next()
    print(f"[DEBUG] Extracted {len(faces)} faces from {shell_count} shell(s)")
    return faces


def visualize_3d_solid(solid_shape, debug_matplot=False, face_polygons=None):
    """Display the 3D solid using matplotlib 3D plotting - showing only polygon boundaries."""
    import matplotlib.pyplot as plt
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    try:
        if solid_shape is None:
            if debug_matplot:
                print("✗ Cannot visualize - shape is None")
            return
        if debug_matplot:
            print("\n" + "="*60)
            print("3D SOLID VISUALIZATION WITH MATPLOTLIB (using wire extraction)")
            print("="*60)
            print("[DEBUG] Starting edge-walking for all faces...")
        all_face_data = []
        if face_polygons is None:
            face_polygons = extract_faces_from_solid(solid_shape)
        for i, face_data in enumerate(face_polygons):
            face = face_data.get('face_obj', None)
            if face is None:
                if debug_matplot:
                    print(f"    Face {i+1}: No face object found")
                continue
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            wire_found = False
            wire_loop_count = 0
            wire_count = 0
            while wire_explorer.More():
                wire = wire_explorer.Current()
                wire_count += 1
                if debug_matplot:
                    print(f"[DEBUG] Walking wire {wire_count} for Face {i+1}...")
                ordered = extract_ordered_wire_vertices(wire)
                if debug_matplot:
                    print(f"[DEBUG] Face {i+1} ordered vertices: {ordered}")
                if ordered and len(ordered) >= 3:
                    all_face_data.append(ordered)
                    if debug_matplot:
                        print(f"    Face {i+1}: {len(ordered)} vertices (edge-walked)")
                else:
                    if debug_matplot:
                        print(f"    Face {i+1}: Could not extract ordered polygon.")
                wire_found = True
                break
            if debug_matplot:
                print(f"[DEBUG] Finished Face {i+1} with {wire_count} wire(s) processed.")
            if not wire_found:
                if debug_matplot:
                    print(f"    Face {i+1}: No wire found.")
        if debug_matplot:
            print(f"[DEBUG] all_face_data length: {len(all_face_data)}")
            for i, face_vertices in enumerate(all_face_data):
                print(f"[DEBUG] Face {i+1} vertices: {face_vertices}")
        if not all_face_data:
            if debug_matplot:
                print("No face data found for 3D visualization.")
            return
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i, face_vertices in enumerate(all_face_data):
            arr = np.array(face_vertices)
            if debug_matplot:
                print(f"[DEBUG] Plotting Face {i+1}: arr shape = {arr.shape}, arr = {arr}")
            if arr.shape[0] == 0:
                if debug_matplot:
                    print(f"[DEBUG] Face {i+1} skipped: no vertices.")
                continue
            if arr.shape[0] <= 2:
                if debug_matplot:
                    print(f"[DEBUG] Face {i+1} skipped: not enough vertices to plot.")
                continue
            # Ensure closed polygon for plotting
            if not np.allclose(arr[0], arr[-1]):
                arr = np.vstack([arr, arr[0]])
            if debug_matplot:
                print(f"[DEBUG] Final arr for plotting: {arr}")
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], label=f'Face {i+1}')
            # Add face number annotation at centroid
            centroid = np.mean(arr, axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], f'{i+1}', color='red', fontsize=12, ha='center', va='center')
        ax.set_title('Extracted Solid Faces (3D Polygon Boundaries)')
        ax.legend()
        plt.tight_layout()
        if debug_matplot:
            print("[DEBUG] Showing 3D plot window (blocking until closed)...")
        plt.show(block=True)
        if debug_matplot:
            print(f"✓ 3D solid visualization complete - {len(all_face_data)} faces plotted.")
            # FINAL: Print connected vertices (rounded for clarity)
            print("\nFINAL: Connected vertices for each face (rounded):")
            for i, face_vertices in enumerate(all_face_data):
                arr = np.array(face_vertices)
                arr_rounded = np.round(arr, 2)
                print(f"Face {i+1}: {arr_rounded.tolist()}")
    except Exception as e:
        print(f"✗ 3D matplotlib visualization failed: {e}")
import numpy as np
from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from config_system import create_default_config, load_config


# OpenCASCADE imports
try:
    from OCC.Core.gp import gp_Vec
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    
    # Try to import TopExp for vertex extraction
    try:
        from OCC.Core.TopExp import topexp, topexp_Vertices
        TOPEXP_AVAILABLE = True
    except Exception:
        TOPEXP_AVAILABLE = False
    
    # Visualization imports
    
    OPENCASCADE_AVAILABLE = True
except ImportError as e:
    print(f"OpenCASCADE not available: {e}")
    OPENCASCADE_AVAILABLE = False
    TOPEXP_AVAILABLE = False

# Dummy placeholder for build_solid_with_polygons_test and plot_four_views
def build_solid_with_polygons_test(config, quiet=False):
    from Base_Solid import build_solid_with_polygons
    seed = config.seed
    print(f"[DEBUG] Calling build_solid_with_polygons(config, seed={seed}, quiet={quiet}) as test...")
    original = build_solid_with_polygons(seed, quiet)
    # (You can add your custom boolean operations here if needed)
    return original

def plot_four_views(solid, user_normal,
    ordered_vertices,
    Vertex_Top_View,
    Vertex_Front_View,
    Vertex_Side_View,
    Vertex_Iso_View,
    face_polygons=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from shapely.geometry import Polygon

    def project_vertex_to_plane(vertex, normal):
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)
        if abs(normal[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        u = temp - np.dot(temp, normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        vertex = np.array(vertex)
        proj_u = np.dot(vertex, u)
        proj_v = np.dot(vertex, v)
        return np.array([proj_u, proj_v])

    views = [
        (np.array([0,0,1]), 'Top View', False),
        (user_normal, 'Isometric View', False),
        (np.array([0,1,0]), 'Front View', True),
        (np.array([1,0,0]), 'Side View', False)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    def plot_polygon_on_ax(ax, polygon, label=None, color='black', linestyle='-', linewidth=2, zorder=2, alpha=1.0):
        if polygon.is_empty:
            return
        if hasattr(polygon, 'exterior'):
            coords = np.array(polygon.exterior.coords)
            ax.plot(coords[:, 0], coords[:, 1], color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder, alpha=alpha)
            if label:
                ax.annotate(label, xy=coords.mean(axis=0), color=color)
        elif hasattr(polygon, 'geoms'):
            for geom in polygon.geoms:
                plot_polygon_on_ax(ax, geom, label=label, color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder, alpha=alpha)

    for i, ax in enumerate(axes):
        normal, label, flip_y = views[i]
        normal = normal / np.linalg.norm(normal)
        if face_polygons is None:
            face_polygons = extract_faces_from_solid(solid)
        _, array_B, array_C = classify_faces_by_projection(face_polygons, normal)
        visible = [(data['polygon'], data.get('name', '')) for data in array_B if 'polygon' in data]
        hidden = [(data['polygon'], data.get('name', '')) for data in array_C if 'polygon' in data]
        print(f"[DEBUG] plot_four_views: {label} projection normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
        print(f"[DEBUG] {label}: {len(visible)} visible, {len(hidden)} hidden polygons")
        # Extra debug: print details of each polygon
        for idx, (poly, name) in enumerate(visible):
            print(f"    [VISIBLE {idx+1}] Name: {name}, Area: {getattr(poly, 'area', None):.4f}, Valid: {poly.is_valid}, Empty: {poly.is_empty}")
        for idx, (poly, name) in enumerate(hidden):
            print(f"    [HIDDEN {idx+1}] Name: {name}, Area: {getattr(poly, 'area', None):.4f}, Valid: {poly.is_valid}, Empty: {poly.is_empty}")
        # Plot hidden polygons (thin dashed gray lines) FIRST
        for poly, name in hidden:
            plot_polygon_on_ax(ax, poly, label=None, color='gray', linestyle='--', linewidth=1, zorder=1, alpha=0.7)
        # Plot visible polygons (thin solid black lines) SECOND
        for poly, name in visible:
            plot_polygon_on_ax(ax, poly, label=name, color='black', linestyle='-', linewidth=1, zorder=2, alpha=1.0)
        ax.set_title(label, fontsize=14)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_facecolor('white')
        ax.grid(True, linestyle=':', color='lightgray', alpha=0.5)
        # Optionally flip y-axis for Front View if needed
        if flip_y:
            ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("four_views.pdf", format="pdf")
    plt.show()
    # FINAL: Print connected vertices for each face (rounded)
    print("\nFINAL: Connected vertices for each face (rounded):")
    if face_polygons is None:
        face_polygons = extract_faces_from_solid(solid)
    for i, face in enumerate(face_polygons):
        arr = np.array(face.get('outer_boundary', []))
        arr_rounded = np.round(arr, 2)
        print(f"Face {i+1}: {arr_rounded.tolist()}")
# def plot_side_view_only(ordered_vertices):
#     """Plot only the Side View in a separate matplotlib window for debugging orientation and limits."""
#     import matplotlib.pyplot as plt
#     import numpy as np
#     # Project to Side View (normal = [1, 0, 0])
#     def project_vertex_to_plane(vertex, normal):
#         normal = np.array(normal)
#         normal = normal / np.linalg.norm(normal)
#         if abs(normal[0]) < 0.9:
#             temp = np.array([1.0, 0.0, 0.0])
#         else:
#             temp = np.array([0.0, 1.0, 0.0])
#         u = temp - np.dot(temp, normal) * normal
#         u = u / np.linalg.norm(u)
#         v = np.cross(normal, u)
#         v = v / np.linalg.norm(v)
#         vertex = np.array(vertex)
#         proj_u = np.dot(vertex, u)
#         proj_v = np.dot(vertex, v)
#         return np.array([proj_u, proj_v])

#     normal = np.array([1, 0, 0])
#     arr2d = np.array([project_vertex_to_plane(v, normal) for v in ordered_vertices])
#     # Ensure closed polygon
#     if not np.allclose(arr2d[0], arr2d[-1]):
#         arr2d = np.vstack([arr2d, arr2d[0]])

#     # Remove criss-crosses: plot only the largest valid simple polygon
#     from shapely.geometry import Polygon, MultiPolygon
#     poly = Polygon(arr2d)
#     if not poly.is_valid or poly.is_empty:
#         print("[WARNING] Side View polygon is not simple (may self-intersect or criss-cross). Attempting to fix...")
#         fixed = poly.buffer(0)
#         # If buffer(0) returns MultiPolygon, pick the largest
#         if isinstance(fixed, MultiPolygon):
#             largest = max(fixed.geoms, key=lambda p: p.area)
#             arr2d = np.array(list(largest.exterior.coords))
#         elif isinstance(fixed, Polygon):
#             arr2d = np.array(list(fixed.exterior.coords))
#         else:
#             print("[ERROR] Could not fix polygon.")

#     # Compute axis limits with margin
#     def add_margin(lims, margin=0.2):
#         span = lims[1] - lims[0]
#         if span == 0:
#             return (lims[0] - 0.5, lims[1] + 0.5)
#         return (lims[0] - margin * span, lims[1] + margin * span)
#     xlim = add_margin((arr2d[:, 0].min(), arr2d[:, 0].max()), 0.2)
#     ylim = add_margin((arr2d[:, 1].min(), arr2d[:, 1].max()), 0.2)

#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.set_facecolor('white')
#     # Plot yellow polygon (main side view)
#     ax.plot(arr2d[:, 0], arr2d[:, 1], color='yellow', linestyle='-', linewidth=3, zorder=2, label='Side View Poly')

#     # Draw unit markers along axes
#     xticks = np.arange(np.floor(xlim[0]), np.ceil(xlim[1])+1, 1)
#     yticks = np.arange(np.floor(ylim[0]), np.ceil(ylim[1])+1, 1)
#     ax.set_xticks(xticks)
#     ax.set_yticks(yticks)
#     ax.grid(True, linestyle=':', color='gray', alpha=0.5)
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#     ax.set_xlabel('U axis (side view) [units]')
#     ax.set_ylabel('V axis (side view) [units]')
#     ax.set_xlim(*xlim)
#     ax.set_ylim(*ylim)
#     ax.set_aspect('equal', adjustable='datalim')
#     ax.set_title('Standalone Side View Only (Units & Ticks)', fontsize=16, color='black')
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

def main():
    # === HELPER FUNCTIONS: must be defined before use ===
    def project_to_view(vertex, normal):
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)
        if np.allclose(normal, [0, 0, 1]):
            u_axis = np.array([1, 0, 0])
            v_axis = np.array([0, 1, 0])
        elif np.allclose(normal, [0, 1, 0]):
            u_axis = np.array([1, 0, 0])
            v_axis = np.array([0, 0, 1])
        elif np.allclose(normal, [1, 0, 0]):
            u_axis = np.array([0, 1, 0])
            v_axis = np.array([0, 0, 1])
        else:
            u_axis = np.cross([0, 0, 1], normal)
            if np.linalg.norm(u_axis) < 1e-8:
                u_axis = np.cross([0, 1, 0], normal)
            u_axis = u_axis / np.linalg.norm(u_axis)
            v_axis = np.cross(normal, u_axis)
            v_axis = v_axis / np.linalg.norm(v_axis)
        vertex = np.array(vertex)
        u = np.dot(vertex, u_axis)
        v = np.dot(vertex, v_axis)
        return u, v

    def filter_possible_vertices(possible_vertices, summary_array, view_name, u_col, v_col, normal, tol=1e-6):
        valid_indices = []
        for idx, vert in enumerate(possible_vertices):
            u, v = project_to_view(vert, normal)
            u_matches = np.isclose(u, summary_array[:, u_col], atol=tol)
            v_matches = np.isclose(v, summary_array[:, v_col], atol=tol)
            match_found = np.any(u_matches & v_matches)
            if match_found:
                valid_indices.append(idx)
        return valid_indices

    def make_summary_array(vertex_array, all_vertices_sorted, proj_normal, view_name):
        try:
            n = vertex_array.shape[0]
            arr = np.zeros((n, 14))
            print(f"[DEBUG] {view_name}: vertex_array shape = {vertex_array.shape}")
            nonzero_row_indices = [i for i in range(n) if np.any(vertex_array[i, :])]
            num_nonzero = len(nonzero_row_indices)
            print(f"[DEBUG] {view_name}: number of nonzero rows = {num_nonzero}")
            def project_vertex(vertex, normal):
                normal = np.array(normal)
                normal = normal / np.linalg.norm(normal)
                if np.allclose(normal, [0, 0, 1]):
                    u_axis = np.array([1, 0, 0])
                    v_axis = np.array([0, 1, 0])
                elif np.allclose(normal, [0, 1, 0]):
                    u_axis = np.array([1, 0, 0])
                    v_axis = np.array([0, 0, 1])
                elif np.allclose(normal, [1, 0, 0]):
                    u_axis = np.array([0, 1, 0])
                    v_axis = np.array([0, 0, 1])
                else:
                    u_axis = np.cross([0, 0, 1], normal)
                    if np.linalg.norm(u_axis) < 1e-8:
                        u_axis = np.cross([0, 1, 0], normal)
                    u_axis = u_axis / np.linalg.norm(u_axis)
                    v_axis = np.cross(normal, u_axis)
                    v_axis = v_axis / np.linalg.norm(v_axis)
                vertex = np.array(vertex)
                u = np.dot(vertex, u_axis)
                v = np.dot(vertex, v_axis)
                return u, v, 0.0
            for row_idx, v_idx in enumerate(nonzero_row_indices):
                x, y, z = all_vertices_sorted[v_idx]
                arr[row_idx, 0:3] = [x, y, z]
                xp, yp, _ = project_vertex([x, y, z], proj_normal)
                if view_name == 'Top View':
                    arr[row_idx, 3] = xp
                    arr[row_idx, 4] = yp
                elif view_name == 'Front View':
                    arr[row_idx, 3] = xp
                    arr[row_idx, 5] = yp
                elif view_name == 'Side View':
                    arr[row_idx, 4] = xp
                    arr[row_idx, 5] = yp
                else:
                    arr[row_idx, 3] = xp
                    arr[row_idx, 4] = yp
                # Only fill as many columns as nonzero_row_indices
                fill_len = len(nonzero_row_indices)
                arr[row_idx, 6:6+fill_len] = vertex_array[v_idx, nonzero_row_indices]
            print(f"\n[DEBUG] Summary array for {view_name} (shape: {arr.shape}):")
            print(arr)
            print(f"[DEBUG] Finished {view_name} summary array.")
            return arr
        except Exception as e:
            print(f"[ERROR] Exception in make_summary_array for {view_name}: {e}")
            import traceback
            traceback.print_exc()
            return None


    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal', type=str, default='1,1,1')
    parser.add_argument('--show_combined', action='store_true')
    parser.add_argument('--show_visible', action='store_true')
    parser.add_argument('--show_hidden', action='store_true')
    parser.add_argument('--seed', type=int, default=47315)
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--save-config', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--debug-matplot', action='store_true', help='Enable matplotlib debug output')
    args = parser.parse_args()

    # Handle configuration loading/creation
    if args.config_file:
        config = load_config(args.config_file)
        seed = config.seed
    else:
        config = create_default_config(args.seed)
        seed = args.seed
    if args.save_config:
        config.save_to_file()
    config.apply_seed()

    # Projection normal
    normal_arg = None
    for i, arg in enumerate(sys.argv):
        if arg == '--normal' and i + 1 < len(sys.argv):
            normal_arg = sys.argv[i + 1]
            break
        elif arg.startswith('--normal='):
            normal_arg = arg.split('=', 1)[1]
            break
    if normal_arg is not None:
        try:
            normal_vals = [float(x) for x in normal_arg.split(',')]
            projection_normal = np.array(normal_vals)
            projection_normal = projection_normal / np.linalg.norm(projection_normal)
        except Exception:
            projection_normal = np.array([1, 1, 1], dtype=float)
            projection_normal = projection_normal / np.linalg.norm(projection_normal)
    else:
        try:
            normal_vals = [float(x) for x in args.normal.split(',')]
            projection_normal = np.array(normal_vals)
            projection_normal = projection_normal / np.linalg.norm(projection_normal)
        except Exception:
            projection_normal = np.array([1, 1, 1], dtype=float)
            projection_normal = projection_normal / np.linalg.norm(projection_normal)


    solid = build_solid_with_polygons_test(config=config, quiet=args.quiet)

    # === Prepare arrays and ordered vertices for Historic algorithm ===
    # Extract all unique vertices from the solid (flatten all face polygons)
    face_polygons = extract_faces_from_solid(solid)
    all_vertices = []
    for face in face_polygons:
        all_vertices.extend(face.get('outer_boundary', []))
        for cutout in face.get('cutouts', []):
            all_vertices.extend(cutout)
    # Remove duplicates (with tolerance)
    all_vertices_np = np.array(all_vertices)
    if len(all_vertices_np) == 0:
        print("[ERROR] No vertices found in solid for further processing.")
        return
    # Use np.unique with rounding for tolerance
    all_vertices_rounded = np.round(all_vertices_np, 6)
    all_vertices_sorted, unique_indices = np.unique(all_vertices_rounded, axis=0, return_index=True)
    all_vertices_sorted = all_vertices_np[np.sort(unique_indices)]
    ordered_vertices = all_vertices_sorted.tolist()
    # FINAL: Print connected vertices for each face (rounded)
    print("\nFINAL: Connected vertices for each face (rounded):")
    for i, face in enumerate(face_polygons):
        arr = np.array(face.get('outer_boundary', []))
        arr_rounded = np.round(arr, 2)
        print(f"Face {i+1}: {arr_rounded.tolist()}")

    n = len(ordered_vertices)
    Vertex_Top_View = np.zeros((n, n), dtype=int)
    Vertex_Front_View = np.zeros((n, n), dtype=int)
    Vertex_Side_View = np.zeros((n, n), dtype=int)
    Vertex_Iso_View = np.zeros((n, n), dtype=int)

    user_normal = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
    # Visualize extracted polygons in 3D before four-view plot
    visualize_3d_solid(solid, debug_matplot=args.debug_matplot, face_polygons=face_polygons)
    plot_four_views(
        solid,
        user_normal,
        ordered_vertices,
        Vertex_Top_View,
        Vertex_Front_View,
        Vertex_Side_View,
        Vertex_Iso_View,
        face_polygons=face_polygons
    )
    # Standalone Side View plot for debugging
    #plot_side_view_only(ordered_vertices)

    # === Post-processing: Find z-levels from Front_View and build Possible_Vertices ===
    print("[DEBUG] Extracting z-levels from Front View summary array...")
    front_view_summary = make_summary_array(Vertex_Front_View, all_vertices_sorted, np.array([0, 1, 0]), 'Front View')
    possible_vertices = np.array([])
    top_view_summary = None
    if front_view_summary is not None and front_view_summary.shape[0] > 0:
        z_coords = np.unique(front_view_summary[:, 2])
        z_coords_sorted = np.sort(z_coords)
        print(f"[DEBUG] Unique z-levels (sorted): {z_coords_sorted}")
        top_view_summary = make_summary_array(Vertex_Top_View, all_vertices_sorted, np.array([0, 0, 1]), 'Top View')
        if top_view_summary is not None and top_view_summary.shape[0] > 0:
            possible_vertices = []
            for row in top_view_summary:
                x, y = row[0], row[1]
                for z in z_coords_sorted:
                    possible_vertices.append([x, y, z])
            possible_vertices = np.array(possible_vertices)
            print("Possible_Vertices array (x, y from Top View, z from z-levels):")
            print(possible_vertices)
        else:
            print("[DEBUG] Top View summary array is empty or None.")
    else:
        print("[DEBUG] Front View summary array is empty or None.")

    # === FINAL: Print valid possible vertices for Top and Side views with debug ===
    print("\n[DEBUG] Entering valid possible vertices filtering block...")
    print(f"[DEBUG] possible_vertices shape: {possible_vertices.shape}")
    print(f"[DEBUG] possible_vertices sample: {possible_vertices[:min(3, len(possible_vertices))]}")
    if top_view_summary is not None:
        print(f"[DEBUG] top_view_summary shape: {top_view_summary.shape}")
        print(f"[DEBUG] top_view_summary sample: {top_view_summary[:min(3, len(top_view_summary))]}")
    else:
        print("[DEBUG] top_view_summary not found!")
    # Top View filtering
    if possible_vertices.size > 0 and top_view_summary is not None:
        valid_top = filter_possible_vertices(
            possible_vertices, top_view_summary, 'Top View', 3, 4, [0, 0, 1]
        )
        print("\n=== Valid Possible_Vertices for Top View ===")
        print(f"Count: {len(valid_top)}")
        for idx in valid_top:
            print(f"  Index {idx}: {possible_vertices[idx]}")
        if not valid_top:
            print("  (None)")
    else:
        print("[DEBUG] Skipping Top View filtering: missing data.")

    # Side View filtering
    side_view_summary = make_summary_array(Vertex_Side_View, all_vertices_sorted, np.array([1, 0, 0]), 'Side View')
    if possible_vertices.size > 0 and side_view_summary is not None:
        print(f"[DEBUG] side_view_summary shape: {side_view_summary.shape}")
        print(f"[DEBUG] side_view_summary sample: {side_view_summary[:min(3, len(side_view_summary))]}")
        valid_side = filter_possible_vertices(
            possible_vertices, side_view_summary, 'Side View', 4, 5, [1, 0, 0]
        )
        print("\n=== Valid Possible_Vertices for Side View ===")
        print(f"Count: {len(valid_side)}")
        for idx in valid_side:
            print(f"  Index {idx}: {possible_vertices[idx]}")
        if not valid_side:
            print("  (None)")
    else:
        print("[DEBUG] Skipping Side View filtering: missing data.")

if __name__ == "__main__":
    main()
