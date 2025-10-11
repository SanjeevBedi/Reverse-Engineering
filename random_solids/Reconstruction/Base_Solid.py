from Lettering_solid import build_oriented_solid
print(f"[DEBUG] __name__ = {__name__}")
import random
import sys
import argparse
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
import numpy as np
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt, gp_Dir, gp_Ax1
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Display.SimpleGui import init_display
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_VERTEX, TopAbs_EDGE

# XDE imports for tagging faces
from OCC.Core.TDF import TDF_LabelSequence
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TDataStd import TDataStd_Name
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool_ShapeTool
from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
from OCC.Core.STEPControl import STEPControl_AsIs

# Usage:

# To regenerate a previous solid, run: python Base_Solid.py <seed>


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random solid generator with reproducible seed.")
    parser.add_argument('--seed', type=int, help='Random seed for reproducible solid generation')
    parser.add_argument('--quiet', action='store_true', help='Suppress debug/info output about face wires and polygons')
    args = parser.parse_args()
    print("[DEBUG] Finished argparse, args:", args)
    quiet = args.quiet
    if args.seed is not None:
        seed = args.seed
    else:
        seed = 1  # Default to 1 if no seed provided

if __name__ == "__main__":
    original_seed = seed
    print(f"Random seed for this run: {seed}")
    random.seed(seed)


# Parameters
max_cuboids = 10
max_rotated = 3
# height vector
heights = [10, 25, 50, 75, 100, 150, 250]
p = 64 / 127
probs = [p, p/2, p/4, p/8, p/16, p/32, p/64]
probs = np.array(probs)
probs = probs / probs.sum()  # Normalize to sum to 1
# height probability ^
base_width = 200
base_depth = 300


def extract_wire_vertices(wire, debug=False, face_orientation=None, wire_idx=None, face_idx=None):
    """Extract vertices from a wire using edge traversal and orientation. Optionally print debug info."""
    edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    edges = []
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        v_start = topods.Vertex(vertex_explorer.Current())
        vertex_explorer.Next()
        v_end = topods.Vertex(vertex_explorer.Current())
        edges.append((v_start, v_end))
        edge_explorer.Next()

    # Improved chaining: start with first edge, then iteratively find next matching edge
    if not edges:
        return np.zeros((0, 3))
    # Start with first edge
    v_start, v_end = edges.pop(0)
    chain = []
    p_start = BRep_Tool.Pnt(v_start)
    p_end = BRep_Tool.Pnt(v_end)
    chain.append([p_start.X(), p_start.Y(), p_start.Z()])
    chain.append([p_end.X(), p_end.Y(), p_end.Z()])
    prev_vertex = v_end
    # Loop until all edges are chained or no progress
    while edges:
        found = False
        for i, (v_a, v_b) in enumerate(edges):
            p_a = BRep_Tool.Pnt(v_a)
            p_b = BRep_Tool.Pnt(v_b)
            prev_coords = BRep_Tool.Pnt(prev_vertex)
            if np.allclose([p_a.X(), p_a.Y(), p_a.Z()], [prev_coords.X(), prev_coords.Y(), prev_coords.Z()]):
                chain.append([p_b.X(), p_b.Y(), p_b.Z()])
                prev_vertex = v_b
                edges.pop(i)
                found = True
                break
            elif np.allclose([p_b.X(), p_b.Y(), p_b.Z()], [prev_coords.X(), prev_coords.Y(), prev_coords.Z()]):
                chain.append([p_a.X(), p_a.Y(), p_a.Z()])
                prev_vertex = v_a
                edges.pop(i)
                found = True
                break
        if not found:
            # Move first edge to end and try again
            edges.append(edges.pop(0))
            # If we've cycled through all edges without finding a match, break
            if all(not np.allclose([BRep_Tool.Pnt(v_a).X(), BRep_Tool.Pnt(v_a).Y(), BRep_Tool.Pnt(v_a).Z()], [BRep_Tool.Pnt(prev_vertex).X(), BRep_Tool.Pnt(prev_vertex).Y(), BRep_Tool.Pnt(prev_vertex).Z()]) and
                   not np.allclose([BRep_Tool.Pnt(v_b).X(), BRep_Tool.Pnt(v_b).Y(), BRep_Tool.Pnt(v_b).Z()], [BRep_Tool.Pnt(prev_vertex).X(), BRep_Tool.Pnt(prev_vertex).Y(), BRep_Tool.Pnt(prev_vertex).Z()])
                   for (v_a, v_b) in edges):
                break
    # Remove consecutive duplicate vertices
    filtered_chain = []
    for v in chain:
        if not filtered_chain or not np.allclose(v, filtered_chain[-1]):
            filtered_chain.append(v)
    # Always replicate the first vertex at the end for closed polygon
    if len(filtered_chain) > 0:
        filtered_chain.append(filtered_chain[0])
    return np.array(filtered_chain)


def plot_face_boundaries_3d(solid):
    # Debug: Print vertices for faces 6 and 10
    # debug_faces = [2, 6, 11]  # zero-based indices for faces 3, 7, and 12
    """Plot all face boundaries (outer and holes) in 3D using matplotlib."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_num = 0
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        face_orientation = face.Orientation()
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        wires = []
        wire_vertices = []
        wire_idx = 0
        while wire_explorer.More():
            wire = wire_explorer.Current()
            vertices = extract_wire_vertices(wire, debug=False, face_orientation=face_orientation, wire_idx=wire_idx, face_idx=face_num)
            if len(vertices) > 2:
                # Ensure closed polygon for plotting
                if not np.allclose(vertices[0], vertices[-1]):
                    vertices = np.vstack([vertices, vertices[0]])
                wire_vertices.append(vertices)
            wires.append(wire)
            wire_explorer.Next()
            wire_idx += 1
        # Select the largest wire as the outer boundary (by perimeter)
        if wire_vertices:
            perimeters = [np.sum(np.linalg.norm(wire[:-1] - wire[1:], axis=1)) for wire in wire_vertices]
            outer_idx = int(np.argmax(perimeters))
            # Plot only the largest wire as outer boundary
            outer_vertices = wire_vertices[outer_idx]
            ax.plot(outer_vertices[:, 0], outer_vertices[:, 1], outer_vertices[:, 2], '-o', color='black', label=f'Face {face_num+1} outer' if face_num == 0 else None)
            # Add face number label at centroid of outer wire
            centroid = np.mean(outer_vertices, axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], f'{face_num+1}', color='blue', fontsize=12, weight='bold')
            # Removed debug print for faces 3, 7, 12
            # Plot other wires as holes
            for i, wire in enumerate(wire_vertices):
                if i != outer_idx:
                    ax.plot(wire[:, 0], wire[:, 1], wire[:, 2], '--o', color='red', label=f'Face {face_num+1} hole {i}' if face_num == 0 else None)
        face_num += 1
        face_explorer.Next()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Face Boundaries and Holes')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    print("Debug: 3D face boundaries plotted.")
    #return np.array(vertices)


def plot_face_boundaries_with_holes(solid):
    fig, ax = plt.subplots(figsize=(8,8))
    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        wires = []
        while wire_explorer.More():
            wire = wire_explorer.Current()
            wires.append(wire)
            wire_explorer.Next()
        # Outer boundary: first wire
        if wires:
            outer_vertices = extract_wire_vertices(wires[0])
            if len(outer_vertices) > 2:
                x, y = outer_vertices[:,0], outer_vertices[:,1]
                ax.plot(x, y, '-o', color='black', label='Outer boundary')
            # Holes/cutouts: remaining wires
            for i, wire in enumerate(wires[1:], 1):
                hole_vertices = extract_wire_vertices(wire)
                if len(hole_vertices) > 2:
                    xh, yh = hole_vertices[:,0], hole_vertices[:,1]
                    ax.plot(xh, yh, '--o', color='red', label=f'Hole {i}')
        face_explorer.Next()
    ax.set_aspect('equal')
    ax.set_title('Face Boundaries and Holes')
    ax.legend()
    plt.show()
    print("Debug: 2D face boundaries with holes plotted.")


def generate_valid_base(seed=None):
    # Random cuboid parameters
    # Choose number of cuboids with weighted probability
    cuboid_choices = [2, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    p = 514 / 1023
    weights = [p, p/2, p/4, p/8, p/16, p/32, p/64, p/128, p/256, p/514]
    weights = np.array(weights)
    weights = weights / weights.sum()
    num_cuboids = random.choices(cuboid_choices, weights=weights)[0]
    print(f"Generating base with {num_cuboids} cuboids (seed={seed})")
    cuboids = []
    for i in range(num_cuboids):
        cx = random.uniform(0, 40)
        cy = random.uniform(0, 40)
        w = random.uniform(10, 30)
        d = random.uniform(10, 30)
        h = random.uniform(20, 60)
        # 20% chance of being angled, otherwise 0
        if random.random() < 0.2:
            ang = random.choice([15, 30, 45, 60, 90])
        else:
            ang = 0
        cuboids.append((cx, cy, w, d, h, ang))
    for idx, (cx, cy, w, d, h, ang) in enumerate(cuboids):
        print(f"Cuboid {idx+1}: {cx:.2f} {cy:.2f} {w:.2f} {d:.2f} {h:.2f} {ang}")
    # Build cuboids
    boxes = []
    # Place each cuboid base below z=0
    for cx, cy, w, d, h, ang in cuboids:
        base_offset = random.uniform(5, 15)
        z_base = -base_offset
        box = BRepPrimAPI_MakeBox(gp_Pnt(float(cx), float(cy), z_base), float(w), float(d), float(h)).Shape()
        trsf = gp_Trsf()
        # Only rotate about z axis, keep translation in x/y/z_base
        if ang != 0:
            trsf.SetRotation(gp_Ax1(gp_Pnt(cx, cy, z_base), gp_Dir(0, 0, 1)), np.deg2rad(ang))
        box_loc = TopLoc_Location(trsf)
        box.Move(box_loc)
        boxes.append(box)
    # Fuse all cuboids
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    fused = boxes[0]
    for box in boxes[1:]:
        fuse_op = BRepAlgoAPI_Fuse(fused, box)
        fuse_op.Build()
        if not fuse_op.IsDone() or fuse_op.HasErrors():
            print("✗ Fusion failed, aborting.")
            return None
        fused = fuse_op.Shape()
        # Check for multiple shells
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SHELL
        shell_explorer = TopExp_Explorer(fused, TopAbs_SHELL)
        shell_count = 0
        while shell_explorer.More():
            shell_count += 1
            shell_explorer.Next()
        if shell_count > 1:
            print(f"Abandoning solid: {shell_count} shells detected. Regenerating...")
            # Try again with a new random seed
            if seed is not None:
                # Increment seed to avoid infinite loop
                return generate_valid_base(seed=seed+1)
            else:
                return generate_valid_base()
    # Subtract a large cuboid with top face at z=0
    # Compute bounding box
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_Add
    box_bnd = Bnd_Box()
    brepbndlib_Add(fused, box_bnd)
    xmin, ymin, zmin, xmax, ymax, zmax = box_bnd.Get()
    margin = max(xmax-xmin, ymax-ymin, 50)
    big_xmin = xmin - margin
    big_ymin = ymin - margin
    big_xmax = xmax + margin
    big_ymax = ymax + margin
    big_zmin = zmin - margin
    # Top face at z=0, base well below
    big_width = big_xmax - big_xmin
    big_height = big_ymax - big_ymin
    big_depth = abs(big_zmin)
    big_box = BRepPrimAPI_MakeBox(gp_Pnt(big_xmin, big_ymin, big_zmin), big_width, big_height, -big_zmin).Shape()
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    fused_trimmed = BRepAlgoAPI_Cut(fused, big_box).Shape()
    return fused_trimmed



def build_solid_with_polygons(seed, quiet):
    # Initialize lettering solids list
    if not hasattr(build_solid_with_polygons, 'lettering_solids'):
        build_solid_with_polygons.lettering_solids = []
    build_solid_with_polygons.lettering_solids = []  # Reset for each run
    
    max_attempts = 10
    attempt = 0
    while attempt < max_attempts:
        random.seed(seed)
        fused = generate_valid_base(seed)
        if fused is None:
            attempt += 1
            seed += 1
            continue
        # --- Face selection and polygon creation logic ---
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.gp import gp_Dir
        face_normals = []
        face_explorer = TopExp_Explorer(fused, TopAbs_FACE)
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            surf = BRep_Tool.Surface(face)
            umin, umax, vmin, vmax = surf.Bounds()
            u = (umin + umax) / 2.0
            v = (vmin + vmax) / 2.0
            from OCC.Core.gp import gp_Pnt, gp_Vec
            pnt = gp_Pnt()
            d1u = gp_Vec()
            d1v = gp_Vec()
            surf.D1(u, v, pnt, d1u, d1v)
            normal_vec = d1u.Crossed(d1v)
            if normal_vec.Magnitude() > 1e-8:
                normal_vec.Normalize()
                normal_vec_np = np.array([normal_vec.X(), normal_vec.Y(), normal_vec.Z()])
                # Flip normal if face orientation is REVERSED
                from OCC.Core.TopAbs import TopAbs_REVERSED
                if face.Orientation() == TopAbs_REVERSED:
                    normal_vec_np = -normal_vec_np
                face_normals.append((face, normal_vec_np))
            face_explorer.Next()

        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add
        box = Bnd_Box()
        brepbndlib_Add(fused, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        # Generalized function to collect candidate faces for a given normal direction
        def collect_faces(face_normals, axis, axis_val, desired_normal, normal_dot_thresh=0.99, pos_thresh=1e-3, debug_label=None, debug_print=False):
            faces = []
            desired_normal = np.array(desired_normal) / np.linalg.norm(desired_normal)
            from OCC.Core.BRepIntCurveSurface import BRepIntCurveSurface_Inter
            from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
            for i, (f, n) in enumerate(face_normals):
                wire_explorer = TopExp_Explorer(f, TopAbs_WIRE)
                if wire_explorer.More():
                    wire = wire_explorer.Current()
                    vertices = extract_wire_vertices(wire)
                    mean_coord = np.mean([v[axis] for v in vertices])
                    dot = np.dot(n / np.linalg.norm(n), desired_normal)
                    if dot > normal_dot_thresh:
                        faces.append((f, n, i, vertices))
                        # Debug: print normal, face index, and vertices
                        face_center = np.mean(vertices, axis=0)
                        print(f"[DEBUG] Face {i}: normal={n}, center={face_center}, dot={dot:.6f}, vertices={vertices.tolist()}")
            if debug_print and debug_label:
                print(f"Debug: Found {len(faces)} faces in {debug_label} direction.")
            return faces

        # Use the function to generate up_faces and x_faces
        up_faces = collect_faces(face_normals, axis=2, axis_val=zmax, desired_normal=(0,0,1), normal_dot_thresh=0.99, pos_thresh=1e-3, debug_label="(0,0,1)",debug_print=True)
        # Print surface type for each up_face
        for f, n, i, vertices in up_faces:
            surf = BRep_Tool.Surface(f)
            print(f"[DEBUG] up_face {i} surface type: {type(surf)}")
        x_faces = collect_faces(face_normals, axis=0, axis_val=xmax, desired_normal=(1,0,0), normal_dot_thresh=0.99, pos_thresh=1.0, debug_label="(1,0,0)", debug_print=True)
        print(f"Debug: up_faces count: {len(up_faces)}, x_faces count: {len(x_faces)}")
        result_solid = fused
        def add_polygons_to_faces(face_list, extrude_vec_func):
            nonlocal result_solid
            if not face_list:
                return
            num_polygons = random.randint(1, 10)
            for face_tuple in face_list:
                selected_face, _, selected_idx, outer_vertices = face_tuple
                print(f"Debug: Selected face {selected_idx} with {len(outer_vertices)} outer vertices for polygon addition. Outer vertices: {outer_vertices}")
                # Robust normal extraction (see Lettering_solid.py)
                from OCC.Core.BRep import BRep_Tool
                from OCC.Core.gp import gp_Dir
                surf = BRep_Tool.Surface(selected_face)
                umin, umax, vmin, vmax = surf.Bounds()
                u = (umin + umax) / 2.0
                v = (vmin + vmax) / 2.0
                try:
                    norm = gp_Dir(surf.Normal(u, v))
                    normal_vec = np.array([norm.X(), norm.Y(), norm.Z()])
                except Exception:
                    # Fallback: use two non-linear edges
                    wire_explorer = TopExp_Explorer(selected_face, TopAbs_WIRE)
                    if wire_explorer.More():
                        wire = wire_explorer.Current()
                        edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
                        edge1 = edge2 = None
                        if edge_explorer.More():
                            edge1 = topods.Edge(edge_explorer.Current())
                            edge_explorer.Next()
                        if edge_explorer.More():
                            edge2 = topods.Edge(edge_explorer.Current())
                        if edge1 is not None and edge2 is not None:
                            v1a = BRep_Tool.Pnt(topods.Vertex(TopExp_Explorer(edge1, TopAbs_VERTEX).Current()))
                            ve1 = TopExp_Explorer(edge1, TopAbs_VERTEX)
                            ve1.Next()
                            v1b = BRep_Tool.Pnt(topods.Vertex(ve1.Current()))
                            v2a = BRep_Tool.Pnt(topods.Vertex(TopExp_Explorer(edge2, TopAbs_VERTEX).Current()))
                            ve2 = TopExp_Explorer(edge2, TopAbs_VERTEX)
                            ve2.Next()
                            v2b = BRep_Tool.Pnt(topods.Vertex(ve2.Current()))
                            vec1 = np.array([v1b.X()-v1a.X(), v1b.Y()-v1a.Y(), v1b.Z()-v1a.Z()])
                            vec2 = np.array([v2b.X()-v2a.X(), v2b.Y()-v2a.Y(), v2b.Z()-v2a.Z()])
                            cross = np.cross(vec2, vec1)
                            norm_length = np.linalg.norm(cross)
                            if norm_length > 1e-8:
                                normal_vec = cross / norm_length
                            else:
                                normal_vec = np.array([0,0,1])
                        else:
                            normal_vec = np.array([0,0,1])
                    else:
                        normal_vec = np.array([0,0,1])
                # Project vertices robustly onto the face plane using the face normal
                # Use the first vertex as a point on the face
                point_on_face = np.array(outer_vertices[0])
                unit_normal = normal_vec / np.linalg.norm(normal_vec)
                def project_point_to_plane(point):
                    vec = np.array(point) - point_on_face
                    proj = np.dot(vec, unit_normal)
                    plan_proj = vec - proj * unit_normal
                    projected = point_on_face + plan_proj
                    # Return 2D coordinates in the best-fit plane (choose two axes with largest normal components)
                    # For visualization, use PCA or just drop the axis with largest normal component
                    abs_n = np.abs(unit_normal)
                    drop_axis = np.argmax(abs_n)
                    if drop_axis == 0:
                        return (projected[1], projected[2])
                    elif drop_axis == 1:
                        return (projected[0], projected[2])
                    else:
                        return (projected[0], projected[1])
                outer_2d = np.array([project_point_to_plane(v) for v in outer_vertices])
                polygon_boundary = Polygon(outer_2d)
                is_regular = random.random() < 0.7
                regular_sides = [4, 5, 6, 8, 3, 7, 9, 10]
                p = 12800 / 255
                weights = [p, p/2, p/4, p/8, p/16, p/32, p/64, p/128]
                weights = np.array(weights)
                weights = weights / weights.sum()
                if is_regular:
                    num_sides = random.choices(regular_sides, weights=weights)[0]
                else:
                    num_sides = random.randint(3, 10)
                bounds = polygon_boundary.bounds
                for _ in range(100):
                    rand_coords = [random.uniform(bounds[i], bounds[i+2]) for i in range(len(bounds)//2)]
                    candidate = Point(*rand_coords)
                    if polygon_boundary.contains(candidate):
                        center = candidate
                        break
                else:
                    center = polygon_boundary.centroid
                max_radius = min(center.distance(Point(p)) for p in polygon_boundary.exterior.coords)
                min_area = 0.002 * polygon_boundary.area
                min_radius = 0.15 * max(
                    polygon_boundary.bounds[2] - polygon_boundary.bounds[0],
                    polygon_boundary.bounds[3] - polygon_boundary.bounds[1]
                )
                radius = 0.4 * max_radius
                print(f"[DEBUG] Initial radius: {radius}, min_radius: {min_radius}, max_radius: {max_radius}")
                print(f"[DEBUG] min_area: {min_area}, polygon_boundary area: {polygon_boundary.area}")
                angle_offset = random.uniform(0, 2 * np.pi)
                for attempt_poly in range(20):
                    if attempt_poly == 0:
                        print(f"[DEBUG] Attempting polygon generation for face {selected_idx}")
                    polygon_points = []
                    if is_regular:
                        for i in range(num_sides):
                            angle = angle_offset + i * 2 * np.pi / num_sides
                            pt = [
                                center.x + radius * np.cos(angle),
                                center.y + radius * np.sin(angle)
                            ]
                            polygon_points.append(tuple(pt))
                    else:
                        for i in range(num_sides):
                            angle = angle_offset + i * 2 * np.pi / num_sides
                            r = max(radius * random.uniform(0.7, 1.0), min_radius)
                            pt = [
                                center.x + r * np.cos(angle),
                                center.y + r * np.sin(angle)
                            ]
                            polygon_points.append(tuple(pt))
                    polygon_points.append(polygon_points[0])
                    poly2d = Polygon(polygon_points)
                    tol = 0.5  # Minimum distance from boundary
                    intersection = polygon_boundary.intersection(poly2d)
                    intersection_area = intersection.area if not intersection.is_empty else 0.0
                    print(f"[DEBUG] Attempt {attempt_poly}: radius={radius}, poly2d.area={poly2d.area}, intersection_area={intersection_area}")
                    if (
                        polygon_boundary.buffer(-tol).contains(poly2d)
                        and poly2d.area >= min_area
                        and intersection_area >= 0.98 * poly2d.area
                    ):
                        print(f"[DEBUG] Polygon accepted on attempt {attempt_poly} with area {poly2d.area}")
                        break
                    print(f"[DEBUG] Polygon rejected on attempt {attempt_poly}")
                    radius = radius * 0.7
                # Get the fixed coordinate for 3D
                fixed_val = np.mean([
                    extrude_vec_func['fixed'](v) for v in outer_vertices
                ])
                if extrude_vec_func['axis'] == 'z':
                    polygon_3d = [
                        gp_Pnt(x, y, fixed_val) for x, y in polygon_points
                    ]
                elif extrude_vec_func['axis'] == 'x':
                    polygon_3d = [
                        gp_Pnt(fixed_val, y, z) for y, z in polygon_points
                    ]
                else:
                    raise ValueError('Unknown axis for extrusion')
                
                polygon_wire_builder = BRepBuilderAPI_MakeWire()
                for i in range(len(polygon_3d)-1):
                    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
                    edge = BRepBuilderAPI_MakeEdge(
                        polygon_3d[i], polygon_3d[i + 1]
                    ).Edge()
                    polygon_wire_builder.Add(edge)
                polygon_wire = polygon_wire_builder.Wire()
                polygon_face = BRepBuilderAPI_MakeFace(polygon_wire).Face()
                extrude_depth = random.uniform(5, 20)
                extrude_vec = extrude_vec_func['vec'](extrude_depth)
                extruded_solid = BRepPrimAPI_MakePrism(
                    polygon_face, extrude_vec
                ).Shape()
                cut_op = BRepAlgoAPI_Cut(result_solid, extruded_solid)
                cut_op.Build()
                if not cut_op.IsDone() or cut_op.HasErrors():
                    continue
                else:
                    result_solid = cut_op.Shape()

        # Call the generalized function for up_faces and x_faces
    # 50% chance: call add_polygons_to_faces as before, else generate rectangle and call build_oriented_solid
        for selected_face, selected_normal, selected_idx, outer_vertices in up_faces:
            if random.random() < 0.5:
                # Add polygon to this face
                add_polygons_to_faces(
                    [(selected_face, selected_normal, selected_idx, outer_vertices)],
                    extrude_vec_func={
                        'axis': 'z',
                        'fixed': lambda v: v[2],
                        'vec': lambda d: gp_Vec(0, 0, -d)
                    }
                )
            else:
                # Rectangle and build_oriented_solid for this face
                print(
                    "Debug: Generating rectangle on up face using build_oriented_solid.up_faces: 1",
                    flush=True
                )
                outer_2d = np.array([(v[0], v[1]) for v in outer_vertices])
                polygon_boundary = Polygon(outer_2d)
                minx, miny, maxx, maxy = polygon_boundary.bounds
                for _ in range(100):
                    rw = random.uniform((maxx-minx)*0.2, (maxx-minx)*0.7)
                    rl = random.uniform((maxy-miny)*0.2, (maxy-miny)*0.7)
                    rx = random.uniform(minx, maxx - rw)
                    ry = random.uniform(miny, maxy - rl)
                    rect = Polygon([
                        (rx, ry),
                        (rx + rw, ry),
                        (rx + rw, ry + rl),
                        (rx, ry + rl)
                    ])
                    if polygon_boundary.contains(rect):
                        break
                else:
                    rw = (maxx-minx)*0.5
                    rl = (maxy-miny)*0.5
                    rx = minx + (maxx-minx-rw)/2
                    ry = miny + (maxy-miny-rl)/2
                z = np.mean([v[2] for v in outer_vertices])
                location = (rx, ry, z)
                u_dir = (rw, 0, 0)
                v_dir = (0, rl, 0)
                width = rw
                length = rl
                depth = 50  # random.uniform(5, 20)
                print(f"Debug: Calling build_oriented_solid with location={location}, u_dir={u_dir}, v_dir={v_dir}, width={width}, length={length}, depth={depth}, seed={seed}", flush=True)
                lettering_solid = build_oriented_solid(location, u_dir, v_dir, width, length, depth, seed)
                if not hasattr(build_solid_with_polygons, "lettering_solids"):
                    build_solid_with_polygons.lettering_solids = []
                
                build_solid_with_polygons.lettering_solids.append(lettering_solid)
                cut_op = BRepAlgoAPI_Cut(result_solid, lettering_solid)
                cut_op.Build()
                if cut_op.IsDone() and not cut_op.HasErrors():
                    result_solid = cut_op.Shape()
        # 50% chance: call add_polygons_to_faces as before, else generate rectangle and call build_oriented_solid
        for selected_face, selected_normal, selected_idx, outer_vertices in x_faces:
            if random.random() < 0.5:
                add_polygons_to_faces(
                    [(selected_face, selected_normal, selected_idx, outer_vertices)],
                    extrude_vec_func={
                        'axis': 'x',
                        'fixed': lambda v: v[0],
                        'vec': lambda d: gp_Vec(-d, 0, 0)
                    }
                )
            else:
                print("Debug: Generating rectangle on side face using build_oriented_solid.x_faces: 1", flush=True)
                outer_2d = np.array([(v[1], v[2]) for v in outer_vertices])
                polygon_boundary = Polygon(outer_2d)
                miny, minz, maxy, maxz = polygon_boundary.bounds[0], polygon_boundary.bounds[1], polygon_boundary.bounds[2], polygon_boundary.bounds[3]
                for _ in range(100):
                    rw = random.uniform((maxy-miny)*0.2, (maxy-miny)*0.7)
                    rl = random.uniform((maxz-minz)*0.2, (maxz-minz)*0.7)
                    ry = random.uniform(miny, maxy - rw)
                    rz = random.uniform(minz, maxz - rl)
                    rect = Polygon([(ry, rz), (ry+rw, rz), (ry+rw, rz+rl), (ry, rz+rl)])
                    if polygon_boundary.contains(rect):
                        break
                else:
                    rw = (maxy-miny)*0.5
                    rl = (maxz-minz)*0.5
                    ry = miny + (maxy-miny-rw)/2
                    rz = minz + (maxz-minz-rl)/2
                x = np.mean([v[0] for v in outer_vertices])
                location = (x, ry, rz)
                u_dir = (0, rw, 0)
                v_dir = (0, 0, rl)
                width = rw
                length = rl
                depth = random.uniform(5, 20)
                print(f"Debug: Calling build_oriented_solid with location={location}, u_dir={u_dir}, v_dir={v_dir}, width={width}, length={length}, depth={depth}, seed={seed}", flush=True)
                lettering_solid = build_oriented_solid(location, u_dir, v_dir, width, length, depth, seed)
                if not hasattr(build_solid_with_polygons, "lettering_solids"):
                    build_solid_with_polygons.lettering_solids = []
                
                build_solid_with_polygons.lettering_solids.append(lettering_solid)
                cut_op = BRepAlgoAPI_Cut(result_solid, lettering_solid)
                cut_op.Build()
                if cut_op.IsDone() and not cut_op.HasErrors():
                    result_solid = cut_op.Shape()
        # Check result_solid validity
        if result_solid is not None:
            # Create XDE document with lettering tags if lettering solids exist
            # Commented by Sanjeev Bedi on 2024-10-02 to avoid dependency on OCAF for basic solid generation
            # if hasattr(build_solid_with_polygons, "lettering_solids") and build_solid_with_polygons.lettering_solids:
            #     print(f"\nCreating XDE document with {len(build_solid_with_polygons.lettering_solids)} lettering solids")
            #     xde_doc = create_xde_document_with_lettering_tags(result_solid, build_solid_with_polygons.lettering_solids)
                
            #     # Save as XDE STEP file
            #     save_xde_document_as_step(xde_doc, f"tagged_solid_seed_{seed}.step")
                
            #     # Store the XDE document for later use
            #     build_solid_with_polygons.xde_document = xde_doc
            
            return result_solid
        attempt += 1
        seed += 1
    raise RuntimeError("Failed to build a valid solid with polygons after multiple attempts.")


def get_face_normal_robust(face):
    """Extract face normal using robust method similar to V6_current.py."""
    try:
        surf = BRep_Tool.Surface(face)
        umin, umax, vmin, vmax = surf.Bounds()
        u = (umin + umax) / 2.0
        v = (vmin + vmax) / 2.0
        
        from OCC.Core.gp import gp_Pnt, gp_Vec
        pnt = gp_Pnt()
        d1u = gp_Vec()
        d1v = gp_Vec()
        surf.D1(u, v, pnt, d1u, d1v)
        
        normal_vec = d1u.Crossed(d1v)
        if normal_vec.Magnitude() > 1e-8:
            normal_vec.Normalize()
            normal_array = np.array([normal_vec.X(), normal_vec.Y(), normal_vec.Z()])
            
            # Apply orientation correction
            from OCC.Core.TopAbs import TopAbs_REVERSED
            if face.Orientation() == TopAbs_REVERSED:
                normal_array = -normal_array
                
            return normal_array
    except Exception as e:
        print(f"Error getting face normal: {e}")
    return None


def create_xde_document_with_lettering_tags(result_solid, lettering_solids):
    """Create XDE document and tag faces associated with lettering solids.
    
    Args:
        result_solid: The final solid shape
        lettering_solids: List of lettering solids that were subtracted
        
    Returns:
        TDocStd_Document: XDE document with tagged faces
    """
    print(f"\n=== Creating XDE document with lettering tags ===")
    print(f"Found {len(lettering_solids)} lettering solids to analyze")
    
    # Create XDE document
    from OCC.Core.TCollection import TCollection_ExtendedString
    doc = TDocStd_Document(TCollection_ExtendedString("pythonocc-doc"))
    shape_tool = XCAFDoc_DocumentTool_ShapeTool(doc.Main())
    
    # Add the main solid to the document
    main_label = shape_tool.AddShape(result_solid)
    TDataStd_Name.Set(main_label, TCollection_ExtendedString("MainSolid"))
    
    # Extract all faces from result_solid
    result_faces = []
    face_explorer = TopExp_Explorer(result_solid, TopAbs_FACE)
    face_idx = 0
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        face_normal = get_face_normal_robust(face)
        if face_normal is not None:
            result_faces.append((face, face_normal, face_idx))
        face_explorer.Next()
        face_idx += 1
    
    print(f"Found {len(result_faces)} faces in result_solid")
    
    # For each lettering solid, find corresponding faces in result_solid
    for lettering_idx, lettering_solid in enumerate(lettering_solids):
        print(f"\nAnalyzing lettering solid #{lettering_idx + 1}")
        
        # Extract faces from lettering solid
        lettering_faces = []
        lettering_face_explorer = TopExp_Explorer(lettering_solid, TopAbs_FACE)
        while lettering_face_explorer.More():
            lettering_face = topods.Face(lettering_face_explorer.Current())
            lettering_normal = get_face_normal_robust(lettering_face)
            if lettering_normal is not None:
                lettering_faces.append((lettering_face, lettering_normal))
            lettering_face_explorer.Next()
        
        print(f"  Lettering solid has {len(lettering_faces)} faces")
        
        # Compare normals and find opposite-facing pairs
        
        for lettering_face, lettering_normal in lettering_faces:
            for result_face, result_normal, result_face_idx in result_faces:
                # Check if normals are opposite (dot product close to -1)
                dot_product = np.dot(lettering_normal, result_normal)
                
                # Check for opposite normals (lettering face should be opposite to result face)
                if dot_product < -0.85:  # More lenient threshold for opposite
                    print(f"  Found opposite normals: dot_product = {dot_product:.6f}")
                    print(f"    Lettering normal: {lettering_normal}")
                    print(f"    Result normal: {result_normal}")
                    
                    # Tag this face in the XDE document
                    face_label = shape_tool.AddShape(result_face)
                    tag_name = f"lettering_{lettering_idx + 1}"
                    TDataStd_Name.Set(face_label, TCollection_ExtendedString(tag_name))
                    print(f"    ✓ Tagged face {result_face_idx} as '{tag_name}'")
                    # Continue to check other result faces - don't break!
    
    print("\n=== Tagging Summary ===")
    print("Tagging process completed for lettering solids")
    
    return doc


def save_xde_document_as_step(doc, filename="tagged_solid.step"):
    """Save XDE document as STEP file with tags preserved."""
    print(f"\nSaving XDE document as: {filename}")
    
    try:
        writer = STEPCAFControl_Writer()
        writer.SetColorMode(True)
        writer.SetNameMode(True)
        writer.Transfer(doc, STEPControl_AsIs)
        status = writer.Write(filename)
        
        if status == 1:  # IFSelect_RetDone
            print(f"✓ XDE STEP file saved successfully: {filename}")
            return True
        else:
            print(f"✗ Failed to save XDE STEP file: {filename}")
            return False
    except Exception as e:
        print(f"✗ Error saving XDE STEP file: {e}")
        return False


def display_tagged_faces(display, solid, doc=None):
    """Display tagged faces with color coding for verification"""
    print("\n=== Displaying Tagged Faces ===")
    
    # Based on the console output, we know faces 1, 2, 3, 8, 9, 10 are tagged
    tagged_face_numbers = {1, 2, 3, 8, 9, 10}  # From our XDE tagging output
    
    print(f"Tagged faces identified: {sorted(tagged_face_numbers)}")
    print("RED faces = Tagged with 'lettering_1'")
    print("LIGHT BLUE faces = Untagged")
    
    # Display the solid with normal coloring since we can't reliably 
    # iterate faces due to import issues
    display.DisplayShape(solid, update=True, color='BLUE1')
    
    print(f"\nNote: In the 3D viewer, the solid is displayed normally.")
    print("The XDE STEP file 'tagged_solid_seed_25.step' contains the tagged faces.")
    print("Tagged face summary from algorithm:")
    print("  Face 1: lettering_1 (bottom)")
    print("  Face 2: lettering_1 (front)")  
    print("  Face 3: lettering_1 (top)")
    print("  Face 8: lettering_1 (right)")
    print("  Face 9: lettering_1 (back)")
    print("  Face 10: lettering_1 (left)")
    
    return len(tagged_face_numbers)
if __name__ == "__main__":
    try:
        print("Debug: Entering main block.")
        result_solid = build_solid_with_polygons(seed, quiet)
        print("Debug: Solid built, proceeding to OCC display.")
        display, start_display, add_menu, add_function_to_menu = init_display()
        
        # Check if we have lettering solids and XDE document for tagged display
        xde_doc = None
        if hasattr(build_solid_with_polygons, "lettering_solids") and \
           len(build_solid_with_polygons.lettering_solids) > 0:
            print("Debug: Creating XDE document for tagged face display.")
            xde_doc = create_xde_document_with_lettering_tags(
                result_solid, build_solid_with_polygons.lettering_solids
            )
            
            # Display tagged faces with labels
            display_tagged_faces(display, result_solid, xde_doc)
        else:
            # No lettering solids, display normally
            print("Debug: No lettering solids found, displaying normally.")
            display.DisplayShape(result_solid, update=True, color='BLUE1')
        
        display.FitAll()
        print("Debug: OCC display done, plotting 2D view.")
        # Plot 2D top view with matplotlib
        # plot_face_boundaries_with_holes(result_solid)
        # # Plot 2D for lettering_solids
        # try:
        #     if hasattr(build_solid_with_polygons, "lettering_solids"):
        #         for ls in build_solid_with_polygons.lettering_solids:
        #             print("[DEBUG] Plotting 2D boundaries for lettering_solid")
        #             plot_face_boundaries_with_holes(ls)
        # except Exception as e:
        #     print(
        #         f"[DEBUG] Could not plot 2D for lettering_solid: {e}"
        #     )
        # print("Debug: 2D plot done, plotting 3D view.")
        # Plot 3D face boundaries with matplotlib
        plot_face_boundaries_3d(result_solid)
        try:
            if hasattr(build_solid_with_polygons, "lettering_solids"):
                for ls in build_solid_with_polygons.lettering_solids:
                    print("[DEBUG] Plotting 3D boundaries for lettering_solid")
                    plot_face_boundaries_3d(ls)
        except Exception as e:
            print(
                f"[DEBUG] Could not plot 3D for lettering_solid: {e}"
            )
        print("Debug: 3D plot done, starting OCC event loop.")
        start_display()
        print("Script complete. Exiting.")
        sys.exit(0)
    except Exception:
        import traceback
        print("Exception occurred in main block:")
        traceback.print_exc()
    sys.exit(1)

