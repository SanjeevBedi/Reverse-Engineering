import numpy as np
import random
import argparse
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax3, gp_Dir
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_REVERSED, TopAbs_FORWARD
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods, topods_Face
from OCC.Display.SimpleGui import init_display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add

def extract_wire_vertices(wire, debug=False, face_orientation=None, wire_idx=None, face_idx=None):
    # ...existing code...
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
    if not edges:
        return np.zeros((0, 3))
    v_start, v_end = edges.pop(0)
    chain = []
    p_start = BRep_Tool.Pnt(v_start)
    p_end = BRep_Tool.Pnt(v_end)
    chain.append([p_start.X(), p_start.Y(), p_start.Z()])
    chain.append([p_end.X(), p_end.Y(), p_end.Z()])
    prev_vertex = v_end
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
            edges.append(edges.pop(0))
            if all(not np.allclose([BRep_Tool.Pnt(v_a).X(), BRep_Tool.Pnt(v_a).Y(), BRep_Tool.Pnt(v_a).Z()], [BRep_Tool.Pnt(prev_vertex).X(), BRep_Tool.Pnt(prev_vertex).Y(), BRep_Tool.Pnt(prev_vertex).Z()]) and
                   not np.allclose([BRep_Tool.Pnt(v_b).X(), BRep_Tool.Pnt(v_b).Y(), BRep_Tool.Pnt(v_b).Z()], [BRep_Tool.Pnt(prev_vertex).X(), BRep_Tool.Pnt(prev_vertex).Y(), BRep_Tool.Pnt(prev_vertex).Z()])
                   for (v_a, v_b) in edges):
                break
    filtered_chain = []
    for v in chain:
        if not filtered_chain or not np.allclose(v, filtered_chain[-1]):
            filtered_chain.append(v)
    if len(filtered_chain) > 0:
        filtered_chain.append(filtered_chain[0])
    return np.array(filtered_chain)

def plot_face_boundaries_3d(solid):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopoDS import topods
    import numpy as np
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
                if not np.allclose(vertices[0], vertices[-1]):
                    vertices = np.vstack([vertices, vertices[0]])
                wire_vertices.append(vertices)
            wires.append(wire)
            wire_explorer.Next()
            wire_idx += 1
        if wire_vertices:
            perimeters = [np.sum(np.linalg.norm(wire[:-1] - wire[1:], axis=1)) for wire in wire_vertices]
            outer_idx = int(np.argmax(perimeters))
            outer_vertices = wire_vertices[outer_idx]
            ax.plot(outer_vertices[:, 0], outer_vertices[:, 1], outer_vertices[:, 2], '-o', color='black', label=f'Face {face_num+1} outer' if face_num == 0 else None)
            centroid = np.mean(outer_vertices, axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], f'{face_num+1}', color='blue', fontsize=12, weight='bold')
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
# --- Seed setup (must be first) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random solid generator with reproducible seed.")
    parser.add_argument('--seed', type=int, help='Random seed for reproducible solid generation')
    args = parser.parse_args()
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 999999)
    print(f"Random seed for this run: {seed}")
    random.seed(seed)
    np.random.seed(seed)


def middle_weighted_rand(a, b, size=1):
    """
    Return a middle-weighted random value between a and b using triangular distribution.
    """
    return np.random.triangular(a, (a + b) / 2, b, size)

def build_oriented_solid(location, u_dir, v_dir, width, length, depth, seed=None):
    """
    Build a rectangular base at 'location' with orientation given by u_dir, v_dir (unit vectors),
    dimensions width, length, extruded by depth along w = u x v, with 2 units margin on all sides.
    Add horizontal and vertical cuboids as in the current logic, in the local u,v,w frame.
    If seed is given, set the random seed.
    Returns the solid.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # Normalize directions and check perpendicularity
    u = np.array(u_dir, dtype=float)
    u = u / np.linalg.norm(u)
    v = np.array(v_dir, dtype=float)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    w = w / np.linalg.norm(w)
    # Debug: print dot products to check perpendicularity
    print(f"u·v = {np.dot(u, v):.6f}, u·w = {np.dot(u, w):.6f}, v·w = {np.dot(v, w):.6f}")
    # Expand rectangle by 2 units on all sides
    width_exp = width + 2
    length_exp = length + 2
    # Build the base box in the local coordinate system (aligned with axes)
    base_box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0.00025), width_exp, length_exp, depth).Shape()
    fused = base_box
    # Print base bounding box
    bbox = Bnd_Box()
    brepbndlib_Add(base_box, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    print(f"[DEBUG] Base bounding box: x=({xmin:.2f},{xmax:.2f}), y=({ymin:.2f},{ymax:.2f}), z=({zmin:.2f},{zmax:.2f})")
    # Place cuboids in the local coordinate system (aligned with axes)
    horiz_choices = [2, 1, 3, 4]
    horiz_p = 8 / 15
    horiz_weights = [horiz_p, horiz_p/2, horiz_p/4, horiz_p/8]
    horiz_weights = np.array(horiz_weights)
    horiz_weights = horiz_weights / horiz_weights.sum()
    cum_weights = np.cumsum(horiz_weights)
    r = random.random()
    num_horiz = horiz_choices[np.searchsorted(cum_weights, r)]
    r = random.random()
    num_vert = horiz_choices[np.searchsorted(cum_weights, r)]
    used_positions = set()
    grid_n = 5
    cell_w = (width) / grid_n
    cell_l = (length) / grid_n
    for _ in range(num_vert):
        while True:
            col = random.randint(0, grid_n - 1)
            row = random.randint(0, grid_n - 1)
            pos = (col, row)
            if pos not in used_positions:
                used_positions.add(pos)
                break
        # Vertical cuboid: max size (0.9*width)x(0.9*length/5)xdepth
        v_w = np.random.uniform(cell_w * 0.6, 0.9 * width)
        v_l = np.random.uniform(cell_l * 0.6, 0.9 * length / grid_n)
        v_d = random.uniform(depth * 0.5, depth)
        v_x = col * cell_w + (cell_w - v_w) / 2
        v_y = row * cell_l + (cell_l - v_l) / 2
        # Clamp so cuboid stays within base
        v_x = max(0, min(v_x, width_exp - v_w))
        v_y = max(0, min(v_y, length_exp - v_l))
        # If cuboid would extend beyond, shrink it to fit
        if v_x + v_w > width_exp:
            v_w = width_exp - v_x
        if v_y + v_l > length_exp:
            v_l = length_exp - v_y
        # Embed cuboid into base: set v_z negative so it always intersects base
        v_z = -v_d * 0.3
        print(f"VERTICAL: origin=({v_x:.2f}, {v_y:.2f}, {v_z:.2f}), size=({v_w:.2f}, {v_l:.2f}, {v_d:.2f})")
        cuboid = BRepPrimAPI_MakeBox(gp_Pnt(v_x, v_y, v_z), v_w, v_l, v_d).Shape()
        # Print cuboid bounding box
        bbox = Bnd_Box()
        brepbndlib_Add(cuboid, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        print(f"[DEBUG] VERTICAL cuboid bbox: x=({xmin:.2f},{xmax:.2f}), y=({ymin:.2f},{ymax:.2f}), z=({zmin:.2f},{zmax:.2f})")
        # Count faces before fusion
        explorer = TopExp_Explorer(fused, TopAbs_FACE)
        face_count_before = 0
        while explorer.More():
            face_count_before += 1
            explorer.Next()
        fused = BRepAlgoAPI_Fuse(fused, cuboid).Shape()
        explorer = TopExp_Explorer(fused, TopAbs_FACE)
        face_count_after = 0
        while explorer.More():
            face_count_after += 1
            explorer.Next()
        print(f"[DEBUG] Faces before fusion: {face_count_before}, after fusion: {face_count_after}")
    for _ in range(num_horiz):
        while True:
            col = random.randint(0, grid_n - 1)
            row = random.randint(0, grid_n - 1)
            pos = (col, row)
            if pos not in used_positions:
                used_positions.add(pos)
                break
        # Horizontal cuboid: max size (0.9*width/5)x(0.9*length)xdepth
        h_w = np.random.uniform(cell_w * 0.6, 0.9 * width / grid_n)
        h_l = np.random.uniform(cell_l * 0.6, 0.9 * length)
        h_d = random.uniform(depth * 0.5, depth)
        h_x = col * cell_w + (cell_w - h_w) / 2
        h_y = row * cell_l + (cell_l - h_l) / 2
        # Clamp so cuboid stays within base
        h_x = max(0, min(h_x, width_exp - h_w))
        h_y = max(0, min(h_y, length_exp - h_l))
        # If cuboid would extend beyond, shrink it to fit
        if h_x + h_w > width_exp:
            h_w = width_exp - h_x
        if h_y + h_l > length_exp:
            h_l = length_exp - h_y
        # Embed cuboid into base: set h_z negative so it always intersects base
        h_z = -h_d * 0.3
        print(f"HORIZONTAL: origin=({h_x:.2f}, {h_y:.2f}, {h_z:.2f}), size=({h_w:.2f}, {h_l:.2f}, {h_d:.2f})")
        cuboid = BRepPrimAPI_MakeBox(gp_Pnt(h_x, h_y, h_z), h_w, h_l, h_d).Shape()
        # Print cuboid bounding box
        bbox = Bnd_Box()
        brepbndlib_Add(cuboid, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        print(f"[DEBUG] HORIZONTAL cuboid bbox: x=({xmin:.2f},{xmax:.2f}), y=({ymin:.2f},{ymax:.2f}), z=({zmin:.2f},{zmax:.2f})")
        # Count faces before fusion
        explorer = TopExp_Explorer(fused, TopAbs_FACE)
        face_count_before = 0
        while explorer.More():
            face_count_before += 1
            explorer.Next()
        fused = BRepAlgoAPI_Fuse(fused, cuboid).Shape()
        explorer = TopExp_Explorer(fused, TopAbs_FACE)
        face_count_after = 0
        while explorer.More():
            face_count_after += 1
            explorer.Next()
        print(f"[DEBUG] Faces before fusion: {face_count_before}, after fusion: {face_count_after}")
    # Debug: count faces after all fusions, before transform
    # ...existing code...
    explorer = TopExp_Explorer(fused, TopAbs_FACE)
    total_faces = 0
    while explorer.More():
        total_faces += 1
        explorer.Next()
    print(f"[DEBUG] Total faces after all fusions (before transform): {total_faces}")
    # Transform: map local axes (x, y, z) to (u, v, w) and move origin to location
    origin = gp_Pnt(0, 0, 0)
    # Source: local frame (x, y, z)
    src_ax3 = gp_Ax3(origin, gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))
    # Target: frame at location, axes u (x), v (y), w (z)
    tgt_ax3 = gp_Ax3(gp_Pnt(*location), gp_Dir(*w), gp_Dir(*u))
    trsf = gp_Trsf()
    trsf.SetDisplacement(src_ax3, tgt_ax3)
    transformed = BRepBuilderAPI_Transform(fused, trsf, True).Shape()
    return transformed

# --- Test parameters ---
if __name__ == "__main__":
    location = (25, 25, 0)
    u_dir = (1, 1, 1)
    v_dir = (-1, 1, 0)
    width = 50
    length = 30
    depth = 10
    seed = None  # Or set to an int for reproducibility

    solid = build_oriented_solid(location, u_dir, v_dir, width, length, depth, seed)
    # --- Display the final solid in OpenCASCADE GUI ---
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(solid, update=True)
    display.FitAll()
    start_display()

    # --- Robust Matplotlib 3D plot using correct vertex extraction ---
    plot_face_boundaries_3d(solid)