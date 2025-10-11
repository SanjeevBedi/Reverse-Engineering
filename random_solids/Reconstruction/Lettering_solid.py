import numpy as np
import random
import argparse

# Import configuration system
try:
    from config_system import ConfigurationManager, create_default_config, load_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("[WARNING] config_system not available, using hardcoded values")

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
    parser.add_argument('--config-file', type=str, help='Load configuration from file instead of generating random values')
    parser.add_argument('--save-config', action='store_true', help='Save configuration parameters to file')
    args = parser.parse_args()
    
    # Handle configuration loading/creation
    if CONFIG_AVAILABLE:
        if args.config_file:
            print(f"Loading configuration from: {args.config_file}")
            config = load_config(args.config_file)
            seed = config.seed
        else:
            seed = args.seed if args.seed is not None else random.randint(0, 999999)
            print(f"Creating default configuration with seed: {seed}")
            config = create_default_config(seed)
        
        # Save configuration if requested
        if args.save_config:
            config.save_to_file()
        
        # Apply seed from configuration
        config.apply_seed()
    else:
        # Fallback to original behavior
        seed = args.seed if args.seed is not None else random.randint(0, 999999)
        config = None
        print(f"Random seed for this run: {seed}")
        random.seed(seed)
        np.random.seed(seed)


def middle_weighted_rand(a, b, size=1):
    """
    Return a middle-weighted random value between a and b using triangular distribution.
    """
    return np.random.triangular(a, (a + b) / 2, b, size)

def build_oriented_solid(location, u_dir, v_dir, width, length, depth, config_or_seed=None):
    """
    Build a rectangular base at 'location' with orientation given by u_dir, v_dir (unit vectors),
    dimensions width, length, extruded by depth along w = u x v, with 2 units margin on all sides.
    Add horizontal and vertical cuboids as in the current logic, in the local u,v,w frame.
    If config_or_seed is given, use it for configuration or as seed.
    Returns the solid.
    """
    
    # Handle both config object and backward compatibility with seed
    if hasattr(config_or_seed, 'get_section'):  # It's a config object
        config = config_or_seed
        cfg = config.get_section('lettering_solid')
        seed = config.seed
        config.apply_seed()
        
        # Get parameters from config
        base_margin = cfg.get('base_expansion_margin', 2)
        base_z_offset = cfg.get('base_z_offset', 0.00025)
        grid_size = cfg.get('grid_size', 5)
        
        # Cuboid choices and probabilities
        cuboid_choices = cfg.get('cuboid_choices', [2, 1, 3, 4])
        base_prob = cfg.get('cuboid_base_prob', 8 / 15)
        
        # Vertical cuboid parameters
        v_width_min_factor = cfg.get('vertical_width_min_factor', 0.6)
        v_width_max_factor = cfg.get('vertical_width_max_factor', 0.9)
        v_length_min_factor = cfg.get('vertical_length_min_factor', 0.6)
        v_length_max_factor = cfg.get('vertical_length_max_factor', 0.9)
        v_length_divisor = cfg.get('vertical_length_divisor', 5)
        v_depth_min_factor = cfg.get('vertical_depth_min_factor', 0.5)
        v_depth_max_factor = cfg.get('vertical_depth_max_factor', 1.0)
        v_embed_factor = cfg.get('vertical_embed_factor', 0.3)
        
        # Horizontal cuboid parameters
        h_width_min_factor = cfg.get('horizontal_width_min_factor', 0.6)
        h_width_max_factor = cfg.get('horizontal_width_max_factor', 0.9)
        h_width_divisor = cfg.get('horizontal_width_divisor', 5)
        h_length_min_factor = cfg.get('horizontal_length_min_factor', 0.6)
        h_length_max_factor = cfg.get('horizontal_length_max_factor', 0.9)
        h_depth_min_factor = cfg.get('horizontal_depth_min_factor', 0.5)
        h_depth_max_factor = cfg.get('horizontal_depth_max_factor', 1.0)
        h_embed_factor = cfg.get('horizontal_embed_factor', 0.3)
        
    else:  # Backward compatibility - it's a seed or None
        seed = config_or_seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Use hardcoded defaults
        base_margin = 2
        base_z_offset = 0.00025
        grid_size = 5
        cuboid_choices = [2, 1, 3, 4]
        base_prob = 8 / 15
        v_width_min_factor = 0.6
        v_width_max_factor = 0.9
        v_length_min_factor = 0.6
        v_length_max_factor = 0.9
        v_length_divisor = 5
        v_depth_min_factor = 0.5
        v_depth_max_factor = 1.0
        v_embed_factor = 0.3
        h_width_min_factor = 0.6
        h_width_max_factor = 0.9
        h_width_divisor = 5
        h_length_min_factor = 0.6
        h_length_max_factor = 0.9
        h_depth_min_factor = 0.5
        h_depth_max_factor = 1.0
        h_embed_factor = 0.3
    # Normalize directions and check perpendicularity
    u = np.array(u_dir, dtype=float)
    u = u / np.linalg.norm(u)
    v = np.array(v_dir, dtype=float)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    w = w / np.linalg.norm(w)
    # Debug: print dot products to check perpendicularity
    print(f"u·v = {np.dot(u, v):.6f}, u·w = {np.dot(u, w):.6f}, v·w = {np.dot(v, w):.6f}")
    
    # Expand rectangle by configured margin on all sides
    width_exp = width + base_margin
    length_exp = length + base_margin
    
    # Build the base box in the local coordinate system (aligned with axes)
    base_box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, base_z_offset), width_exp, length_exp, depth).Shape()
    fused = base_box
    
    # Print base bounding box
    bbox = Bnd_Box()
    brepbndlib_Add(base_box, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    print(f"[DEBUG] Base bounding box: x=({xmin:.2f},{xmax:.2f}), y=({ymin:.2f},{ymax:.2f}), z=({zmin:.2f},{zmax:.2f})")
    
    # Place cuboids in the local coordinate system (aligned with axes)
    weights = np.array([base_prob / (2**i) for i in range(len(cuboid_choices))])
    weights = weights / weights.sum()
    cum_weights = np.cumsum(weights)
    r = random.random()
    num_horiz = cuboid_choices[np.searchsorted(cum_weights, r)]
    r = random.random()
    num_vert = cuboid_choices[np.searchsorted(cum_weights, r)]
    
    used_positions = set()
    cell_w = width / grid_size
    cell_l = length / grid_size
    for _ in range(num_vert):
        while True:
            col = random.randint(0, grid_size - 1)
            row = random.randint(0, grid_size - 1)
            pos = (col, row)
            if pos not in used_positions:
                used_positions.add(pos)
                break
        # Vertical cuboid: parameterized size calculation
        v_w = np.random.uniform(cell_w * v_width_min_factor, v_width_max_factor * width)
        v_l = np.random.uniform(cell_l * v_length_min_factor, v_length_max_factor * length / v_length_divisor)
        v_d = random.uniform(depth * v_depth_min_factor, depth * v_depth_max_factor)
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
        v_z = -v_d * v_embed_factor
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
            col = random.randint(0, grid_size - 1)
            row = random.randint(0, grid_size - 1)
            pos = (col, row)
            if pos not in used_positions:
                used_positions.add(pos)
                break
        # Horizontal cuboid: parameterized size calculation
        h_w = np.random.uniform(cell_w * h_width_min_factor, h_width_max_factor * width / h_width_divisor)
        h_l = np.random.uniform(cell_l * h_length_min_factor, h_length_max_factor * length)
        h_d = random.uniform(depth * h_depth_min_factor, depth * h_depth_max_factor)
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
        h_z = -h_d * h_embed_factor
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

    solid = build_oriented_solid(location, u_dir, v_dir, width, length, depth, config if config else seed)
    # --- Display the final solid in OpenCASCADE GUI ---
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(solid, update=True)
    display.FitAll()
    start_display()

    # --- Robust Matplotlib 3D plot using correct vertex extraction ---
    plot_face_boundaries_3d(solid)