#!/usr/bin/env python3
"""
Build_Solid.py - Solid Generation and Engineering Drawing Creation

This module handles:
1. Building a random 3D solid using OpenCASCADE
2. Extracting face polygons from the solid
3. Creating engineering views (Top, Front, Side, Isometric)
4. Generating connectivity matrices for each view
5. Saving all data to f    # Volume check
    try:
        props = GProp_GProps()
        brepgprop_VolumeProperties(fixed_solid, props)
        volume = props.Mass()
        print(f"[SOLID CHECK] Solid volume: {volume:.6f}")
    except Exception as e:
        print(f"[EXCEPTION] Volume computation failed: {e}")
    print(f"[FACE CHECK] Areas of all faces: {[f'{a:.4f}' for a in face_areas]}")
    return fixed_solidater reconstruction

Output files (named with seed value):
- solid_faces_seed_XXXXX.npy: Face polygon vertices
- connectivity_matrices_seed_XXXXX.npz: View connectivity matrices and projections
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from shapely.geometry import Polygon

# Import OpenCASCADE and reconstruction modules
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX, TopAbs_FACE, TopAbs_SHELL
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_Sewing
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCC.Core.ShapeFix import ShapeFix_Solid, ShapeFix_Shell
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier

import sys
import os
# Ensure Reconstruction is in sys.path for import
recon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Reconstruction')
if recon_path not in sys.path:
    sys.path.append(recon_path)
try:
    from config_system import create_default_config, load_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("[WARNING] config_system not available, using hardcoded values")

# Import necessary functions from V6_current
from V6_current import (
    build_solid_with_polygons_test,
    save_solid_as_step,
    visualize_3d_solid,
    plot_four_views,
    extract_and_visualize_faces
)

from OCC.Display.SimpleGui import init_display


def save_face_polygons(face_polygons, seed, output_dir="Output"):
    """Save face polygons to file"""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"solid_faces_seed_{seed}.npy")
    
    # Convert face polygons to saveable format
    face_data = []
    for face in face_polygons:
        face_dict = {
            'outer_boundary': np.array(face['outer_boundary']),
            'holes': [np.array(hole) for hole in face.get('holes', [])]
        }
        face_data.append(face_dict)
    
    np.save(filename, face_data, allow_pickle=True)
    print(f"\n[SAVE] Saved face polygons to: {filename}")
    return filename


def save_connectivity_matrices(view_matrices, all_vertices, seed, output_dir="Output"):
    """Save connectivity matrices and vertex data to file"""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"connectivity_matrices_seed_{seed}.npz")
    
    # Extract matrices
    top_matrix = view_matrices.get('Top View')
    front_matrix = view_matrices.get('Front View')
    side_matrix = view_matrices.get('Side View')

    # Check for None and replace with empty arrays if needed
    if top_matrix is None:
        print("[WARNING] Top view matrix is None. Saving empty array.")
        top_matrix = np.empty((0,))
    if front_matrix is None:
        print("[WARNING] Front view matrix is None. Saving empty array.")
        front_matrix = np.empty((0,))
    if side_matrix is None:
        print("[WARNING] Side view matrix is None. Saving empty array.")
        side_matrix = np.empty((0,))

    # Save as compressed npz file
    np.savez_compressed(
        filename,
        all_vertices=np.array(all_vertices),
        top_view_matrix=top_matrix,
        front_view_matrix=front_matrix,
        side_view_matrix=side_matrix
    )

    print(f"[SAVE] Saved connectivity matrices to: {filename}")
    print(f"       - Top view matrix shape: {top_matrix.shape}")
    print(f"       - Front view matrix shape: {front_matrix.shape}")
    print(f"       - Side view matrix shape: {side_matrix.shape}")
    print(f"       - Number of vertices: {len(all_vertices)}")

    return filename


def make_face_with_holes(exterior, holes, normal=None):
    """Create an OpenCASCADE face from an exterior polygon and holes."""
    import numpy as np
    from shapely.geometry import Polygon
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    wire_builder = BRepBuilderAPI_MakeWire()
    n = len(exterior)
    # Remove consecutive duplicate vertices
    exterior_np = np.array(exterior)
    filtered_exterior = [exterior_np[0]]
    for pt in exterior_np[1:]:
        if not np.allclose(pt, filtered_exterior[-1]):
            filtered_exterior.append(pt)
    filtered_exterior = np.array(filtered_exterior)
    # Ensure closed
    if not np.allclose(filtered_exterior[0], filtered_exterior[-1]):
        print(f"[WIRE CHECK] Exterior wire not closed, closing it.")
        filtered_exterior = np.vstack([filtered_exterior, filtered_exterior[0]])
    # Check for minimum unique points
    unique_points = {tuple(np.round(pt, 8)) for pt in filtered_exterior}
    print(f"[WIRE CHECK] Exterior wire unique points: {len(unique_points)}")
    if len(unique_points) < 3:
        print(f"[WIRE CHECK] Exterior wire has <3 unique points: {filtered_exterior}")
        return None
    # Project to plane using explicit normal for Shapely validity check
    def project_to_plane(points, normal):
        import numpy as np
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)
        # Find a vector not parallel to normal
        if abs(normal[0]) < 0.9:
            v = np.array([1, 0, 0])
        else:
            v = np.array([0, 1, 0])
        u = np.cross(normal, v)
        u = u / np.linalg.norm(u)
        w = np.cross(normal, u)
        w = w / np.linalg.norm(w)
        proj = np.dot(points, np.vstack([u, w]).T)
        return proj
    if normal is None:
        raise ValueError("Explicit normal must be provided to make_face_with_holes for projection.")
    normal = np.asarray(normal)
    norm_val = np.linalg.norm(normal)
    print(f"[DEBUG] Using explicit normal for projection: {normal}, norm={norm_val}")
    # Check for degenerate normal
    if norm_val < 1e-6 or np.allclose(normal, 0):
        print(f"[WARNING] Degenerate normal detected: {normal}. Falling back to computed normal from exterior vertices.")
        unique_points = np.unique(filtered_exterior, axis=0)
        if unique_points.shape[0] >= 3:
            v1 = unique_points[1] - unique_points[0]
            v2 = unique_points[2] - unique_points[0]
            computed_normal = np.cross(v1, v2)
            computed_normal = computed_normal / np.linalg.norm(computed_normal)
            print(f"[DEBUG] Computed fallback normal: {computed_normal}")
            normal = computed_normal
        else:
            print(f"[ERROR] Not enough unique points to compute fallback normal. Using default [0,0,1].")
            normal = np.array([0,0,1])
    proj_exterior = project_to_plane(filtered_exterior, normal)
    poly2d = Polygon(proj_exterior)
    print(f"[WIRE CHECK] Exterior wire Shapely valid: {poly2d.is_valid} (projected using explicit normal {normal})")
    if not poly2d.is_valid:
        print(f"[WIRE CHECK] Exterior wire is self-intersecting (Shapely invalid): {poly2d.wkt}")
        return None
    for i in range(len(filtered_exterior)-1):
        p1 = gp_Pnt(*filtered_exterior[i])
        p2 = gp_Pnt(*filtered_exterior[i+1])
        edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
        wire_builder.Add(edge)
    # Add holes
    for hidx, hole in enumerate(holes):
        hole_np = np.array(hole)
        filtered_hole = [hole_np[0]]
        for pt in hole_np[1:]:
            if not np.allclose(pt, filtered_hole[-1]):
                filtered_hole.append(pt)
        filtered_hole = np.array(filtered_hole)
        # Ensure closed
        if not np.allclose(filtered_hole[0], filtered_hole[-1]):
            print(f"[WIRE CHECK] Hole wire not closed, closing it.")
            filtered_hole = np.vstack([filtered_hole, filtered_hole[0]])
        unique_hole_points = {tuple(np.round(pt, 8)) for pt in filtered_hole}
        print(f"[WIRE CHECK] Hole {hidx} unique points: {len(unique_hole_points)}")
        if len(unique_hole_points) < 3:
            print(f"[WIRE CHECK] Hole wire has <3 unique points: {filtered_hole}")
            continue
        poly2d_hole = Polygon(filtered_hole[:, :2])
        print(f"[WIRE CHECK] Hole {hidx} Shapely valid: {poly2d_hole.is_valid}")
        if not poly2d_hole.is_valid:
            print(f"[WIRE CHECK] Hole wire is self-intersecting (Shapely invalid): {poly2d_hole.wkt}")
            continue
        hole_wire_builder = BRepBuilderAPI_MakeWire()
        for j in range(len(filtered_hole)-1):
            p1 = gp_Pnt(*filtered_hole[j])
            p2 = gp_Pnt(*filtered_hole[j+1])
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            hole_wire_builder.Add(edge)
        wire_builder.Add(hole_wire_builder.Wire())
    try:
        wire = wire_builder.Wire()
        # Try reversing the wire to flip the face orientation
        wire.Reverse()
        face = BRepBuilderAPI_MakeFace(wire).Face()
        analyzer = BRepCheck_Analyzer(face)
        print(f"[FACE CHECK] OCC face valid: {analyzer.IsValid()}")
        if not analyzer.IsValid():
            print(f"[FACE CHECK] OCC face is INVALID for exterior: {filtered_exterior}")
        return face
    except Exception as e:
        print(f"[EXCEPTION] Error creating OCC face: {e}\n  Exterior: {filtered_exterior}\n  Holes: {holes}")
        return None

def build_solid_from_faces(face_polygons):
    """Build a solid from a list of face polygons (each with outer_boundary and holes)."""
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
    sewing = BRepOffsetAPI_Sewing()
    face_areas = []
    valid_face_count = 0
    for idx, face in enumerate(face_polygons):
        exterior = face['outer_boundary']
        holes = face.get('holes', [])
        # Step 8: Trust that validation was done in Step 7, just build OCC faces
        try:
            occ_face = make_face_with_holes(exterior, holes, normal=face.get('normal', None))
            if occ_face is None:
                print(f"[ERROR] OCC face creation failed for face {idx}")
                continue
            # Area check for diagnostics
            props = GProp_GProps()
            brepgprop_SurfaceProperties(occ_face, props)
            area = props.Mass()
            face_areas.append(area)
            print(f"[STEP 8] Face {idx}: Area = {area:.4f}")
            sewing.Add(occ_face)
            valid_face_count += 1
        except Exception as e:
            print(f"[EXCEPTION] Error in face {idx}: {e}\n  Vertices: {exterior}\n  Holes: {holes}")
    if valid_face_count == 0:
        print("[ERROR] No valid faces to sew. Skipping OCC sewing operation.")
        return None
    
    print("[SEWING] Performing sewing operation...")
    try:
        sewing.Perform()
        print("[SEWING] Sewing complete. Dumping sewing report:")
        shell_shape = sewing.SewedShape()
        # Check for None/null shell before OCC diagnostics
        if shell_shape is None or (hasattr(shell_shape, 'IsNull') and shell_shape.IsNull()):
            print("[ERROR] Sewed shell is None or NULL. Plotting polygons and skipping OCC solid construction.")
            import matplotlib.pyplot as plt
            print("[INFO] OCC shell construction failed. Skipping bottom 2D polygon plot.")
            return None
        # Diagnostics and OCC validity checks
        analyzer = BRepCheck_Analyzer(shell_shape)
        print(f"[SOLID CHECK] Sewed shell valid: {analyzer.IsValid()}")
        if not analyzer.IsValid():
            print(f"[ERROR] Sewed shell is INVALID. Plotting polygons and skipping OCC solid construction.")
            import matplotlib.pyplot as plt
            for idx, face in enumerate(face_polygons):
                exterior = np.array(face['outer_boundary'])
                plt.plot(exterior[:,0], exterior[:,1], label=f'Face {idx} exterior')
                for hidx, hole in enumerate(face.get('holes', [])):
                    hole_np = np.array(hole)
                    plt.plot(hole_np[:,0], hole_np[:,1], '--', label=f'Face {idx} hole {hidx}')
            plt.axis('equal')
            plt.legend()
            plt.title('Polygons (OCC shell construction failed)')
            plt.show()
            return None
        # Extract shell from sewed shape
        print(f"[DEBUG] shell_shape type: {type(shell_shape)} class: {getattr(shell_shape, '__class__', None)}")
        try:
            shape_type = shell_shape.ShapeType()
            if shape_type == 2:  # 2 = TopAbs_SHELL
                print("[DEBUG] shell_shape is already a TopoDS_Shell. Using it directly.")
                shell_found = topods.Shell(shell_shape)
            elif shape_type == 0:  # 0 = TopAbs_COMPOUND
                print("[DEBUG] shell_shape is a TopoDS_Compound. Extracting shell from compound...")
                shell_explorer = TopExp_Explorer(shell_shape, TopAbs_SHELL)
                if shell_explorer.More():
                    shell_found = topods.Shell(shell_explorer.Current())
                else:
                    print("[ERROR] No shell found in compound. Cannot build solid.")
                    return None
            else:
                print(f"[WARNING] shell_shape has unexpected type: {shape_type}. Attempting to use as shell...")
                shell_found = topods.Shell(shell_shape)
        except Exception as e:
            print(f"[EXCEPTION] Error extracting shell: {e}")
            raise
    except Exception as e:
        print(f"[EXCEPTION] Error in sewing operation: {e}")
        raise
    
    if shell_found is None:
        print("[ERROR] No shell found in sewed shape. Cannot build solid.")
        return None
    shell_analyzer = BRepCheck_Analyzer(shell_found)
    print(f"[SHELL CHECK] Shell valid: {shell_analyzer.IsValid()}")
    if not shell_analyzer.IsValid():
        print("[SHELL CHECK] Shell is INVALID!")
    
    # Try to fix shell orientation
    print("[SHELL FIX] Attempting to fix shell orientation...")
    shell_fixer = ShapeFix_Shell(shell_found)
    shell_fixer.Perform()
    fixed_shell = shell_fixer.Shell()
    
    print("[SOLID] Creating solid from shell...")
    solid = BRepBuilderAPI_MakeSolid(fixed_shell).Solid()
    
    # Try to fix solid orientation
    print("[SOLID FIX] Attempting to fix solid orientation...")
    solid_fixer = ShapeFix_Solid(solid)
    solid_fixer.Perform()
    fixed_solid = solid_fixer.Solid()
    
    solid_analyzer = BRepCheck_Analyzer(fixed_solid)
    print(f"[SOLID CHECK] Solid valid: {solid_analyzer.IsValid()}")
    if not solid_analyzer.IsValid():
        print("[SOLID CHECK] Solid is INVALID!")
        # Check if shell is closed
        from OCC.Core.BRepCheck import BRepCheck_Shell
        shell_check = BRepCheck_Shell(fixed_shell)
        print(f"[SHELL CHECK] Shell closed: {shell_check.Closed()}")
        
        # Count edges and check if manifold
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
        from OCC.Core.TopExp import topexp
        
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(fixed_shell, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
        
        non_manifold_edges = 0
        boundary_edges = 0
        total_edges = edge_face_map.Size()
        
        for i in range(1, total_edges + 1):
            face_list = edge_face_map.FindFromIndex(i)
            num_faces = face_list.Size()
            if num_faces == 1:
                boundary_edges += 1
            elif num_faces > 2:
                non_manifold_edges += 1
        
        print(f"[SHELL CHECK] Total edges: {total_edges}")
        print(f"[SHELL CHECK] Boundary edges (only 1 face): {boundary_edges}")
        print(f"[SHELL CHECK] Non-manifold edges (>2 faces): {non_manifold_edges}")
        
        if boundary_edges > 0:
            print("[SHELL CHECK] Shell is NOT closed - has boundary edges!")
    
    # Volume check
    try:
        props = GProp_GProps()
        brepgprop_VolumeProperties(solid, props)
        volume = props.Mass()
        print(f"[SOLID CHECK] Solid volume: {volume:.6f}")
    except Exception as e:
        print(f"[SOLID CHECK] Volume calculation failed: {e}")
    print(f"[FACE CHECK] Areas of all faces: {[f'{a:.4f}' for a in face_areas]}")
    return solid


def plot_built_solid(solid):
    """Plot the built OpenCASCADE solid using pythonocc's viewer."""
    # OCC viewer launch removed

def plot_multiple_solids(solids, labels=None):
    """Plot multiple OpenCASCADE solids in one pythonocc viewer."""
    # OCC viewer launch removed


def main():
    parser = argparse.ArgumentParser(
        description='Build 3D solid and create engineering drawings with connectivity matrices.'
    )
    parser.add_argument(
        '--seed', type=int, default=47315,
        help='Random seed for solid generation (int)'
    )
    parser.add_argument(
        '--normal', type=str, default='1,1,1',
        help='Projection normal for isometric view as comma-separated floats, e.g. "0.75,0.5,1"'
    )
    parser.add_argument(
        '--rotate', type=str, default='0,0,0',
        help='Rotate solid before processing: angles in degrees as "x,y,z"'
    )
    parser.add_argument(
        '--config-file', type=str,
        help='Load configuration from file instead of generating random values'
    )
    parser.add_argument(
        '--save-config', action='store_true',
        help='Save configuration parameters to file'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--output-dir', type=str, default='Output',
        help='Directory to save output files'
    )
    
    args = parser.parse_args()
    
    # Handle configuration
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        config = load_config(args.config_file)
        seed = config.seed
    else:
        print(f"Creating default configuration with seed: {args.seed}")
        config = create_default_config(args.seed)
        seed = args.seed
    
    if args.save_config:
        config.save_to_file()
    
    config.apply_seed()
    
    print("\n" + "="*70)
    print("SOLID GENERATION AND ENGINEERING DRAWING CREATION")
    print("="*70)
    print(f"Seed: {seed}")
    
    # Parse projection normal
    try:
        normal_vals = [float(x) for x in args.normal.split(',')]
        projection_normal = np.array(normal_vals, dtype=float)
        norm = np.linalg.norm(projection_normal)
        if norm == 0:
            raise ValueError("Zero-length normal vector")
        projection_normal = projection_normal / norm
        print(f"Projection normal: {projection_normal}")
    except Exception as e:
        print(f"Could not parse projection normal: {args.normal} ({e})")
        projection_normal = np.array([1, 1, 1], dtype=float)
        projection_normal = projection_normal / np.linalg.norm(projection_normal)
    
    # Build solid
    print("\n[STEP 1] Building 3D solid...")
    solid = build_solid_with_polygons_test(config=config, quiet=args.quiet)
    print(f"Solid created: {type(solid)}")
    
    # Apply rotation if requested
    if args.rotate != '0,0,0':
        try:
            rotation_angles = [float(x) for x in args.rotate.split(',')]
            if len(rotation_angles) != 3:
                raise ValueError("Need 3 angles")
            rx, ry, rz = rotation_angles
            
            if rx != 0 or ry != 0 or rz != 0:
                print(f"\nApplying rotation: X={rx}°, Y={ry}°, Z={rz}°")
                from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir
                from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
                import math
                
                trsf = gp_Trsf()
                
                if rx != 0:
                    axis_x = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
                    trsf_x = gp_Trsf()
                    trsf_x.SetRotation(axis_x, math.radians(rx))
                    trsf.Multiply(trsf_x)
                
                if ry != 0:
                    axis_y = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0))
                    trsf_y = gp_Trsf()
                    trsf_y.SetRotation(axis_y, math.radians(ry))
                    trsf.Multiply(trsf_y)
                
                if rz != 0:
                    axis_z = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
                    trsf_z = gp_Trsf()
                    trsf_z.SetRotation(axis_z, math.radians(rz))
                    trsf.Multiply(trsf_z)
                
                transform = BRepBuilderAPI_Transform(solid, trsf, True)
                transform.Build()
                solid = transform.Shape()
                print("Solid rotated successfully")
        except Exception as e:
            print(f"Could not parse/apply rotation: {args.rotate} ({e})")
    
    # Save STEP file
    os.makedirs("STEPfiles", exist_ok=True)
    save_solid_as_step(solid, "STEPfiles/solid_output.step")
    
    # Extract face polygons
    print("\n[STEP 2] Extracting face polygons from solid...")
    face_polygons = extract_and_visualize_faces(solid, visualize=True)
    print(f"Extracted {len(face_polygons)} faces")
    
    # Build solid from extracted face polygons
    print("\n[DIAGNOSTIC] Attempting to build solid from extracted face polygons...")
    try:
        rebuilt_solid = build_solid_from_faces(face_polygons)
        print("[DIAGNOSTIC] Solid construction complete. Type:", type(rebuilt_solid))
    except Exception as e:
        print("[ERROR] Exception during solid construction:", e)
        rebuilt_solid = None

    # Plot OCC solids BEFORE any matplotlib plt.show()
    print("\n[DIAGNOSTIC] Attempting to launch OCC viewer for original and rebuilt solids...")
    try:
        plot_multiple_solids([solid, rebuilt_solid], labels=["Original Solid", "Rebuilt Solid"])
        print("[DIAGNOSTIC] OCC viewer launched successfully.")
    except Exception as e:
        print("[ERROR] Exception during OCC viewer launch:", e)
    
    # Extract all unique vertices
    print("\n[STEP 3] Extracting unique vertices...")
    vertex_explorer = TopExp_Explorer(solid, TopAbs_VERTEX)
    unique_vertices = []
    seen = set()
    vertex_count = 0
    
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        pnt = BRep_Tool.Pnt(vertex)
        v = (round(pnt.X(), 6), round(pnt.Y(), 6), round(pnt.Z(), 6))
        if v not in seen:
            unique_vertices.append(v)
            seen.add(v)
        vertex_explorer.Next()
        vertex_count += 1
    
    all_vertices_sorted = sorted(unique_vertices, key=lambda v: (v[0], v[1], v[2]))
    print(f"Total unique vertices: {len(all_vertices_sorted)}")
    
    # Display 3D solid
    print("\n[STEP 4] Visualizing 3D solid...")
    visualize_3d_solid(face_polygons, all_vertices_sorted)
    
    # Create engineering views
    print("\n[STEP 5] Creating engineering views (Top, Front, Side, Isometric)...")
    n_vertices = len(all_vertices_sorted)
    Vertex_Top_View = np.zeros((n_vertices, n_vertices), dtype=int)
    Vertex_Front_View = np.zeros((n_vertices, n_vertices), dtype=int)
    Vertex_Side_View = np.zeros((n_vertices, n_vertices), dtype=int)
    Vertex_Iso_View = np.zeros((n_vertices, n_vertices), dtype=int)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(script_dir, "PDFfiles")
    os.makedirs(pdf_dir, exist_ok=True)
    
    view_connectivity_matrices = plot_four_views(
        face_polygons, projection_normal,
        all_vertices_sorted,
        Vertex_Top_View,
        Vertex_Front_View,
        Vertex_Side_View,
        Vertex_Iso_View,
        pdf_dir
    )
    
    # Save data files
    print("\n[STEP 6] Saving face polygons and connectivity matrices...")
    face_file = save_face_polygons(face_polygons, seed, args.output_dir)
    matrix_file = save_connectivity_matrices(
        view_connectivity_matrices, all_vertices_sorted, seed, args.output_dir
    )
    
    print("\n" + "="*70)
    print("SOLID GENERATION COMPLETE")
    print("="*70)
    print(f"Files saved:")
    print(f"  - Face polygons: {face_file}")
    print(f"  - Connectivity matrices: {matrix_file}")
    print(f"\nTo reconstruct the solid, run:")
    print(f"  python Reconstruct_Solid.py --seed {seed}")
    print("="*70)
    
    # Show all plots
    print("[DEBUG] About to call plt.show() for four-view plot...")
    try:
        plt.show()
        print("[DEBUG] plt.show() completed successfully.")
    except Exception as e:
        print(f"[ERROR] Exception during plt.show(): {e}")
    
    # Debug: Check validity of original and rebuilt solids
    print("\n[DEBUG] Checking original solid...")
    print(f"Original solid type: {type(solid)}")
    print(f"Original solid is None: {solid is None}")
    print("\n[DEBUG] Checking rebuilt solid...")
    print(f"Rebuilt solid type: {type(rebuilt_solid)}")
    print(f"Rebuilt solid is None: {rebuilt_solid is None}")

    # OCC viewer launch removed


if __name__ == "__main__":
    main()