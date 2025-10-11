#!/usr/bin/env python3
"""
Investigate why view summaries only contain a subset of solid vertices.
This script will:
1. Load the solid from STEP file
2. Extract ALL unique vertices
3. Load the view summary arrays
4. Compare which vertices are missing from each summary
5. Identify the root cause
"""

import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool

def load_solid_from_step(filename):
    """Load a solid from a STEP file."""
    print(f"Loading solid from: {filename}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)
    
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {filename}")
    
    reader.TransferRoots()
    shape = reader.OneShape()
    
    print(f"✓ Solid loaded successfully")
    return shape

def extract_all_vertices(solid):
    """Extract all unique vertices from a solid."""
    print("\nExtracting all unique vertices from solid...")
    vertex_explorer = TopExp_Explorer(solid, TopAbs_VERTEX)
    unique_vertices = []
    seen = set()
    
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        pnt = BRep_Tool.Pnt(vertex)
        v = (round(pnt.X(), 6), round(pnt.Y(), 6), round(pnt.Z(), 6))
        
        if v not in seen:
            unique_vertices.append(v)
            seen.add(v)
        
        vertex_explorer.Next()
    
    # Sort by x, y, z
    vertices_sorted = sorted(unique_vertices, key=lambda v: (v[0], v[1], v[2]))
    
    print(f"✓ Found {len(vertices_sorted)} unique vertices")
    return vertices_sorted

def analyze_summary_array(summary_array, view_name):
    """Analyze what's in a view summary array."""
    print(f"\n{'='*70}")
    print(f"{view_name} Summary Array Analysis")
    print(f"{'='*70}")
    
    if summary_array is None:
        print("  Summary array is None!")
        return None
    
    print(f"  Shape: {summary_array.shape}")
    print(f"  Number of rows (vertices with edges): {summary_array.shape[0]}")
    
    # Extract 3D coordinates (columns 0-2)
    world_coords = summary_array[:, 0:3]
    print(f"\n  3D World Coordinates (columns 0-2):")
    print(f"    X range: [{world_coords[:, 0].min():.3f}, {world_coords[:, 0].max():.3f}]")
    print(f"    Y range: [{world_coords[:, 1].min():.3f}, {world_coords[:, 1].max():.3f}]")
    print(f"    Z range: [{world_coords[:, 2].min():.3f}, {world_coords[:, 2].max():.3f}]")
    
    # Extract projected coordinates (columns 3-5)
    projected_coords = summary_array[:, 3:6]
    print(f"\n  Projected 2D Coordinates (columns 3-5):")
    
    if view_name == "Top View":
        # Top view uses columns 3,4 for X,Y projection
        unique_x = np.unique(projected_coords[:, 0])
        unique_y = np.unique(projected_coords[:, 1])
        print(f"    Unique X coords (col 3): {len(unique_x)} values")
        print(f"      Range: [{unique_x.min():.3f}, {unique_x.max():.3f}]")
        print(f"      Values: {unique_x}")
        print(f"    Unique Y coords (col 4): {len(unique_y)} values")
        print(f"      Range: [{unique_y.min():.3f}, {unique_y.max():.3f}]")
        print(f"      Values: {unique_y}")
        
        # Unique (x,y) pairs
        xy_pairs = np.unique(projected_coords[:, 0:2], axis=0)
        print(f"    Unique (X,Y) pairs: {len(xy_pairs)}")
        
    elif view_name == "Front View":
        # Front view uses column 3 for X, column 5 for Z projection
        unique_x = np.unique(projected_coords[:, 0])
        unique_z = np.unique(projected_coords[:, 2])
        print(f"    Unique X coords (col 3): {len(unique_x)} values")
        print(f"      Range: [{unique_x.min():.3f}, {unique_x.max():.3f}]")
        print(f"    Unique Z coords (col 5): {len(unique_z)} values")
        print(f"      Range: [{unique_z.min():.3f}, {unique_z.max():.3f}]")
        print(f"      Values: {unique_z}")
        
        # From world coords, extract z-levels
        unique_world_z = np.unique(world_coords[:, 2])
        print(f"    Unique Z-levels from world coords (col 2): {len(unique_world_z)} values")
        print(f"      Values: {unique_world_z}")
    
    return summary_array

def compare_vertices(solid_vertices, summary_vertices_list, view_name):
    """Compare solid vertices against summary array vertices."""
    print(f"\n{'='*70}")
    print(f"Comparing Solid Vertices vs {view_name} Summary")
    print(f"{'='*70}")
    
    # Convert summary vertices to set for fast lookup
    summary_set = set()
    for v in summary_vertices_list:
        v_tuple = tuple(np.round(v, 6))
        summary_set.add(v_tuple)
    
    print(f"  Total vertices in solid: {len(solid_vertices)}")
    print(f"  Vertices in {view_name} summary: {len(summary_set)}")
    
    # Find missing vertices
    missing = []
    for v in solid_vertices:
        if v not in summary_set:
            missing.append(v)
    
    print(f"  Missing vertices: {len(missing)}")
    
    if len(missing) > 0:
        print(f"\n  First 20 missing vertices:")
        for i, v in enumerate(missing[:20]):
            print(f"    {i+1}. ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")
    
    return missing

def main():
    print("="*70)
    print("VIEW SUMMARY INVESTIGATION")
    print("="*70)
    
    # Load solid
    solid = load_solid_from_step("tagged_solid_seed_55.step")
    
    # Extract all vertices
    all_vertices = extract_all_vertices(solid)
    
    # Load summary arrays
    print("\nLoading view summary arrays...")
    try:
        top_view_summary = np.load("top_view_summary.npy")
        print("✓ Loaded top_view_summary.npy")
    except Exception as e:
        print(f"✗ Failed to load top_view_summary.npy: {e}")
        top_view_summary = None
    
    try:
        front_view_summary = np.load("front_view_summary.npy")
        print("✓ Loaded front_view_summary.npy")
    except Exception as e:
        print(f"✗ Failed to load front_view_summary.npy: {e}")
        front_view_summary = None
    
    try:
        side_view_summary = np.load("side_view_summary.npy")
        print("✓ Loaded side_view_summary.npy")
    except Exception as e:
        print(f"✗ Failed to load side_view_summary.npy: {e}")
        side_view_summary = None
    
    # Analyze each summary
    if top_view_summary is not None:
        analyze_summary_array(top_view_summary, "Top View")
        top_summary_vertices = top_view_summary[:, 0:3].tolist()
        compare_vertices(all_vertices, top_summary_vertices, "Top View")
    
    if front_view_summary is not None:
        analyze_summary_array(front_view_summary, "Front View")
        front_summary_vertices = front_view_summary[:, 0:3].tolist()
        compare_vertices(all_vertices, front_summary_vertices, "Front View")
    
    if side_view_summary is not None:
        analyze_summary_array(side_view_summary, "Side View")
        side_summary_vertices = side_view_summary[:, 0:3].tolist()
        compare_vertices(all_vertices, side_summary_vertices, "Side View")
    
    # KEY INSIGHT
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
The view summary arrays only include vertices that participate in edges
detected during the visibility classification phase (classify_faces_by_projection).

If a vertex is not part of any visible or hidden face edge in a particular
view, it won't appear in that view's summary array.

This explains why:
- Top view only has 10 (x,y) coordinate pairs
- Front view only has 3 z-levels

These represent only the vertices that were part of classified face edges,
not all vertices in the solid.

SOLUTION: Modify make_summary_array() to include ALL vertices from the solid,
not just those with non-zero rows in the vertex connectivity matrix.
    """)

if __name__ == "__main__":
    main()
