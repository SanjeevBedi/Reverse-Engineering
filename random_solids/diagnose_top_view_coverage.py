#!/usr/bin/env python3
"""
Diagnose which solid vertices are missing from Top view projected polygons.

This script checks if all vertices from the solid appear in the Top view
visible/hidden polygon projections. This is critical for the reverse
engineering algorithm where candidate vertices MUST come from view data only.
"""

import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool

print("="*70)
print("TOP VIEW COVERAGE DIAGNOSTIC FOR SEED 55")
print("="*70)

# Load the solid
print("\n1. Loading solid from STEP file...")
reader = STEPControl_Reader()
status = reader.ReadFile("tagged_solid_seed_55.step")
if status != IFSelect_RetDone:
    raise RuntimeError("Failed to read STEP file")
reader.TransferRoots()
solid = reader.OneShape()
print("   ✓ Solid loaded")

# Extract all unique vertices from solid
print("\n2. Extracting all unique vertices from solid...")
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

vertices_sorted = sorted(unique_vertices, key=lambda v: (v[0], v[1], v[2]))
print(f"   ✓ Found {len(vertices_sorted)} unique vertices")

# Extract (x,y) coordinates - these are what Top view should see
print("\n3. Extracting (x,y) coordinates from solid vertices...")
xy_coords_from_solid = set()
for v in vertices_sorted:
    xy = (round(v[0], 6), round(v[1], 6))
    xy_coords_from_solid.add(xy)

print(f"   ✓ Solid has {len(xy_coords_from_solid)} unique (x,y) coordinates")
print(f"   Unique (x,y) coordinates:")
xy_sorted = sorted(list(xy_coords_from_solid))
for i, xy in enumerate(xy_sorted[:20]):  # Show first 20
    print(f"      {i+1}. ({xy[0]:.6f}, {xy[1]:.6f})")
if len(xy_sorted) > 20:
    print(f"      ... and {len(xy_sorted) - 20} more")

# Load Top view summary
print("\n4. Loading Top view summary array...")
try:
    top_view_summary = np.load("top_view_summary.npy")
    print(f"   ✓ Loaded top_view_summary.npy with shape {top_view_summary.shape}")
    
    # Extract (x,y) from projected coordinates (columns 3,4)
    projected_xy = top_view_summary[:, 3:5]
    xy_coords_from_top_view = set()
    for row in projected_xy:
        xy = (round(row[0], 6), round(row[1], 6))
        xy_coords_from_top_view.add(xy)
    
    print(f"   ✓ Top view has {len(xy_coords_from_top_view)} unique (x,y) coordinates")
    print(f"   Projected (x,y) coordinates from Top view:")
    xy_top_sorted = sorted(list(xy_coords_from_top_view))
    for i, xy in enumerate(xy_top_sorted):
        print(f"      {i+1}. ({xy[0]:.6f}, {xy[1]:.6f})")
    
except Exception as e:
    print(f"   ✗ Failed to load top_view_summary.npy: {e}")
    xy_coords_from_top_view = set()

# Compare coverage
print("\n5. COVERAGE ANALYSIS")
print("="*70)

if xy_coords_from_top_view:
    missing_xy = xy_coords_from_solid - xy_coords_from_top_view
    extra_xy = xy_coords_from_top_view - xy_coords_from_solid
    
    print(f"Solid (x,y) coordinates: {len(xy_coords_from_solid)}")
    print(f"Top view (x,y) coordinates: {len(xy_coords_from_top_view)}")
    print(f"Missing from Top view: {len(missing_xy)}")
    print(f"Extra in Top view: {len(extra_xy)}")
    
    if len(missing_xy) > 0:
        print(f"\n⚠️  CRITICAL ISSUE: {len(missing_xy)} (x,y) coordinates missing from Top view!")
        print(f"These (x,y) coordinates exist in the solid but not in Top view projections:")
        missing_sorted = sorted(list(missing_xy))
        for i, xy in enumerate(missing_sorted[:30]):  # Show first 30
            # Count how many vertices have this (x,y)
            count = sum(1 for v in vertices_sorted if 
                       abs(v[0] - xy[0]) < 1e-5 and abs(v[1] - xy[1]) < 1e-5)
            print(f"   {i+1}. ({xy[0]:.6f}, {xy[1]:.6f}) - affects {count} vertices")
        if len(missing_sorted) > 30:
            print(f"   ... and {len(missing_sorted) - 30} more")
        
        # Calculate how many vertices are affected
        affected_vertices = []
        for v in vertices_sorted:
            xy = (round(v[0], 6), round(v[1], 6))
            if xy in missing_xy:
                affected_vertices.append(v)
        
        print(f"\n   Total vertices affected: {len(affected_vertices)} out of {len(vertices_sorted)}")
        print(f"   These vertices CANNOT be reconstructed from Top view alone!")
    else:
        print(f"\n✓ All solid (x,y) coordinates are covered by Top view projections")
    
    if len(extra_xy) > 0:
        print(f"\n⚠️  Note: {len(extra_xy)} extra (x,y) coordinates in Top view not in solid")
        print(f"   (This could be from intersection polygons)")
else:
    print("Cannot compare - Top view summary not available")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("""
KEY INSIGHT:
For reverse engineering to work, ALL (x,y) coordinates from the solid
must appear in the Top view projected polygons (visible + hidden + intersections).

If coordinates are missing, it means some faces are not being projected
correctly, or are being filtered out during classification.

SOLUTION:
Ensure classify_faces_by_projection() includes ALL face projections,
even degenerate ones (edge-on faces that project to lines).
""")
