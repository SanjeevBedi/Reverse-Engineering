#!/usr/bin/env python3
"""
Analyze which vertices are missing from the selected set and which faces they belong to.
"""

import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods
from OCC.Extend.TopologyUtils import TopologyExplorer

# Load the solid
print("Loading solid from tagged_solid_seed_55.step...")
step_reader = STEPControl_Reader()
step_reader.ReadFile('tagged_solid_seed_55.step')
step_reader.TransferRoots()
solid = step_reader.Shape()

# Get all unique vertices from solid
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

all_vertices_sorted = sorted(unique_vertices, key=lambda v: (v[0], v[1], v[2]))
print(f"Total unique vertices in solid: {len(all_vertices_sorted)}")

# Load unified summary (selected vertices)
print("\nLoading unified_summary.npy...")
unified_summary = np.load('unified_summary.npy')
selected_vertices = unified_summary[:, :3]
print(f"Total vertices in unified_summary: {len(selected_vertices)}")

# Find missing vertices
print("\nIdentifying missing vertices...")
tolerance = 1e-4
missing_vertices = []
missing_indices = []

for idx, solid_v in enumerate(all_vertices_sorted):
    found = False
    for summary_v in selected_vertices:
        dist = np.sqrt(sum((solid_v[j] - summary_v[j])**2 for j in range(3)))
        if dist < tolerance:
            found = True
            break
    if not found:
        missing_vertices.append(solid_v)
        missing_indices.append(idx)

print(f"\nMissing vertices: {len(missing_vertices)}")
print(f"Missing percentage: {100.0 * len(missing_vertices) / len(all_vertices_sorted):.1f}%")

if missing_vertices:
    print("\nMissing vertex coordinates:")
    for i, (v, idx) in enumerate(zip(missing_vertices, missing_indices)):
        print(f"  V{idx+1} (index {idx}): ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")

# Now find which faces these missing vertices belong to
print("\n" + "="*70)
print("ANALYZING FACES CONTAINING MISSING VERTICES")
print("="*70)

topo = TopologyExplorer(solid)

# Create a mapping of face names to faces
face_map = {}
for face_idx, face in enumerate(topo.faces()):
    # Try to get face name from extended data
    face_name = f"Face_{face_idx}"
    face_map[face_name] = face

print(f"\nTotal faces in solid: {len(face_map)}")

# For each missing vertex, find which faces it belongs to
vertex_to_faces = {}
tolerance = 1e-4

for missing_v in missing_vertices:
    vertex_key = tuple(missing_v)
    vertex_to_faces[vertex_key] = []
    
    for face_name, face in face_map.items():
        # Get vertices of this face
        face_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
        face_vertices = []
        while face_explorer.More():
            vertex = topods.Vertex(face_explorer.Current())
            pnt = BRep_Tool.Pnt(vertex)
            v = (round(pnt.X(), 6), round(pnt.Y(), 6), round(pnt.Z(), 6))
            face_vertices.append(v)
            face_explorer.Next()
        
        # Check if missing vertex is in this face
        for face_v in face_vertices:
            dist = np.sqrt(sum((missing_v[j] - face_v[j])**2 for j in range(3)))
            if dist < tolerance:
                vertex_to_faces[vertex_key].append(face_name)
                break

# Print results
print("\nMissing vertices and their faces:")
for idx, missing_v in enumerate(missing_vertices):
    vertex_key = tuple(missing_v)
    orig_idx = missing_indices[idx]
    faces = vertex_to_faces.get(vertex_key, [])
    print(f"\nV{orig_idx+1}: ({missing_v[0]:.6f}, {missing_v[1]:.6f}, {missing_v[2]:.6f})")
    if faces:
        print(f"  Belongs to faces: {', '.join(faces)}")
    else:
        print(f"  Warning: No faces found!")

# Summary by face
print("\n" + "="*70)
print("SUMMARY: Missing vertices per face")
print("="*70)

face_missing_counts = {}
for vertex_key, faces in vertex_to_faces.items():
    for face_name in faces:
        if face_name not in face_missing_counts:
            face_missing_counts[face_name] = []
        face_missing_counts[face_name].append(vertex_key)

for face_name in sorted(face_missing_counts.keys()):
    vertices = face_missing_counts[face_name]
    print(f"\n{face_name}: {len(vertices)} missing vertices")
    for v in vertices:
        print(f"  ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")

print("\n" + "="*70)
print("DIAGNOSTIC: Why are these vertices missing?")
print("="*70)
print("\nPossible reasons:")
print("1. Vertices not visible in projection filtering (front + side views)")
print("2. Projection tolerance issues")
print("3. Vertices belong to faces that are partially occluded")
print("4. Edge connectivity issues in view summaries")

# Check if missing vertices would have been candidates
print("\n" + "="*70)
print("CHECKING: Would missing vertices be in candidate set?")
print("="*70)

# Load view summaries
top_view_summary = np.load('top_view_summary.npy')
front_view_summary = np.load('front_view_summary.npy')

# Extract unique (x,y) from top view
top_xy_coords = set()
for i in range(top_view_summary.shape[0]):
    x_proj, y_proj = top_view_summary[i, 3], top_view_summary[i, 4]
    top_xy_coords.add((round(x_proj, 6), round(y_proj, 6)))

# Extract z-levels from front view
z_levels = set()
for i in range(front_view_summary.shape[0]):
    z_world = front_view_summary[i, 2]
    z_levels.add(round(z_world, 6))

print(f"\nTop view (x,y) coordinates: {len(top_xy_coords)}")
print(f"Front view z-levels: {len(z_levels)}")

# Check each missing vertex
for idx, missing_v in enumerate(missing_vertices):
    orig_idx = missing_indices[idx]
    x, y, z = missing_v
    
    # Check if (x,y) exists in top view
    xy_key = (round(x, 6), round(y, 6))
    z_key = round(z, 6)
    
    has_xy = xy_key in top_xy_coords
    has_z = z_key in z_levels
    
    print(f"\nV{orig_idx+1}: ({x:.6f}, {y:.6f}, {z:.6f})")
    print(f"  (x,y) in top view: {has_xy}")
    print(f"  z in front view: {has_z}")
    print(f"  Would be candidate: {has_xy and has_z}")
    
    if has_xy and has_z:
        print(f"  ⚠️  This vertex SHOULD have been selected (candidate exists)!")
        print(f"  → Likely failed front/side view projection filtering")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
