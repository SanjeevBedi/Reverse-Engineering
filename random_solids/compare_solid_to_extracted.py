#!/usr/bin/env python3
"""
Compare faces in STEP file to faces in extracted face_polygons.
Find out which faces contain the target edges and whether they were extracted.
"""

import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool

# Load extracted face polygons
face_file = "Output/solid_faces_seed_250.npy"
data = np.load(face_file, allow_pickle=True).item()
extracted_faces = data['faces']

print(f"Extracted {len(extracted_faces)} faces from face_polygons file")

# Load solid from STEP file
step_file = "STEPfiles/solid_output.step"
step_reader = STEPControl_Reader()
status = step_reader.ReadFile(step_file)
if status != IFSelect_RetDone:
    print(f"ERROR: Failed to read STEP file")
    exit(1)

step_reader.TransferRoots()
solid = step_reader.OneShape()

# Target vertices (in cm)
target_vertices_cm = {
    22: (42.715545, 33.357964, 0.0),
    23: (42.715545, 33.357964, 18.636713),
    37: (37.912317, 38.161192, 0.0),
    38: (37.912317, 38.161192, 18.636713),
    40: (36.5, 33.357964, 18.636713),
    41: (36.5, 33.357964, 30.424143),
    42: (29.875562, 30.124438, 0.0),
    43: (29.875562, 30.124438, 18.636713),
    45: (36.5, 23.5, 0.0),
    46: (36.5, 23.5, 18.636713),
}

target_edges = [
    (22, 23, "V22-V23"),
    (37, 38, "V37-V38"),
    (40, 41, "V40-V41"),
    (42, 43, "V42-V43"),
    (45, 46, "V45-V46"),
]

def vertices_match(v1, v2, tol=0.1):
    """Check if two vertices match within tolerance"""
    if isinstance(v1, np.ndarray):
        v1_arr = v1[:3] if len(v1) >= 3 else v1
    elif isinstance(v1, (tuple, list)):
        v1_arr = np.array(v1[:3])
    else:
        v1_arr = np.array([v1.X(), v1.Y(), v1.Z()])
    
    if isinstance(v2, np.ndarray):
        v2_arr = v2[:3] if len(v2) >= 3 else v2
    elif isinstance(v2, (tuple, list)):
        v2_arr = np.array(v2[:3])
    else:
        v2_arr = np.array([v2.X(), v2.Y(), v2.Z()])
    
    return np.allclose(v1_arr, v2_arr, atol=tol)

def get_face_vertices_from_occ(face):
    """Extract vertices from an OpenCASCADE face"""
    vertices = []
    vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        pnt = BRep_Tool.Pnt(vertex)
        vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
        vertex_explorer.Next()
    return vertices

def face_contains_edge(face_vertices, v1_target, v2_target):
    """Check if a face contains both vertices of an edge"""
    v1_found = any(vertices_match(fv, v1_target) for fv in face_vertices)
    v2_found = any(vertices_match(fv, v2_target) for fv in face_vertices)
    return v1_found and v2_found

print(f"\n{'='*70}")
print("STEP 1: Check which faces in SOLID contain target edges")
print(f"{'='*70}")

# Analyze faces in solid
face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
face_count = 0
solid_faces_with_edges = {}

while face_explorer.More():
    face = topods.Face(face_explorer.Current())
    face_count += 1
    face_vertices = get_face_vertices_from_occ(face)
    
    edges_in_face = []
    for v1_num, v2_num, edge_name in target_edges:
        if face_contains_edge(face_vertices, target_vertices_cm[v1_num], target_vertices_cm[v2_num]):
            edges_in_face.append(edge_name)
    
    if edges_in_face:
        solid_faces_with_edges[face_count] = {
            'vertices': face_vertices,
            'edges': edges_in_face,
            'num_vertices': len(face_vertices)
        }
    
    face_explorer.Next()

print(f"\nTotal faces in solid: {face_count}")
print(f"Faces containing target edges: {len(solid_faces_with_edges)}")

for face_num, info in sorted(solid_faces_with_edges.items()):
    print(f"  Face {face_num}: {len(info['vertices'])} vertices, edges: {', '.join(info['edges'])}")

print(f"\n{'='*70}")
print("STEP 2: Check if these faces were extracted to face_polygons")
print(f"{'='*70}")

# Check if extracted faces match the solid faces with target edges
print(f"\nChecking {len(extracted_faces)} extracted faces...")

matched_faces = []
unmatched_solid_faces = list(solid_faces_with_edges.keys())

for ext_idx, ext_face in enumerate(extracted_faces):
    ext_vertices = ext_face['outer_boundary']
    ext_vertex_set = set(tuple(v) for v in ext_vertices)
    
    # Try to match with solid faces
    for solid_face_num, solid_info in solid_faces_with_edges.items():
        solid_vertex_set = set(tuple(np.round(v, 6)) for v in solid_info['vertices'])
        
        # Check if vertices match (with some tolerance for rounding)
        matches = 0
        for ev in ext_vertices:
            ev_tuple = tuple(np.round(ev, 6))
            if ev_tuple in solid_vertex_set:
                matches += 1
        
        # If most vertices match, consider it the same face
        if matches >= min(len(ext_vertices), len(solid_info['vertices'])) * 0.8:
            matched_faces.append({
                'extracted_idx': ext_idx + 1,
                'solid_face': solid_face_num,
                'edges': solid_info['edges']
            })
            if solid_face_num in unmatched_solid_faces:
                unmatched_solid_faces.remove(solid_face_num)
            break

print(f"\nMatched {len(matched_faces)} faces containing target edges:")
for match in matched_faces:
    print(f"  Extracted face #{match['extracted_idx']} = Solid face #{match['solid_face']}: {', '.join(match['edges'])}")

if unmatched_solid_faces:
    print(f"\n⚠️  {len(unmatched_solid_faces)} faces with target edges NOT found in extracted faces!")
    print(f"Missing faces from solid:")
    for face_num in unmatched_solid_faces:
        info = solid_faces_with_edges[face_num]
        print(f"  Face {face_num}: {len(info['vertices'])} vertices, edges: {', '.join(info['edges'])}")
else:
    print(f"\n✓ All faces containing target edges were successfully extracted!")

print(f"\n{'='*70}")
print("STEP 3: Check if missing edges are in EXTRACTED faces")
print(f"{'='*70}")

# For each target edge, check if it's in ANY extracted face
for v1_num, v2_num, edge_name in target_edges:
    found_in_extracted = []
    
    for ext_idx, ext_face in enumerate(extracted_faces):
        ext_vertices = ext_face['outer_boundary']
        if face_contains_edge(ext_vertices, target_vertices_cm[v1_num], target_vertices_cm[v2_num]):
            found_in_extracted.append(ext_idx + 1)
    
    if found_in_extracted:
        print(f"✓ {edge_name}: Found in extracted faces {found_in_extracted}")
    else:
        print(f"✗ {edge_name}: NOT in any extracted face")

print(f"\n{'='*70}")
