# Step 6 REVISED Algorithm - Based on Correct Understanding

## Fundamental Principle
**An edge can only belong to ONE polygon.** Once used, it's marked as consumed.

---

## Algorithm Steps

### 1. Initialize from Step 5
**Input:**
- `edges_on_face`: List of edges with conn=3, e.g., [(v1, v2), (v3, v4), ...]
- `vertices_on_face`: List of all vertices on this face plane

**Create edge tracking:**
```python
unused_edges = set(edges_on_face)  # Track which edges haven't been used
edge_usage = {}  # Maps edge → polygon_id that uses it
```

---

### 2. Build Initial Polygon from Boundary Vertex

**Find starting vertex:**
- Project all vertices to 2D using face normal
- Find bounding box
- Choose vertex closest to bounding box edge (not corner)

**Build longest cycle:**
- Start from boundary vertex
- Follow edges greedily to build longest closed cycle
- This is the **candidate boundary polygon**

**Mark edges as used:**
```python
for edge in polygon_edges:
    unused_edges.remove(edge)
    edge_usage[edge] = 'boundary_candidate'
```

---

### 3. Merge Polygons that Share Edges with Boundary

**Key insight:** If another polygon shares an edge with the boundary, they should be merged into a single outer boundary.

**Algorithm:**
```python
boundary = initial_polygon
boundary_edges = set(get_edges(boundary))

while True:
    merged_any = False
    
    # Try to build another polygon from unused edges
    if not unused_edges:
        break
    
    new_polygon = build_polygon_from_unused_edges(unused_edges)
    if not new_polygon:
        break
    
    new_edges = set(get_edges(new_polygon))
    
    # Check if new polygon shares any edge with boundary
    shared_edges = boundary_edges & new_edges
    
    if shared_edges:
        # Merge into boundary
        boundary = merge_polygons(boundary, new_polygon, shared_edges)
        boundary_edges = set(get_edges(boundary))
        
        # Mark edges as boundary
        for edge in new_edges:
            unused_edges.discard(edge)
            edge_usage[edge] = 'boundary'
        
        merged_any = True
    else:
        # Put edges back - we'll handle this polygon later
        put_back_edges(new_edges)
        break
    
    if not merged_any:
        break

# Now we have the TRUE outer boundary
outer_boundary = boundary
```

**Polygon merging logic:**
When two polygons share an edge, remove that edge and connect the polygons:

```
Polygon A: [1, 2, 3, 4, 1]  (edge 2-3)
Polygon B: [2, 5, 6, 3, 2]  (edge 2-3 shared)

Merged: [1, 2, 5, 6, 3, 4, 1]  (edge 2-3 removed)
```

---

### 4. Classify Remaining Vertices

```python
boundary_vertices = set(outer_boundary)
all_vertices = set(vertices_on_face)
remaining_vertices = all_vertices - boundary_vertices

# Project to 2D and check containment
inside_vertices = []
outside_vertices = []

for v in remaining_vertices:
    if boundary_polygon_2d.contains(Point(v_2d)):
        inside_vertices.append(v)
    else:
        outside_vertices.append(v)
```

---

### 5. Build Polygons from Remaining Vertices

**For inside vertices (holes):**
```python
holes = []
while unused_edges and inside_vertices:
    # Get edges that connect inside vertices
    inside_edges = [e for e in unused_edges 
                    if e[0] in inside_vertices and e[1] in inside_vertices]
    
    if not inside_edges:
        break
    
    # Build polygon from these edges
    hole_polygon = build_polygon_from_edges(inside_edges, 
                                           start_from_any_vertex=True)
    
    if hole_polygon:
        # Verify: hole must NOT share vertices or edges with boundary
        hole_vertices = set(hole_polygon)
        hole_edges = set(get_edges(hole_polygon))
        
        if hole_vertices & boundary_vertices:
            # ERROR: Hole shares vertices with boundary
            # This shouldn't happen with correct merging
            raise ValueError("Hole shares vertices with boundary!")
        
        if hole_edges & set(get_edges(outer_boundary)):
            # ERROR: Hole shares edges with boundary
            raise ValueError("Hole shares edges with boundary!")
        
        holes.append(hole_polygon)
        
        # Mark edges as used
        for edge in hole_edges:
            unused_edges.discard(edge)
            edge_usage[edge] = f'hole_{len(holes)}'
        
        # Remove vertices from available pool
        inside_vertices = [v for v in inside_vertices if v not in hole_vertices]
    else:
        break
```

**For outside vertices (separate faces - DUMMY faces to delete):**
```python
dummy_faces = []
while unused_edges and outside_vertices:
    # Get edges that connect outside vertices
    outside_edges = [e for e in unused_edges 
                     if e[0] in outside_vertices and e[1] in outside_vertices]
    
    if not outside_edges:
        break
    
    # Build polygon from these edges
    dummy_polygon = build_polygon_from_edges(outside_edges,
                                            start_from_any_vertex=True)
    
    if dummy_polygon:
        # Verify: dummy face must NOT share vertices or edges with boundary
        dummy_vertices = set(dummy_polygon)
        dummy_edges = set(get_edges(dummy_polygon))
        
        if dummy_vertices & boundary_vertices:
            # ERROR: Dummy face shares vertices with boundary
            # This shouldn't happen with correct merging
            raise ValueError("Dummy face shares vertices with boundary!")
        
        if dummy_edges & set(get_edges(outer_boundary)):
            # ERROR: Dummy face shares edges with boundary
            raise ValueError("Dummy face shares edges with boundary!")
        
        dummy_faces.append(dummy_polygon)
        
        # Mark edges as DELETED (not used)
        for edge in dummy_edges:
            unused_edges.discard(edge)
            edge_usage[edge] = f'DELETED_dummy_{len(dummy_faces)}'
        
        # Remove vertices from available pool
        outside_vertices = [v for v in outside_vertices 
                           if v not in dummy_vertices]
    else:
        break
```

---

### 6. Verify All Edges Consumed or Deleted

```python
if unused_edges:
    print(f"WARNING: {len(unused_edges)} edges remain unused!")
    print(f"Unused edges: {unused_edges}")
    
    # Diagnose issue
    unused_vertices = set()
    for e in unused_edges:
        unused_vertices.add(e[0])
        unused_vertices.add(e[1])
    
    print(f"Vertices with unused edges: {unused_vertices}")
    print(f"Check if these form disconnected components")
else:
    print("✓ All edges consumed or deleted")

# Return result
return {
    'boundary': outer_boundary,
    'holes': holes,
    'dummy_faces': dummy_faces,
    'edge_usage': edge_usage
}
```

---

## Key Differences from Old Algorithm

### Old Algorithm Issues:
1. ❌ Used "vertex connectivity count" which was redundant
2. ❌ Allowed holes to share vertices with boundary (e.g., vertex 8, 24, 39)
3. ❌ Didn't merge polygons that share edges with boundary
4. ❌ Didn't explicitly mark edges as used/deleted
5. ❌ Treated inside/outside vertices differently in polygon building

### New Algorithm Fixes:
1. ✅ Simply tracks unused edges (no redundant connectivity)
2. ✅ Merges any polygon that shares edges with boundary
3. ✅ Ensures holes NEVER share vertices or edges with boundary
4. ✅ Explicitly tracks edge usage and deletion
5. ✅ Uses same polygon-building algorithm for inside/outside
6. ✅ Identifies dummy faces for deletion

---

## Polygon Merging Example

**Scenario: Face 7 Seed 55**

**Initial polygon built:**
```
Polygon A: [5, 55, 51, 2, 37, 29, 53, 31, 33, 39, 8, 24, 12, 35, 44, 22]
Edges include: (39, 8), (8, 24)
```

**Next polygon found from unused edges:**
```
Polygon B: [8, 24, 46, 39]
Edges: (8, 24), (24, 46), (46, 39), (39, 8)
```

**Shared edges:** (8, 24), (39, 8)

**Action: MERGE!**
```
Remove shared edges (8, 24) and (39, 8)
Connect at vertices 8, 24, 39

Result: [5, 55, 51, 2, 37, 29, 53, 31, 33, 39, 46, 24, 12, 35, 44, 22]
        (polygon B integrated into A)
```

Now vertex 46 is part of the boundary, and vertices 8, 24, 39 remain boundary vertices but polygon [8, 24, 46, 39] no longer exists as a separate entity.

---

## Expected Result for Face 7

**After merging:**
- Outer boundary: Larger polygon that includes vertex 46
- Holes: Possibly [16, 41, 19, 49] if it doesn't share edges with boundary
- No holes should share vertices with boundary

This explains why Face 7 currently has issues - we're incorrectly identifying [8, 24, 46, 39] as a hole when it should be merged into the boundary!
