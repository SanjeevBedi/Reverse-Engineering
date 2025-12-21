# Step 6: Polygon Formation Algorithm - Detailed Review

## Overview
Step 6 takes edges on a face (with connectivity = 3) and forms closed polygons, including detection of holes and internal features.

---

## Algorithm Structure

### Main Function: `build_polygons_from_edges()`

**Input:**
- `edges_to_use`: List of edges (v_i, v_j) to form polygons from
- `selected_verts`: Dictionary mapping vertex index → 3D coordinates
- `normal`: Face normal vector
- `all_vertices_on_face`: All vertex indices that lie on this face plane
- `depth`: Recursion depth (starts at 1 for top-level call)
- `vertex_connectivity`: Dict tracking remaining edge count per vertex

**Output:**
- Tuple: `(list of polygon dictionaries, updated vertex_connectivity)`

---

## Algorithm Phases

### Phase A: 2D Projection & Bounding Box
**Purpose:** Project 3D vertices onto face plane for 2D analysis

1. **Create orthonormal basis** (u, v) on face plane
   - If normal is mostly vertical: `u = normal × [0,0,1]`
   - Otherwise: `u = normal × [1,0,0]`
   - Then: `v = normal × u`

2. **Project all vertices** to 2D using dot products
   - `verts_2d[v_idx] = (dot(vert_3d, u), dot(vert_3d, v))`

3. **Compute bounding box**
   - Find min/max x and y coordinates

---

### Phase B: Find Starting Vertex
**Purpose:** Choose a vertex likely to be on the exterior boundary

**Strategy:**
- Find vertex **closest to bounding box edge** (not corner)
- Distance to boundary = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
- **Fallback:** If no boundary vertex found, use top-left corner vertex

**Why this works:**
- Vertices on bounding box boundary are typically on the outer polygon
- Avoids starting from internal vertices which might create incomplete cycles

---

### Phase C: Edge Refinement (Vertex-on-Edge Detection)
**Purpose:** Handle cases where vertices lie on edges

**For each edge (A, B):**
1. Check if any other vertex V lies on the line segment
2. Use parametric representation: `P = A + t*(B-A)` where `0 < t < 1`
3. If `distance(V, line_AB) < tolerance`:
   - Remove original edge (A, B)
   - Add chain: (A, V₁), (V₁, V₂), ..., (Vₙ, B)
   - Sort intermediate vertices by parameter `t`

**Example:**
```
Original: Edge (33, 55)
Detected: Vertices [39, 8, 27, 5] lie on this edge
Result: Chain [33, 39, 8, 27, 5, 55]
New edges: (33,39), (39,8), (8,27), (27,5), (5,55)
```

**Why important:**
- Ensures vertices on boundary can participate in polygon construction
- Critical for detecting internal features that share edges with outer boundary

---

### Phase D: Build Adjacency Graph
**Purpose:** Create connectivity structure for graph traversal

```python
adjacency[v_i] = [list of vertices connected to v_i]
```

**Vertex Connectivity Tracking (NEW):**
- At depth=1 (top level), initialize: `vertex_connectivity[v] = len(adjacency[v])`
- This counts how many edges each vertex participates in
- **Key insight:** Vertices can be in multiple polygons!

---

### Phase E: Cycle Detection
**Purpose:** Find closed polygon(s) from the edge graph

#### E.1: Try from Boundary Vertex First
- Start from vertex found in Phase B
- Try each neighbor as first step
- Follow greedy path: always choose unvisited neighbor
- Close cycle when returning to start vertex
- Track **longest** cycle found

**Traversal logic:**
```python
path = [start, first_neighbor]
current = first_neighbor
prev = start

while not done:
    neighbors = [n for n in adjacency[current] if n != prev]
    
    for n in neighbors:
        if n == start and len(path) >= 3:
            # Close cycle - found polygon!
            return path
        if n not in path:
            # Continue building path
            path.append(n)
            prev = current
            current = n
            break
```

#### E.2: Fallback to Any Cycle (depth > 1 only)
**Added to handle vertices shared between polygons**

If no cycle from boundary vertex:
- Try starting from **every vertex** in the graph
- Use same traversal logic
- Return first valid cycle found

**Why this works:**
- Internal holes (like [8, 24, 46, 39]) may not include the boundary vertex
- At depth=2+, we're processing unused edges which form disconnected components

---

### Phase F: Connectivity Decrement
**Purpose:** Track which edges have been consumed

**For each edge in the polygon:**
```python
for i in range(len(polygon)):
    v1 = polygon[i]
    v2 = polygon[(i + 1) % len(polygon)]
    vertex_connectivity[v1] -= 1
    vertex_connectivity[v2] -= 1
```

**Result:**
- Vertices with `connectivity > 0` still have unused edges
- These vertices can participate in additional polygons
- **This is the key to detecting shared-vertex holes!**

---

### Phase G: Edge Classification
**Purpose:** Classify remaining edges relative to the polygon

**Categories:**

1. **Exterior edges:** Form the polygon boundary (already used)

2. **Chord edges:** Both endpoints are polygon vertices
   - Connect two points on the boundary
   - Example: Edge (29, 51) where both 29 and 51 are in polygon
   - Used to split polygon into multiple regions

3. **Inside edges:** Both endpoints inside polygon
   - Form internal structures

4. **Outside edges:** Both endpoints outside polygon
   - Form separate polygons

**Classification logic:**
```python
if v1 in polygon_vertices and v2 in polygon_vertices:
    → chord_edge
elif contains(v1) and contains(v2):
    → inside_edge
elif not contains(v1) and not contains(v2):
    → outside_edge
else:
    → crosses boundary (warning)
```

---

### Phase H: Polygon Splitting (if chord edges found)
**Purpose:** Split polygon along chord edges into sub-regions

**For each chord (v1, v2):**
1. Find indices of v1 and v2 in polygon
2. Split into two parts:
   - Part 1: `polygon[idx1 : idx2+1]`
   - Part 2: `polygon[idx2:] + polygon[:idx1+1]` (wraps around)

**Example:**
```
Polygon: [5, 55, 51, 2, 37, 29, 53, 31, 33, 39, 8, 24, 12, 35, 44, 22]
Chord: (29, 51)
Result:
  Part 1: [29, 37, 2, 51]         (4 vertices)
  Part 2: [51, 55, 5, 22, 44, 35, 12, 24, 8, 39, 33, 31, 53, 29]  (14 vertices)
```

---

### Phase I: Iterative Unused Vertex Detection (NEW - Connectivity-Based)
**Purpose:** Find ALL polygons, including those sharing vertices with outer boundary

**Only at depth=1 (top level)**

```python
iteration = 0
while iteration < 10:  # Safety limit
    iteration += 1
    
    # Find vertices with unused edges
    vertices_with_unused_edges = {v for v, count in vertex_connectivity.items() if count > 0}
    
    if not vertices_with_unused_edges:
        break  # All edges consumed
    
    # Get edges between these vertices
    unused_edges = [all edges between vertices_with_unused_edges]
    
    if not unused_edges:
        break  # No more cycles possible
    
    # Recursively build polygons from unused edges
    unused_polygons, vertex_connectivity = build_polygons_from_edges(
        unused_edges, ..., depth + 1, vertex_connectivity)
    
    if not unused_polygons:
        break  # Couldn't form any polygons
    
    # Check if holes (inside outer) or separate faces (outside)
    for poly in unused_polygons:
        if outer_polygon.contains(poly):
            → HOLE detected
        else:
            → separate face
```

**Key innovation:**
- Uses `vertex_connectivity > 0` instead of "completely unused vertices"
- Allows vertices like 8, 24, 39 (used in outer polygon) to be in hole [8, 24, 46, 39]
- Iterates until all edges are consumed or no more cycles found

---

### Phase J: Recursive Processing
**Purpose:** Handle inside/outside edges hierarchically

**Inside edges:**
- Combined with chord edges
- Recursively process at `depth + 1`
- Result polygons added to face

**Outside edges:**
- Recursively process at `depth + 1`
- Form separate polygons (e.g., faces on opposite sides)

---

## Key Features & Innovations

### 1. Connectivity-Based Vertex Tracking
**Problem:** Vertices can be in multiple polygons
**Solution:** Track edge count, decrement on use, check `count > 0`

### 2. Any-Cycle Detection (depth > 1)
**Problem:** Internal holes may not include boundary vertex
**Solution:** Try starting from every vertex to find cycles

### 3. Iterative Unused Edge Processing
**Problem:** Multiple disconnected holes on same face
**Solution:** Loop until no vertices have unused edges

### 4. Edge Refinement
**Problem:** Vertices lying on edges create broken cycles
**Solution:** Detect and insert intermediate vertices into edge chains

### 5. Chord-Based Splitting
**Problem:** Single polygon may represent multiple regions
**Solution:** Split along chord edges connecting boundary vertices

---

## Example Walkthrough: Seed 55 Face 7

**Initial state:**
- 28 edges, 22 vertices
- Edge (33, 55) contains vertices [39, 8, 27, 5]
- Results in 34 refined edges

**Iteration 0 (depth=1):**
- Build outer polygon: 16 vertices
- Connectivity after: `{8: 3, 24: 1, 39: 3, 46: 2, ...}`
- Split along chord (29, 51) → 2 polygons

**Iteration 1 (depth=1):**
- Vertices with unused edges: [5, 8, 16, 19, 24, 27, 29, 33, 39, 41, 44, 46, 49, 51, 53, 55]
- 16 unused edges found
- Recursively build (depth=2):
  - No cycle from boundary vertex 33
  - **Search all vertices** → Find cycle [39, 8, 24, 46] ✓
  - Classified as HOLE (inside outer)
- Recursively process remaining (depth=3):
  - Find cycle [41, 16, 49, 19] ✓
  - Classified as HOLE (inside outer)

**Result:** 
- Face 7 with outer boundary + 2 holes detected! 🎉

---

## Current Limitations

### 1. Face 10 Failure
- 18 edges, 14 vertices
- Cannot find any cycle (even with any-cycle detection)
- Possible causes:
  - Disconnected edge components
  - Non-simple graph structure
  - Need better cycle detection algorithm

### 2. Negative Connectivity
- Vertex 24 reaches connectivity = -1
- Should be clamped to 0 or handled differently

### 3. Performance
- Any-cycle detection tries all vertices (O(V²))
- Could optimize with component detection

---

## Suggested Improvements

### 1. Connected Component Analysis
Before cycle detection, identify disconnected subgraphs:
```python
components = find_connected_components(adjacency)
for component in components:
    polygon = find_cycle_in_component(component)
```

### 2. All Cycles Detection
Instead of longest cycle, find ALL simple cycles:
- Use Johnson's algorithm or DFS-based cycle enumeration
- More robust for complex faces

### 3. Connectivity Bounds
```python
vertex_connectivity[v] = max(0, vertex_connectivity[v] - 1)
```

### 4. Better Start Vertex Heuristic
For depth > 1, choose start vertex with highest connectivity rather than boundary.

---

## Summary

**Step 6 successfully:**
✅ Projects face to 2D for geometric analysis
✅ Detects and refines edges with intermediate vertices  
✅ Builds outer polygons using greedy cycle detection
✅ **NEW:** Tracks vertex connectivity to enable multi-polygon participation
✅ **NEW:** Finds cycles not starting from boundary vertex
✅ **NEW:** Iteratively detects all holes with shared vertices
✅ Splits polygons along chord edges
✅ Classifies edges as inside/outside/chord
✅ Recursively processes internal structures

**Major achievement:**
Successfully detects polygon [8, 24, 46, 39] in Face 7 where vertices 8, 24, 39 are shared with the outer boundary!

**Remaining challenge:**
Face 10 with complex edge structure still cannot form polygons - needs investigation.
