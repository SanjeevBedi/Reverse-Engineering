# V6_current.py Function Analysis

**File:** `/Users/sbedi/Nextcloud/Python/Solid/random_solids/V6_current.py`  
**Total Lines:** 6633  
**Date:** November 28, 2025  

## Overview
This file implements a 3D solid reconstruction system from orthographic projections using connectivity matrices and polygon extraction algorithms.

---

## FUNCTION INVENTORY (24 Functions)

### 1. **print_merged_connectivity_matrix**
- **Line:** 1
- **Parameters:** `reconstructed_vertices, front_proj, side_proj, front_conn, side_conn, top_proj=None, top_conn=None`
- **Description:** Builds and prints merged connectivity matrix using available projections and connectivity matrices. If top_proj/top_conn are not provided, uses zeros.
- **Called by:** Unknown (appears to be utility function)
- **Calls:** `build_merged_connectivity_matrix` (line 12)
- **Call count:** 0 (not called in this file)

---

### 2. **build_merged_connectivity_matrix**
- **Line:** 26
- **Parameters:** `reconstructed_vertices, top_proj, front_proj, side_proj, top_conn, front_conn, side_conn`
- **Description:** Builds merged connectivity matrix from three views. For each pair of reconstructed vertices, counts in how many views the edge exists. Robustly matches reconstructed vertices to projections and view connectivity matrices.
- **Called by:** 
  - `print_merged_connectivity_matrix` (line 12)
  - Within plotting code (line 172)
- **Calls:** Internal helper functions
- **Call count:** 2

---

### 3. **plot_merged_connectivity**
- **Line:** 114
- **Parameters:** `reconstructed_vertices, merged_matrix, original_polygon=None`
- **Description:** Plots reconstructed vertices and edges from merged connectivity matrix. Toggles edges by connectivity (1, 2, 3) with color coding. Optionally plots original polygon.
- **Called by:** Unknown (appears within commented workflow code line 176)
- **Calls:** 
  - `build_merged_connectivity_matrix` (line 172)
  - `plot_polygon` (for visualization)
- **Call count:** 1

---

### 4. **build_solid_with_polygons_test**
- **Line:** 243
- **Parameters:** `config, quiet=False`
- **Description:** Creates a test solid using build_solid_with_polygons function. Applies scaling transformation and optional boolean operations (commented out). Returns scaled shape.
- **Called by:** `main` (line 6149)
- **Calls:** `build_solid_with_polygons` (from Reconstruction.Base_Solid)
- **Call count:** 1

---

### 5. **get_face_normal_from_opencascade**
- **Line:** 353
- **Parameters:** `face`
- **Description:** Extracts the correct face normal from OpenCASCADE face using multiple robust methods. Tries: GeomLProp_SLProps with orientation, surface derivatives with orientation, BRepGProp_Face method, and geometric analysis fallback.
- **Called by:** `extract_and_visualize_faces` (line 662)
- **Calls:** OpenCASCADE API functions
- **Call count:** 1

---

### 6. **extract_wire_vertices_in_sequence**
- **Line:** 458
- **Parameters:** `wire, wire_id`
- **Description:** Extracts vertices from a wire using simplified orientation-based logic. Uses corrected approach: Forward wire selects start vertex from each edge; Reversed wire selects end vertex from each edge. Duplicates first vertex at end to close polygon.
- **Called by:** `extract_and_visualize_faces` (lines 672, 676)
- **Calls:** OpenCASCADE traversal functions (BRepTools_WireExplorer)
- **Call count:** 2+

---

### 7. **extract_and_visualize_faces**
- **Line:** 623
- **Parameters:** `solid, visualize=False`
- **Description:** Extracts face data from an OpenCASCADE solid and optionally visualizes in 3D. Returns a list of face data dicts. Traverses BRep topology: Solid → Shells → Faces → Wires → Edges → Vertices.
- **Called by:** `main` (line 6198)
- **Calls:** 
  - `get_face_normal_from_opencascade` (line 662)
  - `extract_wire_vertices_in_sequence` (lines 672, 676)
- **Call count:** 1

---

### 8. **plot_polygon**
- **Line:** 772
- **Parameters:** `polygon, ax, facecolor='none', edgecolor='black', alpha=1.0, linewidth=1.0, linestyle='-', label=None, outline_only=False`
- **Description:** Simple polygon plotter using matplotlib. Plots polygon exterior and optionally fills it.
- **Called by:** 
  - `plot_merged_connectivity` (visualization)
  - `plot_arrays_visualization` (lines 966, 1040, 1100, 1123)
- **Calls:** Matplotlib functions
- **Call count:** 4+

---

### 9. **find_interior_point**
- **Line:** 783
- **Parameters:** `polygon, debug=False`
- **Description:** Finds an interior point within a polygon. If debug=True, returns (point, method_used). Uses representative_point (guaranteed inside), centroid fallback, or first coordinate of exterior.
- **Called by:** `classify_faces_by_projection` (line 1515)
- **Calls:** Shapely geometry methods
- **Call count:** 1

---

### 10. **intersect_line_with_face**
- **Line:** 810
- **Parameters:** `point_2d, projection_normal, face_vertices_3d`
- **Description:** Intersects a line with a 3D face to find depth. Creates orthogonal basis vectors for projection plane and calculates ray-plane intersection.
- **Called by:** `classify_faces_by_projection` (lines 1525, 1527)
- **Calls:** NumPy operations
- **Call count:** 2+

---

### 11. **calculate_depth_along_normal**
- **Line:** 856
- **Parameters:** `point_3d, projection_normal`
- **Description:** Calculates depth of a 3D point along the projection normal using dot product.
- **Called by:** `classify_faces_by_projection` (lines 1529, 1530)
- **Calls:** NumPy dot product
- **Call count:** 2+

---

### 12. **create_polygon_from_projection**
- **Line:** 865
- **Parameters:** `projected_vertices, allow_invalid=False`
- **Description:** Creates a Shapely polygon from projected vertices. Optionally allows invalid polygons. Attempts fixes using buffer(0) and convex_hull if invalid.
- **Called by:** `classify_faces_by_projection` (line 1369)
- **Calls:** Shapely Polygon creation and validation
- **Call count:** 1+

---

### 13. **plot_arrays_visualization**
- **Line:** 932
- **Parameters:** `array_A, array_B, array_C, unit_projection_normal`
- **Description:** Plots arrays B, C, and B+C with enhanced visualization. Shows visible faces (array_B), hidden faces + intersections (array_C), and combined view. Includes statistics and algorithm info.
- **Called by:** `classify_faces_by_projection` (line 1618)
- **Calls:** `plot_polygon` (lines 966, 1040, 1100, 1123)
- **Call count:** 1

---

### 14. **visualize_3d_solid**
- **Line:** 1194
- **Parameters:** `face_polygons, selected_vertices=None, edges=None, edges_with_class=None`
- **Description:** Displays the 3D solid using matplotlib 3D plotting. Optionally highlights selected vertices and edges with color-coding. Shows both original solid polygons and extracted polygons with toggle controls.
- **Called by:** 
  - `main` (lines 6247, 6603, 6621)
- **Calls:** Matplotlib 3D plotting functions
- **Call count:** 3

---

### 15. **classify_faces_by_projection**
- **Line:** 1315
- **Parameters:** `face_polygons, unit_projection_normal, no_graphics=False`
- **Description:** Enhanced face classification with historic polygon classification algorithm. Classifies faces as visible (array_B) or hidden (array_C) based on dot product and depth-based boolean operations.
- **Called by:** 
  - `plot_four_views` (line 1927)
- **Calls:** 
  - `project_face_to_projection_plane` (lines 1357, 1362)
  - `create_polygon_from_projection` (line 1369)
  - `find_interior_point` (line 1515)
  - `intersect_line_with_face` (lines 1525, 1527)
  - `calculate_depth_along_normal` (lines 1529, 1530)
  - `plot_arrays_visualization` (line 1618)
- **Call count:** 1+

---

### 16. **order_rectangular_vertices**
- **Line:** 1627
- **Parameters:** `vertices`
- **Description:** Trusts OpenCASCADE's natural vertex ordering for rectangular faces. Simply returns vertices as-is since OpenCASCADE provides them in correct topological order.
- **Called by:** Not called in this file (utility function)
- **Calls:** None
- **Call count:** 0

---

### 17. **generate_cuboid_faces**
- **Line:** 1641
- **Parameters:** `width, height, depth`
- **Description:** Generates the 6 faces of a cuboid with given dimensions. Returns face data with vertices and normals.
- **Called by:** Not called in this file (utility function)
- **Calls:** NumPy operations
- **Call count:** 0

---

### 18. **project_face_to_projection_plane**
- **Line:** 1668
- **Parameters:** `face_vertices, projection_normal`
- **Description:** Projects 3D face vertices to a 2D plane for engineering drawing display. Creates two orthogonal vectors in the projection plane.
- **Called by:** `classify_faces_by_projection` (lines 1357, 1362)
- **Calls:** NumPy operations
- **Call count:** 2+

---

### 19. **create_view_connectivity_matrix**
- **Line:** 1709
- **Parameters:** `visible, hidden, projection_normal, view_name, all_vertices_3d=None`
- **Description:** Creates connectivity matrix for a view with unique projected vertices. Extracts projected vertices from polygons and builds NxN connectivity showing which vertices are connected.
- **Called by:** `plot_four_views` (line 1933)
- **Calls:** Internal helper function `project_vertex_to_plane`
- **Call count:** 4 (once per view: top, front, side, isometric)

---

### 20. **plot_four_views**
- **Line:** 1873
- **Parameters:** `face_polygons, user_normal, ordered_vertices, Vertex_Top_View, Vertex_Front_View, Vertex_Side_View, Vertex_Iso_View, pdf_dir="PDFfiles", units="cm", drawing_scale_real=1.0, drawing_scale_drawing=1.0, no_graphics=False, seed=None`
- **Description:** Plots four engineering views (top, front, side, isometric) of the solid. Creates connectivity matrices for each view and saves to PDF. Returns view_connectivity_matrices dict.
- **Called by:** `main` (line 6276)
- **Calls:** 
  - `classify_faces_by_projection` (line 1927)
  - `create_view_connectivity_matrix` (line 1933)
- **Call count:** 1

---

### 21. **split_colinear_edges_in_faces**
- **Line:** 2141
- **Parameters:** `faces, selected_vertices, tolerance=1e-6`
- **Description:** Splits colinear edges in all face polygons (boundaries, holes, and alternates). For each edge, checks if any other vertex lies on that edge and splits accordingly.
- **Called by:** Not called in this file (appears to be unused utility)
- **Calls:** Internal helper functions
- **Call count:** 0

---

### 22. **extract_polygon_faces_from_connectivity**
- **Line:** 2283
- **Parameters:** `selected_vertices, merged_conn, tolerance=1e-6`
- **Description:** **CRITICAL FUNCTION** - Extracts polygon faces from connectivity matrix using planar face detection. Main algorithm:
  1. Generate face normals from non-collinear vector pairs
  2. Create unique list of face equations with plane ranges
  3. Find all vertices on each face using iterative tolerance
  4. Build edges and join into closed polygons
  5. Identify outer boundaries and inner holes
  6. Handle alternates and validate edge-face topology
- **Called by:** `main` (line 6607)
- **Calls:** Many internal helper functions including:
  - `find_all_cycles_from_edges`
  - `merge_polygons_sharing_edge`
  - `build_polygons_from_face_edges`
  - `group_polygons_with_holes`
- **Call count:** 1

---

### 23. **plot_extracted_polygon_faces**
- **Line:** 5926
- **Parameters:** `extracted_faces, selected_vertices, original_faces, units="cm", drawing_scale_real=1.0, drawing_scale_drawing=1.0`
- **Description:** Plots extracted polygon faces with controls to toggle visibility. Unified view showing both original solid faces and extracted polygons using matplotlib 3D.
- **Called by:** `main` (line 6613)
- **Calls:** Matplotlib 3D plotting functions
- **Call count:** 1

---

### 24. **main**
- **Line:** 6067
- **Parameters:** None
- **Description:** **MAIN WORKFLOW FUNCTION** - Entry point for the program. Orchestrates entire reconstruction pipeline:
  1. Parses command-line arguments
  2. Loads/creates configuration
  3. Builds test solid
  4. Extracts face polygons
  5. Extracts vertices from solid
  6. Creates four-view projections
  7. Generates connectivity matrices
  8. Reconstructs vertices from projections
  9. Extracts polygon faces
  10. Visualizes results
- **Called by:** `__main__` block (line 6630)
- **Calls:** 
  - `build_solid_with_polygons_test` (line 6149)
  - `extract_and_visualize_faces` (line 6198)
  - `visualize_3d_solid` (lines 6247, 6603, 6621)
  - `plot_four_views` (line 6276)
  - `extract_polygon_faces_from_connectivity` (line 6607)
  - `plot_extracted_polygon_faces` (line 6613)
- **Call count:** 1 (entry point)

---

## CALL GRAPH HIERARCHY

```
main (entry point)
├── build_solid_with_polygons_test
│   └── build_solid_with_polygons (external)
├── extract_and_visualize_faces
│   ├── get_face_normal_from_opencascade
│   └── extract_wire_vertices_in_sequence (×2+)
├── visualize_3d_solid (×3)
├── plot_four_views
│   ├── classify_faces_by_projection (×4 views)
│   │   ├── project_face_to_projection_plane (×2+)
│   │   ├── create_polygon_from_projection
│   │   ├── find_interior_point
│   │   ├── intersect_line_with_face (×2)
│   │   ├── calculate_depth_along_normal (×2)
│   │   └── plot_arrays_visualization
│   │       └── plot_polygon (×4+)
│   └── create_view_connectivity_matrix (×4 views)
├── extract_polygon_faces_from_connectivity
│   └── [Many internal helper functions]
└── plot_extracted_polygon_faces

Standalone/Utility:
├── print_merged_connectivity_matrix
│   └── build_merged_connectivity_matrix
├── plot_merged_connectivity
│   └── build_merged_connectivity_matrix
├── order_rectangular_vertices (unused)
├── generate_cuboid_faces (unused)
└── split_colinear_edges_in_faces (unused)
```

---

## USAGE STATISTICS

| Function | Calls in File | Called By | Primary Purpose |
|----------|--------------|-----------|-----------------|
| main | 1 | __main__ | Entry point |
| extract_polygon_faces_from_connectivity | 1 | main | **Core algorithm** |
| plot_four_views | 1 | main | Engineering views |
| extract_and_visualize_faces | 1 | main | Face extraction |
| build_solid_with_polygons_test | 1 | main | Solid creation |
| classify_faces_by_projection | 4 | plot_four_views | Visibility classification |
| create_view_connectivity_matrix | 4 | plot_four_views | Connectivity |
| visualize_3d_solid | 3 | main | 3D visualization |
| build_merged_connectivity_matrix | 2 | Various | Merge connections |
| project_face_to_projection_plane | 2+ | classify_faces | Projection |
| intersect_line_with_face | 2+ | classify_faces | Depth calculation |
| calculate_depth_along_normal | 2+ | classify_faces | Depth calculation |
| plot_polygon | 4+ | plot_arrays | 2D plotting |
| plot_merged_connectivity | 1 | Commented | Debugging |
| plot_arrays_visualization | 1 | classify_faces | Debugging |
| plot_extracted_polygon_faces | 1 | main | Final visualization |
| get_face_normal_from_opencascade | 1 | extract_and_visualize | Normal extraction |
| extract_wire_vertices_in_sequence | 2+ | extract_and_visualize | Vertex extraction |
| find_interior_point | 1 | classify_faces | Interior point |
| create_polygon_from_projection | 1+ | classify_faces | Polygon creation |
| order_rectangular_vertices | 0 | None | Unused |
| generate_cuboid_faces | 0 | None | Unused |
| split_colinear_edges_in_faces | 0 | None | Unused |
| print_merged_connectivity_matrix | 0 | None | Unused |

---

## KEY WORKFLOW PATHS

### **Path 1: Solid Creation → Face Extraction**
```
main
→ build_solid_with_polygons_test
→ extract_and_visualize_faces
  → get_face_normal_from_opencascade
  → extract_wire_vertices_in_sequence
```

### **Path 2: View Generation → Connectivity**
```
main
→ plot_four_views
  → classify_faces_by_projection (per view)
    → project_face_to_projection_plane
    → create_polygon_from_projection
    → find_interior_point
    → intersect_line_with_face
    → calculate_depth_along_normal
  → create_view_connectivity_matrix (per view)
```

### **Path 3: Polygon Reconstruction**
```
main
→ extract_polygon_faces_from_connectivity
  → [Complex internal algorithm]
    - Generate face equations
    - Assign vertices to faces
    - Build edge cycles
    - Identify boundaries and holes
    - Validate topology
```

### **Path 4: Visualization**
```
main
→ visualize_3d_solid (×3 at different stages)
→ plot_extracted_polygon_faces
```

---

## FUNCTION COMPLEXITY RANKING

1. **extract_polygon_faces_from_connectivity** - HIGHEST (3644 lines, complex algorithm)
2. **classify_faces_by_projection** - HIGH (312 lines, depth analysis)
3. **plot_four_views** - MEDIUM-HIGH (268 lines, multi-view handling)
4. **main** - MEDIUM-HIGH (563 lines, orchestration)
5. **extract_and_visualize_faces** - MEDIUM (149 lines)
6. **visualize_3d_solid** - MEDIUM (121 lines)
7. **plot_extracted_polygon_faces** - MEDIUM (140 lines)
8. All others - LOW to MEDIUM (< 100 lines)

---

## NOTES FOR LATEX/DOCUMENTATION

- Functions are organized in execution order within the file
- Main workflow starts at line 6067
- Core algorithm (`extract_polygon_faces_from_connectivity`) spans lines 2283-5926
- Three unused utility functions could be removed
- Heavy use of NumPy and Shapely for geometric operations
- OpenCASCADE integration for solid modeling
- Matplotlib for visualization

---

**Generated:** November 28, 2025  
**Analysis Tool:** Claude Sonnet 4.5
