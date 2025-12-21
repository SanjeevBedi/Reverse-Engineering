"""
Classification algorithm module for OpenCASCADE Engineering Drawings Generator.

This module implements the historic polygon classification algorithm.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from shapely.geometry import Polygon, Point


class ClassificationAlgorithm:
    """
    Implements the historic polygon classification algorithm.
    
    This algorithm classifies projected 2D polygons into:
    - Array B: Visible polygons
    - Array C: Hidden polygons and intersections
    
    The algorithm uses depth analysis and polygon intersection detection.
    """
    
    def __init__(self):
        """Initialize the classification algorithm."""
        pass
    
    def classify_polygons(self, projected_polygons: List[dict], 
                          projection_normal: List[float]) -> Tuple[List[dict], 
                                                                  List[dict], 
                                                                  List[dict]]:
        """
        Classify projected polygons using the historic algorithm.
        
        Args:
            projected_polygons: List of projected polygon data
            projection_normal: Projection direction [x, y, z]
            
        Returns:
            Tuple of (array_A, array_B, array_C) containing classified polygons
        """
        print("\n" + "="*60)
        print("HISTORIC POLYGON CLASSIFICATION ALGORITHM")
        print("="*60)
        
        array_A = []  # Initial classification
        array_B = []  # Visible polygons
        array_C = []  # Hidden + intersections
        
        unit_normal = np.array(projection_normal) / np.linalg.norm(
            projection_normal)
        
        print(f"Projection normal: [{unit_normal[0]:.6f}, "
              f"{unit_normal[1]:.6f}, {unit_normal[2]:.6f}]")
        print(f"\nStep 1: Converting projected data to polygons...")
        
        # Convert projected data to Shapely polygons
        valid_polygons = []
        
        for i, poly_data in enumerate(projected_polygons):
            try:
                vertices_2d = poly_data['projected_vertices']
                polygon = self._create_polygon_from_vertices(vertices_2d)
                
                if polygon.area > 1e-6:
                    enhanced_data = {
                        'polygon': polygon,
                        'name': poly_data['name'],
                        'normal': poly_data.get('face_normal'),
                        'parent_face': poly_data.get('original_vertices'),
                        'original_index': i,
                        'dot_product': poly_data.get('dot_product', 0)
                    }
                    
                    valid_polygons.append(enhanced_data)
                    array_A.append(enhanced_data)
                    print(f"  → Added {poly_data['name']} "
                          f"(area: {polygon.area:.2f})")
                    
            except Exception as e:
                print(f"  ✗ Error processing {poly_data.get('name', i)}: {e}")
        
        print(f"\nStep 2: Starting historic classification algorithm...")
        print(f"Initial array_A: {len(array_A)} polygons")
        
        # Display initial array contents
        self._display_array_contents(array_A, "A - INITIAL")
        
        if len(array_A) >= 1:
            # Step 2.1: Move first polygon to array_B as seed
            first_polygon = array_A.pop(0)
            array_B.append(first_polygon)
            print(f"\nMoved {first_polygon['name']} from array_A to array_B "
                  f"as seed")
            
            # Step 2.2: Process remaining polygons with depth-based 
            # classification
            while array_A:
                Pi_data = array_A.pop(0)
                Pi = Pi_data['polygon']
                Pi_name = Pi_data['name']
                Pi_parent_face = Pi_data['parent_face']
                
                print(f"\nProcessing {Pi_name} (area: {Pi.area:.2f})")
                
                intersection_found = False
                
                # Test intersection with all polygons in array_B
                for j, Pj_data in enumerate(array_B):
                    Pj = Pj_data['polygon']
                    Pj_name = Pj_data['name']
                    Pj_parent_face = Pj_data['parent_face']
                    
                    try:
                        intersection = Pi.intersection(Pj)
                        
                        if (hasattr(intersection, 'area') and 
                            intersection.area > 1e-6):
                            
                            print(f"  → Intersection found with {Pj_name} "
                                  f"(area: {intersection.area:.3f})")
                            
                            # Perform depth analysis
                            depth_result = self._analyze_depth_relationship(
                                Pi_parent_face, Pj_parent_face, unit_normal)
                            
                            if depth_result == "Pi_behind":
                                # Pi is behind Pj, move Pi to array_C
                                array_C.append(Pi_data)
                                print(f"  → {Pi_name} is BEHIND {Pj_name}, "
                                      f"moved to array_C")
                                intersection_found = True
                                break
                            elif depth_result == "Pi_front":
                                # Pi is in front of Pj, move Pj to array_C
                                moved_polygon = array_B.pop(j)
                                array_C.append(moved_polygon)
                                print(f"  → {Pi_name} is FRONT of {Pj_name}, "
                                      f"moved {Pj_name} to array_C")
                                # Continue testing Pi against remaining 
                                # polygons in B
                                break
                            else:
                                # Ambiguous case, create intersection polygon
                                intersection_data = {
                                    'polygon': intersection,
                                    'name': f"Intersection_{Pi_name}_{Pj_name}",
                                    'normal': None,
                                    'parent_face': None,
                                    'original_index': -1,
                                    'dot_product': 0
                                }
                                array_C.append(intersection_data)
                                print(f"  → Created intersection polygon in "
                                      f"array_C")
                                
                    except Exception as e:
                        print(f"  → Intersection test error with {Pj_name}: "
                              f"{e}")
                
                # If no intersection found, add to array_B
                if not intersection_found:
                    array_B.append(Pi_data)
                    print(f"  → No intersections found, added {Pi_name} "
                          f"to array_B")
            
            # Step 2.3: Apply final dot product classification
            print(f"\nStep 3: Applying final dot product classification...")
            faces_to_move = []
            
            for i, poly_data in enumerate(array_B):
                dot_product = poly_data.get('dot_product', 0)
                if dot_product < 0:
                    faces_to_move.append(i)
                    print(f"  → {poly_data['name']} has negative dot product "
                          f"({dot_product:.3f}), will move to array_C")
            
            # Move faces with negative dot product to array_C
            for i in reversed(faces_to_move):
                moved_polygon = array_B.pop(i)
                array_C.append(moved_polygon)
                print(f"  → Moved {moved_polygon['name']} to array_C "
                      f"(negative dot product)")
        
        print(f"\n" + "="*60)
        print("CLASSIFICATION COMPLETE")
        print("="*60)
        print(f"Array A (processed): 0 faces (all processed)")
        print(f"Array B (visible): {len(array_B)} faces")
        print(f"Array C (hidden+intersections): {len(array_C)} faces")
        print(f"Total: {len(array_B) + len(array_C)} faces")
        
        return [], array_B, array_C
    
    def _create_polygon_from_vertices(self, vertices_2d: List[List[float]]) -> Polygon:
        """Create a Shapely polygon from 2D vertices."""
        if len(vertices_2d) == 0:
            return Polygon()
        
        # Ensure polygon is closed
        vertices_array = np.array(vertices_2d)
        if len(vertices_array) > 0:
            if not np.allclose(vertices_array[0], vertices_array[-1], 
                               atol=1e-10):
                vertices_array = np.vstack([vertices_array, vertices_array[0]])
        
        try:
            polygon = Polygon(vertices_array)
            
            # For valid polygons, return directly
            if polygon.is_valid and polygon.area > 1e-6:
                return polygon
            
            # Try to fix invalid polygons
            if not polygon.is_valid:
                try:
                    fixed_polygon = polygon.buffer(0)
                    if (fixed_polygon.is_valid and 
                        hasattr(fixed_polygon, 'area') and 
                        fixed_polygon.area > 1e-6):
                        return fixed_polygon
                except Exception:
                    pass
                
                # Fallback to convex hull
                try:
                    hull_polygon = polygon.convex_hull
                    if (hull_polygon.is_valid and 
                        hasattr(hull_polygon, 'area') and 
                        hull_polygon.area > 1e-6):
                        return hull_polygon
                except Exception:
                    pass
            
            return Polygon()
            
        except Exception:
            return Polygon()
    
    def _analyze_depth_relationship(self, face1_vertices: List[List[float]], 
                                    face2_vertices: List[List[float]], 
                                    projection_normal: np.ndarray) -> str:
        """
        Analyze depth relationship between two faces.
        
        Returns:
            "Pi_behind", "Pi_front", or "ambiguous"
        """
        try:
            if (face1_vertices is None or face2_vertices is None or 
                len(face1_vertices) < 3 or len(face2_vertices) < 3):
                return "ambiguous"
            
            # Calculate average depth for each face
            depth1_samples = []
            depth2_samples = []
            
            for vertex in face1_vertices:
                depth = np.dot(vertex, projection_normal)
                depth1_samples.append(depth)
            
            for vertex in face2_vertices:
                depth = np.dot(vertex, projection_normal)
                depth2_samples.append(depth)
            
            avg_depth1 = np.mean(depth1_samples)
            avg_depth2 = np.mean(depth2_samples)
            
            # Compare depths with tolerance
            depth_tolerance = 1e-6
            depth_diff = avg_depth1 - avg_depth2
            
            if depth_diff < -depth_tolerance:
                return "Pi_behind"
            elif depth_diff > depth_tolerance:
                return "Pi_front"
            else:
                return "ambiguous"
                
        except Exception as e:
            print(f"    Error in depth analysis: {e}")
            return "ambiguous"
    
    def _display_array_contents(self, array: List[dict], array_name: str):
        """Display the contents of an array for debugging."""
        if not array:
            return
        
        print(f"\n" + "="*60)
        print(f"ARRAY {array_name} CONTENTS ({len(array)} polygons)")
        print("="*60)
        
        for i, poly_data in enumerate(array):
            polygon = poly_data['polygon']
            name = poly_data['name']
            normal = poly_data.get('normal')
            dot_product = poly_data.get('dot_product', 0)
            
            # Get vertex count
            if hasattr(polygon, 'exterior'):
                vertex_count = len(polygon.exterior.coords) - 1
            elif hasattr(polygon, 'geoms') and len(polygon.geoms) > 0:
                vertex_count = len(polygon.geoms[0].exterior.coords) - 1
            else:
                vertex_count = 0
            
            print(f"  Polygon {i+1} ({name}):")
            print(f"    • Area: {polygon.area:.2f}")
            print(f"    • Vertices: {vertex_count}")
            print(f"    • Dot product: {dot_product:.6f}")
            
            if (normal is not None and hasattr(normal, '__len__') and 
                len(normal) >= 3):
                print(f"    • Normal: [{normal[0]:.3f}, {normal[1]:.3f}, "
                      f"{normal[2]:.3f}]")
            else:
                print(f"    • Normal: Not available")
        
        print("="*60)
