"""
Projection engine module for OpenCASCADE Engineering Drawings Generator.

This module handles 3D to 2D projection operations.
"""

import numpy as np
from typing import List, Tuple, Any


class ProjectionEngine:
    """
    Handles projection of 3D geometry to 2D planes.
    
    This class provides methods to:
    - Get projection normals from user input
    - Project 3D vertices to 2D coordinates
    - Normalize vectors
    - Handle interactive projection direction selection
    """
    
    def __init__(self):
        """Initialize the projection engine."""
        pass
    
    def get_projection_normal_interactive(self) -> List[float]:
        """
        Get projection normal from user input with [1,1,1] as default.
        
        Returns:
            Normalized projection normal vector [x, y, z]
        """
        print("\n" + "="*60)
        print("PROJECTION NORMAL INPUT")
        print("="*60)
        
        default_normal = [1, 1, 1]
        
        try:
            print(f"Enter projection normal vector components "
                  f"(default: {default_normal}):")
            print("Format: x y z (space separated) or press Enter for default")
            
            user_input = input("Projection normal: ").strip()
            
            if not user_input:
                # Use default
                projection_normal = np.array(default_normal, dtype=float)
                print(f"Using default projection normal: {projection_normal}")
            else:
                # Parse user input
                components = user_input.split()
                if len(components) != 3:
                    print(f"Invalid input format. Using default: "
                          f"{default_normal}")
                    projection_normal = np.array(default_normal, dtype=float)
                else:
                    try:
                        projection_normal = np.array([float(x) for x in 
                                                      components])
                        print(f"User input projection normal: "
                              f"{projection_normal}")
                    except ValueError:
                        print(f"Invalid number format. Using default: "
                              f"{default_normal}")
                        projection_normal = np.array(default_normal, 
                                                     dtype=float)
            
            # Convert to unit vector
            unit_projection_normal = self.normalize_vector(projection_normal)
            
            print(f"Original projection normal: "
                  f"[{projection_normal[0]:.3f}, {projection_normal[1]:.3f}, "
                  f"{projection_normal[2]:.3f}]")
            print(f"Unit projection normal: "
                  f"[{unit_projection_normal[0]:.6f}, "
                  f"{unit_projection_normal[1]:.6f}, "
                  f"{unit_projection_normal[2]:.6f}]")
            print(f"Magnitude: {np.linalg.norm(projection_normal):.6f}")
            print("="*60)
            
            return unit_projection_normal.tolist()
            
        except KeyboardInterrupt:
            print(f"\nInterrupted. Using default: {default_normal}")
            projection_normal = np.array(default_normal, dtype=float)
            unit_projection_normal = self.normalize_vector(projection_normal)
            return unit_projection_normal.tolist()
        except Exception as e:
            print(f"Error getting user input: {e}. Using default: "
                  f"{default_normal}")
            projection_normal = np.array(default_normal, dtype=float)
            unit_projection_normal = self.normalize_vector(projection_normal)
            return unit_projection_normal.tolist()
    
    def normalize_vector(self, vector: List[float]) -> np.ndarray:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Input vector [x, y, z]
            
        Returns:
            Normalized unit vector
        """
        vector_array = np.array(vector, dtype=float)
        magnitude = np.linalg.norm(vector_array)
        
        if magnitude < 1e-10:
            print(f"Warning: Zero vector detected. Using default [1,1,1]")
            vector_array = np.array([1, 1, 1], dtype=float)
            magnitude = np.linalg.norm(vector_array)
        
        return vector_array / magnitude
    
    def project_faces_to_2d(self, faces: List[dict], 
                            projection_normal: List[float]) -> List[dict]:
        """
        Project 3D faces to 2D polygons.
        
        Args:
            faces: List of face data dictionaries
            projection_normal: Projection direction [x, y, z]
            
        Returns:
            List of projected polygon data
        """
        projected_polygons = []
        unit_normal = self.normalize_vector(projection_normal)
        
        print(f"\nProjecting {len(faces)} faces to 2D...")
        print(f"Projection normal: [{unit_normal[0]:.6f}, "
              f"{unit_normal[1]:.6f}, {unit_normal[2]:.6f}]")
        
        for i, face_data in enumerate(faces):
            face_id = face_data.get('face_id', i+1)
            outer_boundary = face_data.get('outer_boundary', [])
            face_normal = face_data.get('normal')
            
            if len(outer_boundary) < 3:
                print(f"  Face {face_id}: Insufficient vertices - skipping")
                continue
            
            # Project vertices to 2D
            projected_vertices = self.project_vertices_to_plane(
                outer_boundary, unit_normal)
            
            if len(projected_vertices) >= 3:
                # Calculate dot product for visibility
                dot_product = 0
                if face_normal is not None:
                    face_unit_normal = self.normalize_vector(face_normal)
                    dot_product = np.dot(face_unit_normal, unit_normal)
                
                polygon_data = {
                    'projected_vertices': projected_vertices,
                    'original_vertices': outer_boundary,
                    'face_id': face_id,
                    'face_normal': face_normal,
                    'dot_product': dot_product,
                    'name': f"Face_{face_id}"
                }
                
                projected_polygons.append(polygon_data)
                print(f"  Face {face_id}: ✓ Projected {len(outer_boundary)} "
                      f"→ {len(projected_vertices)} vertices, "
                      f"dot={dot_product:.3f}")
            else:
                print(f"  Face {face_id}: ✗ Projection failed")
        
        print(f"✓ Successfully projected {len(projected_polygons)} faces")
        return projected_polygons
    
    def project_vertices_to_plane(self, vertices_3d: List[List[float]], 
                                  projection_normal: List[float]) -> List[List[float]]:
        """
        Project 3D vertices to a 2D plane perpendicular to projection normal.
        
        Args:
            vertices_3d: List of [x, y, z] coordinates
            projection_normal: Projection direction [x, y, z]
            
        Returns:
            List of [u, v] 2D coordinates
        """
        if not vertices_3d:
            return []
        
        # Normalize the projection normal
        normal = self.normalize_vector(projection_normal)
        
        # Create orthogonal basis vectors for the projection plane
        # Find a vector not parallel to normal
        if abs(normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        # Create first basis vector (orthogonal to normal)
        u_axis = temp - np.dot(temp, normal) * normal
        u_axis = u_axis / np.linalg.norm(u_axis)
        
        # Create second basis vector (orthogonal to both normal and u_axis)
        v_axis = np.cross(normal, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)
        
        # Project each vertex onto the 2D plane
        projected_vertices = []
        for vertex in vertices_3d:
            vertex_array = np.array(vertex)
            
            # Project onto u and v axes
            u_coord = np.dot(vertex_array, u_axis)
            v_coord = np.dot(vertex_array, v_axis)
            
            projected_vertices.append([u_coord, v_coord])
        
        return projected_vertices
    
    def get_common_projection_normals(self) -> dict:
        """
        Get a dictionary of common projection normals.
        
        Returns:
            Dictionary of named projection normals
        """
        return {
            'front': [0, 0, 1],      # Front view (XY plane)
            'back': [0, 0, -1],      # Back view
            'right': [1, 0, 0],      # Right view (YZ plane)
            'left': [-1, 0, 0],      # Left view
            'top': [0, 1, 0],        # Top view (XZ plane)
            'bottom': [0, -1, 0],    # Bottom view
            'isometric': [1, 1, 1],  # Isometric view
            'dimetric': [1, 1, 0.5], # Dimetric view
            'trimetric': [1, 0.8, 0.6] # Trimetric view
        }
    
    def select_projection_normal(self, view_name: str) -> List[float]:
        """
        Select a projection normal by name.
        
        Args:
            view_name: Name of the view ('front', 'isometric', etc.)
            
        Returns:
            Normalized projection normal vector
        """
        normals = self.get_common_projection_normals()
        
        if view_name.lower() in normals:
            normal = normals[view_name.lower()]
            return self.normalize_vector(normal).tolist()
        else:
            print(f"Unknown view name: {view_name}. "
                  f"Available: {list(normals.keys())}")
            return self.normalize_vector([1, 1, 1]).tolist()
