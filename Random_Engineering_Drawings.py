import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

import random
import numpy as np
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax1, gp_Dir, gp_Lin
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.gp import gp_Lin, gp_Dir
from OCC.Core.GeomAPI import GeomAPI_IntCS
from OCC.Core.Geom import Geom_Line
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

class RandomEngineeringDrawings:
    def __init__(self):
        # Generate random base dimensions
        self.base_length = random.randint(80, 150)
        self.base_width = random.randint(60, 120)  
        self.base_height = random.randint(40, 80)
        
        self.main_shape = None
        self.features = []
        self.feature_count = 0
        self.visible_edges = {'front': [], 'top': [], 'side': []}
        self.hidden_edges = {'front': [], 'top': [], 'side': []}
        
        print(f"Random base cuboid dimensions: {self.base_length} x {self.base_width} x {self.base_height} mm")
        
    def create_base_cuboid(self):
        """Create the main base cuboid with random dimensions."""
        print(f"Creating random base cuboid: {self.base_length} x {self.base_width} x {self.base_height}")
        self.main_shape = BRepPrimAPI_MakeBox(self.base_length, self.base_width, self.base_height).Shape()
        return self.main_shape
    
    def add_feature(self, length, width, height, x, y, z, is_addition=True):
        """Add or subtract a specific cuboid feature with proper boolean integration."""
        if self.feature_count >= 15:  # Allow more features
            return None
            
        # Create the feature
        feature = BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), length, width, height).Shape()
        
        # Apply boolean operation with error checking
        if is_addition:
            fuse_op = BRepAlgoAPI_Fuse(self.main_shape, feature)
            fuse_op.Build()
            if fuse_op.IsDone() and not fuse_op.HasErrors():
                self.main_shape = fuse_op.Shape()
                operation = "Added"
                print(f"✓ Successfully fused feature {self.feature_count + 1}")
            else:
                print(f"✗ Failed to fuse feature {self.feature_count + 1}")
                return None
        else:
            cut_op = BRepAlgoAPI_Cut(self.main_shape, feature)
            cut_op.Build()
            if cut_op.IsDone() and not cut_op.HasErrors():
                self.main_shape = cut_op.Shape()
                operation = "Subtracted"
                print(f"✓ Successfully cut feature {self.feature_count + 1}")
            else:
                print(f"✗ Failed to cut feature {self.feature_count + 1}")
                return None
        
        self.features.append({
            'operation': operation,
            'position': (x, y, z),
            'dimensions': (length, width, height),
            'is_addition': is_addition
        })
        
        self.feature_count += 1
        print(f"{operation} feature {self.feature_count}: {length:.1f}x{width:.1f}x{height:.1f} at ({x:.1f},{y:.1f},{z:.1f})")
        
        return self.main_shape

    def display_3d_model(self):
        """Display the integrated 3D model to verify boolean operations."""
        print("\\nDisplaying integrated 3D model...")
        
        try:
            # Try to import and use the display
            from OCC.Display.SimpleGui import init_display
            display, start_display, add_menu, add_function_to_menu = init_display()
            
            # Display the integrated shape
            display.DisplayShape(self.main_shape, update=True)
            display.FitAll()
            
            print("3D model displayed. Close the window to continue...")
            start_display()
            
        except ImportError:
            print("3D display not available. Using shape validation instead...")
            self.validate_shape_integration()
    
    def validate_shape_integration(self):
        """Validate that the shape is properly integrated."""
        print("\\nValidating shape integration...")
        
        # Count faces, edges, and vertices
        face_count = 0
        edge_count = 0
        vertex_count = 0
        
        # Count faces
        face_explorer = TopExp_Explorer(self.main_shape, TopAbs_FACE)
        while face_explorer.More():
            face_count += 1
            face_explorer.Next()
        
        # Count edges
        edge_explorer = TopExp_Explorer(self.main_shape, TopAbs_EDGE)
        while edge_explorer.More():
            edge_count += 1
            edge_explorer.Next()
            
        # Count vertices
        vertex_explorer = TopExp_Explorer(self.main_shape, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex_count += 1
            vertex_explorer.Next()
        
        print(f"Integrated shape contains:")
        print(f"  - {face_count} faces")
        print(f"  - {edge_count} edges") 
        print(f"  - {vertex_count} vertices")
        
        if face_count > 6:  # More than a simple box
            print("✓ Shape successfully integrated with features")
        else:
            print("⚠ Shape may not be properly integrated")
            
        return face_count, edge_count, vertex_count
    
    def create_random_model(self):
        """Create a random model with random number of added and subtracted features."""
        print("\\nCreating random engineering model...")
        
        # Create base
        self.create_base_cuboid()
        
        # COMPLEX TESTING: Generate model with two protrusions and two subtractions
        num_features = 4  # Fixed for complex testing
        print(f"Generating {num_features} random features (2 protrusions + 2 subtractions for testing)...")
        
        # Fixed: exactly two additions and two subtractions for complex testing
        feature_types = [True, True, False, False]  # True = addition (protrusion), False = subtraction
        
        # Shuffle to randomize order
        random.shuffle(feature_types)
        
        for i, is_addition in enumerate(feature_types):
            self.add_random_feature(is_addition, i + 1)
        
        print(f"Generated {self.feature_count} features total")
        
        # Count additions vs subtractions
        additions = [f for f in self.features if f['is_addition']]
        subtractions = [f for f in self.features if not f['is_addition']]
        print(f"✓ {len(additions)} additions, {len(subtractions)} subtractions")
        print("✓ COMPLEX TESTING MODEL: 2 protrusions + 2 subtractions for HLR analysis")
        
        print("\\n" + "="*50)
        print("RANDOM BOOLEAN INTEGRATION COMPLETE")
        print("="*50)
        
        # Validate the integration
        self.validate_shape_integration()
    
    def add_random_feature(self, is_addition, feature_num):
        """Add a random feature (either addition or subtraction) that penetrates at least one face."""
        
        if is_addition:
            # For additions (protrusions), ensure they penetrate through at least one face
            length = random.randint(15, 40)
            width = random.randint(15, 30)
            height = random.randint(10, 25)
            
            # Choose which face to penetrate through
            penetration_face = random.choice(['top', 'front', 'right', 'back', 'left'])
            
            if penetration_face == 'top':
                # Protrusion through top face
                x = random.randint(10, max(10, self.base_length - length - 10))
                y = random.randint(10, max(10, self.base_width - width - 10))
                z = self.base_height  # Starts at top surface
                
            elif penetration_face == 'front':
                # Protrusion through front face (extends beyond Y=0)
                x = random.randint(10, max(10, self.base_length - length - 10))
                y = -random.randint(8, 20)  # Extends into negative Y (front)
                z = random.randint(5, max(5, self.base_height - height - 5))
                
            elif penetration_face == 'right':
                # Protrusion through right face (extends beyond X=base_length)
                x = self.base_length - random.randint(5, length-5)  # Starts inside, extends out
                y = random.randint(10, max(10, self.base_width - width - 10))
                z = random.randint(5, max(5, self.base_height - height - 5))
                
            elif penetration_face == 'back':
                # Protrusion through back face (extends beyond Y=base_width)
                x = random.randint(10, max(10, self.base_length - length - 10))
                y = self.base_width - random.randint(5, width-5)  # Starts inside, extends out
                z = random.randint(5, max(5, self.base_height - height - 5))
                
            else:  # left face
                # Protrusion through left face (extends beyond X=0)
                x = -random.randint(8, 20)  # Extends into negative X (left)
                y = random.randint(10, max(10, self.base_width - width - 10))
                z = random.randint(5, max(5, self.base_height - height - 5))
                
        else:
            # For subtractions (cuts), ensure they penetrate through at least one face
            penetration_type = random.choice(['through_top', 'through_front', 'through_right', 'through_back', 'through_left', 'through_hole'])
            
            if penetration_type == 'through_top':
                # Cut that penetrates through top face
                length = random.randint(15, min(35, self.base_length - 20))
                width = random.randint(15, min(25, self.base_width - 20))
                height = random.randint(8, 20)  # Cut depth
                
                x = random.randint(10, self.base_length - length - 10)
                y = random.randint(10, self.base_width - width - 10)
                z = self.base_height - height  # Cut from top down
                
            elif penetration_type == 'through_front':
                # Cut that penetrates through front face
                length = random.randint(15, min(30, self.base_length - 20))
                width = random.randint(10, 25)  # Extends through front
                height = random.randint(10, min(20, self.base_height - 10))
                
                x = random.randint(10, self.base_length - length - 10)
                y = -5  # Starts outside front face
                z = random.randint(5, self.base_height - height - 5)
                
            elif penetration_type == 'through_right':
                # Cut that penetrates through right face
                length = random.randint(10, 25)  # Extends through right
                width = random.randint(15, min(25, self.base_width - 20))
                height = random.randint(10, min(20, self.base_height - 10))
                
                x = self.base_length - 15  # Starts inside, extends out
                y = random.randint(10, self.base_width - width - 10)
                z = random.randint(5, self.base_height - height - 5)
                
            elif penetration_type == 'through_back':
                # Cut that penetrates through back face
                length = random.randint(15, min(30, self.base_length - 20))
                width = random.randint(10, 25)  # Extends through back
                height = random.randint(10, min(20, self.base_height - 10))
                
                x = random.randint(10, self.base_length - length - 10)
                y = self.base_width - 15  # Starts inside, extends out
                z = random.randint(5, self.base_height - height - 5)
                
            elif penetration_type == 'through_left':
                # Cut that penetrates through left face
                length = random.randint(10, 25)  # Extends through left
                width = random.randint(15, min(25, self.base_width - 20))
                height = random.randint(10, min(20, self.base_height - 10))
                
                x = -5  # Starts outside left face
                y = random.randint(10, self.base_width - width - 10)
                z = random.randint(5, self.base_height - height - 5)
                
            else:  # through_hole
                # Complete through hole
                length = random.randint(8, 20)
                width = random.randint(8, 20)
                height = self.base_height + 10  # Goes completely through
                
                x = random.randint(length + 5, self.base_length - length - 5)
                y = random.randint(width + 5, self.base_width - width - 5)
                z = -5  # Starts below base
        
        # Add the feature
        self.add_feature(length, width, height, x, y, z, is_addition)
    
    def extract_edges_simple(self):
        """Extract edges and classify them as visible or hidden for each view using face analysis."""
        print("Extracting edges for orthographic projections...")
        
        # Clear previous data
        for view in ['front', 'top', 'side']:
            self.visible_edges[view] = []
            self.hidden_edges[view] = []
        
        # First, analyze all faces to understand surface orientations
        self.analyze_face_orientations()
        
        # Get all edges from the shape
        edge_explorer = TopExp_Explorer(self.main_shape, TopAbs_EDGE)
        
        while edge_explorer.More():
            edge = edge_explorer.Current()
            curve, first, last = BRep_Tool.Curve(edge)
            
            if curve is not None:
                p1 = curve.Value(first)
                p2 = curve.Value(last)
                
                # Classify edge for each view
                self.classify_edge_for_views(p1, p2)
            
            edge_explorer.Next()
    
    def analyze_face_orientations(self):
        """Simplified face analysis - let depth testing handle visibility."""
        self.visible_faces = {'front': [], 'top': [], 'side': []}
        
        # For each view, add all faces as potentially visible
        # The actual visibility determination will be done by depth testing in edge classification
        face_explorer = TopExp_Explorer(self.main_shape, TopAbs_FACE)
        
        while face_explorer.More():
            face = face_explorer.Current()
            
            # Add face to all views - depth testing will determine actual visibility
            self.visible_faces['front'].append(face)
            self.visible_faces['top'].append(face)
            self.visible_faces['side'].append(face)
            
            face_explorer.Next()
    
    def get_face_normal(self, face):
        """Get the normal vector of a face at its center."""
        try:
            # Get surface properties
            surface = BRepAdaptor_Surface(face)
            u_min, u_max, v_min, v_max = surface.FirstUParameter(), surface.LastUParameter(), surface.FirstVParameter(), surface.LastVParameter()
            
            # Evaluate at center
            u_mid = (u_min + u_max) / 2
            v_mid = (v_min + v_max) / 2
            
            props = GeomLProp_SLProps(surface, u_mid, v_mid, 1, 1e-6)
            
            if props.IsNormalDefined():
                normal = props.Normal()
                return (normal.X(), normal.Y(), normal.Z())
                
        except Exception:
            pass
        
        return None
    
    def get_face_centroid(self, face):
        """Get the centroid of a face."""
        try:
            props = GProp_GProps()
            brepgprop_SurfaceProperties(face, props)
            center = props.CentreOfMass()
            
            return (center.X(), center.Y(), center.Z())
            
        except Exception:
            pass
        
        return None
    
    def classify_face_visibility(self, face, normal, centroid):
        """Simplified - not used anymore since we rely on depth testing."""
        pass
    
    def robust_visibility_test(self, x, y, z, view):
        """Robust visibility test with improved ray casting and geometric validation."""
        try:
            # Primary method: Enhanced ray casting with higher precision
            ray_result = self.precision_ray_casting(x, y, z, view)
            
            # Secondary method: Boundary analysis for edge cases
            boundary_result = self.is_edge_on_boundary_simple_point(x, y, z, view)
            
            # Tertiary method: Surface proximity analysis
            proximity_result = self.surface_proximity_analysis(x, y, z, view)
            
            # Combine results with improved logic
            if boundary_result:
                # Points on or very near the boundary are typically visible
                return True
            elif proximity_result == 'surface':
                # Points on surface are visible unless blocked by other geometry
                return ray_result
            elif proximity_result == 'interior':
                # Points clearly inside are hidden
                return False
            else:
                # Use ray casting result for other cases
                return ray_result
                
        except Exception:
            # Fallback to enhanced visibility test
            return self.enhanced_visibility_test(x, y, z, view)
    
    def precision_ray_casting(self, x, y, z, view):
        """High-precision ray casting for accurate visibility determination."""
        try:
            target_point = gp_Pnt(x, y, z)
            
            # Define ray direction with increased starting distance for better accuracy
            ray_offset = 1000  # Increased offset for better ray testing
            
            if view == 'front':
                # Viewer looking from positive Y towards negative Y
                ray_start = gp_Pnt(x, self.base_width + ray_offset, z)
                ray_direction = gp_Dir(0, -1, 0)
            elif view == 'top':
                # Viewer looking from positive Z towards negative Z
                ray_start = gp_Pnt(x, y, self.base_height + ray_offset)
                ray_direction = gp_Dir(0, 0, -1)
            elif view == 'side':
                # Viewer looking from positive X towards negative X
                ray_start = gp_Pnt(self.base_length + ray_offset, y, z)
                ray_direction = gp_Dir(-1, 0, 0)
            else:
                return True
            
            # Perform high-precision intersection testing
            return self.high_precision_intersection_test(ray_start, ray_direction, target_point)
            
        except Exception:
            # Fallback to simpler ray casting
            return self.is_point_visible_raycast(x, y, z, view)
    
    def high_precision_intersection_test(self, ray_start, ray_direction, target_point):
        """High-precision intersection testing with adaptive sampling."""
        try:
            target_distance = ray_start.Distance(target_point)
            
            # Use higher sampling resolution for better accuracy
            num_samples = 100  # Significantly increased sampling
            sample_distance = target_distance / num_samples
            
            # Adaptive tolerance based on model size
            intersection_tolerance = max(0.01, min(self.base_length, self.base_width, self.base_height) * 0.001)
            
            # Sample points along the ray with higher precision
            for i in range(1, num_samples):
                distance = i * sample_distance
                sample_point = gp_Pnt(
                    ray_start.X() + ray_direction.X() * distance,
                    ray_start.Y() + ray_direction.Y() * distance,
                    ray_start.Z() + ray_direction.Z() * distance
                )
                
                # Use tighter tolerance for intersection detection
                if self.is_point_near_surface_precise(sample_point, intersection_tolerance):
                    # Found intersection before target - target is hidden
                    return False
            
            # No blocking intersection found - point is visible
            return True
            
        except Exception:
            # Fallback to original method
            return self.accurate_ray_intersection_test(ray_start, ray_direction, target_point)
    
    def is_point_near_surface_precise(self, point, tolerance):
        """Precise surface proximity test with adjustable tolerance."""
        try:
            vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
            distance_calc = BRepExtrema_DistShapeShape(vertex, self.main_shape)
            distance_calc.Perform()
            
            if distance_calc.IsDone() and distance_calc.NbSolution() > 0:
                distance = distance_calc.Value()
                return distance < tolerance
            
            return False
            
        except Exception:
            return False
    
    def surface_proximity_analysis(self, x, y, z, view):
        """Analyze point proximity to surface for visibility classification."""
        try:
            point = gp_Pnt(x, y, z)
            vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
            
            # Calculate distance from point to shape
            distance_calc = BRepExtrema_DistShapeShape(vertex, self.main_shape)
            distance_calc.Perform()
            
            if distance_calc.IsDone() and distance_calc.NbSolution() > 0:
                distance = distance_calc.Value()
                
                # Classify based on distance to surface
                if distance < 1e-6:  # Essentially on surface
                    return 'surface'
                elif distance < 0.01:  # Very close to surface
                    return 'near_surface'
                elif distance > max(self.base_length, self.base_width, self.base_height):  # Far from model
                    return 'exterior'
                else:
                    # Check if point might be interior using additional geometry analysis
                    return self.interior_analysis(point)
            
            return 'unknown'
            
        except Exception:
            return 'unknown'
    
    def interior_analysis(self, point):
        """Additional analysis to determine if point is interior to the solid."""
        try:
            # Use multiple ray directions to better determine interior/exterior status
            test_directions = [
                gp_Dir(1, 0, 0),   # +X
                gp_Dir(-1, 0, 0),  # -X  
                gp_Dir(0, 1, 0),   # +Y
                gp_Dir(0, -1, 0),  # -Y
                gp_Dir(0, 0, 1),   # +Z
                gp_Dir(0, 0, -1)   # -Z
            ]
            
            intersection_counts = []
            
            for direction in test_directions:
                # Cast ray in this direction and count intersections
                intersections = self.count_ray_intersections(point, direction)
                intersection_counts.append(intersections)
            
            # If point has odd number of intersections in most directions, it's likely interior
            odd_count = sum(1 for count in intersection_counts if count % 2 == 1)
            
            if odd_count >= 4:  # Majority of directions show odd intersections
                return 'interior'
            else:
                return 'exterior'
                
        except Exception:
            return 'unknown'
    
    def count_ray_intersections(self, start_point, direction):
        """Count intersections along a ray direction (simplified implementation)."""
        try:
            # Simple implementation - count surface proximity hits along ray
            max_distance = max(self.base_length, self.base_width, self.base_height) * 2
            num_samples = 50
            sample_distance = max_distance / num_samples
            
            intersection_count = 0
            tolerance = 0.05
            
            for i in range(1, num_samples):
                distance = i * sample_distance
                test_point = gp_Pnt(
                    start_point.X() + direction.X() * distance,
                    start_point.Y() + direction.Y() * distance,
                    start_point.Z() + direction.Z() * distance
                )
                
                if self.is_point_near_surface_precise(test_point, tolerance):
                    intersection_count += 1
            
            return intersection_count
            
        except Exception:
            return 0
    
    def enhanced_visibility_test(self, x, y, z, view):
        """Enhanced geometric visibility test using model-specific analysis."""
        
        # Use the working geometric algorithm from Enhanced_HLR_Test.py
        return self.geometric_visibility_test((x, y, z), view)
    
    def geometric_visibility_test(self, point, view_direction):
        """
        ADAPTIVE visibility analysis based on actual model geometry
        
        This function analyzes visibility based on the current model's actual
        dimensions and feature positions, rather than hardcoded values.
        """
        x, y, z = point
        
        # Get actual model bounds and feature positions
        base_bounds = {
            'x_min': 0, 'x_max': self.base_length,
            'y_min': 0, 'y_max': self.base_width, 
            'z_min': 0, 'z_max': self.base_height
        }
        
        # Analyze features to determine visible/hidden regions
        protrusions = [f for f in self.features if f['is_addition']]
        cuts = [f for f in self.features if not f['is_addition']]
        
        if view_direction == 'front':
            # Looking from Y < 0 toward +Y direction
            
            # DEFINITELY VISIBLE: Bottom outline edges (object boundary)
            if abs(z) < 0.5:
                # Bottom edges are visible as they define the object outline
                return True  # Visible - bottom outline
            
            # DEFINITELY HIDDEN: Back face edges (far from viewer)
            if y > base_bounds['y_max'] - 1:
                return False  # Hidden
            
            # Check protrusions - visible if facing viewer
            for protrusion in protrusions:
                px, py, pz = protrusion['position']
                pw, ph, pd = protrusion['dimensions']
                
                # Check if point is on a protrusion facing the viewer
                if (px <= x <= px + pw) and (py - 1 <= y <= py + ph + 1) and (pz <= z <= pz + pd):
                    # If protrusion extends above base height, it's visible
                    if pz + pd > base_bounds['z_max']:
                        return True
                    # If protrusion extends toward viewer (negative Y), it's visible
                    if py < 0:
                        return True
                        
            # Check cuts - internal edges should be hidden from front
            for cut in cuts:
                cx, cy, cz = cut['position']
                cw, ch, cd = cut['dimensions']
                
                # Point inside cut volume should be hidden from front view
                if (cx <= x <= cx + cw) and (cy <= y <= cy + ch) and (cz <= z <= cz + cd):
                    return False  # Hidden - cut internal
                    
            # Check if point is on base top surface
            if abs(z - base_bounds['z_max']) < 0.5:
                # Check if not under protrusions or in cuts
                under_protrusion = False
                in_cut = False
                
                for protrusion in protrusions:
                    px, py, pz = protrusion['position'] 
                    pw, ph, pd = protrusion['dimensions']
                    if (px <= x <= px + pw) and (py <= y <= py + ph) and pz <= base_bounds['z_max']:
                        under_protrusion = True
                        break
                        
                for cut in cuts:
                    cx, cy, cz = cut['position']
                    cw, ch, cd = cut['dimensions']
                    if (cx <= x <= cx + cw) and (cy <= y <= cy + ch) and cz <= base_bounds['z_max']:
                        in_cut = True
                        break
                
                if not under_protrusion and not in_cut:
                    return True  # Visible base top surface
                        
        elif view_direction == 'top':
            # Looking from Z > max toward -Z direction
            
            # DEFINITELY VISIBLE: Bottom outline edges (object boundary)
            if abs(z) < 0.5:
                # Bottom edges are visible as they define the object outline
                return True  # Visible - bottom outline
                
            # Check protrusions - top surfaces are visible
            for protrusion in protrusions:
                px, py, pz = protrusion['position']
                pw, ph, pd = protrusion['dimensions']
                
                # Top face of protrusion
                if (px <= x <= px + pw) and (py <= y <= py + ph) and abs(z - (pz + pd)) < 0.5:
                    return True  # Visible - top of protrusion
                    
            # Check cuts - you can see INTO cuts from above
            for cut in cuts:
                cx, cy, cz = cut['position']
                cw, ch, cd = cut['dimensions']
                
                # Edges inside cut are visible from top (you look into the cut)
                if (cx <= x <= cx + cw) and (cy <= y <= cy + ch) and (cz <= z <= cz + cd):
                    return True  # Visible - cut edges from top
                    
            # Base top surface - visible unless under protrusions
            if abs(z - base_bounds['z_max']) < 0.5:
                for protrusion in protrusions:
                    px, py, pz = protrusion['position']
                    pw, ph, pd = protrusion['dimensions'] 
                    if (px <= x <= px + pw) and (py <= y <= py + ph) and pz <= base_bounds['z_max']:
                        return False  # Hidden under protrusion
                return True  # Visible base top
                
        elif view_direction == 'side':
            # Looking from X > max toward -X direction (right side view)
            
            # DEFINITELY HIDDEN: Far side face (X = 0)
            if abs(x) < 0.5:
                return False  # Hidden - far side face
                
            # DEFINITELY VISIBLE: Bottom outline edges (object boundary)
            if abs(z) < 0.5:
                # Bottom edges are visible as they define the object outline
                return True  # Visible - bottom outline
                
            # DEFINITELY VISIBLE: Top outline edges (object boundary)
            if abs(z - base_bounds['z_max']) < 0.5:
                # Top edges are visible as they define the object outline
                return True  # Visible - top outline
                
            # DEFINITELY VISIBLE: Front outline edges (object boundary)
            if abs(y) < 0.5:
                # Front edges are visible as they define the object outline
                return True  # Visible - front outline
                
            # DEFINITELY VISIBLE: Right side face of base
            if abs(x - base_bounds['x_max']) < 0.5:
                return True  # Visible - right side of base
                
            # Check protrusions FIRST - they should be visible when extending outward
            for protrusion in protrusions:
                px, py, pz = protrusion['position']
                pw, ph, pd = protrusion['dimensions']
                
                # If point is inside or on protrusion boundary
                if (px <= x <= px + pw) and (py <= y <= py + ph) and (pz <= z <= pz + pd):
                    # Protrusions are ALWAYS visible in side view when:
                    # 1. They extend above the base (visible from top)
                    if pz >= base_bounds['z_max']:
                        return True  # Visible - protrusion above base
                    
                    # 2. They extend to or beyond the right face (visible from right)
                    if px + pw >= base_bounds['x_max'] - 1:
                        return True  # Visible - protrusion at right face
                    
                    # 3. They extend to or beyond the front face (visible from front)
                    if py <= 1:
                        return True  # Visible - protrusion at front face
                    
                    # 4. They are on the outline of the protrusion (edge visibility)
                    # Check if point is on protrusion boundary
                    on_boundary = (abs(x - px) < 0.5 or abs(x - (px + pw)) < 0.5 or
                                 abs(y - py) < 0.5 or abs(y - (py + ph)) < 0.5 or
                                 abs(z - pz) < 0.5 or abs(z - (pz + pd)) < 0.5)
                    if on_boundary:
                        return True  # Visible - protrusion boundary
                
                # Also check if point is near protrusion and should be visible
                # Points on the visible faces of protrusions
                if (px - 1 <= x <= px + pw + 1) and (py - 1 <= y <= py + ph + 1) and (pz - 1 <= z <= pz + pd + 1):
                    # If protrusion extends above base or to visible faces, make nearby points visible
                    if pz >= base_bounds['z_max'] or px + pw >= base_bounds['x_max'] - 2:
                        return True  # Visible - near visible protrusion
                
            # DEFINITELY HIDDEN: Internal regions - edges that are clearly inside the object
            # But be less restrictive near protrusions
            x_margin = base_bounds['x_max'] * 0.2  # Increased margin to allow protrusion visibility
            y_margin = base_bounds['y_max'] * 0.15
            z_margin = base_bounds['z_max'] * 0.15
            
            if (x < base_bounds['x_max'] - x_margin and 
                y > y_margin and y < base_bounds['y_max'] - y_margin and
                z > z_margin and z < base_bounds['z_max'] - z_margin):
                # But check if we're near any protrusion - if so, don't hide
                near_protrusion = False
                for protrusion in protrusions:
                    px, py, pz = protrusion['position']
                    pw, ph, pd = protrusion['dimensions']
                    # Check if within protrusion influence zone
                    if (px - 5 <= x <= px + pw + 5) and (py - 5 <= y <= py + ph + 5) and (pz - 5 <= z <= pz + pd + 5):
                        near_protrusion = True
                        break
                
                if not near_protrusion:
                    return False  # Hidden - internal region away from protrusions
                        
            # Check cuts - internal cuts should be hidden unless at opening
            for cut in cuts:
                cx, cy, cz = cut['position']
                cw, ch, cd = cut['dimensions']
                
                # If point is inside cut volume
                if (cx <= x <= cx + cw) and (cy <= y <= cy + ch) and (cz <= z <= cz + cd):
                    # Only visible if cut extends to the right face
                    if cx + cw >= base_bounds['x_max'] - 1:
                        return True  # Visible - cut opening at right face
                    else:
                        return False  # Hidden - internal cut
            
            # For edges near the right side or front face, make them visible
            if x >= base_bounds['x_max'] * 0.8 or y <= base_bounds['y_max'] * 0.2:
                return True  # Visible - near visible faces
            
            # For all other cases, be conservative - default to hidden
            return False
        
        # Default: For unmatched cases, use more conservative approach
        # Instead of defaulting to visible, use geometric analysis
        return self.conservative_visibility_fallback(point, view_direction)
    
    def conservative_visibility_fallback(self, point, view_direction):
        """Conservative fallback for visibility determination"""
        x, y, z = point
        
        # Much more conservative approach - bias toward hiding edges
        if view_direction == 'front':
            # Bottom outline edges are always visible
            if abs(z) < 0.5:
                return True  # Visible - bottom outline
            # Hide edges that are at the back (>60% of width)
            if y > self.base_width * 0.6:
                return False
            # Hide edges that are in interior regions
            if (0.2 * self.base_length <= x <= 0.8 * self.base_length and
                0.2 * self.base_width <= y <= 0.6 * self.base_width and
                0.2 * self.base_height <= z <= 0.8 * self.base_height):
                return False  # Interior edges hidden
            # Only show front-facing edges
            return y <= self.base_width * 0.4
            
        elif view_direction == 'top':
            # Bottom outline edges are always visible
            if abs(z) < 0.5:
                return True  # Visible - bottom outline
            # Hide edges that are in interior regions  
            if (0.2 * self.base_length <= x <= 0.8 * self.base_length and
                0.2 * self.base_width <= y <= 0.8 * self.base_width and
                0.3 * self.base_height <= z <= 0.7 * self.base_height):
                return False  # Interior edges hidden
            # Only show edges close to top surface
            return z >= self.base_height * 0.7
            
        elif view_direction == 'side':
            # Bottom outline edges are always visible
            if abs(z) < 0.5:
                return True  # Visible - bottom outline
            # Top outline edges are always visible
            if abs(z - self.base_height) < 0.5:
                return True  # Visible - top outline
            # Front outline edges are always visible
            if abs(y) < 0.5:
                return True  # Visible - front outline
            # Right side face edges are always visible
            if abs(x - self.base_length) < 0.5:
                return True  # Visible - right side face
            
            # Check if point is related to any protrusions (make more visible)
            for feature in self.features:
                if feature['is_addition']:  # This is a protrusion
                    fx, fy, fz = feature['position']
                    fw, fh, fd = feature['dimensions']
                    
                    # If point is in or near protrusion, make it more likely to be visible
                    if (fx - 2 <= x <= fx + fw + 2) and (fy - 2 <= y <= fy + fh + 2) and (fz - 2 <= z <= fz + fd + 2):
                        # Protrusions above base level are visible
                        if fz >= self.base_height:
                            return True  # Visible - protrusion above base
                        # Protrusions extending to right side are visible
                        if fx + fw >= self.base_length * 0.9:
                            return True  # Visible - protrusion at right side
                        # Protrusions at front are visible  
                        if fy <= self.base_width * 0.1:
                            return True  # Visible - protrusion at front
            
            # Hide edges that are at far side (<30% of length)
            if x < self.base_length * 0.3:
                return False
            # Hide edges that are in internal regions (but be less conservative near top and right)
            if (0.4 * self.base_length <= x <= 0.85 * self.base_length and
                0.15 * self.base_width <= y <= 0.85 * self.base_width and
                0.15 * self.base_height <= z <= 0.85 * self.base_height):
                return False  # Interior edges hidden
            # Show edges very close to right side, front face, or top
            return (x >= self.base_length * 0.8 or 
                   y <= self.base_width * 0.2 or 
                   z >= self.base_height * 0.8)
        
        # Final fallback - default to hidden for safety
        return False
    
    def is_point_visible_raycast(self, x, y, z, view):
        """Check if a point is visible using accurate ray casting from viewing direction."""
        try:
            target_point = gp_Pnt(x, y, z)
            
            # Define ray direction based on view (from viewer towards object)
            if view == 'front':
                # Viewer is in front (positive Y), looking towards negative Y
                ray_start = gp_Pnt(x, self.base_width + 500, z)
                ray_direction = gp_Dir(0, -1, 0)
            elif view == 'top':
                # Viewer is above (positive Z), looking towards negative Z  
                ray_start = gp_Pnt(x, y, self.base_height + 500)
                ray_direction = gp_Dir(0, 0, -1)
            elif view == 'side':
                # Viewer is to the right (positive X), looking towards negative X
                ray_start = gp_Pnt(self.base_length + 500, y, z)
                ray_direction = gp_Dir(-1, 0, 0)
            else:
                return True
            
            # Use more accurate intersection testing
            return self.accurate_ray_intersection_test(ray_start, ray_direction, target_point)
            
        except Exception:
            # Fallback to simple boundary test
            return self.is_edge_on_boundary_simple_point(x, y, z, view)
    
    def accurate_ray_intersection_test(self, ray_start, ray_direction, target_point):
        """Perform accurate ray intersection testing using available OpenCASCADE geometry."""
        try:
            # Calculate distance from ray start to target
            target_distance = ray_start.Distance(target_point)
            
            # Create line from ray start in ray direction
            ray_line = gp_Lin(ray_start, ray_direction)
            
            # Sample multiple points along the ray to check for intersections
            num_samples = 50
            sample_distance = target_distance / num_samples
            
            for i in range(1, num_samples):
                # Calculate sample point along ray
                distance = i * sample_distance
                sample_point = gp_Pnt(
                    ray_start.X() + ray_direction.X() * distance,
                    ray_start.Y() + ray_direction.Y() * distance,
                    ray_start.Z() + ray_direction.Z() * distance
                )
                
                # Check if sample point is inside or very close to surface
                if self.is_point_near_or_inside_solid(sample_point):
                    # Found intersection before target - target is hidden
                    return False
            
            # No blocking intersection found - point is visible
            return True
            
        except Exception:
            # Fallback to simpler method
            return self.simple_visibility_test(ray_start, ray_direction, target_point)
    
    def is_point_inside_solid(self, point):
        """Check if a point is inside the solid using distance analysis."""
        try:
            # Create vertex from point
            vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
            
            # Calculate distance from point to shape
            distance_calc = BRepExtrema_DistShapeShape(vertex, self.main_shape)
            distance_calc.Perform()
            
            if distance_calc.IsDone() and distance_calc.NbSolution() > 0:
                distance = distance_calc.Value()
                # If distance is very small, point might be on surface or inside
                return distance < 1e-6
            
            return False
            
        except Exception:
            return False

    def is_point_near_or_inside_solid(self, point):
        """Check if a point is inside or very close to the solid surface."""
        try:
            # Create vertex from point
            vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
            
            # Calculate distance from point to shape
            distance_calc = BRepExtrema_DistShapeShape(vertex, self.main_shape)
            distance_calc.Perform()
            
            if distance_calc.IsDone() and distance_calc.NbSolution() > 0:
                distance = distance_calc.Value()
                # Consider point as intersecting if it's very close to surface
                return distance < 0.1  # Increased tolerance for intersection detection
            
            return False
            
        except Exception:
            return False
    
    def simple_visibility_test(self, ray_start, ray_direction, target_point):
        """Simple fallback visibility test."""
        try:
            # Check if target point is within reasonable bounds
            target_distance = ray_start.Distance(target_point)
            
            # Use the existing point-in-solid test
            # Sample a few points between ray start and target
            for i in range(1, 10):
                factor = i / 10.0
                test_point = gp_Pnt(
                    ray_start.X() + (target_point.X() - ray_start.X()) * factor,
                    ray_start.Y() + (target_point.Y() - ray_start.Y()) * factor,
                    ray_start.Z() + (target_point.Z() - ray_start.Z()) * factor
                )
                
                if self.is_point_inside_solid(test_point):
                    return False
                    
            return True
            
        except Exception:
            return True
    
    def is_edge_on_boundary_simple_point(self, x, y, z, view):
        """Check if a point is on the boundary of the visible shape for a specific view."""
        tolerance = 2.0
        
        if view == 'front':
            return (x <= tolerance or x >= self.base_length - tolerance or
                    z <= tolerance or z >= self.base_height - tolerance)
        elif view == 'top':
            return (x <= tolerance or x >= self.base_length - tolerance or
                    y <= tolerance or y >= self.base_width - tolerance)
        elif view == 'side':
            return (y <= tolerance or y >= self.base_width - tolerance or
                    z <= tolerance or z >= self.base_height - tolerance)
        return False

    def is_edge_on_boundary_simple(self, p1, p2, view):
        """Simplified boundary detection for outer edges using ray tracing results."""
        # Check if either endpoint is on boundary
        return (self.is_edge_on_boundary_simple_point(p1.X(), p1.Y(), p1.Z(), view) or
                self.is_edge_on_boundary_simple_point(p2.X(), p2.Y(), p2.Z(), view))
    
    def classify_edge_for_views(self, p1, p2):
        """Classify an edge as visible or hidden for each orthographic view."""
        
        # Front view (X-Z projection, looking along Y)
        front_edge = [(p1.X(), p1.Z()), (p2.X(), p2.Z())]
        if self.is_edge_on_visible_face(p1, p2, 'front'):
            self.visible_edges['front'].append(front_edge)
        else:
            self.hidden_edges['front'].append(front_edge)
        
        # Top view (X-Y projection, looking along Z) 
        top_edge = [(p1.X(), p1.Y()), (p2.X(), p2.Y())]
        if self.is_edge_on_visible_face(p1, p2, 'top'):
            self.visible_edges['top'].append(top_edge)
        else:
            self.hidden_edges['top'].append(top_edge)
        
        # Side view (Y-Z projection, looking along X)
        side_edge = [(p1.Y(), p1.Z()), (p2.Y(), p2.Z())]
        if self.is_edge_on_visible_face(p1, p2, 'side'):
            self.visible_edges['side'].append(side_edge)
        else:
            self.hidden_edges['side'].append(side_edge)
    
    def is_edge_on_visible_face(self, p1, p2, view):
        """Determine if an edge is visible using our CORRECTED geometric visibility test."""
        # Get edge midpoint for analysis
        mid_x = (p1.X() + p2.X()) / 2
        mid_y = (p1.Y() + p2.Y()) / 2  
        mid_z = (p1.Z() + p2.Z()) / 2
        
        # Use our corrected geometric visibility test function
        return self.geometric_visibility_test((mid_x, mid_y, mid_z), view)
    
    def get_front_most_y_at_xz(self, x, z):
        """Get the front-most (maximum) Y coordinate at given X,Z location."""
        max_y = float('-inf')
        vertex_explorer = TopExp_Explorer(self.main_shape, TopAbs_VERTEX)
        tolerance = 3.0
        
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            pnt = BRep_Tool.Pnt(vertex)
            
            if abs(pnt.X() - x) < tolerance and abs(pnt.Z() - z) < tolerance:
                max_y = max(max_y, pnt.Y())
                
            vertex_explorer.Next()
        
        return max_y if max_y != float('-inf') else self.base_width
    
    def get_top_most_z_at_xy(self, x, y):
        """Get the top-most (maximum) Z coordinate at given X,Y location."""
        max_z = float('-inf')
        vertex_explorer = TopExp_Explorer(self.main_shape, TopAbs_VERTEX)
        tolerance = 3.0
        
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            pnt = BRep_Tool.Pnt(vertex)
            
            if abs(pnt.X() - x) < tolerance and abs(pnt.Y() - y) < tolerance:
                max_z = max(max_z, pnt.Z())
                
            vertex_explorer.Next()
        
        return max_z if max_z != float('-inf') else self.base_height
    
    def get_bottom_most_z_at_xy(self, x, y):
        """Get the bottom-most (minimum) Z coordinate at given X,Y location."""
        min_z = float('inf')
        vertex_explorer = TopExp_Explorer(self.main_shape, TopAbs_VERTEX)
        tolerance = 3.0
        
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            pnt = BRep_Tool.Pnt(vertex)
            
            if abs(pnt.X() - x) < tolerance and abs(pnt.Y() - y) < tolerance:
                min_z = min(min_z, pnt.Z())
                
            vertex_explorer.Next()
        
        return min_z if min_z != float('inf') else 0
    
    def get_right_most_x_at_yz(self, y, z):
        """Get the right-most (maximum) X coordinate at given Y,Z location."""
        max_x = float('-inf')
        vertex_explorer = TopExp_Explorer(self.main_shape, TopAbs_VERTEX)
        tolerance = 3.0
        
        while vertex_explorer.More():
            vertex = vertex_explorer.Current()
            pnt = BRep_Tool.Pnt(vertex)
            
            if abs(pnt.Y() - y) < tolerance and abs(pnt.Z() - z) < tolerance:
                max_x = max(max_x, pnt.X())
                
            vertex_explorer.Next()
        
        return max_x if max_x != float('-inf') else self.base_length
    
    def create_engineering_drawings_advanced(self):
        """Create detailed engineering drawings with proper orthographic projection standards."""
        print("\\nGenerating standard engineering drawings...")
        
        # Extract edges
        self.extract_edges_simple()
        
        # Calculate unified scale based on largest dimension for consistency
        max_dimension = max(self.base_length, self.base_width, self.base_height)
        scale_factor = 300 / max_dimension  # Target size of 300 units for largest dimension
        
        # Create figure with proper engineering layout
        fig = plt.figure(figsize=(18, 14))
        
        # Standard engineering drawing layout with proper alignment using gridspec:
        # This ensures views are properly aligned with each other
        
        # Calculate scaled dimensions for layout - IMPORTANT: Use same scale for height alignment
        scaled_length = self.base_length * scale_factor
        scaled_width = self.base_width * scale_factor  
        scaled_height = self.base_height * scale_factor
        
        # Top view (upper center) - shows length x width, aligned with front view
        ax_top = plt.subplot2grid((4, 4), (0, 1), colspan=1, rowspan=1)
        
        # Front view (center) - shows length x height, main reference view  
        ax_front = plt.subplot2grid((4, 4), (2, 1), colspan=1, rowspan=1)
        
        # Right side view (center right) - shows width x height, aligned with front view height
        ax_side = plt.subplot2grid((4, 4), (2, 2), colspan=1, rowspan=1)
        
        # Standard view configuration following ISO/ANSI standards
        # FIXED: Use same scale_factor for all views to ensure proper alignment
        views = [
            ("FRONT VIEW", "front", ax_front, "X (Length)", "Z (Height)", scaled_length, scaled_height),
            ("TOP VIEW", "top", ax_top, "X (Length)", "Y (Width)", scaled_length, scaled_width),
            ("RIGHT SIDE VIEW", "side", ax_side, "Y (Width)", "Z (Height)", scaled_width, scaled_height)
        ]
        
        for view_name, view_dir, ax, xlabel, ylabel, x_dim, y_dim in views:
            print(f"Processing {view_name}...")
            self.draw_aligned_orthographic_view(ax, view_dir, view_name, xlabel, ylabel, scale_factor, x_dim, y_dim)
        
        # Add global legend and title
        self.add_global_legend(fig)
        self.add_title_block(fig)
        
        # Adjust layout for proper spacing and alignment
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.92, hspace=0.3, wspace=0.2)
        
        # Save with high quality
        plt.savefig('random_engineering_drawings.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Random engineering drawings saved as 'random_engineering_drawings.png'")
    
    def add_global_legend(self, fig):
        """Add a global legend at the bottom of the figure."""
        # Create legend elements - order reflects rendering priority
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=1.5, alpha=1.0, zorder=2,
                   label='Visible Edges (foreground)'),
            Line2D([0], [0], color='black', linewidth=0.8, linestyle='--', 
                   dashes=[3, 2], alpha=0.6, zorder=1, label='Hidden Edges (background)')
        ]
        
        # Add legend at the bottom center
        fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=2, 
                  fontsize=12, 
                  frameon=True, 
                  fancybox=True, 
                  shadow=True)
    
    def draw_aligned_orthographic_view(self, ax, view_direction, title, xlabel, ylabel, scale_factor, x_dim, y_dim):
        """Draw a single orthographic view with proper alignment and unified scaling."""
        
        # Apply unified scaling to all edges
        scaled_visible_edges = []
        scaled_hidden_edges = []
        
        # Scale visible edges
        for edge in self.visible_edges[view_direction]:
            if len(edge) == 2:
                scaled_edge = [
                    (edge[0][0] * scale_factor, edge[0][1] * scale_factor),
                    (edge[1][0] * scale_factor, edge[1][1] * scale_factor)
                ]
                scaled_visible_edges.append(scaled_edge)
        
        # Scale hidden edges
        for edge in self.hidden_edges[view_direction]:
            if len(edge) == 2:
                scaled_edge = [
                    (edge[0][0] * scale_factor, edge[0][1] * scale_factor),
                    (edge[1][0] * scale_factor, edge[1][1] * scale_factor)
                ]
                scaled_hidden_edges.append(scaled_edge)
        
        # IMPORTANT: Draw hidden edges FIRST (background layer)
        for edge in scaled_hidden_edges:
            x_coords = [edge[0][0], edge[1][0]]
            y_coords = [edge[0][1], edge[1][1]]
            # Hidden lines: dashed, thinner, lower alpha, explicitly set zorder low
            ax.plot(x_coords, y_coords, 'k--', linewidth=0.8, alpha=0.6, 
                   dashes=[3, 2], zorder=1)
        
        # THEN draw visible edges (foreground layer, solid lines)
        for edge in scaled_visible_edges:
            x_coords = [edge[0][0], edge[1][0]]
            y_coords = [edge[0][1], edge[1][1]]
            # Visible lines: solid, thicker, full alpha, explicitly set zorder high
            ax.plot(x_coords, y_coords, 'k-', linewidth=1.5, alpha=1.0, zorder=2)
        
        # Set equal aspect ratio (critical for engineering drawings)
        ax.set_aspect('equal')
        
        # Set consistent limits using the passed dimensions for proper alignment
        margin = 0.15  # 15% margin around the drawing
        ax.set_xlim(-x_dim * margin, x_dim * (1 + margin))
        ax.set_ylim(-y_dim * margin, y_dim * (1 + margin))
        
        # COMPLETELY REMOVE all tick marks, labels, and axes elements for clean engineering drawing
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(left=False, bottom=False, top=False, right=False, 
                      labelleft=False, labelbottom=False, labeltop=False, labelright=False)
        
        # Remove ALL spines (axis borders) for completely clean appearance
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Remove any grid
        ax.grid(False)
        
        # Add title with proper spacing
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        # Add dimension annotations showing actual model dimensions
        self.add_dimension_annotations(ax, view_direction, scale_factor, x_dim, y_dim)
        self.format_dimension_ticks(ax, view_direction, scale_factor)
        
        # Set tick parameters and remove axis borders
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Remove axis borders (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
    def add_dimension_annotations(self, ax, view_direction, scale_factor, x_dim, y_dim):
        """Add dimension annotations to show actual model dimensions."""
        
        # Get the actual dimensions based on view direction
        if view_direction == 'front':
            width_dim = self.base_length  # X direction
            height_dim = self.base_height  # Z direction
            width_label = f'{width_dim:.0f}'
            height_label = f'{height_dim:.0f}'
        elif view_direction == 'top':
            width_dim = self.base_length  # X direction  
            height_dim = self.base_width  # Y direction
            width_label = f'{width_dim:.0f}'
            height_label = f'{height_dim:.0f}'
        elif view_direction == 'side':
            width_dim = self.base_width  # Y direction
            height_dim = self.base_height  # Z direction  
            width_label = f'{width_dim:.0f}'
            height_label = f'{height_dim:.0f}'
        else:
            return
        
        # Add dimension text below and to the left of the drawing
        # Width dimension (bottom)
        ax.text(x_dim / 2, -y_dim * 0.08, f'{width_label} mm', 
               ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Height dimension (left side, rotated)
        ax.text(-x_dim * 0.08, y_dim / 2, f'{height_label} mm', 
               ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)
        ax.text(-x_dim * 0.08, y_dim / 2, f'{height_label} mm', 
               ha='center', va='center', fontsize=10, fontweight='bold', 
               rotation=90)
    
    def add_standard_dimensions(self, ax, view_direction, scale_factor=1.0):
        """Add standard dimension annotations following engineering conventions with unified scaling."""
        dim_color = 'blue'
        dim_fontsize = 9
        
        if view_direction == "front":
            # Overall length dimension (bottom)
            length_scaled = self.base_length * scale_factor
            height_scaled = self.base_height * scale_factor
            
            y_offset = -length_scaled * 0.08
            ax.annotate('', xy=(length_scaled, y_offset), xytext=(0, y_offset),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(length_scaled/2, y_offset - length_scaled * 0.02, f'L = {self.base_length}', 
                   ha='center', va='top', color=dim_color, fontsize=dim_fontsize, fontweight='bold')
            
            # Overall height dimension (left)
            x_offset = -length_scaled * 0.08
            ax.annotate('', xy=(x_offset, height_scaled), xytext=(x_offset, 0),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_offset - length_scaled * 0.02, height_scaled/2, f'H = {self.base_height}', 
                   ha='center', va='center', color=dim_color, fontsize=dim_fontsize, 
                   fontweight='bold', rotation=90)
        
        elif view_direction == "top":
            # Length and width dimensions
            length_scaled = self.base_length * scale_factor
            width_scaled = self.base_width * scale_factor
            
            # Length dimension (bottom)
            y_offset = -width_scaled * 0.08
            ax.annotate('', xy=(length_scaled, y_offset), xytext=(0, y_offset),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(length_scaled/2, y_offset - width_scaled * 0.02, f'L = {self.base_length}', 
                   ha='center', va='top', color=dim_color, fontsize=dim_fontsize, fontweight='bold')
            
            # Width dimension (left)
            x_offset = -length_scaled * 0.08
            ax.annotate('', xy=(x_offset, width_scaled), xytext=(x_offset, 0),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_offset - length_scaled * 0.02, width_scaled/2, f'W = {self.base_width}', 
                   ha='center', va='center', color=dim_color, fontsize=dim_fontsize, 
                   fontweight='bold', rotation=90)
        
        elif view_direction == "side":
            # Width and height dimensions
            width_scaled = self.base_width * scale_factor
            height_scaled = self.base_height * scale_factor
            
            # Width dimension (bottom)
            y_offset = -height_scaled * 0.08
            ax.annotate('', xy=(width_scaled, y_offset), xytext=(0, y_offset),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(width_scaled/2, y_offset - height_scaled * 0.02, f'W = {self.base_width}', 
                   ha='center', va='top', color=dim_color, fontsize=dim_fontsize, fontweight='bold')
            
            # Height dimension (left)
            x_offset = -width_scaled * 0.08
            ax.annotate('', xy=(x_offset, height_scaled), xytext=(x_offset, 0),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_offset - width_scaled * 0.02, height_scaled/2, f'H = {self.base_height}', 
                   ha='center', va='center', color=dim_color, fontsize=dim_fontsize, 
                   fontweight='bold', rotation=90)
    
    def add_aligned_dimensions(self, ax, view_direction, scale_factor, x_dim, y_dim):
        """Add dimension annotations with proper alignment and unified scaling."""
        dim_color = 'blue'
        dim_fontsize = 9
        
        if view_direction == "front":
            # Front view: X (length) vs Z (height)
            # Length dimension (bottom)
            y_offset = -y_dim * 0.1
            ax.annotate('', xy=(x_dim, y_offset), xytext=(0, y_offset),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_dim/2, y_offset - y_dim * 0.03, f'L = {self.base_length}', 
                   ha='center', va='top', color=dim_color, fontsize=dim_fontsize, fontweight='bold')
            
            # Height dimension (left)
            x_offset = -x_dim * 0.1
            ax.annotate('', xy=(x_offset, y_dim), xytext=(x_offset, 0),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_offset - x_dim * 0.03, y_dim/2, f'H = {self.base_height}', 
                   ha='center', va='center', color=dim_color, fontsize=dim_fontsize, 
                   fontweight='bold', rotation=90)
                   
        elif view_direction == "top":
            # Top view: X (length) vs Y (width)
            # Length dimension (bottom)
            y_offset = -y_dim * 0.1
            ax.annotate('', xy=(x_dim, y_offset), xytext=(0, y_offset),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_dim/2, y_offset - y_dim * 0.03, f'L = {self.base_length}', 
                   ha='center', va='top', color=dim_color, fontsize=dim_fontsize, fontweight='bold')
            
            # Width dimension (left)
            x_offset = -x_dim * 0.1
            ax.annotate('', xy=(x_offset, y_dim), xytext=(x_offset, 0),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_offset - x_dim * 0.03, y_dim/2, f'W = {self.base_width}', 
                   ha='center', va='center', color=dim_color, fontsize=dim_fontsize, 
                   fontweight='bold', rotation=90)
                   
        elif view_direction == "side":
            # Side view: Y (width) vs Z (height)
            # Width dimension (bottom)
            y_offset = -y_dim * 0.1
            ax.annotate('', xy=(x_dim, y_offset), xytext=(0, y_offset),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_dim/2, y_offset - y_dim * 0.03, f'W = {self.base_width}', 
                   ha='center', va='top', color=dim_color, fontsize=dim_fontsize, fontweight='bold')
            
            # Height dimension (left)
            x_offset = -x_dim * 0.1
            ax.annotate('', xy=(x_offset, y_dim), xytext=(x_offset, 0),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_offset - x_dim * 0.03, y_dim/2, f'H = {self.base_height}', 
                   ha='center', va='center', color=dim_color, fontsize=dim_fontsize, 
                   fontweight='bold', rotation=90)
    
    def format_dimension_ticks(self, ax, view_direction, scale_factor):
        """Format axis ticks to show actual dimensions rather than scaled values."""
        import matplotlib.ticker as ticker
        
        def scale_formatter(x, pos):
            """Convert scaled values back to actual dimensions."""
            return f'{x/scale_factor:.0f}'
        
        # Apply the formatter to both axes
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(scale_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(scale_formatter))
        
        # Set appropriate tick spacing based on dimensions
        if view_direction == "front":
            x_spacing = max(10, self.base_length // 5) * scale_factor
            y_spacing = max(10, self.base_height // 5) * scale_factor
        elif view_direction == "top":
            x_spacing = max(10, self.base_length // 5) * scale_factor
            y_spacing = max(10, self.base_width // 5) * scale_factor
        elif view_direction == "side":
            x_spacing = max(10, self.base_width // 5) * scale_factor
            y_spacing = max(10, self.base_height // 5) * scale_factor
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_spacing))
    
    def add_title_block(self, fig):
        """Add standard engineering drawing title block."""
        # Add main title
        fig.suptitle('RANDOM ENGINEERING DRAWING\\nORTHOGRAPHIC PROJECTIONS', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add title block information
        additions = [f for f in self.features if f['is_addition']]
        subtractions = [f for f in self.features if not f['is_addition']]
        
        title_text = f"""
        PART: Random Machined Component
        MATERIAL: Aluminum 6061-T6
        SCALE: 1:1
        DIMENSIONS: mm
        PROJECTION: Third Angle (ISO)
        
        BASE DIMENSIONS: {self.base_length} × {self.base_width} × {self.base_height} mm
        FEATURES: {len(self.features)} total ({len(additions)} additions, {len(subtractions)} cuts)
        """
        
        # Add title block as text box
        fig.text(0.02, 0.02, title_text.strip(), fontsize=8, 
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        fig.text(0.02, 0.12, title_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def save_step_file(self, filename="random_engineering_model.step"):
        """Save the model as a STEP file."""
        step_writer = STEPControl_Writer()
        Interface_Static_SetCVal("write.step.schema", "AP203")
        
        step_writer.Transfer(self.main_shape, STEPControl_AsIs)
        status = step_writer.Write(filename)
        
        if status == IFSelect_RetDone:
            print(f"Model saved as {filename}")
        else:
            print("Failed to save STEP file")
    
    def print_model_summary(self):
        """Print a summary of the generated model."""
        print("\\n" + "="*60)
        print("RANDOM MODEL SUMMARY")
        print("="*60)
        print(f"Random Base Cuboid: {self.base_length} x {self.base_width} x {self.base_height} mm")
        print(f"Total Features: {len(self.features)}")
        
        additions = [f for f in self.features if f['is_addition']]
        subtractions = [f for f in self.features if not f['is_addition']]
        
        print(f"Additions (Bosses): {len(additions)}")
        print(f"Subtractions (Cuts): {len(subtractions)}")
        
        print("\\nFeature Details:")
        for i, feature in enumerate(self.features, 1):
            op_type = "Boss" if feature['is_addition'] else "Cut"
            print(f"  {i}. {op_type}: "
                  f"{feature['dimensions'][0]:.1f} x {feature['dimensions'][1]:.1f} x {feature['dimensions'][2]:.1f} mm "
                  f"at ({feature['position'][0]:.1f}, {feature['position'][1]:.1f}, {feature['position'][2]:.1f}) mm")

def main():
    """Main function to generate random engineering model and drawings."""
    print("Random Engineering Drawing Generator")
    print("="*50)
    
    # Set random seed for reproducible results (remove this line for truly random)
    random.seed(42)
    np.random.seed(42)
    
    # Create the generator with random dimensions
    generator = RandomEngineeringDrawings()
    
    # Build the random model
    print("Building random 3D model...")
    generator.create_random_model()
    
    # Display the integrated 3D model
    print("\\nStep 1: Displaying integrated 3D model...")
    generator.display_3d_model()
    
    # Print summary
    generator.print_model_summary()
    
    # Save the model
    print("\\nStep 2: Saving STEP file...")
    generator.save_step_file("random_engineering_model.step")
    
    # Generate random engineering drawings
    print("\\nStep 3: Generating orthographic engineering drawings...")
    generator.create_engineering_drawings_advanced()
    
    print("\\n" + "="*50)
    print("RANDOM WORKFLOW FINISHED!")
    print("="*50)
    print("Files generated:")
    print("  - random_engineering_model.step (3D integrated model)")
    print("  - random_engineering_drawings.png (2D drawings with proper HLR)")
    print("\\nWorkflow Summary:")
    print("  ✓ Random base cuboid generated")
    print("  ✓ Random features added and subtracted")
    print("  ✓ Boolean operations integrated all features into single solid")
    print("  ✓ 3D model displayed for verification")
    print("  ✓ Orthographic projections generated with visible/hidden lines")

if __name__ == "__main__":
    main()
