#!/usr/bin/env python3
"""
Fixed HLR Implementation - Proper hidden line removal using solid classification
"""

import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_ON
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
import random

def test_fixed_hlr():
    """Test the fixed HLR algorithm."""
    print("Fixed HLR Test")
    print("=" * 50)
    
    # Load the STEP file
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile("random_engineering_model.step")
    
    if status != IFSelect_RetDone:
        print("❌ Failed to read STEP file")
        return
    
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    print("✓ Loaded model successfully")
    
    # Test edge visibility with fixed algorithm
    test_edges_with_fixed_hlr(shape)

def test_edges_with_fixed_hlr(shape):
    """Test edge visibility using proper solid classification."""
    print("\nFixed Edge Classification Test:")
    print("-" * 40)
    
    views = {
        'front': gp_Dir(0, -1, 0),   # Looking along -Y
        'top': gp_Dir(0, 0, -1),     # Looking along -Z  
        'side': gp_Dir(-1, 0, 0)     # Looking along -X
    }
    
    # Get edges to test
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    test_edges = []
    
    # Collect first 10 edges for testing
    count = 0
    while edge_explorer.More() and count < 10:
        edge = edge_explorer.Current()
        curve, first, last = BRep_Tool.Curve(edge)
        if curve is not None:
            p1 = curve.Value(first)
            p2 = curve.Value(last)
            test_edges.append((edge, p1, p2))
            count += 1
        edge_explorer.Next()
    
    print(f"Testing {len(test_edges)} edges...")
    
    for view_name, view_dir in views.items():
        visible_count = 0
        hidden_count = 0
        
        for i, (edge, p1, p2) in enumerate(test_edges):
            is_visible = fixed_edge_visibility_test(shape, p1, p2, view_dir)
            
            if is_visible:
                visible_count += 1
                status = "VISIBLE"
            else:
                hidden_count += 1
                status = "HIDDEN"
            
            # Print first few for debugging
            if i < 3:
                print(f"  Edge {i+1} ({p1.X():.1f},{p1.Y():.1f},{p1.Z():.1f}) -> ({p2.X():.1f},{p2.Y():.1f},{p2.Z():.1f}): {status}")
        
        print(f"  {view_name}: {visible_count} visible, {hidden_count} hidden (of {len(test_edges)} tested)")

def fixed_edge_visibility_test(shape, p1, p2, view_direction):
    """Fixed visibility test using proper geometry analysis."""
    
    # Sample multiple points along the edge
    num_samples = 5
    hidden_samples = 0
    
    for i in range(num_samples):
        t = i / max(num_samples - 1, 1)
        sample_point = gp_Pnt(
            p1.X() + t * (p2.X() - p1.X()),
            p1.Y() + t * (p2.Y() - p1.Y()), 
            p1.Z() + t * (p2.Z() - p1.Z())
        )
        
        # Test if this sample point is hidden
        if is_point_hidden_by_solid(shape, sample_point, view_direction):
            hidden_samples += 1
    
    # Edge is hidden if majority of samples are hidden
    return hidden_samples < (num_samples / 2)

def is_point_hidden_by_solid(shape, point, view_direction):
    """Check if a point is hidden by casting a ray in the view direction."""
    
    # Cast ray from point in viewing direction
    ray_length = 1000.0  # Large distance
    
    # Ray start point (moved back along view direction)
    ray_start = gp_Pnt(
        point.X() - view_direction.X() * ray_length,
        point.Y() - view_direction.Y() * ray_length,
        point.Z() - view_direction.Z() * ray_length
    )
    
    # Sample points along the ray between ray_start and point
    num_ray_samples = 20
    for i in range(1, num_ray_samples):
        t = i / num_ray_samples
        ray_point = gp_Pnt(
            ray_start.X() + t * (point.X() - ray_start.X()),
            ray_start.Y() + t * (point.Y() - ray_start.Y()),
            ray_start.Z() + t * (point.Z() - ray_start.Z())
        )
        
        # Check if ray point is inside solid using proper classification
        if is_point_inside_solid_proper(shape, ray_point):
            # Found solid material between viewer and target point
            return True
    
    return False

def is_point_inside_solid_proper(shape, point):
    """Properly determine if a point is inside the solid using BRepClass3d."""
    try:
        # Use OpenCASCADE's solid classifier
        classifier = BRepClass3d_SolidClassifier(shape)
        classifier.Perform(point, 1e-6)  # Small tolerance
        
        state = classifier.State()
        
        # Point is inside if state is IN
        return state == TopAbs_IN
        
    except Exception as e:
        # Fallback to distance-based method
        return is_point_inside_distance_method(shape, point)

def is_point_inside_distance_method(shape, point):
    """Fallback method using distance calculation."""
    try:
        vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
        distance_calc = BRepExtrema_DistShapeShape(vertex, shape)
        distance_calc.Perform()
        
        if distance_calc.IsDone() and distance_calc.NbSolution() > 0:
            distance = distance_calc.Value()
            
            # Very small distance suggests point is on or very close to surface
            # Additional logic needed to determine inside vs outside
            if distance < 1e-6:
                return False  # On surface, consider not inside
            
            # For points not on surface, use a more sophisticated test
            return test_point_inside_by_ray_counting(shape, point)
        
        return False
        
    except Exception:
        return False

def test_point_inside_by_ray_counting(shape, point):
    """Test if point is inside by counting ray intersections."""
    try:
        # Cast rays in multiple directions and count intersections
        directions = [
            gp_Dir(1, 0, 0),
            gp_Dir(0, 1, 0), 
            gp_Dir(0, 0, 1),
            gp_Dir(-1, 0, 0),
            gp_Dir(0, -1, 0),
            gp_Dir(0, 0, -1)
        ]
        
        inside_count = 0
        
        for direction in directions:
            intersections = count_ray_intersections_with_shape(shape, point, direction)
            # Odd number of intersections means point is inside
            if intersections % 2 == 1:
                inside_count += 1
        
        # Point is inside if majority of rays indicate inside
        return inside_count > len(directions) / 2
        
    except Exception:
        return False

def count_ray_intersections_with_shape(shape, point, direction):
    """Count intersections of a ray with the shape (simplified)."""
    try:
        # Sample along ray and count surface proximity hits
        ray_length = 200.0
        num_samples = 50
        intersection_count = 0
        
        for i in range(1, num_samples):
            t = i * ray_length / num_samples
            ray_point = gp_Pnt(
                point.X() + direction.X() * t,
                point.Y() + direction.Y() * t,
                point.Z() + direction.Z() * t
            )
            
            # Check if ray point is very close to surface
            vertex = BRepBuilderAPI_MakeVertex(ray_point).Vertex()
            distance_calc = BRepExtrema_DistShapeShape(vertex, shape)
            distance_calc.Perform()
            
            if distance_calc.IsDone() and distance_calc.NbSolution() > 0:
                distance = distance_calc.Value()
                if distance < 0.1:  # Close to surface
                    intersection_count += 1
        
        return intersection_count
        
    except Exception:
        return 0

if __name__ == "__main__":
    test_fixed_hlr()
