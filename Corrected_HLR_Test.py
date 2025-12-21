#!/usr/bin/env python3
"""
Corrected HLR Implementation - Fixes fundamental visibility issues
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
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Lin
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_ON

def load_step_file(filename):
    """Load STEP file and return the shape."""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    
    if status == IFSelect_RetDone:
        step_reader.TransferRoots()
        return step_reader.OneShape()
    else:
        raise Exception(f"Failed to load STEP file: {filename}")

def analyze_geometry_for_hlr(shape):
    """Analyze the geometry to understand the model structure for HLR."""
    print("Geometry Analysis for HLR:")
    print("=" * 40)
    
    # Get model bounds
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_Add
    
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    print(f"Model bounds:")
    print(f"  X: {xmin:.2f} to {xmax:.2f} (size: {xmax-xmin:.2f})")
    print(f"  Y: {ymin:.2f} to {ymax:.2f} (size: {ymax-ymin:.2f})")
    print(f"  Z: {zmin:.2f} to {zmax:.2f} (size: {zmax-zmin:.2f})")
    
    # Analyze faces and edges
    face_count = 0
    edge_count = 0
    
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face_count += 1
        face_explorer.Next()
    
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge_count += 1
        edge_explorer.Next()
    
    print(f"\nTopology:")
    print(f"  Faces: {face_count}")
    print(f"  Edges: {edge_count}")
    
    return {
        'bounds': (xmin, ymin, zmin, xmax, ymax, zmax),
        'face_count': face_count,
        'edge_count': edge_count
    }

def corrected_visibility_test(point, view_direction, shape, bounds):
    """Corrected visibility test using proper solid classification."""
    try:
        # Create classifier for the solid
        classifier = BRepClass3d_SolidClassifier()
        classifier.Load(shape)
        
        # Classify the point
        classifier.Perform(point, 1e-6)  # Small tolerance
        state = classifier.State()
        
        # If point is inside solid, it's definitely hidden
        if state == TopAbs_IN:
            return False
        
        # If point is outside, use ray casting to check occlusion
        if state == TopAbs_OUT:
            return ray_casting_visibility(point, view_direction, shape, bounds)
        
        # If point is on boundary, it's likely visible
        if state == TopAbs_ON:
            return True
        
        # Default to ray casting
        return ray_casting_visibility(point, view_direction, shape, bounds)
        
    except Exception:
        # Fallback to ray casting
        return ray_casting_visibility(point, view_direction, shape, bounds)

def ray_casting_visibility(point, view_direction, shape, bounds):
    """Improved ray casting for visibility testing."""
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = bounds
        
        # Create ray start point far from the model in view direction
        if view_direction == 'front':  # Looking from +Y towards -Y
            ray_start = gp_Pnt(point.X(), ymax + 100, point.Z())
            ray_dir = gp_Dir(0, -1, 0)
        elif view_direction == 'top':  # Looking from +Z towards -Z
            ray_start = gp_Pnt(point.X(), point.Y(), zmax + 100)
            ray_dir = gp_Dir(0, 0, -1)
        elif view_direction == 'side':  # Looking from +X towards -X
            ray_start = gp_Pnt(xmax + 100, point.Y(), point.Z())
            ray_dir = gp_Dir(-1, 0, 0)
        else:
            return True
        
        # Check if ray intersects solid before reaching the target point
        target_distance = ray_start.Distance(point)
        
        # Sample points along the ray
        num_samples = 50
        for i in range(1, num_samples):
            t = i / num_samples
            sample_distance = t * target_distance
            
            sample_point = gp_Pnt(
                ray_start.X() + ray_dir.X() * sample_distance,
                ray_start.Y() + ray_dir.Y() * sample_distance,
                ray_start.Z() + ray_dir.Z() * sample_distance
            )
            
            # Check if sample point is inside the solid
            classifier = BRepClass3d_SolidClassifier()
            classifier.Load(shape)
            classifier.Perform(sample_point, 1e-6)
            
            if classifier.State() == TopAbs_IN:
                # Found obstruction - point is hidden
                return False
        
        # No obstruction found - point is visible
        return True
        
    except Exception:
        return True

def corrected_edge_visibility(edge, view_direction, shape, bounds):
    """Determine edge visibility using corrected algorithm."""
    try:
        curve, first, last = BRep_Tool.Curve(edge)
        if curve is None:
            return True
        
        # Sample points along the edge
        num_samples = 11
        visible_count = 0
        
        for i in range(num_samples):
            t = first + (last - first) * i / (num_samples - 1)
            sample_point = curve.Value(t)
            
            if corrected_visibility_test(sample_point, view_direction, shape, bounds):
                visible_count += 1
        
        # Edge is visible if majority of points are visible
        visibility_ratio = visible_count / num_samples
        return visibility_ratio > 0.6
        
    except Exception:
        return True

def test_corrected_hlr():
    """Test the corrected HLR algorithm."""
    print("Testing Corrected HLR Algorithm")
    print("=" * 50)
    
    # Load the geometry
    shape = load_step_file("random_engineering_model.step")
    geometry_info = analyze_geometry_for_hlr(shape)
    bounds = geometry_info['bounds']
    
    print(f"\nTesting edge visibility classification...")
    
    # Test specific points for each view
    test_points = {
        'front': [
            gp_Pnt(47, 0, 28.5),    # Center of front face
            gp_Pnt(25, 45, 57),     # Cut area on top
            gp_Pnt(18.5, 14.5, 62), # Boss area
        ],
        'top': [
            gp_Pnt(47, 30.5, 57),   # Center of top face
            gp_Pnt(25, 45, 28.5),   # Cut area 
            gp_Pnt(18.5, 14.5, 57), # Boss base
        ],
        'side': [
            gp_Pnt(94, 30.5, 28.5), # Center of side face
            gp_Pnt(25, 45, 28.5),   # Cut area
            gp_Pnt(35, 14.5, 62),   # Boss side
        ]
    }
    
    for view, points in test_points.items():
        print(f"\n{view.upper()} VIEW:")
        for i, point in enumerate(points, 1):
            visible = corrected_visibility_test(point, view, shape, bounds)
            status = "VISIBLE" if visible else "HIDDEN"
            print(f"  Test point {i}: {status}")
    
    # Test edge classification
    print(f"\nEdge Classification Test:")
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    visible_edges = {'front': 0, 'top': 0, 'side': 0}
    hidden_edges = {'front': 0, 'top': 0, 'side': 0}
    
    edge_count = 0
    while edge_explorer.More() and edge_count < 10:  # Test first 10 edges
        edge = edge_explorer.Current()
        
        for view in ['front', 'top', 'side']:
            visible = corrected_edge_visibility(edge, view, shape, bounds)
            if visible:
                visible_edges[view] += 1
            else:
                hidden_edges[view] += 1
        
        edge_count += 1
        edge_explorer.Next()
    
    for view in ['front', 'top', 'side']:
        total = visible_edges[view] + hidden_edges[view]
        print(f"  {view}: {visible_edges[view]} visible, {hidden_edges[view]} hidden (of {total} tested)")

if __name__ == "__main__":
    test_corrected_hlr()
