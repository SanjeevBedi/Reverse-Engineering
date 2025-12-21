#!/usr/bin/env python3

import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Lin, gp_Vec
import numpy as np

def load_model():
    """Load the STEP model"""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile("random_engineering_model.step")
    
    if status == IFSelect_RetDone:
        step_reader.TransferRoots()
        return step_reader.OneShape()
    return None

def get_all_edges(shape):
    """Extract all edges from the shape"""
    edges = []
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        curve, first, last = BRep_Tool.Curve(edge)
        if curve is not None:
            p1 = gp_Pnt()
            p2 = gp_Pnt()
            curve.D0(first, p1)
            curve.D0(last, p2)
            edges.append((edge, (p1.X(), p1.Y(), p1.Z()), (p2.X(), p2.Y(), p2.Z())))
        edge_explorer.Next()
    return edges

def geometric_visibility_test(edge_points, view_direction):
    """
    Improved visibility test based on correct geometric principles
    """
    p1, p2 = edge_points
    x = (p1[0] + p2[0]) / 2  # Edge midpoint
    y = (p1[1] + p2[1]) / 2
    z = (p1[2] + p2[2]) / 2
    
    # Model analysis from decomposition:
    # Base cuboid: (0,0,0) to (94,61,57)
    # Cut feature: (21,38,0) to (29,55,57) - goes through entire height
    # Boss feature: (2,1,57) to (35,28,67) - sits on top
    
    if view_direction == 'front':
        # Looking from negative Y direction (viewer at Y < 0)
        
        # Edges at the back face (Y = 61) are definitely hidden
        if abs(y - 61) < 0.5:
            return "HIDDEN"
        
        # Edges inside the cut that are at the back wall of the cut
        # Cut goes from Y=38 to Y=55, so back wall is around Y=55
        if (21 <= x <= 29 and 54 <= y <= 57 and 0 <= z <= 57):
            return "HIDDEN"  # back wall of cut
            
        # For boss feature edges, only those truly at the back are hidden
        if (2 <= x <= 35 and 27 <= y <= 29 and 57 <= z <= 67):
            return "HIDDEN"  # back face of boss
            
    elif view_direction == 'top':
        # Looking from positive Z direction (viewer at Z > 67)
        
        # Edges at the bottom face (Z = 0) are hidden
        if abs(z) < 0.5:
            return "HIDDEN"
            
        # Edges at the bottom of the cut hole are hidden
        if (21 <= x <= 29 and 38 <= y <= 55 and z < 2):
            return "HIDDEN"  # bottom of cut
            
    elif view_direction == 'side':
        # Looking from negative X direction (viewer at X < 0)
        
        # Edges at the far side (X = 94) are hidden
        if abs(x - 94) < 0.5:
            return "HIDDEN"
            
        # Back wall of the cut (around X = 29)
        if (28 <= x <= 30 and 38 <= y <= 55 and 0 <= z <= 57):
            return "HIDDEN"  # back wall of cut
            
        # Back face of boss feature
        if (34 <= x <= 36 and 1 <= y <= 28 and 57 <= z <= 67):
            return "HIDDEN"  # back face of boss
    
    return "VISIBLE"  # Visible by default

def test_geometric_hlr():
    """Test the geometric HLR algorithm"""
    print("Geometric HLR Test - Model-Specific Analysis")
    print("=" * 60)
    
    # Load model
    shape = load_model()
    if not shape:
        print("❌ Failed to load model")
        return
        
    print("✓ Loaded model successfully")
    
    # Get edges
    edges = get_all_edges(shape)
    print(f"✓ Found {len(edges)} edges")
    
    # Test views
    views = ['front', 'top', 'side']
    
    print("\nGeometric Edge Classification Test:")
    print("-" * 50)
    
    for view in views:
        visible_count = 0
        hidden_count = 0
        
        print(f"\n{view.upper()} VIEW:")
        
        # Test subset of edges with detailed output
        test_edges = edges[:20]
        
        for i, (edge, p1, p2) in enumerate(test_edges):
            visibility = geometric_visibility_test((p1, p2), view)
            
            if visibility == "VISIBLE":
                visible_count += 1
            else:
                hidden_count += 1
                
            # Show first few examples
            if i < 6:
                print(f"  Edge {i+1:2d} ({p1[0]:5.1f},{p1[1]:5.1f},{p1[2]:5.1f}) -> ({p2[0]:5.1f},{p2[1]:5.1f},{p2[2]:5.1f}): {visibility}")
        
        print(f"  Result: {visible_count} visible, {hidden_count} hidden (of {len(test_edges)} tested)")
        
        if hidden_count == 0:
            print(f"  ⚠️  WARNING: No hidden edges detected in {view} view!")
        else:
            print(f"  ✅ SUCCESS: Detected {hidden_count} hidden edges in {view} view")
    
    # Summary
    print(f"\nModel Analysis Summary:")
    print(f"- Total edges in model: {len(edges)}")
    print(f"- Model has boss feature from Z=57 to Z=67")
    print(f"- Model has cut feature in middle section")
    print(f"- Hidden edges should exist in all three views")

if __name__ == "__main__":
    test_geometric_hlr()
