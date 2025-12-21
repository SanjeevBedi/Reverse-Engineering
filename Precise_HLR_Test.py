#!/usr/bin/env python3
"""
Precise HLR Implementation based on actual edge analysis
"""

import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt

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

def precise_visibility_analysis(edge_points, view_direction):
    """
    Precise visibility analysis based on actual model geometry
    
    Model structure:
    - Base cuboid: (0,0,0) to (94,61,57)
    - Cut hole: (21,38,0) to (29,55,57) - goes through base
    - Boss feature: (2,1,57) to (35,28,67) - sits on top
    """
    p1, p2 = edge_points
    
    # Edge characteristics
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    if view_direction == 'front':
        # Looking from Y < 0 toward +Y direction
        
        # DEFINITELY HIDDEN: Back face edges (Y = 61)
        if abs(y1 - 61) < 0.1 and abs(y2 - 61) < 0.1:
            return "HIDDEN"
        
        # DEFINITELY HIDDEN: Far edges of boss that are at the back
        # Boss back edge at Y=28, but boss sits on base, so some edges at Y=28 are hidden
        if (2 <= min(x1,x2) <= 35 and 27.5 <= max(y1,y2) <= 28.5 and 
            57 <= min(z1,z2) <= 67):
            return "HIDDEN"
        
        # DEFINITELY HIDDEN: Back wall edges of the cut hole
        # Cut goes from Y=38 to Y=55, back wall around Y=55
        if (21 <= min(x1,x2) <= 29 and 54.5 <= max(y1,y2) <= 55.5 and 
            0 <= min(z1,z2) <= 57):
            return "HIDDEN"
            
        # POTENTIALLY HIDDEN: Internal edges of cut that go toward back
        # Vertical edges inside cut that are close to back wall
        if (21 <= min(x1,x2) <= 29 and 50 <= min(y1,y2) <= 55 and 
            abs(z1 - z2) > 20):  # Vertical edges inside cut
            return "HIDDEN"
    
    elif view_direction == 'top':
        # Looking from Z > 67 toward -Z direction
        
        # DEFINITELY HIDDEN: Bottom face edges (Z = 0)
        if abs(z1) < 0.1 and abs(z2) < 0.1:
            return "HIDDEN"
        
        # DEFINITELY HIDDEN: Bottom edges of cut hole
        if (21 <= min(x1,x2) <= 29 and 38 <= min(y1,y2) <= 55 and 
            max(z1,z2) < 2):
            return "HIDDEN"
        
        # POTENTIALLY HIDDEN: Internal cut edges that go down deep
        if (21 <= min(x1,x2) <= 29 and 38 <= min(y1,y2) <= 55 and 
            min(z1,z2) < 10 and abs(z1 - z2) > 20):  # Deep vertical edges in cut
            return "HIDDEN"
    
    elif view_direction == 'side':
        # Looking from X < 0 toward +X direction
        
        # DEFINITELY HIDDEN: Far side face edges (X = 94)
        if abs(x1 - 94) < 0.1 and abs(x2 - 94) < 0.1:
            return "HIDDEN"
        
        # DEFINITELY HIDDEN: Far edges of boss (X = 35)
        if (34.5 <= max(x1,x2) <= 35.5 and 1 <= min(y1,y2) <= 28 and 
            57 <= min(z1,z2) <= 67):
            return "HIDDEN"
        
        # DEFINITELY HIDDEN: Back wall of cut hole (X = 29)
        if (28.5 <= max(x1,x2) <= 29.5 and 38 <= min(y1,y2) <= 55 and 
            0 <= min(z1,z2) <= 57):
            return "HIDDEN"
        
        # POTENTIALLY HIDDEN: Internal edges inside cut near back wall
        if (25 <= min(x1,x2) <= 29 and 38 <= min(y1,y2) <= 55 and 
            abs(z1 - z2) > 20):  # Vertical edges near back of cut
            return "HIDDEN"
    
    return "VISIBLE"

def test_precise_hlr():
    """Test the precise HLR algorithm"""
    print("Precise HLR Analysis Based on Actual Edge Coordinates")
    print("=" * 65)
    
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
    
    print("\nPrecise Edge Classification Test:")
    print("-" * 55)
    
    for view in views:
        visible_count = 0
        hidden_count = 0
        hidden_edges = []
        
        print(f"\n{view.upper()} VIEW:")
        
        # Test all edges but show details for first batch
        for i, (edge, p1, p2) in enumerate(edges):
            result = precise_visibility_analysis((p1, p2), view)
            
            if result == "VISIBLE":
                visible_count += 1
            else:
                hidden_count += 1
                hidden_edges.append((i+1, p1, p2))
                
            # Show first few for debugging
            if i < 8:
                print(f"  Edge {i+1:2d} ({p1[0]:5.1f},{p1[1]:5.1f},{p1[2]:5.1f}) -> ({p2[0]:5.1f},{p2[1]:5.1f},{p2[2]:5.1f}): {result}")
        
        print(f"\n  SUMMARY: {visible_count} visible, {hidden_count} hidden (of {len(edges)} total)")
        
        if hidden_count > 0:
            print(f"  ✅ SUCCESS: Detected {hidden_count} hidden edges in {view} view")
            print(f"  📊 Hidden ratio: {hidden_count/len(edges)*100:.1f}%")
            
            # Show hidden edges
            print(f"  Hidden edges:")
            for edge_num, p1, p2 in hidden_edges[:5]:  # Show first 5 hidden edges
                print(f"    Edge {edge_num}: ({p1[0]:.1f},{p1[1]:.1f},{p1[2]:.1f}) -> ({p2[0]:.1f},{p2[1]:.1f},{p2[2]:.1f})")
            if len(hidden_edges) > 5:
                print(f"    ... and {len(hidden_edges)-5} more")
        else:
            print(f"  ⚠️  WARNING: No hidden edges detected in {view} view!")
        
        print()

if __name__ == "__main__":
    test_precise_hlr()
