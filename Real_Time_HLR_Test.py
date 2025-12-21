#!/usr/bin/env python3
"""
Real-time HLR Test
==================
Test the actual edge classification in the running program
"""

import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

from OCC.Core.gp import gp_Pnt
from Random_Engineering_Drawings import EngineeringDrawingsHLR

print("Real-time HLR Test")
print("="*40)

# Create an instance of our HLR engine
hlr = EngineeringDrawingsHLR()

# Test boss top face edge (should be VISIBLE)
boss_p1 = gp_Pnt(2.0, 1.0, 67.0)
boss_p2 = gp_Pnt(35.0, 1.0, 67.0)

print("Testing Boss Top Face Edge:")
print(f"  From: ({boss_p1.X():.1f}, {boss_p1.Y():.1f}, {boss_p1.Z():.1f})")
print(f"  To:   ({boss_p2.X():.1f}, {boss_p2.Y():.1f}, {boss_p2.Z():.1f})")

# Test using the actual function that's called in the program
result_front = hlr.is_edge_on_visible_face(boss_p1, boss_p2, 'front')
result_top = hlr.is_edge_on_visible_face(boss_p1, boss_p2, 'top')

print(f"  Front view: {'✅ VISIBLE' if result_front else '❌ HIDDEN'}")
print(f"  Top view:   {'✅ VISIBLE' if result_top else '❌ HIDDEN'}")
print()

# Test cut hole edge (should be HIDDEN)
hole_p1 = gp_Pnt(21.0, 38.0, 0.0)
hole_p2 = gp_Pnt(21.0, 38.0, 57.0)

print("Testing Cut Hole Edge:")
print(f"  From: ({hole_p1.X():.1f}, {hole_p1.Y():.1f}, {hole_p1.Z():.1f})")
print(f"  To:   ({hole_p2.X():.1f}, {hole_p2.Y():.1f}, {hole_p2.Z():.1f})")

result_front = hlr.is_edge_on_visible_face(hole_p1, hole_p2, 'front')
result_top = hlr.is_edge_on_visible_face(hole_p1, hole_p2, 'top')

print(f"  Front view: {'✅ HIDDEN' if not result_front else '❌ VISIBLE'}")
print(f"  Top view:   {'✅ HIDDEN' if not result_top else '❌ VISIBLE'}")

print()
print("="*40)
print("✅ CONCLUSION:")
print("If boss edges show as VISIBLE and hole edges show as HIDDEN,")
print("then the corrected HLR algorithm is working properly!")
