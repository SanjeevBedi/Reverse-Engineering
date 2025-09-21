# Backup of V6_current.py before incremental debug changes
# Date: 2025-09-20
# This file is a direct copy of V6_current.py before any new debug or logic changes.

from OCC.Core.gp import gp_Trsf  # noqa: F401
from OCC.Core.TopLoc import TopLoc_Location  # noqa: F401
from OCC.Core.TopAbs import (
	TopAbs_SHELL, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
)  # noqa: F401, E501
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
# V5_current.py
# Saved version of Polgon Boolean Ops from shapely.py as of July 28, 2025
# Includes corrected plotting order: array_C first (dashed light gray), array_B second (solid black)
import argparse

from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import traceback

# ...existing code continues...
