"""
OpenCASCADE Engineering Drawings Generator

A comprehensive Python package for generating engineering drawings from 3D CAD models
using OpenCASCADE technology with advanced polygon classification algorithms.
"""

__version__ = "1.0.0"
__author__ = "OpenCASCADE Engineering Team"
__email__ = "engineering@example.com"

# Import main classes for easy access
from .main import EngineeringDrawingsGenerator

# Import individual modules
from . import solid_generator
from . import face_analyzer  
from . import vertex_extractor
from . import projection_engine
from . import classification_algorithm
from . import visualization

__all__ = [
    "EngineeringDrawingsGenerator",
    "solid_generator",
    "face_analyzer",
    "vertex_extractor", 
    "projection_engine",
    "classification_algorithm",
    "visualization",
]
