#!/usr/bin/env python3
"""
Command-line entry point for OpenCASCADE Engineering Drawings Generator.

This script provides a simple command-line interface for generating engineering
drawings from 3D CAD models using OpenCASCADE technology.
"""

import sys
import os

# Add the source directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import the main application
from engineering_drawings.main import main

if __name__ == "__main__":
    main()
