"""
OpenCASCADE Engineering Drawings Generator
A comprehensive Python package for generating engineering drawings from 3D CAD models.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="engineering-drawings",
    version="1.0.0",
    author="OpenCASCADE Engineering Team",
    author_email="engineering@example.com",
    description="Generate engineering drawings from 3D CAD models using OpenCASCADE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/engineering-team/opencascade-engineering-drawings",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[req for req in requirements if not req.startswith("pythonocc")],
    extras_require={
        "dev": ["pytest>=6.0.0", "pytest-cov>=3.0.0", "black>=22.0.0", "flake8>=4.0.0"],
        "docs": ["sphinx>=4.0.0", "sphinx-rtd-theme>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "engineering-drawings=engineering_drawings.main:main",
        ],
    },
    keywords="opencascade engineering drawings cad visualization",
    project_urls={
        "Bug Reports": "https://github.com/engineering-team/opencascade-engineering-drawings/issues",
        "Source": "https://github.com/engineering-team/opencascade-engineering-drawings",
        "Documentation": "https://opencascade-engineering-drawings.readthedocs.io/",
    },
)
