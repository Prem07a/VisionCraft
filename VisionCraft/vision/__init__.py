"""
VisionCraft - A custom image processing library.

This library provides a collection of functions for image filtering, processing, and transformations.

Author: Prem Gaikwad
Date: Feb 2024

Directory Structure:
- vision/
    - __init__.py
    - filter.py
    - processing.py
    - transform.py
    - utils.py

Usage:
import VisionCraft

> Example: Importing filter functions
from VisionCraft.vision.filter import boxFilter

> Example: Importing processing functions
from VisionCraft.vision.processing import imgNegative

> Example: Importing transform functions
from VisionCraft.vision.transform import grayLevelSlicing

> Example: Importing utility functions
from VisionCraft.vision.utils import imShow
"""

from VisionCraft.vision import *