# ===============================
# ABLA Package Initialization
# ===============================

"""
ABLA (Automated Bacterial Labeling Annotations)

A tool for automated membrane segmentation in cryo-electron tomography.
"""

# Import the API functions for easy access
from .abla_api import ABLAProcessor, process_tomogram

# Define package version
__version__ = "1.0.0"

# Define what's available when using "from ABLA import *"
__all__ = ['ABLAProcessor', 'process_tomogram'] 