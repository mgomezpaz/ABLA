from .basic import Basic
from .analysis_base import Analysis
from .membrane_segmenter import MembraneSegmenter

def get_analysis(analysis_type):
    """
    Factory function to get the appropriate analysis class.
    
    Args:
        analysis_type (str): Type of analysis to perform
        
    Returns:
        Analysis class corresponding to the requested type
        
    Raises:
        Exception: If the requested analysis type doesn't exist
    """
    analysis_type = analysis_type.lower()

    if analysis_type == "basic":
        return Basic
    else:
        raise Exception("Analysis Doesn't Exist")
