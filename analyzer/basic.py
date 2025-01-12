from .analysis_base import Analysis

class Basic(Analysis):
    """
    A basic analyzer class that inherits from Analysis.
    Provides simple file path analysis functionality.
    """
    def analyze(self, data, results, **kwargs):
        """
        Performs basic analysis by storing and printing the file path.
        
        Args:
            data: Input file path to analyze
            results: List to store analysis results
            **kwargs: Additional keyword arguments (unused)
        """
        results.append({"file_path": data})
        print({"file_path": data})