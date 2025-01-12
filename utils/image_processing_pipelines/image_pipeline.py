from abc import ABC, abstractmethod

class ImagePipeline(ABC):
    """
    Abstract base class defining the interface for image processing pipelines.
    All image processing pipelines should inherit from this class and implement
    the process_image method.
    """
    def __init__(self):
        """Initialize the pipeline."""
        pass
    
    @abstractmethod
    def process_image(
        self,
        image,
        **kwargs
    ):
        """
        Abstract method that must be implemented by all pipeline classes.
        
        Args:
            image: Input image to be processed
            **kwargs: Additional keyword arguments specific to each pipeline
            
        Returns:
            Processed image data in format specific to each pipeline implementation
        """
        pass