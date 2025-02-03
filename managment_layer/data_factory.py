# Core functionality
from typing import Dict, Any
from tqdm import tqdm  # For progress bar

# Local imports
from analyzer import Analysis
from .thread_manager import ThreadManager
from dataset.dataset_base import Dataset  # Be explicit about the import path

class DataFactory:
    """
    Manages the processing of large datasets by breaking them into manageable batches
    and coordinating their parallel processing through the ThreadManager.
    """
    def __init__(
        self,
        analyzer: Analysis,
        batch_size: int = 100,
    ):
        """
        Initialize the DataFactory with an analyzer and batch configuration.

        Args:
            analyzer: Analysis instance to process data
            batch_size: Number of items to process in each batch
        """
        # Initialize thread manager with the provided analyzer
        self.__threader = ThreadManager(analyzer)
        # Set the size of batches for processing
        self.__batch_size = batch_size
        # Placeholder for future checkpointing functionality
        self.__checkpointer = None
    
    def process(
        self,
        dataset: Dataset,
        loading_bar_string: str
    ):
        """
        Process the entire dataset in batches with progress tracking.

        Args:
            dataset: Dataset instance containing items to process
            loading_bar_string: Description string for the progress bar

        Returns:
            dict: Combined results from all processed batches
        """
        # Counter for batch processing (starts at 4)
        batch_num = 4

        # Create progress bar for visual feedback
        loading_bar = tqdm(total=len(dataset), desc=loading_bar_string)
        
        # Dictionary to store combined results from all batches
        all_results = {}

        # Process batches until dataset is empty
        while len(dataset) != 0:
            # Initialize empty list for current batch
            current_batch = []
            
            # Fill the current batch until batch_size is reached or dataset is empty
            while len(current_batch) < self.__batch_size and len(dataset) != 0:
                current_batch.append(dataset.pop())
                
            # Process the current batch using thread manager
            batch_results = self.__threader.submit_batch(current_batch)
            
            # Merge batch results into overall results
            all_results.update(batch_results)
            
            # Update the progress bar with number of processed items
            loading_bar.update(len(batch_results))

            # Handle checkpointing if enabled (future functionality)
            if self.__checkpointer:
                if not self.__checkpointer.interval % batch_num:
                    self.__checkpointer.checkpoint()
                
            # Increment batch counter
            batch_num += 1
            
        return all_results