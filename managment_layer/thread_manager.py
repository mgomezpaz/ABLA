from analyzer import Analysis
import threading


class ThreadManager:
    """
    Manages concurrent processing of data batches using threading.
    Coordinates multiple analysis tasks running in parallel.
    """
    def __init__(
        self,
        analyzer: Analysis
    ):
        """
        Initialize the thread manager with an analyzer instance.

        Args:
            analyzer: Analysis instance to process data
        """
        # Store analyzer instance for processing data
        self.analyzer = analyzer
    
    def submit_batch(
        self,
        batch: list[dict] # A list of dictionaries with data_paths as keys and other required arguments as values
    ):
        """
        Submits a batch of data points for concurrent processing.

        Args:
            batch: List of dictionaries, each containing a single key-value pair
                  where key is the identifier and value is the data path

        Returns:
            dict: Results from processing all data points in the batch

        Raises:
            Exception: If datapoint format is incorrect
        """
        # Dictionary to store results from all threads
        results = {}
        # List to store thread objects for tracking
        threads = []

        # Process each datapoint in the batch
        for datapoint in batch:
            # Ensure datapoint is in dictionary format
            if type(datapoint) != dict:
                raise Exception("Datapoint Must be in Dictionary Format")
            
            # Ensure datapoint contains exactly one key-value pair
            if len(datapoint) != 1:
                raise Exception("Datapoint dictionary must contain exactly one key-value pair")
            
            # Extract the first (and only) key-value pair from the datapoint
            key, data_path = next(iter(datapoint.items()))
            
            # Prepare arguments for the analyzer
            analyze_args = {
                'data_path': data_path,
                'key': key,
                'results': results
            }
            
            
            # Create and start a new thread for processing this datapoint
            thread = threading.Thread(target=self.analyzer.analyze, kwargs=analyze_args)
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete processing
        for thread in threads:
            thread.join()

        # Return results
        return results
