#!/usr/bin/env python3
# ===============================
# ABLA API Usage Example
# ===============================

"""
This script demonstrates how to use ABLA programmatically from another program.
It shows both the class-based approach and the convenience function approach.
"""

import os
import sys
from abla_api import ABLAProcessor, process_tomogram

def example_1_using_class():
    """
    Example of using the ABLAProcessor class to process tomogram data.
    This approach is useful when you need to process multiple datasets
    with the same configuration.
    """
    print("\n=== Example 1: Using ABLAProcessor class ===")
    
    # Initialize the processor with custom settings
    processor = ABLAProcessor(
        device="cuda",  # Use GPU if available
        batch_size=2,   # Process 2 files at a time
        num_negative_points=3,
        num_positive_points=2
    )
    
    # Process a single file
    results = processor.process_data(
        data_path="/path/to/your/tomogram.rec",  # Replace with your actual file path
        dataset_name="example_dataset_1",
        file_extension=".rec"
    )
    
    print(f"Processing completed. Results saved in results/example_dataset_1/")
    
    # You can also process a directory of files
    # results = processor.process_data(
    #     data_path="/path/to/your/data/directory",
    #     dataset_name="example_dataset_2",
    #     file_extension=".rec"
    # )

def example_2_using_function():
    """
    Example of using the convenience function to process tomogram data.
    This approach is simpler when you only need to process a single dataset.
    """
    print("\n=== Example 2: Using convenience function ===")
    
    # Process a single file using the convenience function
    results = process_tomogram(
        data_path="/path/to/your/tomogram.rec",  # Replace with your actual file path
        dataset_name="example_dataset_3",
        file_extension=".rec",
        device="cuda",
        batch_size=1
    )
    
    print(f"Processing completed. Results saved in results/example_dataset_3/")

def example_3_processing_multiple_files():
    """
    Example of processing multiple specific files.
    """
    print("\n=== Example 3: Processing multiple specific files ===")
    
    # List of specific files to process
    file_list = [
        "/path/to/your/tomogram1.rec",
        "/path/to/your/tomogram2.rec",
        "/path/to/your/tomogram3.rec"
    ]
    
    # Process the list of files
    results = process_tomogram(
        data_path=file_list,
        dataset_name="example_dataset_4",
        file_extension=".rec"
    )
    
    print(f"Processing completed. Results saved in results/example_dataset_4/")

def example_4_custom_results_directory():
    """
    Example of specifying a custom results directory.
    """
    print("\n=== Example 4: Using custom results directory ===")
    
    # Process with custom results directory
    results = process_tomogram(
        data_path="/path/to/your/tomogram.rec",  # Replace with your actual file path
        dataset_name="example_dataset_5",
        file_extension=".rec",
        results_dir="/path/to/custom/results"
    )
    
    print(f"Processing completed. Results saved in /path/to/custom/results/example_dataset_5/")

if __name__ == "__main__":
    # Uncomment the examples you want to run
    # example_1_using_class()
    # example_2_using_function()
    # example_3_processing_multiple_files()
    # example_4_custom_results_directory()
    
    print("Please modify the file paths in this example script before running.")
    print("Uncomment the example functions you want to run.") 