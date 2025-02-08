# ===============================
# IMPORT LIBRARIES
# ===============================

import numpy as np
# Basic utilities (fast imports)
import os
import sys
import shutil
import glob
from pprint import pprint
import torch

# ===============================
# IMPORT CUSTOM MODULES
# ===============================
from config import *
from analyzer.membrane_segmenter import MembraneSegmenter
from managment_layer.data_factory import DataFactory
from dataset.sc_dataset import SCDataset
from utils.image_processing_pipelines import BacteriaCentroidPipeline

# Remove unused imports
# import pandas as pd            # Not used
# import matplotlib.pyplot as plt # Not used
# import seaborn as sns         # Not used

def cleanup_temp_files(dataset_name):
    """
    Clean up temporary files and directories created during processing.
    
    Args:
        dataset_name (str): Name of the dataset used in processing
    """
    try:
        # Clean up temporary image databases
        temp_pattern = f"temporary_files/temp_image_database_{dataset_name}*"
        temp_dirs = glob.glob(temp_pattern)
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def main():
    """
    Main function to run the bacterial membrane segmentation analysis pipeline.
    Handles data loading, processing, and result generation.
    """
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(ABLA_ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(ABLA_ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ABLA_ROOT, "temporary_files"), exist_ok=True)
    
    # ===============================
    # INPUT DATA PATHS
    # ===============================
    # Default path for dataset
    default_path = FILE_SETTINGS['default_dataset_path']
    
    # Get dataset path from user input or use default
    data_path = input(f"Enter the path to your dataset (press Enter for default): ")
    # Get dataset name for results directory
    dataset_name = input("Enter a name for this dataset: ").strip()
    while not dataset_name:  # Ensure a name is provided
        print("Dataset name cannot be empty")
        dataset_name = input("Enter a name for this dataset: ").strip()
    
    if data_path.strip():
        # If it's a file path, use the file path directly
        if os.path.isfile(data_path):
            data_folder_paths = [data_path]  # Pass the full file path
        else:
            data_folder_paths = [data_path]
    else:
        data_folder_paths = [default_path]  # Use the full default path
    
    # Get file extension from user input or use default
    default_file_ext = FILE_SETTINGS['default_file_extension']
    file_ext_input = input(f"Enter file extension (press Enter for default={default_file_ext}): ")
    file_ext = file_ext_input.strip() if file_ext_input.strip() else default_file_ext
    
    # Initialize dataset with specified paths and file extension
    dataset = SCDataset(data_folder_paths, file_ext=file_ext)
    
    # ===============================
    # SETUP ANALYSIS PARAMETERS
    # ===============================
    print("Setting up Analysis Pipeline...")
    device = MODEL_SETTINGS['device']
    
    # --- Batch Size Configuration ---
    default_batch_size = MODEL_SETTINGS['batch_size']
    batch_input = input(f"Enter batch size (press Enter for default={default_batch_size}): ")
    
    # Validate batch size input
    if batch_input.strip():
        while True:
            try:
                batch_size = int(batch_input)
                if batch_size > 0:
                    break
                print("Batch size must be positive")
                batch_input = input(f"Enter batch size (press Enter for default={default_batch_size}): ")
            except ValueError:
                print("Please enter a valid number")
                batch_input = input(f"Enter batch size (press Enter for default={default_batch_size}): ")
    else:
        batch_size = default_batch_size
    
    # --- Segmentation Points Configuration ---
    # Default values for segmentation points
    default_neg_points = MODEL_SETTINGS['num_negative_points']
    default_pos_points = MODEL_SETTINGS['num_positive_points']
    
    # Get user input for points
    neg_input = input(f"Enter number of negative points (press Enter for default={default_neg_points}): ")
    pos_input = input(f"Enter number of positive points (press Enter for default={default_pos_points}): ")
    
    # Initialize with defaults
    num_negative_points = default_neg_points
    num_positive_points = default_pos_points
    
    # Validate negative points input
    if neg_input.strip():
        while True:
            try:
                num_negative_points = int(neg_input)
                if num_negative_points > 0:
                    break
                print("Number of points must be positive")
                neg_input = input(f"Enter number of negative points (press Enter for default={default_neg_points}): ")
            except ValueError:
                print("Please enter a valid number")
                neg_input = input(f"Enter number of negative points (press Enter for default={default_neg_points}): ")
    
    # Validate positive points input
    if pos_input.strip():
        while True:
            try:
                num_positive_points = int(pos_input)
                if num_positive_points > 0:
                    break
                print("Number of points must be positive")
                pos_input = input(f"Enter number of positive points (press Enter for default={default_pos_points}): ")
            except ValueError:
                print("Please enter a valid number")
                pos_input = input(f"Enter number of positive points (press Enter for default={default_pos_points}): ")
    
    # ===============================
    # INITIALIZE ANALYZER
    # ===============================
    # Create results directory with dataset name
    results_dir = os.path.join(RESULTS_SETTINGS['results_dir'], dataset_name)
    
    # Setup the membrane segmentation analyzer with configured parameters
    analyzer = MembraneSegmenter(
        image_processing_pipeline=BacteriaCentroidPipeline(
            num_negative_points=num_negative_points,
            num_positive_points=num_positive_points,
        ),
        device=device,
        sam2_checkpoint=MODEL_PATHS['sam2_checkpoint'],
        model_cfg=MODEL_PATHS['model_config'],
        results_dir=results_dir  # Use the modified results directory
    )
    
    # ===============================
    # DATA PROCESSING
    # ===============================
    print("Setting up DataFactory...")
    # Initialize DataFactory for batch processing
    data_factory = DataFactory(analyzer, batch_size=batch_size)
    
    print("Processing Dataset...")
    # Process the dataset with error handling
    results = None
    try:
        results = data_factory.process(dataset, "Processing Dataset")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    # Print results if processing was successful
    # if results is not None:
    #     pprint(results)
    # else:
    #     print("No results due to an error during processing.")
    
    # ===============================
    # DATA VISUALIZATION
    # ===============================
    # TODO: Add data visualization
    
    # ===============================
    # PROGRAM TERMINATION
    # ===============================
    # Add cleanup after processing is complete
    print("\nCleaning up temporary files...")
    cleanup_temp_files(dataset_name)
    
    print("\nProgram completed successfully.")
    try:
        # Clean up any GPU memory if CUDA was used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    sys.exit(0)  # Exit with success status code
    
# ===============================
# SCRIPT ENTRY POINT
# ===============================
if __name__ == "__main__":
    main()
    