# ===============================
# ABLA API - Programmatic Interface
# ===============================

import os
import sys
import torch
from typing import Dict, Any, Optional, Union, List

# Import ABLA components
from config import *
from analyzer.membrane_segmenter import MembraneSegmenter
from managment_layer.data_factory import DataFactory
from dataset.sc_dataset import SCDataset
from utils.image_processing_pipelines import BacteriaCentroidPipeline

def cleanup_temp_files(dataset_name):
    """
    Clean up temporary files and directories created during processing.
    
    Args:
        dataset_name (str): Name of the dataset used in processing
    """
    import glob
    import shutil
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

class ABLAProcessor:
    """
    A class that provides programmatic access to ABLA functionality.
    This allows ABLA to be imported and used as a module in other programs.
    """
    
    def __init__(
        self,
        device: str = None,
        batch_size: int = None,
        num_negative_points: int = None,
        num_positive_points: int = None,
        sam2_checkpoint: str = None,
        model_config: str = None
    ):
        """
        Initialize the ABLA processor with configurable parameters.
        
        Args:
            device (str, optional): Computing device ('cuda', 'cpu'). Defaults to config value.
            batch_size (int, optional): Batch size for processing. Defaults to config value.
            num_negative_points (int, optional): Number of negative points. Defaults to config value.
            num_positive_points (int, optional): Number of positive points. Defaults to config value.
            sam2_checkpoint (str, optional): Path to SAM2 checkpoint. Defaults to config value.
            model_config (str, optional): Path to model config. Defaults to config value.
        """
        # Create necessary directories
        os.makedirs(os.path.join(ABLA_ROOT, "data"), exist_ok=True)
        os.makedirs(os.path.join(ABLA_ROOT, "results"), exist_ok=True)
        os.makedirs(os.path.join(ABLA_ROOT, "temporary_files"), exist_ok=True)
        
        # Use provided values or defaults from config
        self.device = device if device is not None else MODEL_SETTINGS['device']
        self.batch_size = batch_size if batch_size is not None else MODEL_SETTINGS['batch_size']
        self.num_negative_points = num_negative_points if num_negative_points is not None else MODEL_SETTINGS['num_negative_points']
        self.num_positive_points = num_positive_points if num_positive_points is not None else MODEL_SETTINGS['num_positive_points']
        self.sam2_checkpoint = sam2_checkpoint if sam2_checkpoint is not None else MODEL_PATHS['sam2_checkpoint']
        self.model_config = model_config if model_config is not None else MODEL_PATHS['model_config']
        
    def process_data(
        self,
        data_path: Union[str, List[str]],
        dataset_name: str,
        file_extension: str = None,
        results_dir: str = None
    ) -> Dict[str, Any]:
        """
        Process data using ABLA.
        
        Args:
            data_path (str or list): Path(s) to data file(s) or directory containing data files
            dataset_name (str): Name for this dataset (used for results directory)
            file_extension (str, optional): File extension to filter for. Defaults to config value.
            results_dir (str, optional): Directory to save results. Defaults to config value.
            
        Returns:
            dict: Results of the processing
        """
        # Validate inputs
        if not dataset_name:
            raise ValueError("Dataset name cannot be empty")
        
        # Convert single path to list
        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = data_path
            
        # Use provided values or defaults
        file_ext = file_extension if file_extension is not None else FILE_SETTINGS['default_file_extension']
        
        # Initialize dataset
        dataset = SCDataset(data_paths, file_ext=file_ext)
        
        # Set up results directory
        if results_dir is None:
            results_dir = os.path.join(RESULTS_SETTINGS['results_dir'], dataset_name)
        else:
            results_dir = os.path.join(results_dir, dataset_name)
            
        # Initialize analyzer
        analyzer = MembraneSegmenter(
            image_processing_pipeline=BacteriaCentroidPipeline(
                num_negative_points=self.num_negative_points,
                num_positive_points=self.num_positive_points,
            ),
            device=self.device,
            sam2_checkpoint=self.sam2_checkpoint,
            model_cfg=self.model_config,
            results_dir=results_dir
        )
        
        # Initialize data factory
        data_factory = DataFactory(analyzer, batch_size=self.batch_size)
        
        # Process dataset
        results = None
        try:
            results = data_factory.process(dataset, "Processing Dataset")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
        
        # Cleanup
        cleanup_temp_files(dataset_name)
        
        # Clean up GPU memory if CUDA was used
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
            
        return results

# Convenience function for direct use
def process_tomogram(
    data_path: Union[str, List[str]],
    dataset_name: str,
    file_extension: str = None,
    device: str = None,
    batch_size: int = None,
    num_negative_points: int = None,
    num_positive_points: int = None,
    results_dir: str = None
) -> Dict[str, Any]:
    """
    Convenience function to process tomogram data without creating an ABLAProcessor instance.
    
    Args:
        data_path (str or list): Path(s) to data file(s) or directory containing data files
        dataset_name (str): Name for this dataset (used for results directory)
        file_extension (str, optional): File extension to filter for. Defaults to config value.
        device (str, optional): Computing device ('cuda', 'cpu'). Defaults to config value.
        batch_size (int, optional): Batch size for processing. Defaults to config value.
        num_negative_points (int, optional): Number of negative points. Defaults to config value.
        num_positive_points (int, optional): Number of positive points. Defaults to config value.
        results_dir (str, optional): Directory to save results. Defaults to config value.
        
    Returns:
        dict: Results of the processing
    """
    processor = ABLAProcessor(
        device=device,
        batch_size=batch_size,
        num_negative_points=num_negative_points,
        num_positive_points=num_positive_points
    )
    
    return processor.process_data(
        data_path=data_path,
        dataset_name=dataset_name,
        file_extension=file_extension,
        results_dir=results_dir
    ) 