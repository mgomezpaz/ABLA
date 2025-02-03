# Core functionality
from typing import Dict, Any, List
import os
from glob import glob
import random

# Local imports
from .dataset_base import Dataset  # Changed from dataset to dataset_base

class SCDataset(Dataset):
    """A dataset class specifically for handling .rec files"""
    
    def __init__(self, data_folder_paths: list, file_ext: str = '.rec', max_files: int = None):
        """
        Initialize the SCDataset
        Args:
            data_folder_paths (list): List of paths to folders containing data
            file_ext (str): File extension to filter for ('.rec' or '.mrc')
            max_files (int, optional): Maximum number of files to process
        """
        super().__init__(data_folder_paths, max_files)
        self.file_ext = file_ext
        
        # If we're given a direct file path instead of a directory
        if len(data_folder_paths) == 1 and os.path.isfile(data_folder_paths[0]):
            self.fpaths = data_folder_paths
        # Add this else block to handle directory paths
        else:
            self.fpaths = []
            for folder_path in data_folder_paths:
                if os.path.isdir(folder_path):
                    # Walk through the directory and collect all files
                    for root, _, files in os.walk(folder_path):
                        for file in files:
                            if file.endswith(self.file_ext):
                                self.fpaths.append(os.path.join(root, file))
        
        self.file_paths = self._get_file_paths()

    def _get_file_paths(self):
        """
        Extract all files with specified extension from the file paths
        Returns:
            dict: Dictionary mapping filenames to their full paths
        """
        file_paths = {}
        for path in self.fpaths:
            if path.endswith(self.file_ext):
                file_name = os.path.basename(path)
                file_paths[file_name] = path
        return file_paths

    def get_file_paths(self):
        """
        Getter method for file_paths
        Returns:
            dict: Dictionary of file paths
        """
        return self.file_paths

    def __len__(self):
        """
        Get the number of .rec files in the dataset
        Returns:
            int: Number of .rec files
        """
        return len(self.file_paths)

    def pop(self, index: int = -1):
        """
        Remove and return a .rec file at the specified index
        Args:
            index (int): Index of file to pop (default: -1 for last item)
        Returns:
            dict: Single-item dictionary with filename and path
        Raises:
            IndexError: If no files available or index out of range
        """
        if not self.file_paths:
            raise IndexError("No .rec files available to pop.")
        
        keys = list(self.file_paths.keys())
        if index < 0:
            index += len(keys)
        
        if index >= len(keys) or index < 0:
            raise IndexError("Index out of range")
        
        key = keys[index]
        path = self.file_paths.pop(key)
        return {key: path}

    def random_pop(self):
        """
        Remove and return a random .rec file
        Returns:
            dict: Single-item dictionary with filename and path
        Raises:
            IndexError: If no files available
        """
        if not self.file_paths:
            raise IndexError("No .rec files available for random_pop()")
        
        key = random.choice(list(self.file_paths.keys()))
        path = self.file_paths.pop(key)
        return {key: path}
