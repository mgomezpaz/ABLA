# Core functionality
from typing import Dict, Any, List
import os
from glob import glob
import random

# Local imports
from .dataset_base import Dataset  # Changed from dataset to dataset_base

class SCDataset(Dataset):
    """A dataset class specifically for handling .rec and .mrc files"""
    
    def __init__(self, data_folder_paths: list, file_ext: str = '.rec', max_files: int = None):
        """
        Initialize the SCDataset
        Args:
            data_folder_paths (list): List of paths to folders containing data
            file_ext (str): File extension to filter for ('.rec' or '.mrc')
            max_files (int, optional): Maximum number of files to process
        """
        super().__init__(data_folder_paths, max_files)
        # Support both .rec and .mrc files
        self.file_ext = file_ext.lower()  # Normalize extension
        self.data_paths = []
        
        # Handle both single files and directories
        for path in data_folder_paths:
            if os.path.isfile(path):
                # For direct file paths, check extension
                if path.lower().endswith(self.file_ext):
                    self.data_paths.append(path)
            else:
                # For directories, walk through all subdirectories
                for root, _, files in os.walk(path):
                    for file in files:
                        # Check for both .rec and .mrc files (case insensitive)
                        if file.lower().endswith(self.file_ext):
                            full_path = os.path.join(root, file)
                            self.data_paths.append(full_path)
        
        # Sort paths for consistent ordering
        self.data_paths.sort()
        
        if not self.data_paths:
            print(f"Warning: No files with extension {self.file_ext} found in the specified paths.")
            print("Searched in:", data_folder_paths)
        else:
            print(f"Found {len(self.data_paths)} files with extension {self.file_ext}")
            
        self.file_paths = self._get_file_paths()

    def _get_file_paths(self):
        """
        Extract all files with specified extension from the file paths
        Returns:
            dict: Dictionary mapping filenames to their full paths
        """
        file_paths = {}
        for path in self.data_paths:
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

    def __getitem__(self, idx):
        return {os.path.basename(self.data_paths[idx]): self.data_paths[idx]}
