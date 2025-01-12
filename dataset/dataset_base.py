import os
from abc import ABC, abstractmethod
import subprocess

class Dataset(ABC):
    """Abstract base class for dataset handling"""
    
    def __init__(self, data_folder_path: list, max_files: int = None):
        """
        Initialize the dataset
        Args:
            data_folder_path (list): List of paths to data folders
            max_files (int, optional): Maximum number of files to process
        """
        self.max_files = max_files if max_files is not None else float('inf')
        self.fpaths = []  # List to store clean results
        
        file_count = 0  # To keep track of the number of files processed
        
        for path in data_folder_path:
            try:
                # Run the lfs find command to locate .rec files
                result = subprocess.run(["lfs", "find", path, "-type", "f", "--name", "*.rec"],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        check=True) 
                # Process the output of the command
                for name in result.stdout.splitlines():
                    if file_count >= self.max_files:
                        break

                    # Check if path contains 'peet' or 'align' (case insensitive)
                    components = name.split('/')
                    found_peet = False
                    for component in components:
                        if 'peet' in component.lower() or "align" in component.lower():
                            found_peet = True
                            break
                    
                    # Add path if it doesn't contain excluded terms
                    if not found_peet:
                        self.fpaths.append(name)
                        file_count += 1

            except subprocess.CalledProcessError as e:
                print(f"An error occurred while running the subprocess: {e}")
    
    @abstractmethod
    def __len__(self):
        """
        Abstract method to get dataset length
        Must be implemented by child classes
        """
        pass
    
    @abstractmethod
    def pop(self, index: int = -1):
        """
        Abstract method to remove and return an item
        Must be implemented by child classes
        Args:
            index (int): Index of item to pop
        """
        pass