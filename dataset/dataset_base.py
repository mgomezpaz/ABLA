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
                # Try lfs first, fall back to regular find if lfs isn't available
                try:
                    result = subprocess.run(["lfs", "find", path, "-type", "f", "--name", "*.rec"],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          text=True,
                                          check=True)
                except FileNotFoundError:
                    result = subprocess.run(["find", path, "-type", "f", "-name", "*.rec"],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          text=True,
                                          check=True)
                
                # Process the output
                for name in result.stdout.splitlines():
                    if file_count >= self.max_files:
                        break
                    
                    components = name.split('/')
                    found_peet = False
                    for component in components:
                        if 'peet' in component.lower() or "align" in component.lower():
                            found_peet = True
                            break
                    
                    if not found_peet:
                        self.fpaths.append(name)
                        file_count += 1

            except subprocess.CalledProcessError as e:
                print(f"An error occurred while running the find command: {e}")
                print("Falling back to Python's os.walk...")
                
                # Fallback to os.walk if both find commands fail
                for root, _, files in os.walk(path):
                    for file in files:
                        if file_count >= self.max_files:
                            break
                            
                        if file.endswith('.rec'):
                            full_path = os.path.join(root, file)
                            components = full_path.split(os.sep)
                            found_peet = False
                            for component in components:
                                if 'peet' in component.lower() or "align" in component.lower():
                                    found_peet = True
                                    break
                            
                            if not found_peet:
                                self.fpaths.append(full_path)
                                file_count += 1
    
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