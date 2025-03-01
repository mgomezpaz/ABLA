# ABLA API: Using ABLA as a Module

This document explains how to use ABLA (Automated Bacterial Labeling Annotations) as a module in your own Python programs.

## Installation

To use ABLA as a module in your project, you can clone the repository using git:

```bash
git clone https://github.com/matiasgp/ABLA.git
```

After cloning, make sure to follow the installation instructions in the main README.md file to set up the environment and dependencies.

## Basic Usage

ABLA provides a programmatic API that allows you to integrate it into your own Python programs. There are two main ways to use ABLA:

1. Using the `ABLAProcessor` class for more control
2. Using the `process_tomogram` convenience function for simpler usage

### Example 1: Using the ABLAProcessor class

```python
from ABLA.abla_api import ABLAProcessor

# Initialize the processor with custom settings
processor = ABLAProcessor(
    device="cuda",  # Use GPU if available
    batch_size=2,   # Process 2 files at a time
    num_negative_points=3,
    num_positive_points=2
)

# Process a single file
results = processor.process_data(
    data_path="/path/to/your/tomogram.rec",
    dataset_name="my_dataset",
    file_extension=".rec"
)

# Process a directory of files
results = processor.process_data(
    data_path="/path/to/your/data/directory",
    dataset_name="my_directory_dataset",
    file_extension=".rec"
)
```

### Example 2: Using the convenience function

```python
from ABLA.abla_api import process_tomogram

# Process a single file
results = process_tomogram(
    data_path="/path/to/your/tomogram.rec",
    dataset_name="my_dataset",
    file_extension=".rec",
    device="cuda",
    batch_size=1
)
```

### Example 3: Processing multiple specific files

```python
from ABLA.abla_api import process_tomogram

# List of specific files to process
file_list = [
    "/path/to/your/tomogram1.rec",
    "/path/to/your/tomogram2.rec",
    "/path/to/your/tomogram3.rec"
]

# Process the list of files
results = process_tomogram(
    data_path=file_list,
    dataset_name="multiple_files_dataset",
    file_extension=".rec"
)
```

### Example 4: Using a custom results directory

```python
from ABLA.abla_api import process_tomogram

# Process with custom results directory
results = process_tomogram(
    data_path="/path/to/your/tomogram.rec",
    dataset_name="my_dataset",
    file_extension=".rec",
    results_dir="/path/to/custom/results"
)
```

## API Reference

### ABLAProcessor

```python
class ABLAProcessor:
    def __init__(
        self,
        device: str = None,
        batch_size: int = None,
        num_negative_points: int = None,
        num_positive_points: int = None,
        sam2_checkpoint: str = None,
        model_config: str = None
    ):
        # Initialize the processor
        
    def process_data(
        self,
        data_path: Union[str, List[str]],
        dataset_name: str,
        file_extension: str = None,
        results_dir: str = None
    ) -> Dict[str, Any]:
        # Process data and return results
```

### process_tomogram

```python
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
    # Process tomogram data and return results
```

## Parameters

- `data_path`: Path to a single file, a directory containing files, or a list of file paths
- `dataset_name`: Name for this dataset (used for results directory)
- `file_extension`: File extension to filter for (default: ".rec")
- `device`: Computing device ("cuda", "cpu") (default: from config)
- `batch_size`: Batch size for processing (default: from config)
- `num_negative_points`: Number of negative points (default: from config)
- `num_positive_points`: Number of positive points (default: from config)
- `results_dir`: Directory to save results (default: from config)

## Results

The processing functions return a dictionary containing the results of the analysis. The results are also saved to disk in the specified results directory.

## Example Project

For a complete example of how to use ABLA as a module, see the `example_usage.py` file in the ABLA directory.

## Troubleshooting

If you encounter issues when using ABLA as a module:

1. Make sure you have followed all installation steps in the main README.md
2. Check that the SAM2 model is properly installed and the checkpoints are downloaded
3. Verify that your input data is in the correct format
4. Check the paths to your data files and make sure they exist

For more help, please create an issue on the GitHub repository. 