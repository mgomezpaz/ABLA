# ===============================
# CONFIGURATION SETTINGS
# ===============================
import os

# Get ABLA root directory
ABLA_ROOT = os.path.dirname(os.path.abspath(__file__))

# Model Settings
MODEL_SETTINGS = {
    'device': 'cuda',  # Use 'cpu' if no GPU available
    'batch_size': 1,
    'num_negative_points': 3,
    'num_positive_points': 2
}

# File Settings
FILE_SETTINGS = {
    'default_file_extension': '.rec',
    # Use environment variable or fallback to a default path
    'default_dataset_path': os.getenv('ABLA_DATASET_PATH', '/path/to/your/data/example.rec')
}

# Model Paths
MODEL_PATHS = {
    # FOR DEFAULT MODEL
    'sam2_checkpoint': os.path.join(ABLA_ROOT, "analyzer/sam2/checkpoints/sam2.1_hiera_large.pt"),
    'model_config': "configs/sam2.1/sam2.1_hiera_l.yaml",
    # FOR FINE-TUNED MODEL
    #'sam2_checkpoint': os.path.join(ABLA_ROOT, "analyzer/sam2/checkpoints/bacteria2.pt"),
    #'model_config': "configs/sam2.1/sam2.1_hiera_b+.yaml"
}

# Results Settings
RESULTS_SETTINGS = {
    'results_dir': os.getenv('ABLA_RESULTS_DIR', os.path.join(ABLA_ROOT, "results")),
}
