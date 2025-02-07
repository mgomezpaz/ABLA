# ===============================
# CONFIGURATION SETTINGS
# ===============================

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
    'default_dataset_path': "/home/matiasgp/groups/grp_tomo_db1_d1/nobackup/archive/TomoDB1_d1/FlagellarMotor_P1/CaulobacterCrescentus_UnpluggedPL/lza2018-10-20-13/targ65_full.rec"
}

# Model Paths
MODEL_PATHS = {
    # FOR DEFAULT MODEL
    'sam2_checkpoint': "/home/matiasgp/Desktop/ABLA/analyzer/sam2/checkpoints/sam2.1_hiera_large.pt",
    'model_config': "configs/sam2.1/sam2.1_hiera_l.yaml",
    # FOR FINE-TUNED MODEL
    #'sam2_checkpoint': "/home/matiasgp/Desktop/ABLA/analyzer/sam2/checkpoints/bacteria2.pt",
    #'model_config': "configs/sam2.1/sam2.1_hiera_b+.yaml"
}

# Results Settings
RESULTS_SETTINGS = {
    'results_dir': "/home/matiasgp/Desktop/data/annotations/",
}
