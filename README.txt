ABLA Installation Guide
======================

Prerequisites
------------
- Python >= 3.10
- CUDA toolkit (recommended for GPU support)
- Git
- Wget
- Conda or Mamba package manager

Basic Installation
-----------------
1. Clone the repository: 
```
git clone https://github.com/yourusername/ABLA.git
cd ABLA
```
2. Run the setup script:
```
chmod +x setup_environment.sh
./setup_environment.sh
```

3. Install SAM2 in the correct location:
```bash
# Activate your environment
conda activate your_env_name

# Create and navigate to the SAM2 directory
mkdir -p analyzer/sam2
cd analyzer/sam2

# Clone and install SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[notebooks]"
```

4. Download SAM2 checkpoints:
```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

Alternatively, download individual checkpoints to `ABLA/analyzer/sam2/sam2/checkpoints/`:
- sam2.1_hiera_tiny.pt
- sam2.1_hiera_small.pt
- sam2.1_hiera_base_plus.pt
- sam2.1_hiera_large.pt

The script will:
- Ask for your preferred environment name
- Ask whether to use conda or mamba
- Set up the environment with all dependencies
- Configure ABLA for use

Dependencies
-----------
The installation will handle all required dependencies, including:
- PyTorch >= 2.3.1
- TorchVision >= 0.18.1
- SAM2 (Segment Anything Model 2)
- Other required Python packages

GPU Support
----------
For GPU support:
- CUDA toolkit must be installed and match your PyTorch CUDA version
- The setup script will detect CUDA availability
- If CUDA toolkit is missing, you'll be warned but can continue installation

Windows Users
------------
It's strongly recommended to use Windows Subsystem for Linux (WSL) with Ubuntu for installation.

SAM2 Integration Notes
---------------------
- SAM2 requires Python >= 3.10
- CUDA toolkit must match your PyTorch CUDA version
- If you see "Failed to build the SAM 2 CUDA extension" during installation:
  - This is not critical
  - SAM2 will still work
  - Some post-processing functionality may be limited
  - Core functionality remains unaffected

Model Checkpoints
----------------
The setup script will automatically download required checkpoints. If you need to manually download or update checkpoints:

1. Navigate to the checkpoints directory:
   ```bash
   cd ABLA/analyzer/sam2/checkpoints
   ```

2. Run the download script:
   ```bash
   ./download_ckpts.sh
   ```

This will download:
- sam2.1_hiera_large.pt (SAM2 model weights)

Note: Make sure you have sufficient disk space (~2GB) for the checkpoint files.

Troubleshooting
--------------
1. If mamba is selected but not found:
   ```
   conda install mamba -n base -c conda-forge
   ```

2. If CUDA toolkit is missing:
   - Install CUDA toolkit matching your PyTorch version
   - Reinstall if needed

3. For WSL users experiencing issues:
   - Ensure WSL2 is installed
   - Install Ubuntu from Microsoft Store
   - Follow Linux installation steps within WSL

4. If PyTorch installation fails:
   - The script will attempt to install the CPU version
   - You can manually install GPU version later:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

5. If you see "No module named 'sam2.sam2_video_predictor'" error:
   - Check SAM2 installation directory structure
   - Ensure PYTHONPATH includes SAM2 directory:
     ```bash
     export PYTHONPATH=$PYTHONPATH:path/to/ABLA/analyzer/sam2
     ```
   - Reinstall SAM2 following steps above

Verification
-----------
After installation, verify setup:
1. Activate environment:
   ```
   conda activate your_env_name
   ```

2. Run Python and verify imports:
   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   ```

3. Test ABLA:
   ```python
   from ABLA.abla import main
   ```

Additional Notes
--------------
- Keep your GPU drivers updated
- For production use, consider using specific version numbers in environment.yml
- Regular updates may be required for security and performance improvements

Support
-------
For issues:
1. Check the troubleshooting section
2. Verify all prerequisites are met
3. Create an issue on the GitHub repository with:
   - Full error message
   - System information
   - CUDA version (if applicable)
   - Installation method used

Directory Structure
------------------
After installation, your ABLA directory should look like this:
```
ABLA/
├── analyzer/
│   └── sam2/
│       └── sam2/           # SAM2 repository
│           ├── checkpoints/
│           │   └── ...     # Model weights
│           └── ...         # SAM2 source files
├── environment.yml
├── setup_environment.sh
└── ... other ABLA files
```

Make sure SAM2 is installed in the correct location as shown above. This structure is required for ABLA to properly integrate with SAM2.

Future Improvements
------------------
The following parts of the project could be enhanced or completed:

1. Environment Setup
   - Add support for different Python versions
   - Include automated CUDA toolkit installation
   - Add validation steps for environment setup

2. Documentation
   - Add detailed API documentation
   - Include more usage examples
   - Create troubleshooting guide for common issues

3. Testing
   - Add unit tests
   - Create integration tests
   - Implement automated testing workflow

4. Features
   - Add support for additional model architectures
   - Implement batch processing capabilities
   - Create visualization tools

5. Performance
   - Optimize memory usage
   - Improve processing speed
   - Add multi-GPU support

6. User Interface
   - Create command-line interface
   - Develop web interface
   - Add progress bars for long operations

If you'd like to contribute to any of these improvements, please submit a pull request or open an issue for discussion.

Acknowledgments
--------------
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}