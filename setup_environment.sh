#!/bin/bash

# Function to check if CUDA is available and get version
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA is available"
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
        echo "CUDA Version: $CUDA_VERSION"
        return 0
    else
        echo "CUDA is not available"
        return 1
    fi
}

# Function to check CUDA toolkit
check_cuda_toolkit() {
    if command -v nvcc &> /dev/null; then
        echo "CUDA toolkit is available"
        nvcc --version
        return 0
    else
        echo "CUDA toolkit is not available"
        echo "Please install CUDA toolkit that matches your PyTorch CUDA version"
        return 1
    fi
}

# Function to create and activate environment
setup_environment() {
    echo "Setting up environment..."
    
    # Ask for environment name
    default_name="abla_env"
    read -p "Enter environment name (default: $default_name): " env_name
    env_name=${env_name:-$default_name}
    
    # Ask for package manager preference
    while true; do
        read -p "Use mamba instead of conda? (y/n): " use_mamba
        case $use_mamba in
            [Yy]* )
                if ! command -v mamba &> /dev/null; then
                    echo "Mamba not found. Please install mamba first or use conda."
                    echo "You can install mamba with: conda install mamba -n base -c conda-forge"
                    exit 1
                fi
                pkg_manager="mamba"
                break;;
            [Nn]* )
                pkg_manager="conda"
                break;;
            * ) echo "Please answer y or n.";;
        esac
    done
    
    # Remove existing environment if it exists
    $pkg_manager env remove -n $env_name
    
    # Create new environment from yml file
    sed -i "1i name: $env_name" environment.yml
    $pkg_manager env create -f environment.yml
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    
    # Install SAM2 from official repository
    echo "Installing SAM2 from official repository..."
    if [ ! -d "analyzer/sam2" ]; then
        mkdir -p analyzer/sam2
        git clone https://github.com/facebookresearch/sam2.git analyzer/sam2
        cd analyzer/sam2
        pip install -e ".[notebooks]"
        cd ../..
    fi
    
    # Download SAM2 checkpoints
    echo "Downloading SAM2 checkpoints..."
    cd analyzer/sam2/checkpoints
    if [ ! -f "download_ckpts.sh" ]; then
        echo "Error: download_ckpts.sh not found"
        exit 1
    fi
    chmod +x download_ckpts.sh
    ./download_ckpts.sh
    cd ../../..
    
    # Verify PyTorch installation
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
}

# Main execution
echo "Setting up ABLA environment..."

# Check if running in WSL for Windows users
if grep -q Microsoft /proc/version; then
    echo "Running in Windows Subsystem for Linux (WSL)"
else
    echo "Running in native Linux/Unix environment"
fi

# Check for required tools
for cmd in conda git wget; do
    if ! command -v $cmd &> /dev/null; then
        echo "$cmd is required but not installed. Please install it first."
        exit 1
    fi
done

# Check CUDA availability
check_cuda

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "environment.yml not found in current directory"
    exit 1
fi

# Setup the environment
setup_environment

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate $env_name"