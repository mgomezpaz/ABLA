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
    if ! command -v mamba &> /dev/null; then
        echo "Mamba is not installed but highly recommended for faster installation."
        echo "Would you like to install mamba first? This will speed up the installation significantly."
        read -p "Install mamba? (y/n): " install_mamba
        case $install_mamba in
            [Yy]* )
                echo "Installing mamba..."
                conda install mamba -n base -c conda-forge -y
                pkg_manager="mamba"
                ;;
            [Nn]* )
                echo "Continuing with conda (slower). You can install mamba later with:"
                echo "conda install mamba -n base -c conda-forge"
                pkg_manager="conda"
                ;;
            * )
                echo "Invalid response. Continuing with conda."
                pkg_manager="conda"
                ;;
        esac
    else
        # Mamba is installed, recommend using it
        echo "Mamba is installed (recommended for faster installation)."
        pkg_manager="mamba"
    fi
    
    # Remove existing environment if it exists
    $pkg_manager env remove -n $env_name
    
    # Create new environment from yml file
    # Make a temporary copy of environment.yml
    cp environment.yml environment_temp.yml
    # Remove any existing name field
    sed -i '/^name:/d' environment_temp.yml
    # Add environment name to the temporary file
    sed -i "1i name: $env_name" environment_temp.yml
    
    # Create environment from temporary file with verbose output
    echo "Creating conda environment (this may take a few minutes)..."
    if [ "$pkg_manager" = "mamba" ]; then
        # For mamba, remove the --no-deps flag
        if ! timeout 1800 mamba env create -f environment_temp.yml --verbose; then
            echo "Environment creation timed out after 30 minutes"
            echo "Try running: conda clean -a"
            echo "Then try the installation again"
            rm environment_temp.yml
            exit 1
        fi
    else
        # For conda, keep the --no-deps flag
        if ! timeout 1800 conda env create -f environment_temp.yml --no-deps --verbose; then
            echo "Environment creation timed out after 30 minutes"
            echo "Try running: conda clean -a"
            echo "Then try the installation again"
            rm environment_temp.yml
            exit 1
        fi
    fi
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate $env_name
    
    # Install SAM2 from official repository
    echo "Installing SAM2 from official repository..."
    if [ ! -d "analyzer/sam2" ]; then
        mkdir -p analyzer/sam2
        git clone https://github.com/facebookresearch/sam2.git analyzer/sam2
        cd analyzer/sam2
        pip install -e ".[notebooks]" --no-deps
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
echo "Note: This installation process might take about 30 minutes. Please be patient."

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