#!/bin/bash

# Define the environment name
ENV_NAME="unico"
ENV_FILE="environment.yml"

# Create the conda environment if it doesn't exist
if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment ${ENV_NAME}..."
    conda env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

# Activate the conda environment
echo "Activating environment ${ENV_NAME}..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# ---- Build chamfer_dist extension ----
echo ">>> Building chamfer_dist extension..."
cd extensions/chamfer_dist || { echo "Directory not found"; exit 1; }

python setup.py install
echo ">>> chamfer_dist built successfully!"

# ---- Build pointnet2_ops_lib extension ----
echo ">>> Building pointnet2_ops_lib extension..."
cd ../pointnet2_ops_lib || { echo "Directory not found"; exit 1; }

python setup.py install
echo ">>> pointnet2_ops_lib built successfully!"