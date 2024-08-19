#!/bin/bash
echo "Running script: $0"

# Ensure conda is initialized
source /shared/ucl/apps/miniconda/4.10.3/etc/profile.d/conda.sh

# Activate the virtual environment
conda activate pod_density_env

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Environment activated successfully."
else
    echo "Failed to activate the environment."
    exit 1
fi

# Deactivate the virtual environment
conda deactivate
