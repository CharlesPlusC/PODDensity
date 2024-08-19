#!/bin/bash
echo "Running script: $0"

# Define paths
SCRIPT_DIR="/home/zcesccc/PODDensity/"
LOG_DIR="/home/zcesccc/output/logs"
EMAIL="zcesccc@ucl.ac.uk"
VENV_NAME="pod_density_env"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Log file for this run
LOG_FILE="$LOG_DIR/$(date +'%Y%m%d_%H%M%S').log"

# Ensure conda is initialized
source /shared/ucl/apps/miniconda/4.10.3/etc/profile.d/conda.sh

# Activate the virtual environment
conda activate $VENV_NAME

# Run the Python script and redirect output to log file
cd $SCRIPT_DIR
python -m source.DensityInversion.AutoDensity > $LOG_FILE 2>&1

# Check if the script was successful
if [ $? -eq 0 ]; then
    STATUS="SUCCESS"
else
    STATUS="FAILURE"
fi

# Deactivate the virtual environment
conda deactivate

# Send email notification using sendmail
SUBJECT="Density Inversion Job - $STATUS"
MAIL_BODY="The density inversion job completed with status: $STATUS.\n\nLog file: $LOG_FILE\n\n-- Log Output --\n$(tail -n 100 $LOG_FILE)"
echo -e "Subject: $SUBJECT\n\n$MAIL_BODY" | /usr/sbin/sendmail
