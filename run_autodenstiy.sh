#!/bin/bash

# Define paths
SCRIPT_DIR="/home/zcesccc/PODDensity/"
LOG_DIR="/home/zcesccc/output/logs"
EMAIL="zcesccc@ucl.ac.uk"
VENV_DIR="/home/zcesccc/.conda/envs/pod_density_env/bin/python"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Log file for this run
LOG_FILE="$LOG_DIR/$(date +'%Y%m%d_%H%M%S').log"

# Activate virtual environment
source $VENV_DIR/bin/activate

# Run the Python script and redirect output to log file
cd $SCRIPT_DIR
python3 -m source.DensityInversion.AutoDensity > $LOG_FILE 2>&1

# Check if the script was successful
if [ $? -eq 0 ]; then
    STATUS="SUCCESS"
else
    STATUS="FAILURE"
fi

# Deactivate virtual environment
deactivate

# Send email notification
SUBJECT="Density Inversion Job - $STATUS"
MAIL_BODY="The density inversion job completed with status: $STATUS.\n\nLog file: $LOG_FILE\n\n-- Log Output --\n$(tail -n 100 $LOG_FILE)"
echo -e $MAIL_BODY | mail -s "$SUBJECT" $EMAIL
