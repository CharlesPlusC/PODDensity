import os
import glob
import pandas as pd

def create_and_submit_density_jobs():
    """
    Create and submit job scripts for EDR density retrieval for each spacecraft folder.

    Returns:
        None
    """
    # Automatically set directories
    user_home_dir = os.getenv("HOME")
    project_root_dir = f"{user_home_dir}/EDRDensity/PODDensity/"
    ephemerides_folder = f"{project_root_dir}/output/PODDensityInversion/Data/StormAnalysis"
    folder_for_jobs = f"{user_home_dir}/Scratch/EDR_in/sge_jobs"
    work_dir = f"{user_home_dir}/Scratch/EDR_in/working"
    logs_folder = f"{user_home_dir}/Scratch/EDR_in/logs"
    output_folder = f"{user_home_dir}/Scratch/EDR_in/output"

    # Create necessary directories
    os.makedirs(folder_for_jobs, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    spacecraft_folders = [f for f in os.listdir(ephemerides_folder) if os.path.isdir(os.path.join(ephemerides_folder, f))]
    print(f"spacecraft_folders: {spacecraft_folders}")
    job_count = 0

    for spacecraft in spacecraft_folders:
        spacecraft_folder = os.path.join(ephemerides_folder, spacecraft)
        print(f"spacecraft_folder: {spacecraft_folder}")
        # Find all CSV files for the spacecraft
        storm_files = glob.glob(os.path.join(spacecraft_folder, "*.csv"))
        if not storm_files:
            print(f"No CSV files found in {spacecraft_folder}.")
            continue

        # Create job script
        script_filename = os.path.join(folder_for_jobs, f"{spacecraft}_density_inversion.sh")
        script_content = f"""#!/bin/bash -l
#$ -l h_rt=0:5:0
#$ -l mem=4G
#$ -N {spacecraft}_density_inversion
#$ -t 1-{len(storm_files)}
#$ -wd {work_dir}
#$ -o {logs_folder}/{spacecraft}_$TASK_ID.out
#$ -e {logs_folder}/{spacecraft}_$TASK_ID.err

source $UCL_CONDA_PATH/etc/profile.d/conda.sh
module load python/miniconda3/4.10.3
conda activate pod_density_env

export PYTHONPATH=$PYTHONPATH:/home/zcesccc/EDRDensity/PODDensity/source

cd $TMPDIR

storm_file=$(ls {spacecraft_folder}/*.csv | sed -n "${{SGE_TASK_ID}}p")

python -c "
import pandas as pd
from source.EDRDensity import density_inversion_edr

# Load the storm file
ephemeris_df = pd.read_csv('$storm_file', parse_dates=['UTC'])
ephemeris_df.set_index('UTC', inplace=True)

# Run the density inversion
density_inversion_edr(
    sat_name='{spacecraft}',
    ephemeris_df=ephemeris_df,
    models_to_query=[None],
    freq='1S'
)
"
"""
        # Write the job script
        with open(script_filename, "w") as script_file:
            script_file.write(script_content)

        # Submit the job
        os.system(f"qsub {script_filename}")
        job_count += 1

    print(f"Submitted {job_count} jobs.")

if __name__ == "__main__":
    create_and_submit_density_jobs()
