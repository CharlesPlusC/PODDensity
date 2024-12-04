import os
import sys
import pandas as pd
import glob
from source.DensityInversion.EDRDensity import density_inversion_edr

def run_density_inversion(storm_file, satellite):
    """
    Run the density inversion for a given storm file and satellite.

    Args:
        storm_file (str): Path to the storm CSV file.
        satellite (str): Satellite name.

    Returns:
        None
    """
    try:
        print(f"Running density inversion for storm file: {storm_file}, satellite: {satellite}")
        
        ephemeris_df = pd.read_csv(storm_file)
        
        # Define the save folder
        save_folder = f"/home/zcesccc/Scratch/EDR_in/output/{satellite}/"
        os.makedirs(save_folder, exist_ok=True)

        # Run the density inversion
        density_inversion_edr(
            sat_name=satellite,
            ephemeris_df=ephemeris_df,
            models_to_query=[None],
            freq='1S',
            save_folder=save_folder
        )
        print(f"Density inversion completed for {satellite}. Results saved to {save_folder}.")
    except Exception as e:
        print(f"Error during density inversion for {satellite}: {e}")
        raise

def create_and_submit_density_jobs(spacecraft_to_do=None):
    """
    Create and submit job scripts for EDR density retrieval for each spacecraft folder.

    Returns:
        None
    """
    # Automatically set directories
    user_home_dir = os.getenv("HOME")
    project_root_dir = f"{user_home_dir}/PODDensity/"
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

    # Find spacecraft folders and filter for CHAMP only
    if spacecraft_to_do:
        spacecraft_folders = [
            f for f in os.listdir(ephemerides_folder)
            if os.path.isdir(os.path.join(ephemerides_folder, f)) and spacecraft_to_do in f
        ]
    else:
        spacecraft_folders = [
            f for f in os.listdir(ephemerides_folder)
            if os.path.isdir(os.path.join(ephemerides_folder, f))
        ]
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
#$ -l h_rt=30:00:0
#$ -l mem=8G
#$ -l tmpfs=8G
#$ -N {spacecraft}_density_inversion
#$ -t 1-{len(storm_files)}
#$ -wd {work_dir}
#$ -o {logs_folder}/{spacecraft}_$TASK_ID.out
#$ -e {logs_folder}/{spacecraft}_$TASK_ID.err

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate pod_density_env
export PYTHONPATH="{project_root_dir}:$PYTHONPATH"

cp -r {user_home_dir}/PODDensity/ $TMPDIR

# Navigate to the copied directory
cd $TMPDIR/PODDensity/

storm_file=$(ls {spacecraft_folder}/*.csv | sed -n "${{SGE_TASK_ID}}p")

/home/{os.getenv('USER')}/.conda/envs/pod_density_env/bin/python -m source.DensityInversion.StormTimeEDRDensity {spacecraft} "$storm_file"
"""
        # Write the job script
        with open(script_filename, "w") as script_file:
            script_file.write(script_content)

        # Submit the job
        os.system(f"qsub {script_filename}")
        job_count += 1

    print(f"Submitted {job_count} jobs.")

def main_script(satellite, storm_file):
    """
    Main function to process a specific storm file and run density inversion.

    Args:
        satellite (str): Satellite name.
        storm_file (str): Path to the storm CSV file.

    Returns:
        None
    """
    run_density_inversion(storm_file, satellite)

if __name__ == "__main__":
    # Command-line argument handling
    if len(sys.argv) == 3:
        satellite = sys.argv[1]
        storm_file = sys.argv[2]
        main_script(satellite, storm_file)
    else:
        # Default action: create and submit jobs
        create_and_submit_density_jobs(spacecraft_to_do=None) #pass the name of the spacecraft to do, e.g. "CHAMP" otherwise it will do them all
