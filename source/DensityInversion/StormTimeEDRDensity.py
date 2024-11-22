#Files to run the density inversion for the storms outlined in the selected_storms_v1.txt file
# the select_storms() function in SWIndices.py is used to select the storms

from ..tools.Get_SP3_from_GFZ_FTP import download_sp3
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df
from .EDRDensity import density_inversion_edr
from ..tools.utilities import interpolate_positions, calculate_acceleration
import os
import json
import ftplib
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

def load_sp3_codes(json_path):
    """
    Load SP3 codes from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing SP3 codes.

    Returns:
        dict: A dictionary containing SP3 codes.
    """
    with open(json_path, 'r') as file:
        return json.load(file)

def download_files(ftp_server, path, local_directory):
    """
    Download files from an FTP server.

    Args:
        ftp_server (str): FTP server address.
        path (str): Path on the FTP server to download files from.
        local_directory (str): Local directory to save downloaded files.

    Returns:
        None
    """
    try:
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login()
            ftp.cwd(path)
            files = ftp.nlst()
            for filename in files:
                local_path = os.path.join(local_directory, filename)
                with open(local_path, 'wb') as local_file:
                    ftp.retrbinary('RETR ' + filename, local_file.write)
            print(f"Downloaded files to {local_directory}")
    except ftplib.all_errors as e:
        print(f"FTP error: {e}")

def main():
    """
    Main function to process storm time ephemeris data and perform density inversion.

    Args:
        None

    Returns:
        None
    """
    force_model_config = {
        '90x90gravity': True, '3BP': True, 'solid_tides': True,
        'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True
    }

    storm_data = download_storm_time_ephems()
    storm_ephem_data = load_storm_sp3(storm_data)
    seen_identifiers = set()

    for satellite, data_groups in storm_ephem_data.items():
        new_df_list = []
        for df_group in data_groups:
            filtered_group = filter_non_empty_dataframes(df_group)
            if filtered_group:  # Only add non-empty groups
                new_df_list.append(filtered_group)
        storm_ephem_data[satellite] = new_df_list

    # for satellite, df_list in storm_ephem_data.items():
    #same line as above but using tqdm to show progress
    for satellite, df_list in tqdm(storm_ephem_data.items(), desc="Density Inversion"):
        print(f"Processing {satellite}")
        for storm_period_index, df_period in enumerate(df_list):
            print(f"Processing storm period {storm_period_index} for {satellite}")
            for storm_df_index, storm_df in enumerate(df_period):
                if not storm_df.empty:
                    identifier = tuple(storm_df['UTC'].tolist())
                    if identifier in seen_identifiers:
                        print(f"Duplicate detected: Skipping storm {storm_df} for {satellite}")
                        continue
                    seen_identifiers.add(identifier)
                    print(f"Processing storm {storm_df_index} for {satellite}")
                    print(f"storm start: {storm_df['UTC'].iloc[0]}")
                    print(f"storm end: {storm_df['UTC'].iloc[-1]}")
                    interp_ephemeris_df = interpolate_positions(storm_df, '0.01S')
                    velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
                
                    density_inversion_df = density_inversion("GRACE-FO-A", velacc_ephem, 'vel_acc_x', 'vel_acc_y', 'vel_acc_z', force_model_config, nc_accs=False, 
                                models_to_query=['JB08', 'DTM2000', "NRLMSISE00"], density_freq='15S')

                    datenow = datetime.now().strftime("%Y%m%d%H%M%S")
                    savepath = f"output/DensityInversion/PODDensityInversion/Data/StormAnalysis/{satellite}"
                    os.makedirs(savepath, exist_ok=True)
                    output_path = savepath + f"/{satellite}_storm_density_{storm_period_index}_{datenow}.csv"
                    density_inversion_df.to_csv(output_path)
                    print(f"Data saved to {output_path}")
                else:
                    print(f"Skipping processing for {satellite} storm {storm_df_index} due to empty DataFrame.")

def create_and_submit_density_jobs():
    """
    Create and submit density inversion jobs to a job scheduler.

    Args:
        None

    Returns:
        None
    """
    import os
    import json

    user_home_dir = os.getenv("HOME")
    project_root_dir = f"{user_home_dir}/Rhoin/POD-Density-Inversion/"
    folder_for_jobs = f"{user_home_dir}/Scratch/Rhoin/sge_jobs"
    work_dir = f"{user_home_dir}/Scratch/Rhoin/working"
    logs_folder = f"{user_home_dir}/Scratch/Rhoin/logs"
    output_folder = f"{user_home_dir}/Scratch/Rhoin/output"

    os.makedirs(folder_for_jobs, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    storm_data = download_storm_time_ephems()
    storm_ephem_data = load_storm_sp3(storm_data)

    index = 0
    for satellite, periods in storm_ephem_data.items():
        for period_index, df_group in enumerate(periods):
            if df_group:
                script_filename = f"{folder_for_jobs}/{satellite}_period{period_index}.sh"
                script_content = f"""#!/bin/bash -l
#$ -l h_rt=24:0:0
#$ -l mem=8G
#$ -l tmpfs=15G
#$ -N {satellite}_period{period_index}
#$ -t 1-{len(df_group)}
#$ -wd {work_dir}
#$ -o {logs_folder}/out_{satellite}_period{period_index}_$TASK_ID.txt
#$ -e {logs_folder}/err_{satellite}_period{period_index}_$TASK_ID.txt

module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate pod_density_env
export PYTHONPATH="{project_root_dir}:$PYTHONPATH"

cp -r {user_home_dir}/Rhoin/EDR_Density_Inversion $TMPDIR

cd $TMPDIR/POD-Density-Inversion

/home/{os.getenv('USER')}/.conda/envs/POD_Density_Inversion/bin/python -m source.DensityInversion.StormTimeDensity {satellite} {period_index} $SGE_TASK_ID {output_folder}
"""
                with open(script_filename, 'w') as file:
                    file.write(script_content)

                os.system(f"qsub {script_filename}")
                index += 1

def main_script(satellite, period_index, df_index, output_folder):
    """
    Main script to process a specific storm period and perform density inversion.

    Args:
        satellite (str): Satellite name.
        period_index (int): Index of the storm period.
        df_index (int): Index of the DataFrame within the storm period.
        output_folder (str): Folder to save the output data.

    Returns:
        None
    """
    from datetime import datetime

    storm_data = download_storm_time_ephems()
    storm_ephem_data = load_storm_sp3(storm_data)

    df_list = storm_ephem_data[satellite][int(period_index)]
    if 0 <= int(df_index) - 1 < len(df_list):
        df = df_list[int(df_index) - 1]
        if not df.empty:
            force_model_config = {
                '90x90gravity': True, '3BP': True, 'solid_tides': True,
                'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True
            }

            interp_ephemeris_df = interpolate_positions(df, '0.01S')
            velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
            
            density_inversion_df = density_inversion(satellite, velacc_ephem, 'vel_acc_x', 'vel_acc_y', 'vel_acc_z', force_model_config, nc_accs=False, 
                        models_to_query=[None], density_freq='15S')

            datenow = datetime.now().strftime("%Y%m%d%H%M%S")
            savepath = f"{output_folder}/StormAnalysis/{satellite}"
            os.makedirs(savepath, exist_ok=True)
            output_path = savepath + f"/{satellite}_storm_density_{period_index}_{df_index}_{datenow}.csv"
            density_inversion_df.to_csv(output_path)
            print(f"Data saved to {output_path}")
        else:
            print(f"Skipping processing for {satellite} storm {df_index} due to empty DataFrame.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5:
        main_script(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        create_and_submit_density_jobs()