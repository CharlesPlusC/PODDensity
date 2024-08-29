import datetime
import os
import pandas as pd
import orekit
from orekit.pyhelpers import setup_orekit_curdir
# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir()
from ..tools.SWIndices import get_current_kp_index, update_kp_ap_Ap_SN_F107
from ..tools.utilities import interpolate_positions, calculate_acceleration
from ..tools.Get_SP3_from_GFZ_FTP import download_sp3
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df, process_satellite_for_date_range
from source.DensityInversion.Plotting.PODDensityPlotting import plot_arglat_density_and_kp, plot_density_and_kp
from .PODDensity import density_inversion

# Define paths and constants
BASE_DIR = "output/DensityInversion/PODDensityInversion/Data"
SATELLITES = ["GRACE-FO-A", "TerraSAR-X"]
Kp_THRESHOLD = 5
STORM_END_DELAY_HOURS = 18

# Function to generate storm name based on the start date
def generate_storm_name(start_time):
    return start_time.strftime("%d-%m-%Y")

# Function to construct file paths
def construct_file_path(satellite, storm_name, filename):
    return os.path.join(BASE_DIR, satellite, storm_name, filename)

# Function to check if storm status file exists and read the status
def read_storm_status(satellite):
    storm_status_file = construct_file_path(satellite, "", "storm_status.txt")
    if os.path.exists(storm_status_file):
        with open(storm_status_file, "r") as file:
            data = file.readlines()
            start_time_str = data[0].strip()
            kp_history = [(datetime.datetime.strptime(line.split(',')[0].strip(), "%Y-%m-%d %H:%M:%S"),
                           float(line.split(',')[1].strip())) for line in data[1:]]
            start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
            storm_name = generate_storm_name(start_time)
            return start_time, storm_name, kp_history
    return None, None, []

# Function to update storm status file with current Kp index
def update_storm_status(satellite, start_time, kp_index):
    storm_name = generate_storm_name(start_time)
    storm_status_file = construct_file_path(satellite, "", "storm_status.txt")
    os.makedirs(os.path.dirname(storm_status_file), exist_ok=True)
    with open(storm_status_file, "a") as file:
        if os.path.getsize(storm_status_file) == 0:
            file.write(f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}, {kp_index}\n")

# Function to check if the storm is still ongoing
def is_storm_ongoing(kp_history):
    now = datetime.datetime.now(datetime.timezone.utc)
    # Filter the Kp history to only include the last 18 hours
    recent_history = [(time, kp) for time, kp in kp_history if now - time <= datetime.timedelta(hours=STORM_END_DELAY_HOURS)]
    
    # Check if Kp has been above the threshold in the recent history
    if any(kp >= Kp_THRESHOLD for _, kp in recent_history):
        return True
    
    # Check if the last Kp measurement was above the threshold and within the delay period
    if recent_history:
        last_time, last_kp = recent_history[-1]
        if last_kp < Kp_THRESHOLD and (now - last_time).total_seconds() / 3600 <= STORM_END_DELAY_HOURS:
            return True
    
    return False

# Function to find the last timestamp of processed density data
def find_last_density_timestamp(density_output_file):
    if os.path.exists(density_output_file):
        density_df = pd.read_csv(density_output_file)
        if not density_df.empty:
            return pd.to_datetime(density_df['UTC'].max())
    return None

# Function to append new density data without duplications
def append_density_data(density_df, density_output_file):
    if os.path.exists(density_output_file):
        existing_df = pd.read_csv(density_output_file, parse_dates=['UTC'])
        combined_df = pd.concat([existing_df, density_df]).drop_duplicates(subset=['UTC']).sort_values(by='UTC')
        combined_df.to_csv(density_output_file, index=False)
    else:
        #make sure the directory exists
        os.makedirs(os.path.dirname(density_output_file), exist_ok=True)
        density_df.to_csv(density_output_file, index=False)

# Function to perform the workflow if a storm is ongoing
def handle_storm():
    for satellite in SATELLITES:
        storm_start_time, storm_name, kp_history = read_storm_status(satellite)
        
        if storm_start_time is None:
            storm_start_time = datetime.datetime.now(datetime.timezone.utc)
            storm_name = generate_storm_name(storm_start_time)
            update_storm_status(satellite, storm_start_time, get_current_kp_index())
        
        # Determine the start and end dates for SP3 data processing
        storm_end_time = datetime.datetime.now(datetime.timezone.utc)
        #add one day to the storm end time to ensure all data in that range is processed
        storm_end_time += datetime.timedelta(days=1)
        
        # Process SP3 data for the satellite within the storm date range
        process_satellite_for_date_range(satellite, storm_start_time.strftime("%Y-%m-%d"), storm_end_time.strftime("%Y-%m-%d"))
        
        # Determine the output file path for the density estimates
        density_output_file = construct_file_path(satellite, storm_name, "density_estimates.csv")
        last_density_time = find_last_density_timestamp(density_output_file)
        
        sp3_ephem_df = sp3_ephem_to_df(satellite, storm_start_time.strftime("%Y-%m-%d"))
        force_model_config = {'90x90gravity': True, '3BP': True, 'solid_tides': True,
                              'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
        
        # Filter the ephemeris data to only include times after the last computed density
        if last_density_time is not None:
            sp3_ephem_df = sp3_ephem_df[sp3_ephem_df['UTC'] > last_density_time]
        
        #check it is not empty and it is longer than 3
        if not sp3_ephem_df.empty and len(sp3_ephem_df) > 3:
            interp_ephemeris_df = interpolate_positions(sp3_ephem_df, '0.01S')
            # Numerically differentiate the interpolated ephemeris data to get acceleration
            velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
            # Perform density inversion
            density_df = density_inversion(satellite, velacc_ephem, 'accx', 'accy', 'accz', force_model_config, models_to_query=[None], density_freq='60S') # The frequency you select will drastically change compute time.
            
            # Append the new density data while avoiding duplicates
            append_density_data(density_df, density_output_file)
            
            # Load the density estimates CSV into a DataFrame
            density_estimates_df = pd.read_csv(density_output_file, parse_dates=['UTC'])
            # Generate the path to the storm folder to save the plot to
            arglat_plot_path_storm_folder = construct_file_path(satellite, storm_name, "arglat_latest_plot")
            plot_arglat_density_and_kp([density_estimates_df], moving_avg_minutes=45, sat_name=satellite, save_path=arglat_plot_path_storm_folder)
            lineplot_path_storm_folder = construct_file_path(satellite, storm_name, "lineplot_latest_plot")
            plot_density_and_kp([density_estimates_df], moving_avg_minutes=45, sat_name=satellite, save_path=lineplot_path_storm_folder)
        else:
            print(f"No new data to process for {satellite} since last density computation.")

# Main script execution
def main():
    update_kp_ap_Ap_SN_F107()
    current_kp = get_current_kp_index()
    print(f"Current Kp index is {current_kp}")
    # current_kp = 5.3 # For testing purposes

    if current_kp >= Kp_THRESHOLD:
        handle_storm()
    else:
        for satellite in SATELLITES:
            storm_start_time, storm_name, kp_history = read_storm_status(satellite)
            if storm_start_time and not is_storm_ongoing(kp_history):
                # Storm has ended, so clear the storm status
                storm_status_file = construct_file_path(satellite, "", "storm_status.txt")
                if os.path.exists(storm_status_file):
                    os.remove(storm_status_file)
                print(f"Storm '{storm_name}' has ended for {satellite}. Status file removed.")

if __name__ == "__main__":
    main()
