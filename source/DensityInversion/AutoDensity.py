import datetime
import os
from ..tools.SWIndices import get_current_kp_index, update_kp_ap_Ap_SN_F107
from ..tools.utilities import interpolate_positions, calculate_acceleration
from ..tools.Get_SP3_from_GFZ_FTP import download_sp3
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df, process_satellite_for_date_range
from .PODDensity import density_inversion

# Define paths and constants
BASE_DIR = "output/DensityInversion/PODDensityInversion/Data"
SATELLITES = ["GRACE-FO-A", "TerraSAR-X"]
Kp_THRESHOLD = 5

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
        file.write(f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}, {kp_index}\n")

# Function to check if the storm is still ongoing
def is_storm_ongoing(kp_history):
    now = datetime.datetime.now(datetime.timezone.utc)
    kp_history = [(time, kp) for time, kp in kp_history if now - time <= datetime.timedelta(hours=12)]
    if any(kp >= Kp_THRESHOLD for _, kp in kp_history):
        return True
    return False

# Function to verify if SP3 data has been downloaded successfully
def verify_sp3_download(satellite, storm_start_time, storm_end_time):
    sp3_codes = {
        "GRACE-FO-A": "L64",
        "TerraSAR-X": "L13",
    }
    
    sp3_code = sp3_codes.get(satellite)
    if not sp3_code:
        print(f"Unknown satellite code for {satellite}")
        return False
    
    current_date = storm_start_time
    while current_date <= storm_end_time:
        year = current_date.year
        day_of_year = current_date.strftime("%j")
        expected_directory = f"external/sp3_files/{sp3_code}/{year}/{day_of_year}"
        
        # Check if the directory exists and contains files
        if not os.path.exists(expected_directory) or not os.listdir(expected_directory):
            return False  # If any expected directory or files are missing, return False
        current_date += datetime.timedelta(days=1)
    
    return True  # All expected files are present

# Function to perform the workflow if a storm is ongoing
def handle_storm():
    for satellite in SATELLITES:
        storm_start_time, storm_name, kp_history = read_storm_status(satellite)
        
        if storm_start_time is None:
            storm_start_time = datetime.datetime.now(datetime.timezone.utc)
            storm_name = generate_storm_name(storm_start_time)
            update_storm_status(satellite, storm_start_time, get_current_kp_index())
        
        # Determine the start and end dates for SP3 data download
        storm_end_time = datetime.datetime.now(datetime.timezone.utc)
        
        # Check if SP3 data was successfully downloaded
        if not verify_sp3_download(satellite, storm_start_time, storm_end_time):
            print(f"SP3 data not found for {satellite}, attempting to download again...")
            download_sp3(storm_start_time, storm_end_time, satellite, orbit_type="NRT")
        
        # If download was successful, proceed with data processing
        if verify_sp3_download(satellite, storm_start_time, storm_end_time):
            storm_start_time_str = storm_start_time.strftime("%Y-%m-%d")
            storm_end_time_str = storm_end_time.strftime("%Y-%m-%d")
            print(f"storm start time: {storm_start_time_str}, storm end time: {storm_end_time_str}")
            print(f"satellite: {satellite}")
            process_satellite_for_date_range(satellite, storm_start_time_str, storm_end_time_str)

            sp3_ephem_df = sp3_ephem_to_df(satellite, storm_start_time_str)
            force_model_config = {'90x90gravity': True, '3BP': True, 'solid_tides': True,
                                  'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
            # Interpolate the ephemeris data to desired resolution (0.01S)
            print(f"head of sp3_ephem_df: {sp3_ephem_df.head()}")
            interp_ephemeris_df = interpolate_positions(sp3_ephem_df, '0.01S')
            # Numerically differentiate the interpolated ephemeris data to get acceleration
            velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
            # Perform density inversion
            print(f"length of velacc_ephem: {len(velacc_ephem)}")
            density_df = density_inversion(satellite, velacc_ephem, 'accx', 'accy', 'accz', force_model_config, models_to_query=[None], density_freq='360S')
            
            # Save or append density inversion result to the output file
            density_output_file = construct_file_path(satellite, storm_name, "density_estimates.csv")
            os.makedirs(os.path.dirname(density_output_file), exist_ok=True)
            if os.path.exists(density_output_file):
                density_df.to_csv(density_output_file, mode='a', header=False, index=False)
            else:
                density_df.to_csv(density_output_file, index=False)
        else:
            print(f"Failed to download SP3 data for {satellite}. Will retry in the next cycle.")

# Main script execution
def main():
    update_kp_ap_Ap_SN_F107()
    current_kp = get_current_kp_index()
    current_kp = 6  # For testing
    print(f"Current Kp index is {current_kp}")

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

#TODO: Delete the local SP3 files after the storm has ended (but keep the ephemeris)