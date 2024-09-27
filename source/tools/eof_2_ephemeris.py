#read in .EOF data to pandas dataframe
#convert pos/vels to EME2000
# Write out ephemeris to .txt file 
# 2001-04-11 09:59:47.000000 -4015.0921448672484 -4840.464357089205 -2691.41391220233 2.2188680290067504 2.057530378290252 -6.992561959392307
# 0.5 0.5 0.5 0.001 0.001 0.001

#TODO: use the Sentinel API to get the ephemeris data. Currently I'm just getting it from the Coperincus website :) 

import os
import pandas as pd
import json
import xml.etree.ElementTree as ET
from ..tools.utilities import SP3_to_EME2000, utc_to_mjd

# Define the root folder where the EOF files are stored
root_folder = 'external/eof_files/'

# Define the list of spacecraft directories to process (e.g., S1A, S2A, etc.)
spacecraft_folders = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
print(f"Spacecraft folders: {spacecraft_folders}")

# Function to parse the EOF XML file and extract ephemeris data
def parse_eof_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    ephemeris_data = []
    # Find all OSV elements
    for osv in root.findall(".//OSV"):
        utc = osv.find("UTC").text.split("=")[-1]
        x = float(osv.find("X").text)
        y = float(osv.find("Y").text)
        z = float(osv.find("Z").text)
        vx = float(osv.find("VX").text)
        vy = float(osv.find("VY").text)
        vz = float(osv.find("VZ").text)

        ephemeris_data.append([utc, x, y, z, vx, vy, vz])

    return pd.DataFrame(ephemeris_data, columns=['UTC', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ'])

# Function to check for time gaps greater than 61 seconds and split the dataframe if necessary
def check_and_split_time_gaps(df, spacecraft):
    df['UTC'] = pd.to_datetime(df['UTC'])  # Convert UTC to datetime objects
    df.sort_values(by='UTC', inplace=True)  # Sort the dataframe by UTC

    time_diffs = df['UTC'].diff().dt.total_seconds().fillna(0)  # Calculate time differences

    # Check if any time difference is greater than 61 seconds
    if (time_diffs > 61).any():
        gap_indices = time_diffs[time_diffs > 61].index  # Find indices where gaps occur
        print(f"Time gaps greater than 61 seconds found in {spacecraft}. Splitting the dataframe.")
        
        # Split the dataframe at the gap indices and return as separate dataframes
        split_dfs = []
        previous_index = 0
        for gap_index in gap_indices:
            split_dfs.append(df.iloc[previous_index:gap_index])
            previous_index = gap_index

        split_dfs.append(df.iloc[previous_index:])  # Append the remaining part of the dataframe
        return split_dfs
    else:
        return [df]  # If no gaps, return the dataframe as a single list

def write_ephemeris_file(file_name, df, satellite_info, output_dir="external/ephems"):
    sat_dir = os.path.join(output_dir, file_name.split('_')[0])
    os.makedirs(sat_dir, exist_ok=True)

    start_day = df.index.min().strftime("%Y-%m-%d")
    end_day = df.index.max().strftime("%Y-%m-%d")
    norad_id = satellite_info['norad_id']
    file_name = f"NORAD{norad_id}-{start_day}-{end_day}.txt"
    file_path = os.path.join(sat_dir, file_name)

    with open(file_path, 'w') as file:
        for idx, row in df.iterrows():
            utc = idx.strftime("%Y-%m-%d %H:%M:%S.%f")
            line1 = f"{utc} {row['pos_x_eci']} {row['pos_y_eci']} {row['pos_z_eci']} {row['vel_x_eci']} {row['vel_y_eci']} {row['vel_z_eci']}\n"
            line2 = f"{row['sigma_x']} {row['sigma_y']} {row['sigma_z']} {row['sigma_xv']} {row['sigma_yv']} {row['sigma_zv']}\n"
            file.write(line1)
            file.write(line2)

# Function to convert from ITRS (SP3) to EME2000 and add error covariances
def convert_to_eme2000_and_write(ephemeris_dataframes, satellite):

    sat_list_path="misc/sat_list.json"
    
    with open(sat_list_path, 'r') as file:
        sat_dict = json.load(file)

    for df_index, df in enumerate(ephemeris_dataframes):
        print(f"Processing {satellite}_{df_index}")
        
        # Ensure index is in datetime format
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)


        # Convert UTC to MJD for conversion
        mjd_times = [utc_to_mjd(dt) for dt in df['UTC']]
        df['MJD'] = mjd_times

        # Extract ITRS (SP3) positions and velocities
        itrs_positions = df[['X', 'Y', 'Z']].values
        itrs_velocities = df[['VX', 'VY', 'VZ']].values
        #divide by 1000 to convert to km
        itrs_positions /= 1000
        itrs_velocities /= 1000

        # Convert from ITRS (SP3) to EME2000
        print(f"Converting ITRS to EME2000 for {satellite}_{df_index}")
        eme2000_positions, eme2000_velocities = SP3_to_EME2000(itrs_positions, itrs_velocities, df['MJD'])

        # Add converted positions and velocities to the dataframe
        df['pos_x_eci'], df['pos_y_eci'], df['pos_z_eci'] = eme2000_positions.T
        df['vel_x_eci'], df['vel_y_eci'], df['vel_z_eci'] = eme2000_velocities.T

        # Add sigma error values (assumed constants)
        df['sigma_x'] = 5e-1  # in km
        df['sigma_y'] = 5e-1  # in km
        df['sigma_z'] = 5e-1  # in km
        df['sigma_xv'] = 1e-3 # in km/s
        df['sigma_yv'] = 1e-3 # in km/s
        df['sigma_zv'] = 1e-3 # in km/s

        # Write ephemeris to file
        file_name = f"{satellite}_{df_index}"
        print(f"Writing ephemeris file for {file_name}")
        #make UTC the index
        df.set_index('UTC', inplace=True)
        write_ephemeris_file(file_name, df, sat_dict[satellite])

# Dictionary to store dataframes for each spacecraft
spacecraft_dataframes = {}

# Iterate over all spacecraft folders
for spacecraft in spacecraft_folders:
    spacecraft_folder = os.path.join(root_folder, spacecraft)
    all_ephemeris = pd.DataFrame()

    # Iterate over all EOF files in the spacecraft folder
    for file_name in os.listdir(spacecraft_folder):
        if file_name.endswith('.EOF'):
            file_path = os.path.join(spacecraft_folder, file_name)
            # Parse each EOF file and append to the combined dataframe
            ephemeris_df = parse_eof_file(file_path)
            all_ephemeris = pd.concat([all_ephemeris, ephemeris_df])

    # Remove overlapping times by keeping only the first occurrence of each UTC time
    all_ephemeris.drop_duplicates(subset='UTC', keep='first', inplace=True)

    # Check for gaps greater than 61 seconds and split the dataframe if necessary
    split_ephemerides = check_and_split_time_gaps(all_ephemeris, spacecraft)

    # Store the resulting dataframes for the spacecraft
    spacecraft_dataframes[spacecraft] = split_ephemerides

# Apply conversion for each spacecraft's ephemeris dataframes
for spacecraft, dfs in spacecraft_dataframes.items():
    convert_to_eme2000_and_write(dfs, spacecraft)


# # Example: To access the dataframe for S1A (Sentinel-1A)
# s1a_ephemeris_dfs = spacecraft_dataframes.get('S1A')

# # You can display or save the dataframes, for example:
# for spacecraft, dfs in spacecraft_dataframes.items():
#     for i, df in enumerate(dfs):
#         print(f"Spacecraft: {spacecraft}, Ephemeris Segment {i+1}")
#         print(df.head())  # Show first few rows of the dataframe

#         # Plot the norm of the position vector for each segment
#         df['R'] = (df['X']**2 + df['Y']**2 + df['Z']**2)**0.5 / 1000  # Convert to kilometers
#         import matplotlib.pyplot as plt
#         plt.plot(df['UTC'], df['R'])
#         plt.xlabel('Time')
#         plt.ylabel('Position Norm (km)')
#         plt.title(f"{spacecraft} Position Norm (Segment {i+1})")
#         plt.show()