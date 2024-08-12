# for all the satellites listed in main, this script will take the sp3 files in sp3_files, and concatenate them into a single dataframe per spacecraft. 
# the datatframe will then be used to write out an ephemeris file for each spacecraft that will be saved in the specified directory.

import json
import pandas as pd
import sp3
import gzip
import tempfile
from datetime import datetime
import os
import glob
from ..tools.utilities import SP3_to_EME2000, utc_to_mjd
#run from CLI from root using: python source/tools/sp3_2_ephemeris.py

def read_sp3_gz_file(sp3_gz_file_path):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.sp3')
    temp_file_path = temp_file.name

    with gzip.open(sp3_gz_file_path, 'rb') as gz_file:
        temp_file.write(gz_file.read())
    temp_file.close()

    print(f"reading sp3 file: {sp3_gz_file_path}")

    product = sp3.Product.from_file(temp_file_path)
    
    satellite = product.satellites[0]
    records = satellite.records

    times = []
    positions = []
    velocities = []

    for record in records:
        times.append(record.time)
        positions.append(record.position)
        velocities.append(record.velocity)

    df = pd.DataFrame({
        'Time': times,
        'Position_X': [pos[0]/1000 for pos in positions],
        'Position_Y': [pos[1]/1000 for pos in positions],
        'Position_Z': [pos[2]/1000 for pos in positions],
        'Velocity_X': [vel[0]/1000 for vel in velocities],
        'Velocity_Y': [vel[1]/1000 for vel in velocities],
        'Velocity_Z': [vel[2]/1000 for vel in velocities]
    })

    print(f"Read {len(df)} records from {temp_file_path}")
    os.remove(temp_file_path)
    return df

def process_sp3_files(base_path, sat_list):
    all_dataframes = {sat_name: [] for sat_name in sat_list}

    for sat_name, sat_info in sat_list.items():
        sp3_c_code = sat_info['sp3-c_code']
        satellite_path = os.path.join(base_path, sp3_c_code)
        for year_folder in glob.glob(f"{satellite_path}/*"):
            for day_folder in glob.glob(f"{year_folder}/*"):
                for sp3_gz_file in glob.glob(f"{day_folder}/*.sp3.gz"):
                    df = read_sp3_gz_file(sp3_gz_file)
                    df['Time'] = pd.to_datetime(df['Time'])  # Ensure 'Time' is in datetime format
                    all_dataframes[sat_name].append(df)

    contiguous_dataframes = {}
    for sat_name, dfs in all_dataframes.items():
        if dfs:
            concatenated_df = pd.concat(dfs).drop_duplicates(subset='Time').set_index('Time').sort_index()
            
            date_diffs = concatenated_df.index.to_series().diff().dt.total_seconds() > 1800  # 30 minutes
            split_points = concatenated_df[date_diffs].index
            
            grouped_dfs = []
            last_idx = concatenated_df.index[0] if not split_points.empty else None
            
            for idx in split_points:
                current_df = concatenated_df.loc[last_idx:idx - pd.Timedelta(seconds=1)]
                grouped_dfs.append(current_df)
                last_idx = idx
                
            if last_idx is not None and last_idx != concatenated_df.index[-1]:
                grouped_dfs.append(concatenated_df.loc[last_idx:])
            elif split_points.empty:
                # When there are no split points, the whole concatenated_df should be added.
                grouped_dfs.append(concatenated_df)
                
            contiguous_dataframes[sat_name] = grouped_dfs
        else:
            contiguous_dataframes[sat_name] = []
        
    return contiguous_dataframes

def write_ephemeris_file(file_name, df, satellite_info, output_dir="external/ephems"):
    # Create directory for satellite if it doesn't exist
    sat_dir = os.path.join(output_dir, file_name.split('_')[0])  # Adjust directory naming
    os.makedirs(sat_dir, exist_ok=True)

    # Define the file name
    start_day = df.index.min().strftime("%Y-%m-%d")
    end_day = df.index.max().strftime("%Y-%m-%d")
    norad_id = satellite_info['norad_id']  # Directly access from passed dict
    file_name = f"NORAD{norad_id}-{start_day}-{end_day}.txt"
    file_path = os.path.join(sat_dir, file_name)

    # Write data to the ephemeris file
    with open(file_path, 'w') as file:
        for idx, row in df.iterrows():
            # Convert index to UTC string without timezone information
            utc = idx.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S.%f")
            line1 = f"{utc} {row['pos_x_eci']} {row['pos_y_eci']} {row['pos_z_eci']} {row['vel_x_eci']} {row['vel_y_eci']} {row['vel_z_eci']}\n"
            line2 = f"{row['sigma_x']} {row['sigma_y']} {row['sigma_z']} {row['sigma_xv']} {row['sigma_yv']} {row['sigma_zv']}\n"
            file.write(line1)
            file.write(line2)

def sp3_ephem_to_df(satellite, date=None, ephemeris_dir="external/ephems"):
    sat_dir = os.path.join(ephemeris_dir, satellite)
    ephemeris_files = glob.glob(os.path.join(sat_dir, "*.txt"))

    if date is not None:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        selected_file = None

        for file_path in ephemeris_files:
            basename = os.path.basename(file_path)
            parts = basename.split('-')
            try:
                # Start Date
                start_year = parts[1]
                start_month = parts[2]
                start_day = parts[3]
                start_date_str = f"{start_year}-{start_month}-{start_day}"
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()

                # End Date
                end_year = parts[4]
                end_month = parts[5]
                end_day = parts[6].split('.')[0]  # To remove the '.txt' extension
                end_date_str = f"{end_year}-{end_month}-{end_day}"
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

                if start_date <= target_date <= end_date:
                    selected_file = file_path
                    break
            except ValueError as e:
                print(f"Error parsing date from filename {basename}: {e}")
                continue

        ephemeris_files = [selected_file] if selected_file else []

    df = pd.DataFrame()

    for file_path in ephemeris_files:
        with open(file_path, 'r') as file:
            data = []
            while True:
                line1 = file.readline()
                line2 = file.readline()
                if not line2:
                    break

                line1_parts = line1.strip().split()
                utc = ' '.join(line1_parts[:2])
                ephemeris_values = line1_parts[2:]
                sigma_values = line2.strip().split()

                if len(ephemeris_values) != 6 or len(sigma_values) != 6:
                    raise ValueError("Incorrect number of values in ephemeris or sigma lines.")

                converted_values = [float(val) * 1000 if i < 6 else float(val)
                                    for i, val in enumerate(ephemeris_values)]

                row = [pd.to_datetime(utc)] + converted_values + sigma_values
                data.append(row)

            file_df = pd.DataFrame(data, columns=['UTC', 'x', 'y', 'z',
                                                  'xv', 'yv', 'zv',
                                                  'sigma_x', 'sigma_y', 'sigma_z',
                                                  'sigma_xv', 'sigma_yv', 'sigma_zv'])

            df = pd.concat([df, file_df], ignore_index=True)

    return df

def main(satellite_list=None):
    #satellite_list is a list of satellite names to process. If None, all satellites will be processed.
    sat_list_path = "misc/sat_list.json"
    sp3_files_path = "external/sp3_files"
    with open(sat_list_path, 'r') as file:
        sat_dict = json.load(file)

    if satellite_list:
        sat_dict = {k: v for k, v in sat_dict.items() if k in satellite_list}

    print(f"sat_dict: {sat_dict}")


    sp3_dataframes = process_sp3_files(sp3_files_path, sat_dict)
    print(f"sp3_dataframes: {sp3_dataframes}")

    for satellite, dfs in sp3_dataframes.items():
        print(f"Processing {satellite}")
        base_satellite_name = satellite.split('_')[0]  # Extract the original satellite name
        for df_index, df in enumerate(dfs):
            print(f"Processing {base_satellite_name}_{df_index}")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            mjd_times = [utc_to_mjd(dt) for dt in df.index]
            df['MJD'] = mjd_times

            itrs_positions = df[['Position_X', 'Position_Y', 'Position_Z']].values
            itrs_velocities = df[['Velocity_X', 'Velocity_Y', 'Velocity_Z']].values
            print(f"Converting ITRS to EME2000 for {base_satellite_name}_{df_index}")
            icrs_positions, icrs_velocities = SP3_to_EME2000(itrs_positions, itrs_velocities, df['MJD'])
            df['pos_x_eci'], df['pos_y_eci'], df['pos_z_eci'] = icrs_positions.T
            df['vel_x_eci'], df['vel_y_eci'], df['vel_z_eci'] = icrs_velocities.T

            df['sigma_x'] = 5e-1  # in km
            df['sigma_y'] = 5e-1  # in km
            df['sigma_z'] = 5e-1  # in km
            df['sigma_xv'] = 1e-3 # in km/s
            df['sigma_yv'] = 1e-3 # in km/s
            df['sigma_zv'] = 1e-3 # in km/s

            file_name = f"{base_satellite_name}_{df_index}"
            print(f"Writing ephemeris file for {file_name}")
            write_ephemeris_file(file_name, df, sat_dict[base_satellite_name])  


if __name__ == "__main__":
    main(satellite_list = ['TerraSAR-X', 'TanDEM-X']) #'GRACE-FO-A',
