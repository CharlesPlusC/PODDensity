import os
import zipfile
import pandas as pd
import datetime
# Step 1: Extract zip files if not already extracted
def extract_zip_files(source_folder, destination_folder):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                dest_path = os.path.join(destination_folder, os.path.basename(root))
                os.makedirs(dest_path, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_path)

# Step 2: Process each density data file, skipping headers
def process_density_file(density_folder, storm_csv):
    column_names = ["Date", "Time", "System", "Alt", "Lon", "Lat", "LST", "ArgLat", 
                    "dens_x", "rho_mean", "Flag1", "Flag2"]

    data_rows = []
    with open(density_folder, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            data_rows.append(line.strip().split())

    density_data = pd.DataFrame(data_rows, columns=column_names)
    density_data["UTC"] = pd.to_datetime(density_data["Date"] + " " + density_data["Time"], format="%Y-%m-%d %H:%M:%S.%f")
    density_data["dens_x"] = density_data["dens_x"].astype(float)
    density_data.sort_values("UTC", inplace=True)

    storm_csv = storm_csv.drop(columns=[col for col in storm_csv.columns if "AccelerometerDensity" in col], errors="ignore")

    density_matches = []
    for storm_time in storm_csv["UTC"]:
        nearest_idx = density_data["UTC"].searchsorted(storm_time, side="left")
        if nearest_idx == 0:
            nearest_idx = 1
        elif nearest_idx >= len(density_data):
            nearest_idx = len(density_data) - 1

        prev_idx = nearest_idx - 1 if nearest_idx > 0 else nearest_idx
        next_idx = nearest_idx if nearest_idx < len(density_data) else nearest_idx - 1

        time_diff_prev = abs((storm_time - density_data.iloc[prev_idx]["UTC"]).total_seconds())
        time_diff_next = abs((storm_time - density_data.iloc[next_idx]["UTC"]).total_seconds())

        if time_diff_prev <= 10 and (time_diff_prev <= time_diff_next):
            matched_density = density_data.iloc[prev_idx]["dens_x"]
        elif time_diff_next <= 10:
            matched_density = density_data.iloc[next_idx]["dens_x"]
        else:
            matched_density = None

        if matched_density is not None:
            density_matches.append({"UTC": storm_time, "AccelerometerDensity": matched_density})

    density_matches_df = pd.DataFrame(density_matches)
    if "UTC" not in density_matches_df.columns:
        print(f"Error: No 'UTC' column in density_matches_df for file {density_folder}")
        density_matches_df["UTC"] = pd.NaT  

    storm_csv = storm_csv.merge(density_matches_df, on="UTC", how="left")
    print(f"head of storm_csv: {storm_csv.head()}")
    return storm_csv

# Step 3: Cache available density files and their dates
def get_available_density_files(mission_folder):
    print(f"Mission folder checked: {mission_folder}...")
    available_files = {}
    for file in os.listdir(mission_folder):
        print(f"Checking file: {file}...")
        if file.endswith(".txt"):
            try:
                # For filenames like CH_DNS_ACC_2008-03_v02.txt
                parts = file.split('_')
                if len(parts) >= 4:
                    year_month = parts[3].split('-')
                    year = int(year_month[0])  # Extract the year
                    month = int(year_month[1])  # Extract the month
                    available_files[(year, month)] = os.path.join(mission_folder, file)
                    print(f"Added {(year, month)} -> {file}")
            except (IndexError, ValueError) as e:
                print(f"Error parsing {file}: {e}")
    print(f"Available files after parsing: {available_files.keys()}")
    return available_files

# Step 4: Find closest available file based on the cache
def find_closest_density_file(available_files, target_year, target_month):
    min_diff = float('inf')
    closest_file = None
    print(f"Available files being checked: {available_files}...")
    for (year, month), file_path in available_files.items():
        diff = abs((year - target_year) * 12 + (month - target_month))
        if diff < min_diff:
            min_diff = diff
            closest_file = file_path
    print(f"Closest file found: {closest_file}")
    return closest_file

# Step 5: Process density data and add to StormAnaly
# sis CSVs
def process_density_data(data_folder, storm_folder):
    for mission in ["GRACE-FO"]: 
        mission_folder = os.path.join(data_folder, f"version_02_{mission}_data")
        storm_csv_folder = os.path.join(storm_folder, mission)
        if not os.path.isdir(mission_folder) or not os.path.isdir(storm_csv_folder):
            print(f"Directory not found: {mission_folder} or {storm_csv_folder}")
            continue

        available_files = get_available_density_files(mission_folder)
        if not available_files:
            print(f"No files found for mission {mission} in {mission_folder}.")

        for storm_file in os.listdir(storm_csv_folder):
            if storm_file.endswith("withEDR.csv"):
                print(f"Processing {storm_file} for {mission}...")
                storm_csv_path = os.path.join(storm_csv_folder, storm_file)
                storm_csv = pd.read_csv(storm_csv_path, parse_dates=["UTC"])
                storm_csv.sort_values("UTC", inplace=True)

                if storm_csv.empty:
                    print(f"Warning: {storm_file} has no data.")
                    continue

                storm_year = storm_csv["UTC"].iloc[0].year
                storm_month = storm_csv["UTC"].iloc[0].month

                density_file_path = find_closest_density_file(available_files, storm_year, storm_month)

                if density_file_path:
                    storm_csv = process_density_file(density_file_path, storm_csv)
                else:
                    print(f"No density file available within a year of {storm_year}-{storm_month:02d} for {mission}")

                storm_csv.to_csv(storm_csv_path, index=False)


def process_density_folder_to_csv(density_folder, start_date, end_date, csv_output_path):
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    # Prepare DataFrame to accumulate data
    full_density_data = pd.DataFrame(columns=["UTC", "Lat", "Lon", "Alt", "dens_x"])

    # Iterate over files in the folder
    for file in sorted(os.listdir(density_folder)):
        if file.endswith(".txt"):
            # Extract year and month from filename
            parts = file.split('_')
            print(f"parts: {parts}")
            year = int(parts[3])  # Expected format: YYYY-MM
            month = int(parts[4])
            file_start_date = datetime.datetime(year, month, 1)
            file_end_date = (file_start_date + datetime.timedelta(days=31)).replace(day=1) - datetime.timedelta(days=1)
            
            # Check if file overlaps with the specified date range
            if file_start_date <= end_date and file_end_date >= start_date:
                print(f"Processing file: {file}")
                file_path = os.path.join(density_folder, file)
                
                # Read file data
                column_names = ["Date", "Time", "System", "Alt", "Lon", "Lat", "LST", 
                                "ArgLat", "dens_x", "rho_mean", "Flag1", "Flag2"]
                data_rows = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if not line.startswith("#"):
                            data_rows.append(line.strip().split())
                
                # Convert to DataFrame
                density_data = pd.DataFrame(data_rows, columns=column_names)
                density_data["UTC"] = pd.to_datetime(density_data["Date"] + " " + density_data["Time"], format="%Y-%m-%d %H:%M:%S.%f")
                density_data["dens_x"] = density_data["dens_x"].astype(float)
                
                # Filter to date range
                density_data = density_data[(density_data["UTC"] >= start_date) & (density_data["UTC"] <= end_date)]
                
                if not density_data.empty:
                    # Select required columns and accumulate data
                    density_data = density_data[["UTC", "Lat", "Lon", "Alt", "dens_x"]]
                    full_density_data = pd.concat([full_density_data, density_data], ignore_index=True)
                else:
                    print(f"No data in {file} within specified date range.")
    
    # Save to CSV if data was found
    if not full_density_data.empty:
        full_density_data.sort_values("UTC", inplace=True)
        full_density_data.to_csv(csv_output_path, index=False)
        print(f"Densities from {start_date.date()} to {end_date.date()} saved to {csv_output_path}")
    else:
        print("No data found within the specified date range across all files.")


def add_lat_lon_alt_density(csv_file, extracted_data_folder):
    data = pd.read_csv(csv_file, parse_dates=["UTC"])
    print(f'Head of data: {data.head()}')

    data["latitude"] = None
    data["longitude"] = None
    data["altitude"] = None
    data["AccelerometerDensity"] = None

    # Select the correct mission folder based on the CSV filename
    if "CHAMP" in csv_file:
        mission_folder = os.path.join(extracted_data_folder, "version_02_CHAMP_data")
    elif "GRACE" in csv_file:
        mission_folder = os.path.join(extracted_data_folder, "version_02_GRACE-FO_data")
    else:
        print("CSV file does not specify a recognized mission (CHAMP or GRACE).")
        return data

    # Identify the closest file in the selected mission folder
    storm_start = data["UTC"].iloc[0]
    min_diff = float('inf')
    nearest_file = None

    for root, _, files in os.walk(mission_folder):
        for file in files:
            if file.endswith(".txt"):
                # Extract the relevant parts of the filename
                parts = file.split('_')
                if len(parts) >= 4:  # Ensure the file has enough parts for both cases
                    try:
                        if '-' in parts[3]:  # Case 1: Year and month in "2008-03" format
                            year_month = parts[3].split('-')
                            year = int(year_month[0])  # Extract the year
                            month = int(year_month[1])  # Extract the month
                        elif len(parts) >= 5:  # Case 2: Separate year and month
                            year = int(parts[3])  # Extract year from parts[3]
                            month = int(parts[4].split('.')[0])  # Extract month from parts[4]
                        else:
                            print(f"Skipping file due to unexpected structure: {file}")
                            continue
                        
                        print(f"Year: {year}, Month: {month}")
                        file_date = datetime(year, month, 1)
                        
                        # Calculate the difference in months
                        diff = abs((file_date.year - storm_start.year) * 12 + (file_date.month - storm_start.month))
                        if diff < min_diff:
                            min_diff = diff
                            nearest_file = os.path.join(root, file)
                    except ValueError as e:
                        print(f"Skipping file with unexpected date format in: {file}, error: {e}")
                else:
                    print(f"Skipping file due to unexpected structure: {file}")

    if not nearest_file:
        print("No suitable density file found.")
        return data

    print(f"Nearest file found: {nearest_file}")

    # Load and process only the nearest file
    column_names = ["Date", "Time", "System", "Alt", "Lon", "Lat", "LST", "ArgLat", "dens_x", "rho_mean", "Flag1", "Flag2"]
    density_data = pd.read_csv(nearest_file, delim_whitespace=True, comment="#", names=column_names, parse_dates={"UTC": ["Date", "Time"]})
    density_data.sort_values("UTC", inplace=True)

    # Match Alt, Lon, Lat, and AccelerometerDensity
    for idx, storm_time in data["UTC"].items():
        nearest_idx = density_data["UTC"].searchsorted(storm_time, side="left")
        if nearest_idx == 0:
            nearest_idx = 1
        elif nearest_idx >= len(density_data):
            nearest_idx = len(density_data) - 1

        prev_idx = nearest_idx - 1 if nearest_idx > 0 else nearest_idx
        next_idx = nearest_idx if nearest_idx < len(density_data) else nearest_idx - 1

        time_diff_prev = abs((storm_time - density_data.iloc[prev_idx]["UTC"]).total_seconds())
        time_diff_next = abs((storm_time - density_data.iloc[next_idx]["UTC"]).total_seconds())

        if time_diff_prev <= 10 and time_diff_prev <= time_diff_next:
            matched_row = density_data.iloc[prev_idx]
        elif time_diff_next <= 10:
            matched_row = density_data.iloc[next_idx]
        else:
            matched_row = None

        if matched_row is not None:
            data.at[idx, "latitude"] = matched_row["Lat"]
            data.at[idx, "longitude"] = matched_row["Lon"]
            data.at[idx, "altitude"] = matched_row["Alt"]
            data.at[idx, "AccelerometerDensity"] = matched_row["dens_x"]

    # Save the modified CSV
    data.to_csv(csv_file, index=False)
    return data



if __name__ == "__main__":

    # Main execution
    source_zip_folder = "external/TUDelft_Densities"
    extracted_data_folder = "external/TUDelft_Densities/extracted_density_data"
    storm_folder = "output/PODDensityInversion/Data/StormAnalysis"

    # extract_zip_files(source_zip_folder, extracted_data_folder)
    # process_density_data(extracted_data_folder, storm_folder)

    # Get density for a specific date range and satellite and save to CSV
    density_folder = "external/TUDelft_Densities/extracted_density_data/version_02_GRACE-FO_data"
    process_density_folder_to_csv(density_folder, "2022-02-01", "2022-02-07", "GRACE_density_feb_2022.csv")
    # Add Lat, Lon, Alt, and AccelerometerDensity to storm data CSV
    # add_lat_lon_alt_density(csv_file="output/EDR/Data/GRACE-FO-A/EDR_GRACE-FO-A_2023-05-06_density_inversion.csv",extracted_data_folder=extracted_data_folder)

