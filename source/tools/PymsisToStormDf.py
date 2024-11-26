import os
import pandas as pd
import numpy as np
from pymsis import msis
from pymsis.utils import get_f107_ap

def append_nrlmsise_density(csv_path):
    # Load the data frame
    df = pd.read_csv(csv_path)
    print(f"Processing file: {csv_path}")

    # Extract latitude, longitude, altitude, and UTC values
    dates = pd.to_datetime(df['UTC']).to_numpy()
    lons = df['longitude'].to_numpy()
    lats = df['latitude'].to_numpy()
    alts = df['altitude'].to_numpy() / 1000  # Convert meters to kilometers

    # Retrieve F10.7 and Ap indices for the given dates
    f107, f107a, ap_values = get_f107_ap(dates)
    
    # Calculate density using MSIS model with geomagnetic activity set to -1
    try:
        msis_output = msis.run(dates, lons, lats, alts, f107s=f107, f107as=f107a, aps=ap_values, geomagnetic_activity=-1)
    except ValueError as e:
        print(f"Error in msis.run for {csv_path}: {e}")
        raise

    # Extract total mass density (1st variable in output) and add to DataFrame
    density_values = np.squeeze(msis_output)[:, 0]
    df['NRLMSISE-00'] = density_values

    # Save the updated DataFrame
    df.to_csv(csv_path, index=False)
    print(f"Updated file saved: {csv_path}")

def process_storm_data(storm_folder, missions):
    for mission in missions:
        mission_folder = os.path.join(storm_folder, mission)
        if not os.path.isdir(mission_folder):
            print(f"Directory {mission_folder} not found. Skipping.")
            continue

        for storm_file in os.listdir(mission_folder):
            if storm_file.endswith(".csv"):
                storm_csv_path = os.path.join(mission_folder, storm_file)
                append_nrlmsise_density(storm_csv_path)

# Specify the storm folder and missions
storm_folder = "output/PODDensityInversion/Data/StormAnalysis/"
missions = ["CHAMP"] # "GRACE-FO","TerraSAR-X"

# Process the storm data
process_storm_data(storm_folder, missions)

#if you just want to compute NRLMSISE-00 for a single file:
# storm_csv_path = "output/DensityInversion/PODDensityInversion/Data/OneStormAllMethods/ACT_vs_POD_2023_05_06_GRACE-FOA.csv"
# append_nrlmsise_density(storm_csv_path)