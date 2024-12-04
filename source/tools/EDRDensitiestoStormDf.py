import os
import pandas as pd

def find_edr_storm_matches():
    # Set paths
    storm_folder = "output/PODDensityInversion/Data/StormAnalysis"
    edr_folder = "output/EDR/Data"

    missions = ["GRACE-FO"]#"CHAMP",

    for mission in missions:
        print(f"\nProcessing mission: {mission}")

        # Paths for mission-specific StormAnalysis and EDR files
        mission_storm_folder = os.path.join(storm_folder, mission)
        mission_edr_folder = os.path.join(edr_folder, mission)

        # Validate folders
        if not os.path.isdir(mission_storm_folder):
            print(f"StormAnalysis directory {mission_storm_folder} not found. Skipping.")
            continue

        if not os.path.isdir(mission_edr_folder):
            print(f"EDR directory {mission_edr_folder} not found. Skipping.")
            continue

        # List all EDR and StormAnalysis files
        edr_files = [os.path.join(mission_edr_folder, f) for f in os.listdir(mission_edr_folder) if f.startswith("EDR") and f.endswith(".csv")]
        storm_files = [os.path.join(mission_storm_folder, f) for f in os.listdir(mission_storm_folder) if f.endswith(".csv")]

        print(f"Found {len(edr_files)} EDR files and {len(storm_files)} StormAnalysis files.")

        # Iterate over EDR files
        for edr_file in edr_files:
            print(f"\nProcessing EDR file: {edr_file}")

            # Read the EDR file to get the first timestamp
            edr_df = pd.read_csv(edr_file)
            edr_df["UTC"] = pd.to_datetime(edr_df["UTC"])

            if edr_df.empty:
                print(f"EDR file {edr_file} is empty. Skipping.")
                continue

            edr_first_timestamp = edr_df["UTC"].iloc[0]
            print(f"First timestamp in EDR file: {edr_first_timestamp}")

            # Check StormAnalysis files for matching timestamp
            for storm_file in storm_files:
                storm_df = pd.read_csv(storm_file)
                storm_df["UTC"] = pd.to_datetime(storm_df["UTC"])

                if edr_first_timestamp in storm_df["UTC"].values:
                    print(f"Match found! EDR file: {edr_file} | StormAnalysis file: {storm_file}")

                    # Merge EDR densities into a copy of the StormAnalysis DataFrame
                    merged_df = storm_df.copy()
                    merged_df = merged_df.merge(
                        edr_df[["UTC", "EDR Density"]],
                        on="UTC",
                        how="left"
                    )

                    # Save the merged DataFrame to a new CSV
                    output_csv_path = storm_file.replace(".csv", "_withEDR.csv")
                    merged_df.to_csv(output_csv_path, index=False)
                    print(f"Saved merged file: {output_csv_path}")


if __name__ == "__main__":
    find_edr_storm_matches()
