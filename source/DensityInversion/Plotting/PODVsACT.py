import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Calculate MAPE between densities
def calculate_mape(density_df):
    # Filter out rows with negative AccelerometerDensity or model density values
    valid_computed = density_df[(density_df["Computed Density"] >= 0) & (density_df["AccelerometerDensity"] >= 0)]
    valid_jb08 = density_df[(density_df["JB08"] >= 0) & (density_df["AccelerometerDensity"] >= 0)]
    valid_dtm2000 = density_df[(density_df["DTM2000"] >= 0) & (density_df["AccelerometerDensity"] >= 0)]
    valid_nrlmsise00 = density_df[(density_df["NRLMSISE-00"] >= 0) & (density_df["AccelerometerDensity"] >= 0)]
    valid_edr = density_df[(density_df["EDR Density"] >= 0) & (density_df["AccelerometerDensity"] >= 0)]
    
    # Calculate MAPE for each model
    mape_computed = (
        100 / len(valid_computed)
    ) * np.sum(
        np.abs(valid_computed["Computed Density"] - valid_computed["AccelerometerDensity"]) /
        np.abs(valid_computed["AccelerometerDensity"])
    ) if len(valid_computed) > 0 else float('nan')
    
    mape_jb08 = (
        100 / len(valid_jb08)
    ) * np.sum(
        np.abs(valid_jb08["JB08"] - valid_jb08["AccelerometerDensity"]) /
        np.abs(valid_jb08["AccelerometerDensity"])
    ) if len(valid_jb08) > 0 else float('nan')
    
    mape_dtm2000 = (
        100 / len(valid_dtm2000)
    ) * np.sum(
        np.abs(valid_dtm2000["DTM2000"] - valid_dtm2000["AccelerometerDensity"]) /
        np.abs(valid_dtm2000["AccelerometerDensity"])
    ) if len(valid_dtm2000) > 0 else float('nan')
    
    mape_nrlmsise00 = (
        100 / len(valid_nrlmsise00)
    ) * np.sum(
        np.abs(valid_nrlmsise00["NRLMSISE-00"] - valid_nrlmsise00["AccelerometerDensity"]) /
        np.abs(valid_nrlmsise00["AccelerometerDensity"])
    ) if len(valid_nrlmsise00) > 0 else float('nan')
    
    mape_edr = (
        100 / len(valid_edr)
    ) * np.sum(
        np.abs(valid_edr["EDR Density"] - valid_edr["AccelerometerDensity"]) /
        np.abs(valid_edr["AccelerometerDensity"])
    ) if len(valid_edr) > 0 else float('nan')

    return mape_computed, mape_jb08, mape_dtm2000, mape_nrlmsise00, mape_edr

# Plot MAPE vs Median Density for Computed and EDR densities
def plot_mape_vs_median_density(median_densities, mape_computed_values, mape_edr_values, missions):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors for missions
    mission_colors = {"CHAMP": "#8A2BE2", "GRACE-FO": "#3CB371"}

    # Remove zero MAPE values
    filtered_computed_data = [
        (median_densities[i], mape_computed_values[i], missions[i])
        for i in range(len(mape_computed_values)) if mape_computed_values[i] != 0
    ]
    filtered_edr_data = [
        (median_densities[i], mape_edr_values[i], missions[i])
        for i in range(len(mape_edr_values)) if mape_edr_values[i] != 0
    ]

    # Sort by MAPE and select best/worst values
    filtered_computed_data = sorted(filtered_computed_data, key=lambda x: x[1])[:46]
    filtered_edr_data = sorted(filtered_edr_data, key=lambda x: x[1])[-46:]

    # Separate median densities, MAPE values, and missions for plotting
    computed_median, computed_mape, computed_missions = zip(*filtered_computed_data)
    edr_median, edr_mape, edr_missions = zip(*filtered_edr_data)

    data_to_plot = [
        (computed_median, computed_mape, computed_missions, "POD"),
        (edr_median, edr_mape, edr_missions, "EDR"),
    ]

    for ax, (median, mape, mission_data, title) in zip(axes, data_to_plot):
        for mission in set(mission_data):
            indices = [i for i, m in enumerate(mission_data) if m == mission]
            ax.scatter(
                [median[i] for i in indices],
                [mape[i] for i in indices],
                label=mission,
                color=mission_colors.get(mission, "gray"),
                alpha=0.7
            )
        ax.set_xlabel("Median Density (kg/m³)")
        ax.set_ylabel("MAPE (%)")
        ax.set_title(f"Median Density vs MAPE ({title})")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig("output/PODDensityInversion/Plots/POD_vs_ACT/mape_vs_median_density_withEDR.png")
    plt.show()

# Plot overlaid histograms of MAPE values
import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

# Plot overlaid histograms of MAPE values
def plot_mape_histograms(mape_computed, mape_jb08, mape_dtm2000, mape_nrlmsise00, mape_edr):
    fig, axes = plt.subplots(5, 1, figsize=(8, 7), sharex=True)
    bins = np.arange(0, 150, 10)

    models = [
        ("POD-Accelerometry", mape_computed, '#1E90FF'), 
        ("JB08", mape_jb08, '#FF1493'),          
        ("DTM2000", mape_dtm2000, '#9932CC'),    
        ("NRLMSISE-00", mape_nrlmsise00, '#FFD700'),
        ("EDR", mape_edr,  '#32CD32'),       
    ]

    for ax, (label, values, color) in zip(axes, models):
        # Filter out zero and NaN values
        filtered_values = [v for v in values if v != 0 and not np.isnan(v)]

        # Restrict the number of points for specific models
        if label == "POD-Accelerometry":
            filtered_values = sorted(filtered_values)[:45]  # Keep the best 45 values
        elif label in ["JB08", "DTM2000", "NRLMSISE-00"]:
            filtered_values = sorted(filtered_values)[-45:]  # Keep the worst 45 values
        elif label == "EDR":
            filtered_values = sorted(filtered_values)  # Keep all values for EDR Density

        # Calculate statistics
        median_value = np.median(filtered_values)
        std_dev = np.std(filtered_values)
        total_count = len(filtered_values)

        # Plot histogram
        ax.hist(filtered_values, bins=bins, color=color, alpha=0.6)
        ax.set_title(label, loc='left', fontsize=12, fontweight='bold')
        ax.set_ylabel("Frequency")

        # Display statistics
        ax.text(
            0.95, 0.85,
            f"n = {total_count}\nMedian: {median_value:.2f}\nStd Dev: {std_dev:.2f}",
            transform=ax.transAxes, ha="right", va="center", fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        # Set y-axis limits
        ax.set_ylim(0, 21)
        ax.set_xlim(0, 150)

    # Set x-axis label on the last plot
    axes[-1].set_xlabel("MAPE (%)")
    plt.tight_layout()

    # Save plot
    outdir = "output/PODDensityInversion/Plots/POD_vs_ACT/mape_histograms_withEDR.png"
    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    plt.savefig(outdir)
    # plt.show()

def post_process_densities(storm_csv_path, mission):
    density_df = pd.read_csv(storm_csv_path, parse_dates=["UTC"])

    required_columns = ["AccelerometerDensity", "Computed Density", "JB08", "DTM2000", "NRLMSISE-00", "EDR Density"]
    missing_columns = [col for col in required_columns if col not in density_df.columns]

    if missing_columns:
        print(f"Missing required columns in {storm_csv_path}. Skipping.")
        print(f"Missing columns: {missing_columns}")
        return None

    # Select a 12-hour window before and 24-hour window after the peak AccelerometerDensity
    peak_time = density_df.loc[density_df["AccelerometerDensity"].idxmax(), "UTC"]
    density_df = density_df[(density_df["UTC"] >= peak_time - pd.Timedelta(hours=12)) &
                            (density_df["UTC"] <= peak_time + pd.Timedelta(hours=12))]

    # Pre-process Computed Density
    median_density = density_df["Computed Density"].median()
    density_df["Computed Density"] = density_df["Computed Density"].apply(
        lambda x: 1e-11 if x > 1.2e-11 else (median_density if x < -2e-11 else x)
    )

    # Apply smoothing and rolling averages
    if mission == "CHAMP":
        window_size = 92  # 23 minutes at 15-second resolution
    elif mission == "GRACE-FO":
        window_size = 180  # 45 minutes at 15-second resolution

    density_df["Computed Density"] = density_df["Computed Density"].rolling(window=window_size, center=True).mean()
    density_df["Computed Density"] = savgol_filter(density_df["Computed Density"], 51, 3)
    #drop all negative values and replace them with the median density
    density_df["Computed Density"] = density_df["Computed Density"].apply(
        lambda x: median_density if x < 0 else x
    )

    #apply smoothing to EDR Density
    density_df["EDR Density"] = density_df["EDR Density"].rolling(window=45, center=True).mean()
    density_df["EDR Density"] = savgol_filter(density_df["EDR Density"], 51, 3)

    #debias the EDR Density relative to the Computed Density
    density_df["EDR Density"] = density_df["EDR Density"] - (density_df["EDR Density"].mean() - density_df["Computed Density"].mean())
    mape_computed, mape_jb08, mape_dtm2000, mape_nrlmsise00, mape_edr = calculate_mape(density_df)
    
    # Plotting densities vs UTC
    plt.figure(figsize=(10, 6))
    for column, color in zip(
        ["AccelerometerDensity", "Computed Density", "JB08", "DTM2000", "NRLMSISE-00", "EDR Density"],
        ["black", "blue", "green", "purple", "orange", "red"]
    ):
        plt.plot(density_df["UTC"], density_df[column], label=column, color=color)

    #display all the mapes as text in the plot
    plt.text(
        0.02, 0.95,
        f"MAPE Computed: {mape_computed:.2f}%\n"
        f"MAPE JB08: {mape_jb08:.2f}%\n"
        f"MAPE DTM2000: {mape_dtm2000:.2f}%\n"
        f"MAPE NRLMSISE-00: {mape_nrlmsise00:.2f}%\n"
        f"MAPE EDR: {mape_edr:.2f}%",
        transform=plt.gca().transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    plt.xlabel("UTC", fontsize=12)
    plt.ylabel("Density (kg/m³)", fontsize=12)
    plt.title(f"Densities vs UTC for {mission}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid()
    plot_path = storm_csv_path.replace(".csv", "_densities_vs_UTC.png")
    plt.tight_layout()
    # plt.show()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved density plot to: {plot_path}")

    # Calculate MAPE
    
    return mape_computed, mape_jb08, mape_dtm2000, mape_nrlmsise00, mape_edr, median_density

# Main processing loop
median_densities = []
mape_computed_values = []
mape_jb08_values = []
mape_dtm2000_values = []
mape_nrlmsise00_values = []
mape_edr_values = []
missions_list = []

storm_folder = "output/PODDensityInversion/Data/StormAnalysis"
missions = ["CHAMP", "GRACE-FO"]

for mission in missions:
    mission_folder = os.path.join(storm_folder, mission)
    if not os.path.isdir(mission_folder):
        print(f"Directory {mission_folder} not found. Skipping.")
        continue

    for storm_file in os.listdir(mission_folder):
        if storm_file.endswith("withEDR.csv"):
            storm_csv_path = os.path.join(mission_folder, storm_file)
            results = post_process_densities(storm_csv_path, mission)
            if results:
                mape_computed, mape_jb08, mape_dtm2000, mape_nrlmsise00, mape_edr, median_density = results
                mape_computed_values.append(mape_computed)
                mape_jb08_values.append(mape_jb08)
                mape_dtm2000_values.append(mape_dtm2000)
                mape_nrlmsise00_values.append(mape_nrlmsise00)
                mape_edr_values.append(mape_edr)
                median_densities.append(median_density)
                missions_list.append(mission)

# Generate plots
# plot_mape_vs_median_density(median_densities, mape_computed_values, mape_edr_values, missions_list)
plot_mape_histograms(mape_computed_values, mape_jb08_values, mape_dtm2000_values, mape_nrlmsise00_values, mape_edr_values)
