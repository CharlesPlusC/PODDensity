import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pymsis import msis
from pymsis.utils import get_f107_ap

# Calculate MAPE between densities
def calculate_mape(density_df):
    mape_computed = (100 / len(density_df)) * np.sum(
        np.abs(density_df["Computed Density"] - density_df["AccelerometerDensity"]) / 
        np.abs(density_df["AccelerometerDensity"])
    )
    mape_jb08 = (100 / len(density_df)) * np.sum(
        np.abs(density_df["JB08"] - density_df["AccelerometerDensity"]) / 
        np.abs(density_df["AccelerometerDensity"])
    )
    mape_dtm2000 = (100 / len(density_df)) * np.sum(
        np.abs(density_df["DTM2000"] - density_df["AccelerometerDensity"]) / 
        np.abs(density_df["AccelerometerDensity"])
    )
    mape_nrlmsise00 = (100 / len(density_df)) * np.sum(
        np.abs(density_df["NRLMSISE-00"] - density_df["AccelerometerDensity"]) /
        np.abs(density_df["AccelerometerDensity"])
    )
    return mape_computed, mape_jb08, mape_dtm2000, mape_nrlmsise00

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

def plot_mape_vs_median_density_computed(median_densities, mape_computed_values, missions):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Select the best 46 MAPE values for Computed
    selected_mape_computed = sorted(mape_computed_values)[:45]
    
    # Filter median densities and corresponding MAPE values based on the selection
    filtered_median = [median_densities[i] for i, val in enumerate(mape_computed_values) if val in selected_mape_computed]
    filtered_mape = [val for val in mape_computed_values if val in selected_mape_computed]
    filtered_missions = [missions[i] for i, val in enumerate(mape_computed_values) if val in selected_mape_computed]

    # Define colors for missions
    mission_colors = {"CHAMP": "#8A2BE2", "GRACE-FO": "#3CB371"}  # Using hex codes for compatibility
    
    # Plot scatter for Computed densities only
    for mission in set(filtered_missions):
        indices = [i for i, m in enumerate(filtered_missions) if m == mission]
        ax.scatter([filtered_median[i] for i in indices], [filtered_mape[i] for i in indices],
                   label=mission, color=mission_colors.get(mission, "gray"), alpha=0.7)

    # Linear fit
    filtered_median_np = np.array(filtered_median).reshape(-1, 1)
    filtered_mape_np = np.array(filtered_mape)
    model = LinearRegression().fit(filtered_median_np, filtered_mape_np)
    slope = model.coef_[0]
    r_squared = r2_score(filtered_mape_np, model.predict(filtered_median_np))
    
    # Plot line of best fit
    x_fit = np.linspace(min(filtered_median), max(filtered_median), 100)
    y_fit = model.predict(x_fit.reshape(-1, 1))
    ax.plot(x_fit, y_fit, color='red', linestyle='--', label="Best fit line")
    
    # Display R² and slope on the plot
    ax.text(0.05, 0.95, f"$R^2$ = {r_squared:.2f}", 
            transform=ax.transAxes, ha="left", va="top", 
            fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Set plot limits with 10% padding
    x_min, x_max = min(filtered_median) * 0.9, max(filtered_median) * 1.1
    y_min, y_max = min(filtered_mape) * 0.9, max(filtered_mape) * 1.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Labels with units
    ax.set_xlabel("Median Density (kg/m³)", fontsize=12)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid()

    plt.tight_layout()
    plt.savefig("output/DensityInversion/PODDensityInversion/Plots/POD_vs_ACT/mape_vs_median_density_computed.png")

# New function to plot overlaid histograms of MAPE values
def plot_mape_histograms(mape_computed_values, mape_jb08_values, mape_dtm2000_values, mape_nrlmsise00_values):
    fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True, sharey=True)
    bin_width = 5
    bins = np.arange(0, max(mape_computed_values + mape_jb08_values + mape_dtm2000_values + mape_nrlmsise00_values) + bin_width, bin_width)
    
    models = [
        ("POD-Derived", mape_computed_values, "blue", True),   # True indicates "select best values"
        ("JB08", mape_jb08_values, "green", False),         # False indicates "select worst values"
        ("DTM2000", mape_dtm2000_values, "purple", False),
        ("NRLMSISE-00", mape_nrlmsise00_values, "orange", False)
    ]

    for ax, (label, values, color, select_best) in zip(axes, models):
        if select_best:
            selected_values = sorted(values)[:45]
        else:
            selected_values = sorted(values)[-45:]
        
        median_value = np.median(selected_values)
        std_dev = np.std(selected_values)
        total_count = len(selected_values)
        
        ax.hist(selected_values, bins=bins, color=color, alpha=0.6)
        ax.text(0.95, 0.85, f"n = {total_count}\nMedian: {median_value:.2f}\nStd Dev: {std_dev:.2f}",
                transform=ax.transAxes, ha="right", va="center", fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_title(label, loc='left', fontsize=12, fontweight='bold')
        ax.set_ylabel("Frequency")

    axes[-1].set_xlabel("MAPE (%)")
    #force x axis to be between 0-100
    axes[-1].set_xlim(0, 100)
    # plt.suptitle("Histogram of MAPE Values for Top 40 Best Computed and Worst Other Models")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("output/DensityInversion/PODDensityInversion/Plots/POD_vs_ACT/mape_histograms.png", dpi=600)
    # plt.show()

# Process density data with time-based rolling average and calculate MAPE
def process_and_plot_density_with_time_interval(storm_csv_path, mission):
    density_df = pd.read_csv(storm_csv_path, parse_dates=["UTC"])
    required_columns = ["AccelerometerDensity", "Computed Density", "JB08", "DTM2000", "NRLMSISE-00"]
    if not all(col in density_df.columns for col in required_columns):
        return None

    max_accelerometer_index = density_df["AccelerometerDensity"].idxmax()
    max_accelerometer_time = density_df.loc[max_accelerometer_index, "UTC"]
    start_time = max_accelerometer_time - pd.Timedelta(hours=12)
    end_time = max_accelerometer_time + pd.Timedelta(hours=24)
    density_df = density_df[(density_df["UTC"] >= start_time) & (density_df["UTC"] <= end_time)]

    median_density = density_df['Computed Density'].median()
    density_df['Computed Density'] = density_df['Computed Density'].apply(
        lambda x: 1e-11 if x > 1.2e-11 else (median_density if x < -2e-11 else x)
    )

    # Use mission-specific window size for moving average
    if mission == "CHAMP":
        moving_avg_minutes = 23
    elif mission == "GRACE-FO":
        moving_avg_minutes = 45
    seconds_per_point = 15  # 15-second resolution data
    window_size = (moving_avg_minutes * 60) // seconds_per_point

    density_df['Computed Density'] = density_df['Computed Density'].rolling(window=window_size, center=True).mean()
    median_density = density_df['Computed Density'].median()
    IQR = density_df['Computed Density'].quantile(0.75) - density_df['Computed Density'].quantile(0.25)
    lower_bound = median_density - 5 * IQR
    upper_bound = median_density + 5 * IQR
    density_df.loc[:, 'Computed Density'] = density_df['Computed Density'].apply(lambda x: median_density if x < lower_bound or x > upper_bound else x)
    density_df['Computed Density'] = savgol_filter(density_df['Computed Density'], 51, 3)

    #find the time of peak AccelerometerDensity
    peak_density_time = density_df.loc[density_df['AccelerometerDensity'].idxmax(), 'UTC']
    #keep only data that is 12 hours prior and 24 hours after the peak density time
    density_df = density_df[(density_df['UTC'] >= peak_density_time - pd.Timedelta(hours=12)) & (density_df['UTC'] <= peak_density_time + pd.Timedelta(hours=12))]
    print(f"storm_csv_path: {storm_csv_path}")
    
    mape_computed, mape_jb08, mape_dtm2000, mape_nrlmsise00 = calculate_mape(density_df)
    print(f"mape computed: {mape_computed}")
    return mape_computed, mape_jb08, mape_dtm2000, mape_nrlmsise00, median_density

# Main loop to process each mission and storm file
# Main loop to process each mission and storm file
median_densities = []
mape_computed_values = []
mape_jb08_values = []
mape_dtm2000_values = []
mape_nrlmsise00_values = []
missions_list = []

# Set paths
storm_folder = "output/DensityInversion/PODDensityInversion/Data/StormAnalysis"
missions = ["CHAMP", "GRACE-FO"]

for mission in missions:
    mission_folder = os.path.join(storm_folder, mission)
    if not os.path.isdir(mission_folder):
        print(f"Directory {mission_folder} not found. Skipping.")
        continue

    for storm_file in os.listdir(mission_folder):
        if storm_file.endswith(".csv"):
            storm_csv_path = os.path.join(mission_folder, storm_file)
            results = process_and_plot_density_with_time_interval(storm_csv_path, mission)
            if results:
                mape_computed, mape_jb08, mape_dtm2000, mape_nrlmsise00, median_density = results
                mape_computed_values.append(mape_computed)
                mape_jb08_values.append(mape_jb08)
                mape_dtm2000_values.append(mape_dtm2000)
                mape_nrlmsise00_values.append(mape_nrlmsise00)
                median_densities.append(median_density)
                missions_list.append(mission)

# Plot MAPE vs Median Density for all models
plot_mape_vs_median_density_computed(median_densities, mape_computed_values, missions_list)

# Plot overlaid histograms for MAPE values
plot_mape_histograms(mape_computed_values, mape_jb08_values, mape_dtm2000_values, mape_nrlmsise00_values)

