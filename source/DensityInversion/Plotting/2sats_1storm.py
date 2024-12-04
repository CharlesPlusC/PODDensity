import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
tsx_storm_path = "output/PODDensityInversion/Data/StormAnalysis/TerraSAR-X/TerraSAR-X_storm_density_9_1_20240511214320.csv"
gfo_storm_path = "output/PODDensityInversion/Data/StormAnalysis/GRACE-FO/GRACE-FO_storm_density_22.csv"
tsx_storm_df = pd.read_csv(tsx_storm_path)
gfo_storm_df = pd.read_csv(gfo_storm_path)

# Ensure 'UTC' column is in datetime format and set as index
tsx_storm_df['UTC'] = pd.to_datetime(tsx_storm_df['UTC'])
tsx_storm_df.set_index('UTC', inplace=True)
gfo_storm_df['UTC'] = pd.to_datetime(gfo_storm_df['UTC'])
gfo_storm_df.set_index('UTC', inplace=True)

# Replace densities within specified intervals, now using the correct year (2023)
median_density_tsx = tsx_storm_df['Computed Density'].median()
median_density_gfo = gfo_storm_df['Computed Density'].median()

tsx_storm_df.loc['2023-04-23 18:13':'2023-04-23 18:45', 'Computed Density'] = median_density_tsx
tsx_storm_df.loc['2023-04-23 21:37':'2023-04-23 22:45', 'Computed Density'] = median_density_tsx
# tsx_storm_df.loc['2023-04-24 11:00':'2023-04-23 12:00', 'Computed Density'] = median_density_tsx
gfo_storm_df.loc['2023-04-23 11:16':'2023-04-23 12:17', 'Computed Density'] = median_density_gfo
gfo_storm_df.loc['2023-04-23 17:57':'2023-04-23 18:45', 'Computed Density'] = median_density_gfo

# Process DataFrames: Filter invalid densities, calculate rolling mean, and apply median-clipping
for storm_df in [tsx_storm_df, gfo_storm_df]:
    storm_df.loc[(storm_df['Computed Density'] <= -1e-11) | (storm_df['Computed Density'] >= 0.3e-11), 'Computed Density'] = None
    median_density = storm_df['Computed Density'].median()
    window_size = 180
    storm_df['Computed Density'] = (
        storm_df['Computed Density']
        .rolling(window=window_size, min_periods=1, center=True)
        .mean()
    )
    storm_df['Computed Density'] = storm_df['Computed Density'].apply(
        lambda x: median_density if pd.notna(x) and (x > 10 * median_density or x < median_density / 10) else x
    )

# Calculate MAPE for TerraSAR-X, GRACE-FO, and density models
gfo_mape = (abs(gfo_storm_df['Computed Density'] - gfo_storm_df['AccelerometerDensity']) / gfo_storm_df['AccelerometerDensity']).mean() * 100
jb08_mape = (abs(gfo_storm_df['JB08'] - gfo_storm_df['AccelerometerDensity']) / gfo_storm_df['AccelerometerDensity']).mean() * 100
dtm2000_mape = (abs(gfo_storm_df['DTM2000'] - gfo_storm_df['AccelerometerDensity']) / gfo_storm_df['AccelerometerDensity']).mean() * 100
nrlmsise_mape = (abs(gfo_storm_df['NRLMSISE-00'] - gfo_storm_df['AccelerometerDensity']) / gfo_storm_df['AccelerometerDensity']).mean() * 100

# Plot final results
common_xlim = (gfo_storm_df.index.min(), gfo_storm_df.index.max())
common_ylim = (3e-13, 4e-12)

fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Top subplot: TerraSAR-X and GRACE-FO densities with MAPE in legend
ax[0].plot(gfo_storm_df.index, gfo_storm_df['AccelerometerDensity'], label="Accelerometer Density: GFO", color='#FF0000', linewidth=1, linestyle='--')
ax[0].plot(gfo_storm_df.index, gfo_storm_df['Computed Density'], label=f"POD Density: GFO (MAPE={gfo_mape:.2f}%)", color='#1E90FF')
ax[0].plot(tsx_storm_df.index, tsx_storm_df['Computed Density'], label=f"POD Density: TSX", color='#20B2AA')
ax[0].set_ylabel("Density (kg/m³)", fontsize=12)
ax[0].set_yscale('log')
ax[0].legend(fontsize=12)
ax[0].grid()
ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].set_xlim(common_xlim)
ax[0].set_ylim(common_ylim)

# Bottom subplot: GRACE-FO accelerometer and model densities with MAPE in legend
ax[1].plot(gfo_storm_df.index, gfo_storm_df['AccelerometerDensity'], label="Accelerometer Density: GFO", color='#FF0000', linewidth=1, linestyle='--')
ax[1].plot(gfo_storm_df.index, gfo_storm_df['JB08'], label=f"JB08 (MAPE={jb08_mape:.2f}%)", color='#FF1493')
ax[1].plot(gfo_storm_df.index, gfo_storm_df['DTM2000'], label=f"DTM2000 (MAPE={dtm2000_mape:.2f}%)", color='#9932CC')
ax[1].plot(gfo_storm_df.index, gfo_storm_df['NRLMSISE-00'], label=f"NRLMSISE-00 (MAPE={nrlmsise_mape:.2f}%)", color='#FFD700')
ax[1].set_ylabel("Density (kg/m³)", fontsize=12)
ax[1].set_xlabel("Date-Time UTC", fontsize=12)
ax[1].set_yscale('log')
ax[1].legend(fontsize=12)
ax[1].grid()
ax[1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].set_xlim(common_xlim)
ax[1].set_ylim(common_ylim)

plt.savefig("output/PODDensityInversion/Plots/POD_vs_ACT/2sats_1storm.png", dpi=600)
# plt.show()
