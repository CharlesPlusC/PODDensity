import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data CSVs
tsx_plot_data = pd.read_csv("output/PaperFigures/2sats_1storm/TSX_plot_data.csv")
gfo_plot_data = pd.read_csv("output/PaperFigures/2sats_1storm/GFO_plot_data.csv")

tsx_plot_data['Timestamp'] = pd.to_datetime(tsx_plot_data['Timestamp'])
tsx_plot_data.set_index('Timestamp', inplace=True)

gfo_plot_data['Timestamp'] = pd.to_datetime(gfo_plot_data['Timestamp'])
gfo_plot_data.set_index('Timestamp', inplace=True)

# Calculate MAPE for GRACE-FO
gfo_mape = (abs(gfo_plot_data['GFO_Computed_Density'] - gfo_plot_data['GFO_Accelerometer_Density']) / gfo_plot_data['GFO_Accelerometer_Density']).mean() * 100
jb08_mape = (abs(gfo_plot_data['GFO_JB08'] - gfo_plot_data['GFO_Accelerometer_Density']) / gfo_plot_data['GFO_Accelerometer_Density']).mean() * 100
dtm2000_mape = (abs(gfo_plot_data['GFO_DTM2000'] - gfo_plot_data['GFO_Accelerometer_Density']) / gfo_plot_data['GFO_Accelerometer_Density']).mean() * 100
nrlmsise_mape = (abs(gfo_plot_data['GFO_NRLMSISE-00'] - gfo_plot_data['GFO_Accelerometer_Density']) / gfo_plot_data['GFO_Accelerometer_Density']).mean() * 100

# Plot final results
common_xlim = (gfo_plot_data.index.min(), gfo_plot_data.index.max())
common_ylim = (3e-13, 4e-12)

fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Top subplot: TerraSAR-X and GRACE-FO densities with MAPE in legend
ax[0].plot(gfo_plot_data.index, gfo_plot_data['GFO_Accelerometer_Density'], label="Accelerometer Density: GFO", color='#FF0000', linewidth=1, linestyle='--')
ax[0].plot(gfo_plot_data.index, gfo_plot_data['GFO_Computed_Density'], label=f"POD Density: GFO (MAPE={gfo_mape:.2f}%)", color='#1E90FF')
ax[0].plot(tsx_plot_data.index, tsx_plot_data['TSX_Computed_Density'], label=f"POD Density: TSX", color='#20B2AA')
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
ax[1].plot(gfo_plot_data.index, gfo_plot_data['GFO_Accelerometer_Density'], label="Accelerometer Density: GFO", color='#FF0000', linewidth=1, linestyle='--')
ax[1].plot(gfo_plot_data.index, gfo_plot_data['GFO_JB08'], label=f"JB08 (MAPE={jb08_mape:.2f}%)", color='#FF1493')
ax[1].plot(gfo_plot_data.index, gfo_plot_data['GFO_DTM2000'], label=f"DTM2000 (MAPE={dtm2000_mape:.2f}%)", color='#9932CC')
ax[1].plot(gfo_plot_data.index, gfo_plot_data['GFO_NRLMSISE-00'], label=f"NRLMSISE-00 (MAPE={nrlmsise_mape:.2f}%)", color='#FFD700')
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

plt.show()