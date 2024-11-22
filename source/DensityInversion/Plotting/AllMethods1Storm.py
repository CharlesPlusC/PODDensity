from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

storm_df = "output/DensityInversion/PODDensityInversion/Data/OneStormAllMethods/ACT_vs_POD_2023_05_06_GRACE-FOA.csv"
edr_df = "output/DensityInversion/PODDensityInversion/Data/OneStormAllMethods/EDR_2023_05_06_GFOA.csv"
nrt_df = "output/DensityInversion/PODDensityInversion/Data/OneStormAllMethods/NRT_2023_05_06_GFOA.csv"
nrt_df = pd.read_csv(nrt_df)
storm_df = pd.read_csv(storm_df)
edr_df = pd.read_csv(edr_df)
storm_df['UTC'] = pd.to_datetime(storm_df['UTC'])
edr_df['UTC'] = pd.to_datetime(edr_df['UTC'])
nrt_df['UTC'] = pd.to_datetime(nrt_df['UTC'])
# rename the POD_Density column to Computed Density
storm_df.rename(columns={'POD_Density': 'Computed Density'}, inplace=True)
storm_df = storm_df[storm_df['Computed Density'] > -1.7e-11]
storm_df = storm_df[storm_df['Computed Density'] < 1e-11]
median_density = storm_df['Computed Density'].median()
window_size = 90
storm_df['Computed Density'] = storm_df['Computed Density'].rolling(window=window_size, min_periods=1, center=True).mean()
storm_df['Computed Density'] = storm_df['Computed Density'].apply(lambda x: median_density if x > 12 * median_density or x < median_density / 12 else x)
storm_df['Computed Density'] = savgol_filter(storm_df['Computed Density'], 51, 3)

#rename 'Computed Density' to NRT_Density
nrt_df.rename(columns={'Computed Density': 'NRT Density'}, inplace=True)
nrt_df = nrt_df[nrt_df['NRT Density'] > -1.7e-11]
nrt_df = nrt_df[nrt_df['NRT Density'] < 1e-11]
window_size = 210
nrt_df['NRT Density'] = nrt_df['NRT Density'].rolling(window=window_size, min_periods=1, center=False).mean()
nrt_df['NRT Density'] = nrt_df['NRT Density'].apply(lambda x: median_density if x > 12 * median_density or x < median_density / 12 else x)
nrt_df['NRT Density'] = savgol_filter(nrt_df['NRT Density'], 51, 3)

# Define the custom time period for MAPE calculation
start_date = datetime(2023, 5, 6, 0, 0)
end_date = datetime(2023, 5, 6, 17, 0)

def calculate_MAPE(df, model, start_date, end_date):
    # Filter the dataframe to the specified date range
    filtered_df = df[(df['UTC'] >= start_date) & (df['UTC'] <= end_date)]
    # Calculate MAPE on the filtered data
    filtered_df['MAPE'] = (abs(filtered_df['AccelerometerDensity'] - filtered_df[model]) / filtered_df['AccelerometerDensity']) * 100
    return filtered_df['MAPE'].median()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top subplot: Accelerometer and Computed Density
# ax1.plot(storm_df['UTC'], storm_df['Computed Density'], label=f'POD Density, MAPE: {calculate_MAPE(storm_df, "Computed Density", start_date, end_date):.2f}', color='blue')
ax1.plot(nrt_df['UTC'], nrt_df['NRT Density'], label=f'NRT Density', color='pink')
ax1.plot(storm_df['UTC'], storm_df['AccelerometerDensity'], label='Accelerometer', color='purple', linewidth=2, alpha=0.8)
ax1.plot(edr_df['UTC'], edr_df['rho_eff'].rolling(window=200, center=True).mean()*10, label='EDR', color='orange')
ax1.set_yscale('log')
ax1.set_ylabel('Neutral Density (kg/m³)')
#set y-axis limits
ax1.set_ylim(2e-13, 2e-12)
ax1.legend()
ax1.grid(True, which='both', axis='y', linestyle='dotted', color='gray', linewidth=0.75)
ax1.set_title('Density Inversion for GRACE-FO')

# Bottom subplot: Accelerometer, NRLMSISE-00, and JB08
ax2.plot(storm_df['UTC'], storm_df['NRLMSISE-00'], 
         label=f'NRLMSISE-00, MAPE: {calculate_MAPE(storm_df, "NRLMSISE-00", start_date, end_date):.2f}', color='red')
ax2.plot(storm_df['UTC'], storm_df['JB08'], 
         label=f'JB08, MAPE: {calculate_MAPE(storm_df, "JB08", start_date, end_date):.2f}', color='green')
ax2.plot(storm_df['UTC'], storm_df['DTM2000'], 
         label=f'DTM2000, MAPE: {calculate_MAPE(storm_df, "DTM2000", start_date, end_date):.2f}', color='orange')
ax2.plot(storm_df['UTC'], storm_df['AccelerometerDensity'], label='Accelerometer', color='purple', linewidth=2, alpha=0.8)
ax2.set_yscale('log')
ax2.set_xlabel('Time (UTC)')
ax2.set_ylabel('Neutral Density (kg/m³)')
ax2.legend()
ax2.set_ylim(2e-13, 2e-12)
ax2.grid(True, which='both', axis='y', linestyle='dotted', color='gray', linewidth=0.75)

# Define the custom x-axis limits
plt.xlim(start_date, end_date)
plt.ylim(2e-13, 2e-12)

# Format x-axis to display date and hour
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
fig.autofmt_xdate()  # Auto-format for better readability

plt.show()
