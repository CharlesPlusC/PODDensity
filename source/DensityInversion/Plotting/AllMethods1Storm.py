from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, welch
import numpy as np

# File paths
storm_df_path = "output/PODDensityInversion/Data/GRACE-FO-A/GRACE-FO-A_2023-05-06_density_inversion.csv"
edr_df_path = "output/EDR/Data/GRACE-FO-A/EDR_GRACE-FO-A_2023-05-06_density_inversion.csv"
nrt_df_path = "output/PODDensityInversion/Data/GRACE-FO-A/NRT_GRACE-FO-A_2023-05-06_density_inversion.csv"

# Load data
storm_df = pd.read_csv(storm_df_path)
edr_df = pd.read_csv(edr_df_path)
nrt_df = pd.read_csv(nrt_df_path)

# Convert UTC columns to datetime
storm_df['UTC'] = pd.to_datetime(storm_df['UTC'])
edr_df['UTC'] = pd.to_datetime(edr_df['UTC'])
nrt_df['UTC'] = pd.to_datetime(nrt_df['UTC'])

# Preprocess storm_df
storm_df.rename(columns={'POD_Density': 'Computed Density'}, inplace=True)
storm_df['Computed Density'] = storm_df['Computed Density'].rolling(window=180, center=True).mean()
median_density = storm_df['Computed Density'].mean()
storm_df['Computed Density'] = storm_df['Computed Density'].apply(
    lambda x: 1e-11 if x > 1.2e-11 else (median_density if x < -2e-11 else x)
)
storm_df['Computed Density'] = savgol_filter(storm_df['Computed Density'], 51, 3)

# Preprocess edr_df
edr_df['EDR Density'] = edr_df['EDR Density'].rolling(window=180, center=False).mean()
edr_df['EDR Density'] = savgol_filter(edr_df['EDR Density'], 51, 3)

# Preprocess nrt_df
nrt_df.rename(columns={'Computed Density': 'NRT Density'}, inplace=True)
nrt_df['NRT Density'] = nrt_df['NRT Density'].rolling(window=180, center=True).mean()
nrt_df['NRT Density'] = nrt_df['NRT Density'].apply(
    lambda x: 1e-11 if x > 1.2e-11 else (median_density if x < -2e-11 else x)
)
nrt_df['NRT Density'] = savgol_filter(nrt_df['NRT Density'], 51, 3)

# Merge data frames on UTC
mega_df = pd.merge(storm_df[['UTC', 'Computed Density', 'JB08', 'DTM2000']],
                   edr_df[['UTC', 'EDR Density', 'AccelerometerDensity', 'NRLMSISE-00']], on='UTC', how='inner')
mega_df = pd.merge(mega_df, nrt_df[['UTC', 'NRT Density']], on='UTC', how='inner')

#force the df tp start at 2023-05-06 00:00:45 and end at 2023-05-06 17:30:00
mega_df = mega_df[(mega_df['UTC'] >= '2023-05-06 00:00:45') & (mega_df['UTC'] <= '2023-05-06 17:30:00')]

# Function to calculate MAPE
def calculate_MAPE(df, model, reference='AccelerometerDensity'):
    if model == 'EDR Density':  # Check if the model is EDR Density
        df = df.copy()  # Avoid modifying the original dataframe
        df[model] = df[model] / 1.8  # Divide EDR Density values by 2

    valid = df[reference] != 0  # Avoid division by zero
    mape = (abs(df[model] - df[reference]) / df[reference])[valid].mean() * 100
    return mape

# Calculate MAPE for each model
mape_pod = calculate_MAPE(mega_df, 'Computed Density')
mape_edr = calculate_MAPE(mega_df, 'EDR Density')
mape_nrt = calculate_MAPE(mega_df, 'NRT Density')
mape_jb08 = calculate_MAPE(mega_df, 'JB08')
mape_nrlmsise = calculate_MAPE(mega_df, 'NRLMSISE-00')
mape_dtm2000 = calculate_MAPE(mega_df, 'DTM2000')

# Plot time series
# Plot time series
def plot_time_series(mega_df):
    colors = {
        'EDR': '#32CD32',
        'POD': '#1E90FF',
        'NRT': '#20B2AA',
        'ACT': '#FF0000',
        'JB08': '#FF1493',
        'DTM2000': '#9932CC',
        'NRLMSISE-00': '#FFD700'
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    
    # Top subplot: Computed Densities
    ax1.plot(mega_df['UTC'], mega_df['Computed Density'], label=f'POD Density, MAPE: {mape_pod:.2f}%', color=colors['POD'])
    ax1.plot(mega_df['UTC'], mega_df['NRT Density'], label=f'NRT Density, MAPE: {mape_nrt:.2f}%', color=colors['NRT'])
    ax1.plot(mega_df['UTC'], mega_df['AccelerometerDensity'], label='Accelerometer', color=colors['ACT'], linewidth=1, alpha=1, linestyle='--')
    
    #de-bias the EDR Denstiy relative to the Accelerometer Density
    mega_df['EDR Density'] = mega_df['EDR Density']/1.8
    
    ax1.plot(mega_df['UTC'], mega_df['EDR Density'], label=f'EDR, MAPE: {mape_edr:.2f}%', color=colors['EDR'])

    ax1.set_yscale('log')
    ax1.set_ylabel('Neutral Density (kg/m³)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='dotted')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    #add a vertical black dotted line at 2023-05-06 00:50:00
    ax1.axvline(datetime.strptime('2023-05-06 00:50:00', '%Y-%m-%d %H:%M:%S'), color='black', linestyle='--')
    ax1.text(datetime.strptime('2023-05-06 00:50:00', '%Y-%m-%d %H:%M:%S'), 1e-12, 'SSC', fontsize=10, rotation=90, verticalalignment='bottom')

    # Bottom subplot: Models
    ax2.plot(mega_df['UTC'], mega_df['JB08'], label=f'JB08, MAPE: {mape_jb08:.2f}%', color=colors['JB08'])
    ax2.plot(mega_df['UTC'], mega_df['NRLMSISE-00'], label=f'NRLMSISE-00, MAPE: {mape_nrlmsise:.2f}%', color=colors['NRLMSISE-00'])
    ax2.plot(mega_df['UTC'], mega_df['DTM2000'], label=f'DTM2000, MAPE: {mape_dtm2000:.2f}%', color=colors['DTM2000'])
    ax2.plot(mega_df['UTC'], mega_df['AccelerometerDensity'], label='Accelerometer', color=colors['ACT'], linewidth=1, alpha=1, linestyle='--')
    ax2.set_yscale('log')
    ax2.set_xlabel('Date-Time (UTC)', fontsize=12)
    ax2.set_ylabel('Neutral Density (kg/m³)', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, which='both', linestyle='dotted')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.axvline(datetime.strptime('2023-05-06 00:50:00', '%Y-%m-%d %H:%M:%S'), color='black', linestyle='--')
    #label the vertical line black text saying "SSC"
    ax2.text(datetime.strptime('2023-05-06 00:50:00', '%Y-%m-%d %H:%M:%S'), 1e-12, 'SSC', fontsize=10, rotation=90, verticalalignment='bottom')


    # Format x-axis
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))

    ax2.set_xlim([datetime.strptime('2023-05-06 00:35:00', '%Y-%m-%d %H:%M:%S'), mega_df['UTC'].max()])
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("output/PODDensityInversion/Plots/POD_vs_ACT/POD_vs_ACT_vs_EDR_vs_NRT_vs_Models_tseries.png", dpi=600)
    # plt.show()

# Call the plotting functions
# plot_time_series(mega_df)

############################################################################################################################################################################
# from datetime import datetime
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.signal import savgol_filter, welch
# import numpy as np

# # File paths
# storm_df_path = "output/PODDensityInversion/Data/GRACE-FO-A/GRACE-FO-A_2023-05-06_density_inversion.csv"
# edr_df_path = "output/EDR/Data/GRACE-FO-A/EDR_GRACE-FO-A_2023-05-06_density_inversion.csv"
# nrt_df_path = "output/PODDensityInversion/Plots/GRACE-FO-A/Accelerometer_benchmark/ACTvsEDRvsPOD/NRT_2023_05_06_GFOA.csv"

# # Load data
# storm_df = pd.read_csv(storm_df_path)
# edr_df = pd.read_csv(edr_df_path)
# nrt_df = pd.read_csv(nrt_df_path)

# # Convert UTC columns to datetime
# storm_df['UTC'] = pd.to_datetime(storm_df['UTC'])
# edr_df['UTC'] = pd.to_datetime(edr_df['UTC'])
# nrt_df['UTC'] = pd.to_datetime(nrt_df['UTC'])

# # Preprocess storm_df
# storm_df.rename(columns={'POD_Density': 'Computed Density'}, inplace=True)
# storm_df['Computed Density'] = storm_df['Computed Density'].rolling(window=180, center=True).mean()
# median_density = storm_df['Computed Density'].median()
# storm_df['Computed Density'] = storm_df['Computed Density'].apply(
#     lambda x: 1e-11 if x > 1.2e-11 else (median_density if x < -2e-11 else x)
# )
# storm_df['Computed Density'] = savgol_filter(storm_df['Computed Density'], 51, 3)

# # Preprocess edr_df
# edr_df['EDR Density'] = edr_df['EDR Density'].rolling(window=180, center=False).mean()
# edr_df['EDR Density'] = savgol_filter(edr_df['EDR Density'], 51, 3)

# # Preprocess nrt_df
# nrt_df.rename(columns={'Computed Density': 'NRT Density'}, inplace=True)
# nrt_df['NRT Density'] = nrt_df['NRT Density'].rolling(window=180, center=True).mean()
# nrt_df['NRT Density'] = nrt_df['NRT Density'].apply(
#     lambda x: 1e-11 if x > 1.2e-11 else (median_density if x < -2e-11 else x)
# )
# nrt_df['NRT Density'] = savgol_filter(nrt_df['NRT Density'], 51, 3)

# # Merge data frames on UTC
# mega_df = pd.merge(storm_df[['UTC', 'Computed Density', 'JB08', 'DTM2000']],
#                    edr_df[['UTC', 'EDR Density', 'AccelerometerDensity', 'NRLMSISE-00']], on='UTC', how='inner')
# mega_df = pd.merge(mega_df, nrt_df[['UTC', 'NRT Density']], on='UTC', how='inner')

def act_edr_pod_nrt_ASD(POD_and_ACT_data_path, EDR_data_path, NRT_data_path):
    from scipy.signal import welch
    import matplotlib.pyplot as plt
    import numpy as np

    POD_and_ACT_data = pd.read_csv(POD_and_ACT_data_path)
    EDR_data = pd.read_csv(EDR_data_path)
    NRT_data = pd.read_csv(NRT_data_path)

    # Load and process data
    POD_and_ACT_data['UTC'] = pd.to_datetime(POD_and_ACT_data['UTC'])
    EDR_data['UTC'] = pd.to_datetime(EDR_data['UTC'])
    NRT_data['UTC'] = pd.to_datetime(NRT_data['UTC'])

    merged_data = pd.merge(POD_and_ACT_data, EDR_data, on='UTC')
    merged_data = pd.merge(merged_data, NRT_data, on='UTC', suffixes=('', '_NRT'))
    merged_data['UTC'] = pd.to_datetime(merged_data['UTC'])
    merged_data['EDR_rolling'] = (merged_data['EDR Density']).rolling(window=180, center=True).mean()
    POD_and_ACT_data['POD_rolling'] = POD_and_ACT_data['Computed Density'].rolling(window=180, center=True).mean()
    merged_data['NRT_rolling'] = merged_data['Computed Density'].rolling(window=180, center=True).mean()

    start_time = pd.to_datetime('2023-05-06 00:00:00')
    end_time = pd.to_datetime('2023-05-06 17:30:00')
    merged_data = merged_data[merged_data['UTC'] <= end_time]
    merged_data = merged_data[merged_data['UTC'] >= start_time]
    POD_and_ACT_data = POD_and_ACT_data[POD_and_ACT_data['UTC'] <= end_time]
    POD_and_ACT_data = POD_and_ACT_data[POD_and_ACT_data['UTC'] >= start_time]

    # Extract time series
    time_series_top = {
        'ACT': merged_data['AccelerometerDensity'],
        'EDR': merged_data['EDR_rolling'],
        'POD': POD_and_ACT_data['POD_rolling'],
        'NRT': merged_data['NRT_rolling']
    }

    time_series_bottom = {
        'ACT': merged_data['AccelerometerDensity'],
        'DTM2000': POD_and_ACT_data['DTM2000'],
        'JB08': POD_and_ACT_data['JB08'],
        'NRLMSISE-00': POD_and_ACT_data['NRLMSISE00']
    }

    colors = {
        'EDR': '#32CD32',
        'POD': '#1E90FF',
        'NRT': '#20B2AA',
        'ACT': '#FF0000',
        'NRLMSISE-00': '#FFD700',
        'DTM2000': '#9932CC',
        'JB08': '#FF1493'
    }

    # Orbital frequencies based on data sampled every 15 seconds
    orbit_frequency = 1 / (5400)  # Once per orbit (5400 seconds)
    half_orbit_frequency = 1 / (2700)  # Every half orbit
    quarter_orbit_frequency = 1 / (1350)  # Every quarter orbit

    # Plot Amplitude Spectral Density
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Top subplot
    for label, data in time_series_top.items():
        f, Pxx = welch(data.dropna(), fs=1 / 15, nperseg=1024)  # Adjust fs based on actual time differences
        ASD = np.sqrt(Pxx)
        ax1.loglog(f, ASD, label=label, color=colors[label])
    ax1.axvline(orbit_frequency, color='black', linestyle='--')
    ax1.axvline(half_orbit_frequency, color='black', linestyle='--')
    ax1.axvline(quarter_orbit_frequency, color='black', linestyle='--')
    ax1.set_title('ACT, EDR, POD, and NRT', fontsize=12)
    ax1.set_ylabel('ASD (kg/m³/Hz⁰⋅⁵)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Bottom subplot
    for label, data in time_series_bottom.items():
        f, Pxx = welch(data.dropna(), fs=1 / 15, nperseg=1024)
        ASD = np.sqrt(Pxx)
        ax2.loglog(f, ASD, label=label, color=colors[label])
    ax2.axvline(orbit_frequency, color='black', linestyle='--')
    ax2.axvline(half_orbit_frequency, color='black', linestyle='--')
    ax2.axvline(quarter_orbit_frequency, color='black', linestyle='--')
    ax2.set_title('ACT, DTM2000, JB08, and NRLMSISE-00', fontsize=12)
    ax2.set_xlabel('Frequency', fontsize=12)
    ax2.set_ylabel('ASD (kg/m³/Hz⁰⋅⁵)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Force both x-axes to start at 10^-4 and stop at 10^-2
    ax1.set_xlim([1e-4, 1e-2])
    ax2.set_xlim([1e-4, 1e-2])
    # Set y-limits for both subplots
    ax1.set_ylim([1e-14, 3e-11])
    ax2.set_ylim([1e-14, 3e-11])

    # Set font size for ticks
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # Add text annotations for vertical lines
    ax1.text(orbit_frequency, 1e-13, '~Once per orbit', fontsize=10, rotation=90, verticalalignment='bottom')
    ax1.text(half_orbit_frequency, 1e-13, '~Half-orbit', fontsize=10, rotation=90, verticalalignment='bottom')
    ax1.text(quarter_orbit_frequency, 1e-13, '~Quarter-orbit', fontsize=10, rotation=90, verticalalignment='bottom')

    ax2.text(orbit_frequency, 1e-13, '~Once per orbit', fontsize=10, rotation=90, verticalalignment='bottom')
    ax2.text(half_orbit_frequency, 1e-13, '~Half-orbit', fontsize=10, rotation=90, verticalalignment='bottom')
    ax2.text(quarter_orbit_frequency, 1e-13, '~Quarter-orbit', fontsize=10, rotation=90, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig("output/PODDensityInversion/Plots/POD_vs_ACT/POD_vs_ACT_vs_EDR_vs_NRT_vs_Models_ASD.png", dpi=600)
# Call the ASD plotting function
act_edr_pod_nrt_ASD(storm_df_path, edr_df_path, nrt_df_path)

