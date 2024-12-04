from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, welch
import numpy as np

# File paths
storm_df_path = "output/PODDensityInversion/Data/StormAnalysis/GRACE-FO/GRACE-FO-A_2023-05-06_density_inversion.csv"
edr_df_path = "output/EDR/Data/GRACE-FO/EDR_GRACE-FO-A_2023-05-06_density_inversion.csv"
nrt_df_path = "output/PODDensityInversion/Data/StormAnalysis/GRACE-FO/NRT_GRACE-FO-A_2023-05-06_density_inversion.csv"

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
    ax1.plot(mega_df['UTC'], mega_df['Computed Density'], label=f'POD Density', color=colors['POD'])
    ax1.plot(mega_df['UTC'], mega_df['NRT Density'], label=f'NRT Density', color=colors['NRT'])
    ax1.plot(mega_df['UTC'], mega_df['AccelerometerDensity'], label='Accelerometer', color=colors['ACT'], linewidth=1, alpha=1, linestyle='--')
    ax1.plot(mega_df['UTC'], mega_df['EDR Density'], label=f'EDR', color=colors['EDR'])

    ax1.set_yscale('log')
    ax1.set_ylabel('Neutral Density (kg/m³)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='dotted')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.axvline(datetime.strptime('2023-05-06 00:50:00', '%Y-%m-%d %H:%M:%S'), color='black', linestyle='--')
    ax1.text(datetime.strptime('2023-05-06 00:50:00', '%Y-%m-%d %H:%M:%S'), 1e-12, 'SSC', fontsize=10, rotation=90, verticalalignment='bottom')

    # Bottom subplot: Models
    ax2.plot(mega_df['UTC'], mega_df['JB08'], label=f'JB08', color=colors['JB08'])
    ax2.plot(mega_df['UTC'], mega_df['NRLMSISE-00'], label=f'NRLMSISE-00', color=colors['NRLMSISE-00'])
    ax2.plot(mega_df['UTC'], mega_df['DTM2000'], label=f'DTM2000', color=colors['DTM2000'])
    ax2.plot(mega_df['UTC'], mega_df['AccelerometerDensity'], label='Accelerometer', color=colors['ACT'], linewidth=1, alpha=1, linestyle='--')
    ax2.set_yscale('log')
    ax2.set_xlabel('Date-Time (UTC)', fontsize=12)
    ax2.set_ylabel('Neutral Density (kg/m³)', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, which='both', linestyle='dotted')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.axvline(datetime.strptime('2023-05-06 00:50:00', '%Y-%m-%d %H:%M:%S'), color='black', linestyle='--')
    ax2.text(datetime.strptime('2023-05-06 00:50:00', '%Y-%m-%d %H:%M:%S'), 1e-12, 'SSC', fontsize=10, rotation=90, verticalalignment='bottom')

    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
    ax2.set_xlim([datetime.strptime('2023-05-06 00:35:00', '%Y-%m-%d %H:%M:%S'), mega_df['UTC'].max()])
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_asd_from_csv(csv_path):
    data = pd.read_csv(csv_path)
    data['UTC'] = pd.to_datetime(data['UTC'])

    # Extract time series for plotting
    time_series_top = {
        'ACT': data['AccelerometerDensity'],
        'EDR': data['EDR_rolling'],
        'POD': data['POD_rolling'],
        'NRT': data['NRT_rolling']
    }

    time_series_bottom = {
        'ACT': data['AccelerometerDensity'],
        'DTM2000': data['DTM2000'],
        'JB08': data['JB08'],
        'NRLMSISE-00': data['NRLMSISE00']
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

    # Orbital frequencies
    orbit_frequency = 1 / 5400
    half_orbit_frequency = 1 / 2700
    quarter_orbit_frequency = 1 / 1350

    # Plot Amplitude Spectral Density
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Top subplot
    for label, series in time_series_top.items():
        f, Pxx = welch(series.dropna(), fs=1 / 15, nperseg=1024)
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
    for label, series in time_series_bottom.items():
        f, Pxx = welch(series.dropna(), fs=1 / 15, nperseg=1024)
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

    # Set x-axis and y-axis limits
    ax1.set_xlim([1e-4, 1e-2])
    ax2.set_xlim([1e-4, 1e-2])
    ax1.set_ylim([1e-14, 3e-11])
    ax2.set_ylim([1e-14, 3e-11])

    # Add text annotations
    for ax in [ax1, ax2]:
        ax.text(orbit_frequency, 1e-13, '~Once per orbit', fontsize=10, rotation=90, verticalalignment='bottom')
        ax.text(half_orbit_frequency, 1e-13, '~Half-orbit', fontsize=10, rotation=90, verticalalignment='bottom')
        ax.text(quarter_orbit_frequency, 1e-13, '~Quarter-orbit', fontsize=10, rotation=90, verticalalignment='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # Call the function
    plot_asd_from_csv("output/PaperFigures/AllMethods1Storm/Processed_ASD_Data_For_Plotting.csv")
    
    # Reload processed data for plotting
    mega_df = pd.read_csv("output/PaperFigures/AllMethods1Storm/Data_For_Plotting.csv")
    mega_df['UTC'] = pd.to_datetime(mega_df['UTC'])
    plot_time_series(mega_df)