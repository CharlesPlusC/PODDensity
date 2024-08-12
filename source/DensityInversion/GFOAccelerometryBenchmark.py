import pandas as pd
import matplotlib.pyplot as plt
import orekit
from orekit.pyhelpers import setup_orekit_curdir
# download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants
from ..tools.utilities import interpolate_positions,calculate_acceleration, get_satellite_info, project_acc_into_HCL
from ..tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df
from ..tools.GFODataReadTools import get_gfo_inertial_accelerations
from .PODDensity import density_inversion
import numpy as np
import datetime
import matplotlib.dates as mdates

# podaac-data-downloader -c GRACEFO_L1B_ASCII_GRAV_JPL_RL04 -d ./GRACE-FO_A_DATA -sd 2023-05-10T06:00:00Z -ed 2023-05-11T23:59:59Z -e ".*" --verbose
# podaac-data-downloader -c GRACEFO_L1B_ASCII_GRAV_JPL_RL04 -d ./GRACE-FO_A_DATA -sd 2024-05-10T06:00:00Z -ed 2024-05-12T23:59:59Z -e ".*" --verbose

def ACT_vs_EDR_vs_POD_NRT_plot(POD_and_ACT_data, EDR_data, NRT_data):
    POD_and_ACT_data['UTC'] = pd.to_datetime(POD_and_ACT_data['UTC'])
    EDR_data['UTC'] = pd.to_datetime(EDR_data['UTC'])
    NRT_data['UTC'] = pd.to_datetime(NRT_data['UTC'])

    merged_data = pd.merge(POD_and_ACT_data, EDR_data, on='UTC')
    merged_data = pd.merge(merged_data, NRT_data, on='UTC', suffixes=('', '_NRT'))
    merged_data['UTC'] = pd.to_datetime(merged_data['UTC'])
    merged_data['EDR_rolling'] = (merged_data['rho_eff'] * 10).rolling(window=180, center=True).mean()
    POD_and_ACT_data['POD_rolling'] = POD_and_ACT_data['POD_Density'].rolling(window=90, center=True).mean()
    merged_data['NRT_rolling'] = merged_data['Computed Density'].rolling(window=90, center=True).mean()

    merged_data = merged_data.iloc[20:]

    start_time = pd.to_datetime('2023-05-06 00:23:00')
    end_time = pd.to_datetime('2023-05-06 17:15:00')
    merged_data = merged_data[merged_data['UTC'] <= end_time]
    merged_data = merged_data[merged_data['UTC'] >= start_time]
    POD_and_ACT_data = POD_and_ACT_data[POD_and_ACT_data['UTC'] <= end_time]
    POD_and_ACT_data = POD_and_ACT_data[POD_and_ACT_data['UTC'] >= start_time]

    median_ACT = merged_data['ACT_Density'].median()
    merged_data['ACT_Density'] = 2 * median_ACT - merged_data['ACT_Density']
    date_of_data = merged_data['UTC'].iloc[0].strftime("%Y-%m-%d")

    # Create a figure with 2 subplots (one for the densities and one for the residuals)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)

    merged_data['ACT_Density'] = -1 * merged_data['ACT_Density']

    colors = {
        'ACT': '#FF4500',
        'EDR': '#32CD32',
        'POD': '#1E90FF',
        'JB08': '#FF1493',
        'DTM2000': '#9932CC',
        'MSISE00': '#FFD700',
        'NRT': '#20B2AA'
    }

    ax1.plot(mdates.date2num(merged_data['UTC']), merged_data['ACT_Density'], label='ACT', color=colors['ACT'], linewidth=1)
    ax1.plot(mdates.date2num(merged_data['UTC']), merged_data['EDR_rolling'], label='EDR', color=colors['EDR'], linewidth=1)
    ax1.plot(mdates.date2num(POD_and_ACT_data['UTC']), POD_and_ACT_data['POD_rolling'], label='POD', color=colors['POD'], linewidth=1)
    ax1.plot(mdates.date2num(merged_data['UTC']), merged_data['NRT_rolling'], label='NRT', linestyle='-', color=colors['NRT'], linewidth=1)
    ax1.plot(mdates.date2num(merged_data['UTC']), merged_data['jb08_rho'], label='JB08', linestyle='--', color=colors['JB08'], linewidth=1)
    ax1.plot(mdates.date2num(merged_data['UTC']), merged_data['dtm2000_rho'], label='DTM2000', linestyle='--', color=colors['DTM2000'], linewidth=1)
    ax1.plot(mdates.date2num(merged_data['UTC']), merged_data['nrlmsise00_rho'], label='NRLMSISE-00', linestyle='--', color=colors['MSISE00'], linewidth=1)

    ax1.set_ylabel('Density (kg/m³)')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax1.grid(which='both', linestyle='-.')
    ax1.set_yscale('log')
    ax1.set_title('GRACE-FO-A Density Comparison:\n' + date_of_data)

    # Add a thick vertical line at 00:27 and label it "SSC"
    ax1.axvline(x=mdates.date2num(pd.to_datetime('2023-05-06 00:27:00')), color='k', linestyle='--', linewidth=2)
    ax1.text(mdates.date2num(pd.to_datetime('2023-05-06 00:27:00')), 2e-12, 'SSC', rotation=90, verticalalignment='bottom')

    # Calculate residuals
    merged_data['Residual_EDR'] = merged_data['ACT_Density'] - merged_data['EDR_rolling']
    POD_and_ACT_data['Residual_POD'] = merged_data['ACT_Density'] - POD_and_ACT_data['POD_rolling']
    merged_data['Residual_JB08'] = merged_data['ACT_Density'] - merged_data['jb08_rho']
    merged_data['Residual_DTM2000'] = merged_data['ACT_Density'] - merged_data['dtm2000_rho']
    merged_data['Residual_MSISE00'] = merged_data['ACT_Density'] - merged_data['nrlmsise00_rho']
    merged_data['Residual_NRT'] = merged_data['ACT_Density'] - merged_data['NRT_rolling']

    ax2.plot(mdates.date2num(merged_data['UTC']), merged_data['Residual_EDR'], color=colors['EDR'], linewidth=1)
    ax2.plot(mdates.date2num(POD_and_ACT_data['UTC']), POD_and_ACT_data['Residual_POD'], color=colors['POD'], linewidth=1)
    ax2.plot(mdates.date2num(merged_data['UTC']), merged_data['Residual_NRT'], color=colors['NRT'], linewidth=1)
    ax2.plot(mdates.date2num(merged_data['UTC']), merged_data['Residual_JB08'], linestyle='--', color=colors['JB08'], linewidth=1)
    ax2.plot(mdates.date2num(merged_data['UTC']), merged_data['Residual_DTM2000'], linestyle='--', color=colors['DTM2000'], linewidth=1)
    ax2.plot(mdates.date2num(merged_data['UTC']), merged_data['Residual_MSISE00'], linestyle='--', color=colors['MSISE00'], linewidth=1)

    # Calculate and display biases
    biases = {
        'EDR': merged_data['Residual_EDR'].mean(),
        'POD': POD_and_ACT_data['Residual_POD'].mean(),
        'JB08': merged_data['Residual_JB08'].mean(),
        'DTM2000': merged_data['Residual_DTM2000'].mean(),
        'MSISE00': merged_data['Residual_MSISE00'].mean(),
        'NRT': merged_data['Residual_NRT'].mean()
    }

    bias_text = "Bias of Residuals:\n"
    for key, value in biases.items():
        bias_text += f"{key}: {value:.2e}\n"

    ax2.text(1.02, 0.5, bias_text, transform=ax2.transAxes, fontsize='small', verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5), color='black')

    ax2.set_ylabel('Residuals (kg/m³)')
    ax2.grid(which='both', linestyle='-.')
    ax2.set_xlabel('UTC Time')

    # Add a thick vertical line at 00:27 and label it "SSC"
    ax2.axvline(x=mdates.date2num(pd.to_datetime('2023-05-06 00:27:00')), color='k', linestyle='--', linewidth=2)
    ax2.text(mdates.date2num(pd.to_datetime('2023-05-06 00:27:00')), 1e12, 'SSC', rotation=90, verticalalignment='bottom')

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Set the x-axis limits
    ax1.set_xlim(start_time, end_time)

    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"output/DensityInversion/PODDensityInversion/Plots/GRACE-FO-A/Accelerometer_benchmark/ACTvsEDRvsPOD/ACT_EDR_POD_NRT_Comparison_{timenow}.png", dpi=600)
    plt.show()

def act_edr_pod_nrt_stats(POD_and_ACT_data, EDR_data, NRT_data):
    # Load and process data
    POD_and_ACT_data['UTC'] = pd.to_datetime(POD_and_ACT_data['UTC'])
    EDR_data['UTC'] = pd.to_datetime(EDR_data['UTC'])
    NRT_data['UTC'] = pd.to_datetime(NRT_data['UTC'])

    merged_data = pd.merge(POD_and_ACT_data, EDR_data, on='UTC')
    merged_data = pd.merge(merged_data, NRT_data, on='UTC', suffixes=('', '_NRT'))
    merged_data['UTC'] = pd.to_datetime(merged_data['UTC'])
    merged_data['EDR_rolling'] = (merged_data['rho_eff'] * 10).rolling(window=180, center=True).mean()
    POD_and_ACT_data['POD_rolling'] = POD_and_ACT_data['POD_Density'].rolling(window=90, center=True).mean()
    merged_data['NRT_rolling'] = merged_data['Computed Density'].rolling(window=90, center=True).mean()

    merged_data = merged_data.iloc[22:]

    start_time = pd.to_datetime('2023-05-06 00:23:00')
    end_time = pd.to_datetime('2023-05-06 17:15:00')
    merged_data = merged_data[merged_data['UTC'] <= end_time]
    merged_data = merged_data[merged_data['UTC'] >= start_time]
    POD_and_ACT_data = POD_and_ACT_data[POD_and_ACT_data['UTC'] <= end_time]
    POD_and_ACT_data = POD_and_ACT_data[POD_and_ACT_data['UTC'] >= start_time]

    median_ACT = merged_data['ACT_Density'].median()
    merged_data['ACT_Density'] = 2 * median_ACT - merged_data['ACT_Density']
    date_of_data = merged_data['UTC'].iloc[0].strftime("%Y-%m-%d")

    merged_data['ACT_Density'] = -1 * merged_data['ACT_Density']

    # Calculate MAPE
    def mape(A, F):
        return 100 * np.mean(np.abs(F - A) / np.abs(A))

    mape_EDR = mape(merged_data['ACT_Density'], merged_data['EDR_rolling'])
    mape_POD = mape(merged_data['ACT_Density'], POD_and_ACT_data['POD_rolling'])
    mape_JB08 = mape(merged_data['ACT_Density'], merged_data['jb08_rho'])
    mape_DTM2000 = mape(merged_data['ACT_Density'], merged_data['dtm2000_rho'])
    mape_MSISE00 = mape(merged_data['ACT_Density'], merged_data['nrlmsise00_rho'])
    mape_NRT = mape(merged_data['ACT_Density'], merged_data['NRT_rolling'])

    # Calculate RMSE
    def rmse(A, F):
        return np.sqrt(np.mean((F - A) ** 2))

    rmse_EDR = rmse(merged_data['ACT_Density'], merged_data['EDR_rolling'])
    rmse_POD = rmse(merged_data['ACT_Density'], POD_and_ACT_data['POD_rolling'])
    rmse_JB08 = rmse(merged_data['ACT_Density'], merged_data['jb08_rho'])
    rmse_DTM2000 = rmse(merged_data['ACT_Density'], merged_data['dtm2000_rho'])
    rmse_MSISE00 = rmse(merged_data['ACT_Density'], merged_data['nrlmsise00_rho'])
    rmse_NRT = rmse(merged_data['ACT_Density'], merged_data['NRT_rolling'])

    # Prepare data for plotting
    mape_residuals = [mape_EDR, mape_POD, mape_JB08, mape_DTM2000, mape_MSISE00, mape_NRT]
    rmse_residuals = [rmse_EDR, rmse_POD, rmse_JB08, rmse_DTM2000, rmse_MSISE00, rmse_NRT]

    colors = {
        'EDR': '#32CD32',
        'POD': '#1E90FF',
        'JB08': '#FF1493',
        'DTM2000': '#9932CC',
        'MSISE00': '#FFD700',
        'NRT': '#20B2AA'
    }
    labels = ['EDR', 'POD', 'JB08', 'DTM2000', 'MSISE00', 'NRT']
    colors = [colors[label] for label in labels]

    # Create subplots for MAPE
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Plot MAPE
    bars_mape = ax.bar(labels, mape_residuals, color=colors)
    ax.set_title('MAPE of Residuals')
    ax.set_ylabel('MAPE (%)')
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis='y', linestyle='--')
    #force y axis to be 0-50
    ax.set_ylim([0, 50])

    for i, bar in enumerate(bars_mape):
        yval = bar.get_height()
        rmse_val = rmse_residuals[i]
        text_yval = yval * 1 if labels[i] == 'NRT' else yval * 1.01
        ax.text(bar.get_x() + bar.get_width() / 2, text_yval, f"{yval:.2f}%\n ({rmse_val:.2e})", ha='center', va='bottom')

    plt.tight_layout()
    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"output/DensityInversion/PODDensityInversion/Plots/GRACE-FO-A/Accelerometer_benchmark/ACTvsEDRvsPOD/ACT_EDR_POD_NRT_MAPE_{timenow}.png", dpi=600)
    plt.show()

def act_edr_pod_nrt_ASD(POD_and_ACT_data, EDR_data, NRT_data):
    from scipy.signal import welch
    import matplotlib.pyplot as plt
    import numpy as np

    POD_and_ACT_data['UTC'] = pd.to_datetime(POD_and_ACT_data['UTC'])
    EDR_data['UTC'] = pd.to_datetime(EDR_data['UTC'])
    NRT_data['UTC'] = pd.to_datetime(NRT_data['UTC'])

    merged_data = pd.merge(POD_and_ACT_data, EDR_data, on='UTC')
    merged_data = pd.merge(merged_data, NRT_data, on='UTC', suffixes=('', '_NRT'))
    merged_data['UTC'] = pd.to_datetime(merged_data['UTC'])
    merged_data['EDR_rolling'] = (merged_data['rho_eff'] * 10).rolling(window=180, center=True).mean()
    POD_and_ACT_data['POD_rolling'] = POD_and_ACT_data['POD_Density'].rolling(window=90, center=True).mean()
    merged_data['NRT_rolling'] = merged_data['Computed Density'].rolling(window=90, center=True).mean()

    merged_data = merged_data.iloc[22:]

    start_time = pd.to_datetime('2023-05-06 00:23:00')
    end_time = pd.to_datetime('2023-05-06 17:15:00')
    merged_data = merged_data[merged_data['UTC'] <= end_time]
    merged_data = merged_data[merged_data['UTC'] >= start_time]
    POD_and_ACT_data = POD_and_ACT_data[POD_and_ACT_data['UTC'] <= end_time]
    POD_and_ACT_data = POD_and_ACT_data[POD_and_ACT_data['UTC'] >= start_time]

    median_ACT = merged_data['ACT_Density'].median()
    merged_data['ACT_Density'] = 2 * median_ACT - merged_data['ACT_Density']
    date_of_data = merged_data['UTC'].iloc[0].strftime("%Y-%m-%d")

    merged_data['ACT_Density'] = -1 * merged_data['ACT_Density']

    time_diff = NRT_data['UTC'].diff().dt.total_seconds().dropna().mode()[0]
    fs = 1.0 / time_diff

    time_series_top = {
        'ACT': merged_data['ACT_Density'],
        'EDR': merged_data['EDR_rolling'],
        'POD': POD_and_ACT_data['POD_rolling'],
        'NRT': merged_data['NRT_rolling'],
        
    }

    time_series_bottom = {
        'ACT': merged_data['ACT_Density'],
        'DTM2000': merged_data['dtm2000_rho'],
        'JB08': merged_data['jb08_rho'],
        'NRLMSISE00': merged_data['nrlmsise00_rho']
    }

    colors = {
        'ACT': '#FF4500',
        'EDR': '#32CD32',
        'POD': '#1E90FF',
        'NRT': '#20B2AA',
        'JB08': '#FF1493',
        'DTM2000': '#9932CC',
        'NRLMSISE00': '#FFD700'
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    for label, data in time_series_top.items():
        f, Pxx = welch(data.dropna(), fs=fs, nperseg=1024)
        ASD = np.sqrt(Pxx)
        ax1.loglog(f, ASD, label=label, color=colors[label])
    
    for label, data in time_series_bottom.items():
        f, Pxx = welch(data.dropna(), fs=fs, nperseg=1024)
        ASD = np.sqrt(Pxx)
        ax2.loglog(f, ASD, label=label, color=colors[label])

    ax1.set_ylabel('ASD (kg/m^3/Hz^0.5)')
    ax1.set_title('Amplitude Spectral Density of: ACT, POD, EDR, NRT')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('ASD (kg/m^3/Hz^0.5)')
    ax2.set_title('Amplitude Spectral Density of: ACT, NRLMSISE00, DTM2000, JB08')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-4, 3e-2])
    plt.ylim([1e-16, 4e-11])
    plt.tight_layout()
    
    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"output/DensityInversion/PODDensityInversion/Plots/GRACE-FO-A/Accelerometer_benchmark/ACTvsEDRvsPOD/ACT_EDR_POD_NRT_ASD_{timenow}.png", dpi=600)
    plt.show()

def ACT_vs_POD(acc_data_path, quat_data_path, sat_name="GRACE-FO-A", max_time=24, calculate_rho_from_vel=True, models_to_query=['JB08', 'DTM2000', 'NRLMSISE00']):
    force_model_config = {
        '90x90gravity': True,
        '3BP': True,
        'solid_tides': True,
        'ocean_tides': True,
        'knocke_erp': True,
        'relativity': True,
        'SRP': True
    }

    inertial_gfo_data = get_gfo_inertial_accelerations(acc_data_path, quat_data_path)
    inertial_gfo_data['UTC'] = pd.to_datetime(inertial_gfo_data['utc_time'])
    inertial_gfo_data.drop(columns=['utc_time'], inplace=True)

    intertial_t0 = inertial_gfo_data['UTC'].iloc[0]
    inertial_act_gfo_data = inertial_gfo_data[
        (inertial_gfo_data['UTC'] >= intertial_t0) & 
        (inertial_gfo_data['UTC'] <= intertial_t0 + pd.Timedelta(hours=max_time))
    ]
    
    sp3_ephemeris_df = sp3_ephem_to_df(satellite=sat_name, date=intertial_t0.strftime("%Y-%m-%d"))
    sp3_ephemeris_df['UTC'] = pd.to_datetime(sp3_ephemeris_df['UTC'])
    sp3_ephemeris_df = sp3_ephemeris_df[
        (sp3_ephemeris_df['UTC'] >= intertial_t0) & 
        (sp3_ephemeris_df['UTC'] <= intertial_t0 + pd.Timedelta(hours=max_time))
    ]

    inertial_act_gfo_ephem = pd.merge(inertial_act_gfo_data, sp3_ephemeris_df, on='UTC', how='inner')
    print(f"head of inertial_act_gfo_ephem: {inertial_act_gfo_ephem.head()}")
    print(f"columns of inertial_act_gfo_ephem: {inertial_act_gfo_ephem.columns}")

    act_x_acc_col, act_y_acc_col, act_z_acc_col = 'inertial_x_acc', 'inertial_y_acc', 'inertial_z_acc'
    rp_fm_config = {
    'knocke_erp': True,
    'SRP': True}
    rho_from_ACT = density_inversion(
        sat_name, inertial_act_gfo_ephem, 
        act_x_acc_col, act_y_acc_col, act_z_acc_col, 
        rp_fm_config, nc_accs=True, 
        models_to_query=models_to_query, density_freq='15S'
    )
    rho_from_ACT.rename(columns={'Computed Density': 'ACT_Density'}, inplace=True)

    if calculate_rho_from_vel:
        interp_ephemeris_df = interpolate_positions(sp3_ephemeris_df, '0.01S')
        sp3_velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
        sp3_vel_acc_col_x, sp3_vel_acc_col_y, sp3_vel_acc_col_z = 'vel_acc_x', 'vel_acc_y', 'vel_acc_z'
        rho_from_vel = density_inversion(
            sat_name, sp3_velacc_ephem, 
            sp3_vel_acc_col_x, sp3_vel_acc_col_y, sp3_vel_acc_col_z, 
            force_model_config=force_model_config, nc_accs=False, 
            models_to_query=[None], density_freq='15S'
        )
        rho_from_vel.rename(columns={'Computed Density': 'POD_Density'}, inplace=True)

        merged_df = pd.merge(
            rho_from_ACT[['UTC', 'ACT_Density'] + models_to_query], 
            rho_from_vel[['UTC', 'POD_Density']], 
            on='UTC', 
            how='inner'
        )
    else:
        merged_df = rho_from_ACT[['UTC', 'ACT_Density'] + models_to_query]

    timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    merged_df.dropna(inplace=True)

    print(f"head of merged_df: {merged_df.head()}")
    print(f"columns of merged_df: {merged_df.columns}")
    merged_df.to_csv(f"output/DensityInversion/PODDensityInversion/Data/{sat_name}/Accelerometer_benchmark/{timenow}_bench.csv", index=False)

if __name__ == '__main__':

    # File paths
    # acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-06_C_04.txt"
    # quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-06_C_04.txt"

    # # Call the function with appropriate parameters
    # ACT_vs_POD(
    #     acc_data_path=acc_data_path,
    #     quat_data_path=quat_data_path,
    #     sat_name="GRACE-FO-A",
    #     max_time=24,
    #     calculate_rho_from_vel=True,
    #     models_to_query=['JB08', 'DTM2000', 'NRLMSISE00'])

    POD_and_ACT_data = pd.read_csv("output/DensityInversion/PODDensityInversion/Plots/GRACE-FO-A/Accelerometer_benchmark/ACTvsEDRvsPOD/ACT_vs_POD_2023_05_06_GFOA.csv")
    EDR_data = pd.read_csv("output/DensityInversion/PODDensityInversion/Plots/GRACE-FO-A/Accelerometer_benchmark/ACTvsEDRvsPOD/EDR_2023_05_06_GFOA.csv")
    NRT_data = pd.read_csv("output/DensityInversion/PODDensityInversion/Plots/GRACE-FO-A/Accelerometer_benchmark/ACTvsEDRvsPOD/NRT_2023_05_06_GFOA.csv")
    median_NRT = NRT_data['Computed Density'].median()
    NRT_data['Computed Density'] = NRT_data['Computed Density'].apply(lambda x: median_NRT if x < -2e-12 else x)
    POD_and_ACT_data['POD_Density'] = POD_and_ACT_data['POD_Density'].apply(lambda x: median_NRT if x < -2e-12 else x)
    NRT_data = NRT_data.iloc[3:]
    NRT_data = NRT_data.iloc[:-5]
    POD_and_ACT_data = POD_and_ACT_data.iloc[3:]
    POD_and_ACT_data = POD_and_ACT_data.iloc[:-5]

    act_edr_pod_nrt_stats(POD_and_ACT_data, EDR_data, NRT_data)
    # act_edr_pod_nrt_ASD(POD_and_ACT_data, EDR_data, NRT_data)
    # ACT_vs_EDR_vs_POD_NRT_plot(POD_and_ACT_data, EDR_data, NRT_data)

#     sat_name = "GRACE-FO-A"
#     force_model_config = {'90x90gravity': True, '3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
#     max_time = 24
#     acc_data_path = "external/GFOInstrumentData/ACT1B_2023-05-06_C_04.txt"
#     quat_data_path = "external/GFOInstrumentData/SCA1B_2023-05-06_C_04.txt"
#     inertial_gfo_data = get_gfo_inertial_accelerations(acc_data_path, quat_data_path)
#     inertial_gfo_data['UTC'] = pd.to_datetime(inertial_gfo_data['utc_time'])
#     inertial_gfo_data.drop(columns=['utc_time'], inplace=True)

#     intertial_t0 = inertial_gfo_data['UTC'].iloc[0]
#     inertial_act_gfo_data = inertial_gfo_data[(inertial_gfo_data['UTC'] >= intertial_t0) & (inertial_gfo_data['UTC'] <= intertial_t0 + pd.Timedelta(hours=max_time))]
#     sp3_ephemeris_df = sp3_ephem_to_df(satellite="GRACE-FO-A", date="2023-05-06")
#     sp3_ephemeris_df['UTC'] = pd.to_datetime(sp3_ephemeris_df['UTC'])
#     sp3_ephemeris_df = sp3_ephemeris_df[(sp3_ephemeris_df['UTC'] >= intertial_t0) & (sp3_ephemeris_df['UTC'] <= intertial_t0 + pd.Timedelta(hours=max_time))]

#     inertial_act_gfo_ephem = pd.merge(inertial_act_gfo_data, sp3_ephemeris_df, on='UTC', how='inner')
#     print(f"head of inertial_act_gfo_ephem: {inertial_act_gfo_ephem.head()}")
#     print(f"columns of inertial_act_gfo_ephem: {inertial_act_gfo_ephem.columns}")

#     act_x_acc_col, act_y_acc_col, act_z_acc_col = 'inertial_x_acc', 'inertial_y_acc', 'inertial_z_acc'
#     rho_from_ACT = density_inversion(sat_name, inertial_act_gfo_ephem, 
#                                      act_x_acc_col, act_y_acc_col, act_z_acc_col, 
#                                      force_model_config, nc_accs=True, 
#                                      models_to_query=['JB08', "DTM2000", "NRLMSISE00"], density_freq='15S')
#     rho_from_ACT.rename(columns={'Computed Density': 'ACT_Density'}, inplace=True)

#     print(f"head of rho_from_ACT: {rho_from_ACT.head()}")

#     interp_ephemeris_df = interpolate_positions(sp3_ephemeris_df, '0.01S')
#     sp3_velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
#     sp3_vel_acc_col_x, sp3_vel_acc_col_y, sp3_vel_acc_col_z = 'vel_acc_x', 'vel_acc_y', 'vel_acc_z'
#     rho_from_vel = density_inversion(sat_name, sp3_velacc_ephem, 
#                                     sp3_vel_acc_col_x, sp3_vel_acc_col_y, sp3_vel_acc_col_z, 
#                                     force_model_config=force_model_config, nc_accs=False, 
#                                     models_to_query=[None], density_freq='15S')
#     rho_from_vel.rename(columns={'Computed Density': 'POD_Density'}, inplace=True)

#     print(f"head of rho_from_vel: {rho_from_vel.head()}")

#     merged_df = pd.merge(rho_from_ACT[['UTC', 'ACT_Density', 'JB08', 'DTM2000', 'NRLMSISE00']], rho_from_vel[['UTC', 'POD_Density']], on='UTC', how='inner')

#     timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     merged_df.dropna(inplace=True)

#     print(f"head of merged_df: {merged_df.head()}")
#     print(f"columns of merged_df: {merged_df.columns}")
#     merged_df.to_csv(f"output/DensityInversion/PODDensityInversion/Plots/GRACE-FO-A/Accelerometer_benchmark/{timenow}_bench.csv", index=False)

