import os
import orekit
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from scipy.integrate import trapezoid as trapz
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir
from source.tools.utilities import get_satellite_info, utc_to_mjd, interpolate_positions
from source.tools.orekit_tools import state2acceleration

# Initialize Orekit VM and setup
vm = orekit.initVM()
# download_orekit_data_curdir()
setup_orekit_curdir()

def density_inversion_edr(sat_name, ephemeris_df, models_to_query=[None], freq='1S'):

    # Path to store and load precomputed accelerations
    # acc_csv_path = f"output/EDR/Data/GRACE-FO/precomputed_accelerations_2023_03_23.csv"
    print(f"columns in ephemeris_df: {ephemeris_df.columns}")

    ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    ephemeris_df.set_index('UTC', inplace=True)
    mjd_times = [utc_to_mjd(dt) for dt in ephemeris_df.index]
    ephemeris_df['MJD'] = mjd_times

    #find the peak NRLMSISE-00 density value
    peak_density = max(ephemeris_df['NRLMSISE-00'])
    #now slice the dataframe so that it starts 12 hours before this and ends 24 hours after this
    peak_density_time = ephemeris_df[ephemeris_df['NRLMSISE-00'] == peak_density].index[0]
    start_time = peak_density_time - datetime.timedelta(hours=12)
    end_time = peak_density_time + datetime.timedelta(hours=24)
    ephemeris_df = ephemeris_df[start_time:end_time]

    ephemeris_df = interpolate_positions(ephemeris_df, freq)


    ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    ephemeris_df.set_index('UTC', inplace=True)
    mjd_times = [utc_to_mjd(dt) for dt in ephemeris_df.index]
    ephemeris_df['MJD'] = mjd_times

    eci_positions = ephemeris_df[['x', 'y', 'z']].values
    eci_velocities = ephemeris_df[['xv', 'yv', 'zv']].values
    times = ephemeris_df.index

    sat_info = get_satellite_info(sat_name)
    settings = {
        'cr': sat_info['cr'],
        'cd': sat_info['cd'],
        'cross_section': sat_info['cross_section'],
        'mass': sat_info['mass'],
    }
    force_model_config = {'knocke_erp': True, 'SRP': True, '90x90gravity': True, '3BP': True, 'ocean_tides': True, 'solid_tides': True}

    # If precomputed accelerations exist, load them; otherwise, compute and save them
    # if os.path.exists(acc_csv_path):
    #     print(f"Loading precomputed accelerations from {acc_csv_path}")
    #     acc_df = pd.read_csv(acc_csv_path, parse_dates=['UTC'])
    #     acc_df.set_index('UTC', inplace=True)
    #     nc_eci_accs = acc_df[['nc_x', 'nc_y', 'nc_z']].values
    #     c_eci_accs = acc_df[['c_x', 'c_y', 'c_z']].values
    # else:
    print("Computing accelerations...")
    nc_eci_accs = []
    c_eci_accs = []
    rows = []

    for i in tqdm(range(len(times)), desc="Computing Accelerations"):
        r = eci_positions[i]
        v = eci_velocities[i]
        time = times[i]

        acc_eci = state2acceleration(
            np.hstack((r, v)), time, settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], **force_model_config
        )

        # Extract accelerations
        nc_acc_eci_sum = acc_eci['knocke_erp'] + acc_eci['SRP']
        c_acc_eci_sum = acc_eci['90x90gravity'] + acc_eci['3BP'] + acc_eci['ocean_tides'] + acc_eci['solid_tides']

        nc_eci_accs.append(nc_acc_eci_sum)
        c_eci_accs.append(c_acc_eci_sum)

        # Save to rows for CSV
        rows.append({
            'UTC': time,
            'nc_x': nc_acc_eci_sum[0], 'nc_y': nc_acc_eci_sum[1], 'nc_z': nc_acc_eci_sum[2],
            'c_x': c_acc_eci_sum[0], 'c_y': c_acc_eci_sum[1], 'c_z': c_acc_eci_sum[2]
        })

    # Save precomputed accelerations to a CSV
    acc_df = pd.DataFrame(rows)
    yyyy_mm_dd = datetime.datetime.strftime(ephemeris_df.index[0], "%Y-%m-%d")
    acc_csv_path = f"output/EDR/Data/{sat_name}/precomp_accs_{yyyy_mm_dd}.csv"
    os.makedirs(os.path.dirname(acc_csv_path), exist_ok=True)
    acc_df.to_csv(acc_csv_path, index=False)
    nc_eci_accs = np.array(nc_eci_accs)
    c_eci_accs = np.array(c_eci_accs)

    # Resample accelerations to match the desired frequency
    acc_df = acc_df.asfreq(freq)
    nc_eci_accs = acc_df[['nc_x', 'nc_y', 'nc_z']].values
    c_eci_accs = acc_df[['c_x', 'c_y', 'c_z']].values

    rows_list = []
    arc_length = 40 * 60
    sample_time = int(pd.to_timedelta(freq).total_seconds())
    num_points_per_arc = arc_length // sample_time

    for i in tqdm(range(num_points_per_arc, len(ephemeris_df), sample_time), desc='Processing Density Inversion'):
        time = ephemeris_df.index[i]

        # Define window
        start_idx = max(0, i - num_points_per_arc)
        end_idx = i + 1  # Include current point

        # Compute non-conservative integral
        g_noncon_integ = sum(
            [trapz(nc_eci_accs[start_idx:end_idx, j], eci_positions[start_idx:end_idx, j]) for j in range(3)]
        )

        # Compute conservative integral
        g_con_integ = -sum(
            [trapz(c_eci_accs[start_idx:end_idx, j], eci_positions[start_idx:end_idx, j]) for j in range(3)]
        )

        # Kinetic energy change
        vel_window = np.linalg.norm(eci_velocities[start_idx:end_idx], axis=1)
        delta_v = 0.5 * (vel_window[-1]**2 - vel_window[0]**2)

        # Total energy change
        delta_ener = delta_v + g_con_integ

        # Drag work
        drag_work = -delta_ener - g_noncon_integ

        # Relative velocity
        v = eci_velocities[i]
        r = eci_positions[i]
        atm_rot = np.array([0, 0, 72.9211e-6])  # Earth's angular velocity
        v_rel = np.linalg.norm(v - np.cross(atm_rot, r))

        # Density calculation
        density = drag_work / (settings['cd'] * settings['cross_section'] * v_rel**3)

        # Append results to row data
        row_data = {
            'UTC': time,
            'x': eci_positions[i][0], 'y': eci_positions[i][1], 'z': eci_positions[i][2],
            'xv': eci_velocities[i][0], 'yv': eci_velocities[i][1], 'zv': eci_velocities[i][2],
            'EDR Density': density, 'Drag Work': drag_work, 'Delta Energy': delta_ener,
            'g_con_integ': g_con_integ, 'g_noncon_integ': g_noncon_integ, 'delta_v': delta_v
        }

        # Query models and add results
        for model in models_to_query:
            if model and globals().get(f"query_{model.lower()}"):
                model_func = globals()[f"query_{model.lower()}"]
                row_data[model] = model_func(position=eci_positions[i], datetime=time)
        rows_list.append(row_data)

    # Create DataFrame and save results
    if rows_list:
        density_inversion_df = pd.DataFrame(rows_list)
        yyyy_mm_dd = datetime.datetime.strftime(ephemeris_df.index[0], "%Y-%m-%d")
        output_dir = f"output/DensityInversion/EDRDensityInversion/Data/{sat_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"EDR_{sat_name}_{yyyy_mm_dd}_density_inversion.csv")
        density_inversion_df.to_csv(output_path, index=False)

    return density_inversion_df, drag_work

# def add_jb08_accelerations(ephemeris_df, output_path, freq='1S'):
#     # Interpolate ephemeris to desired frequency
#     ephemeris_df = interpolate_positions(ephemeris_df, freq)
#     ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
#     ephemeris_df.set_index('UTC', inplace=True)

#     mjd_times = [utc_to_mjd(dt) for dt in ephemeris_df.index]
#     ephemeris_df['MJD'] = mjd_times

#     eci_positions = ephemeris_df[['x', 'y', 'z']].values
#     times = ephemeris_df.index

#     # Check if the output file exists
#     if os.path.exists(output_path):
#         print(f"Loading existing data from {output_path}")
#         existing_df = pd.read_csv(output_path, parse_dates=['UTC'])
#         existing_df.set_index('UTC', inplace=True)
#     else:
#         print("No existing file found. Creating a new one.")
#         existing_df = pd.DataFrame()

#     # Query JB08 accelerations and add to the existing DataFrame
#     rows = []
#     print("Querying JB08 accelerations...")
#     for i in tqdm(range(len(times)), desc="Computing JB08 Accelerations"):
#         r = eci_positions[i]
#         time = times[i]

#         # Query JB08 model
#         jb08_rho = query_jb08(position=r, datetime=time)

#         # Create a new row for JB08 accelerations
#         row = {
#             'UTC': time,
#             'JB08': jb08_rho,
#         }
#         rows.append(row)

#     jb08_df = pd.DataFrame(rows).set_index('UTC')

#     # Merge with existing DataFrame
#     updated_df = existing_df.combine_first(jb08_df)

#     # Save the updated DataFrame
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     updated_df.to_csv(output_path)
#     print(f"Updated file saved to {output_path}")

if __name__ == "__main__": 
    pass
    # storm_df_path = "output/PODDensityInversion/Data/StormAnalysis/GRACE-FO/GRACE-FO-A_2024-05-08_density_inversion.csv"
    # storm_df_path = "output/PODDensityInversion/Data/StormAnalysis/GRACE-FO/GRACE-FO_storm_density_15_1_20240511215004.csv"
    # storm_df = pd.read_csv(storm_df_path)
    # satellite = "GRACE-FO"
    # density_inversion_edr(
    #     sat_name=satellite,
    #     ephemeris_df=storm_df,
    #     models_to_query=[None],
    #     freq='1S'
    #     )

#     ephem_date_str = "2023-03-23"
#     sp3_ephem_gfo = sp3_ephem_to_df("GRACE-FO-A", ephem_date_str)
#     sp3_ephem_gfo = sp3_ephem_gfo.iloc[1000:3500]
#     print(f"first and last 5 rows of SP3 Ephemeris DataFrame:\n{sp3_ephem_gfo.head()}\n{sp3_ephem_gfo.tail()}")
    
#     import matplotlib.dates as mdates
#     output_file = "output/EDR/Data/GRACE-FO/precomputed_accelerations_2023_03_23.csv"
#     # add_jb08_accelerations(sp3_ephem_gfo, output_file, freq='30S')
#     # print(f"SP3 Ephemeris DataFrame:\n{sp3_ephem_gfo.head()}")
    
#     edr_density_df, drag_works = density_inversion_edr("GRACE-FO", sp3_ephem_gfo, models_to_query=[None], freq='1S')

#     edr_density_df_jb = pd.read_csv("output/EDR/Data/GRACE-FO/precomputed_accelerations_2023_03_23.csv")
#     edr_density_df_jb['UTC'] = pd.to_datetime(edr_density_df_jb['UTC'])
#     #drop all rows with NaN values
#     edr_density_df_jb = edr_density_df_jb.dropna()
#     print(f"EDR Density DataFrame:\n{edr_density_df_jb.head()}")
    
#     # Create subplots
#     fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

#     axes[0].plot(edr_density_df['UTC'], edr_density_df['delta_v'], label='Kinetic Energy', color='blue')
#     axes[0].plot(edr_density_df['UTC'], -edr_density_df['g_con_integ'], label='Conservative Integral', color='red', linestyle='dashed')
#     axes[0].set_ylabel('Delta V')
#     axes[0].set_title('Kinetic Energy')
#     axes[0].legend()
#     axes[0].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

#     # Plot each line on its own subplot
#     axes[1].plot(edr_density_df['UTC'], edr_density_df['Delta Energy'], label='Kinetic-Potential', color='blue')
#     axes[1].set_ylabel('Delta Energy')
#     axes[1].set_title('Kinetic-Potential Energy')
#     axes[1].legend()
#     axes[1].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

#     axes[2].plot(edr_density_df['UTC'], edr_density_df['g_con_integ'], label='Conservative Integral', color='blue')
#     axes[2].set_ylabel('Conservative Integral')
#     axes[2].set_title('Conservative Integral')
#     axes[2].legend()
#     axes[2].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

#     axes[3].plot(edr_density_df['UTC'], edr_density_df['g_noncon_integ'], label='Non-Conservative Integral', color='blue')
#     axes[3].set_ylabel('Non-Conservative Integral')
#     axes[3].set_title('Non-Conservative Integral')
#     axes[3].legend()
#     axes[3].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

#     axes[4].plot(edr_density_df['UTC'], edr_density_df['Drag Work'], label='Drag Work (Kinetic-Potential + Non-Con)', color='blue')
#     axes[4].set_ylabel('Drag Work')
#     axes[4].set_title('Drag Work')
#     axes[4].legend()
#     axes[4].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

#     # Plot the last two lines on the same subplot
#     axes[5].plot(edr_density_df['UTC'], edr_density_df['EDR Density'].rolling(window=45*40,center=True).mean(), label='EDR Density Rolling', color='blue')
#     axes[5].plot(edr_density_df['UTC'], edr_density_df['EDR Density'], label='EDR Density', color='orange')
#     axes[5].plot(edr_density_df_jb['UTC'], edr_density_df_jb['JB08'], label='JB08', color='green')
#     axes[5].set_ylabel('Density (kg/m³)')
#     axes[5].set_title('EDR Density vs JB08')
#     #log scale
#     axes[5].set_yscale('log')
#     axes[5].legend()
#     axes[5].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

#     date_formatter = mdates.DateFormatter("%Y-%m-%d %H:%M")  # Include date and hour
#     for ax in axes:
#         ax.xaxis.set_major_formatter(date_formatter)

#     # Limit x-axis labels to 10 evenly spaced ticks
#     x_ticks = edr_density_df['UTC'][::max(len(edr_density_df) // 10, 1)]
#     axes[5].set_xticks(x_ticks)

#     axes[5].tick_params(axis='x', rotation=45)
#     axes[5].set_xlabel('UTC')

#     # Tight layout for better spacing
#     plt.tight_layout()

#     # Show the plot
#     plt.show()  