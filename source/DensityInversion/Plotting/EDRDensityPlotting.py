import os
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from scipy.integrate import trapz
from orekit.pyhelpers import setup_orekit_curdir
from source.tools.sp3_2_ephemeris import sp3_ephem_to_df
from source.tools.utilities import get_satellite_info, utc_to_mjd, interpolate_positions


sp3_ephem_gfo = sp3_ephem_to_df("GRACE-FO-A", "2023-03-23")
ephemeris_df = sp3_ephem_gfo.iloc[1000:3500]

print(f"SP3 Ephemeris DataFrame:\n{sp3_ephem_gfo.head()}")    
# edr_density_df, drag_works = density_inversion_edr("GRACE-FO", sp3_ephem_gfo, models_to_query=[None], freq='1S')
    
freq = '1S'
# Path to store and load precomputed accelerations
acc_csv_path = f"output/DensityInversion/EDRDensityInversion/Data/GRACE-FO/precomputed_accelerations_2023_03_23.csv"
sat_name = "GRACE-FO"

# Interpolate ephemeris to the desired frequency
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

print(f"Loading precomputed accelerations from {acc_csv_path}")
acc_df = pd.read_csv(acc_csv_path, parse_dates=['UTC'])
acc_df.set_index('UTC', inplace=True)
nc_eci_accs = acc_df[['nc_x', 'nc_y', 'nc_z']].values
c_eci_accs = acc_df[['c_x', 'c_y', 'c_z']].values

# Resample accelerations to match the desired frequency
acc_df = acc_df.asfreq(freq)
nc_eci_accs = acc_df[['nc_x', 'nc_y', 'nc_z']].values
c_eci_accs = acc_df[['c_x', 'c_y', 'c_z']].values

rows_list = []
arc_length = 25 * 60
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
    density = drag_work / (3.2 * settings['cross_section'] * v_rel**3)

    # Append results to row data
    row_data = {
        'UTC': time,
        'x': eci_positions[i][0], 'y': eci_positions[i][1], 'z': eci_positions[i][2],
        'xv': eci_velocities[i][0], 'yv': eci_velocities[i][1], 'zv': eci_velocities[i][2],
        'EDR Density': density, 'Drag Work': drag_work, 'Delta Energy': delta_ener,
        'g_con_integ': g_con_integ, 'g_noncon_integ': g_noncon_integ, 'delta_v': delta_v
    }

    #put all the rows in a dataframe
    rows_list.append(row_data)

# Create a DataFrame from the list of rows
edr_density_df = pd.DataFrame(rows_list)
# edr_density_df.set_index('UTC', inplace=True)

print(f"EDR Density DataFrame:\n{edr_density_df.head()}")

import matplotlib.pyplot as plt
# Create subplots
fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

axes[0].plot(edr_density_df['UTC'], edr_density_df['delta_v'], label='Kinetic Energy', color='blue')
axes[0].plot(edr_density_df['UTC'], -edr_density_df['g_con_integ'], label='Conservative Integral', color='red', linestyle='dashed')
axes[0].set_ylabel('Delta V')
axes[0].set_title('Kinetic Energy')
axes[0].legend()
axes[0].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

# Plot each line on its own subplot
axes[1].plot(edr_density_df['UTC'], edr_density_df['Delta Energy'], label='Kinetic-Potential', color='blue')
axes[1].set_ylabel('Delta Energy')
axes[1].set_title('Kinetic-Potential Energy')
axes[1].legend()
axes[1].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

axes[2].plot(edr_density_df['UTC'], edr_density_df['g_con_integ'], label='Conservative Integral', color='blue')
axes[2].set_ylabel('Conservative Integral')
axes[2].set_title('Conservative Integral')
axes[2].legend()
axes[2].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

axes[3].plot(edr_density_df['UTC'], edr_density_df['g_noncon_integ'], label='Non-Conservative Integral', color='blue')
axes[3].set_ylabel('Non-Conservative Integral')
axes[3].set_title('Non-Conservative Integral')
axes[3].legend()
axes[3].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

axes[4].plot(edr_density_df['UTC'], edr_density_df['Drag Work'], label='Drag Work (Kinetic-Potential + Non-Con)', color='blue')
axes[4].set_ylabel('Drag Work')
axes[4].set_title('Drag Work')
axes[4].legend()
axes[4].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

# Plot the last two lines on the same subplot
edr_density_df_jb = pd.read_csv("output/DensityInversion/EDRDensityInversion/Data/GRACE-FO/precomputed_accelerations_2023_03_23.csv")
edr_density_df_jb['UTC'] = pd.to_datetime(edr_density_df_jb['UTC'])
#drop all nan rows in the dataframe
edr_density_df_jb.dropna(inplace=True)
axes[5].plot(edr_density_df['UTC'], edr_density_df['EDR Density'].rolling(window=60*45,center=True).mean(), label='EDR Density Rolling', color='orange')
# axes[5].plot(edr_density_df['UTC'], edr_density_df['EDR Density'], label='EDR Density', color='blue')
axes[5].plot(edr_density_df_jb['UTC'], edr_density_df_jb['JB08'], label='JB08', color='green')
axes[5].set_ylabel('Density (kg/m³)')
axes[5].set_title('EDR Density vs JB08')
#log scale
axes[5].set_yscale('log')
axes[5].legend()
axes[5].grid(True, linestyle='dotted', color='gray', linewidth=0.75)

# Set x-axis properties
axes[5].set_xticks(edr_density_df['UTC'][::int(len(edr_density_df)/10)])
axes[5].tick_params(axis='x', rotation=45)
axes[5].set_xlabel('UTC')

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()