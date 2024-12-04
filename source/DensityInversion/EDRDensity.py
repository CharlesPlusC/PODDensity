import os
import orekit
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from scipy.integrate import trapezoid as trapz
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir
from source.tools.sp3_2_ephemeris import sp3_ephem_to_df
from source.tools.utilities import get_satellite_info, utc_to_mjd, interpolate_positions
from source.tools.orekit_tools import state2acceleration, query_jb08

# Initialize Orekit VM and setup
vm = orekit.initVM()
# download_orekit_data_curdir()
setup_orekit_curdir()

def compute_accelerations(sat_name, ephemeris_df, freq='1S', save_folder=None):
    ephemeris_df = interpolate_positions(ephemeris_df, freq)
    ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    ephemeris_df.set_index('UTC', inplace=True)
    ephemeris_df['UTC'] = ephemeris_df.index
    ephemeris_df['MJD'] = [utc_to_mjd(dt) for dt in ephemeris_df.index]

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

        nc_acc_eci_sum = acc_eci['knocke_erp'] + acc_eci['SRP']
        c_acc_eci_sum = acc_eci['90x90gravity'] + acc_eci['3BP'] + acc_eci['ocean_tides'] + acc_eci['solid_tides']

        nc_eci_accs.append(nc_acc_eci_sum)
        c_eci_accs.append(c_acc_eci_sum)

        rows.append({
            'UTC': time,
            'nc_x': nc_acc_eci_sum[0], 'nc_y': nc_acc_eci_sum[1], 'nc_z': nc_acc_eci_sum[2],
            'c_x': c_acc_eci_sum[0], 'c_y': c_acc_eci_sum[1], 'c_z': c_acc_eci_sum[2]
        })

    acc_df = pd.DataFrame(rows)
    yyyy_mm_dd = datetime.datetime.strftime(ephemeris_df.index[0], "%Y-%m-%d")
    acc_csv_path = os.path.join(save_folder or f"output/EDR/Data/{sat_name}/", f"precomp_accs_{sat_name}_{yyyy_mm_dd}.csv")
    os.makedirs(os.path.dirname(acc_csv_path), exist_ok=True)
    acc_df.to_csv(acc_csv_path, index=False)

    return acc_df

def perform_density_inversion(sat_name, ephemeris_df, acc_df, freq='1S', save_folder=None, models_to_query=[None]):
    # Slice the ephemeris dataframe to match the same start and end UTC as the acc_df
    ephemeris_df = ephemeris_df[(ephemeris_df['UTC'] >= acc_df['UTC'].iloc[0]) & (ephemeris_df['UTC'] <= acc_df['UTC'].iloc[-1])]
    ephemeris_df = interpolate_positions(ephemeris_df, freq)

    ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    ephemeris_df.set_index('UTC', inplace=True)
    ephemeris_df['MJD'] = [utc_to_mjd(dt) for dt in ephemeris_df.index]

    #make both dataframes have the same start and end times
    acc_df['UTC'] = pd.to_datetime(acc_df['UTC'])
    acc_df.set_index('UTC', inplace=True)
    acc_df = acc_df[(acc_df.index >= ephemeris_df.index[0]) & (acc_df.index <= ephemeris_df.index[-1])]

    eci_positions = ephemeris_df[['x', 'y', 'z']].values
    eci_velocities = ephemeris_df[['xv', 'yv', 'zv']].values

    nc_eci_accs = acc_df[['nc_x', 'nc_y', 'nc_z']].values
    c_eci_accs = acc_df[['c_x', 'c_y', 'c_z']].values

    rows_list = []
    arc_length = 45*60
    sample_time = 1
    num_points_per_arc = arc_length // sample_time

    for i in tqdm(range(num_points_per_arc, len(ephemeris_df), sample_time), desc='Processing Density Inversion'):
        time = ephemeris_df.index[i]
        start_idx = max(0, i - num_points_per_arc)
        end_idx = i + 1

        g_noncon_integ = sum(
            [trapz(nc_eci_accs[start_idx:end_idx, j], eci_positions[start_idx:end_idx, j]) for j in range(3)]
        )

        g_con_integ = -sum(
            [trapz(c_eci_accs[start_idx:end_idx, j], eci_positions[start_idx:end_idx, j]) for j in range(3)]
        )

        vel_window = np.linalg.norm(eci_velocities[start_idx:end_idx], axis=1)
        delta_v = 0.5 * (vel_window[-1]**2 - vel_window[0]**2)
        delta_ener = delta_v + g_con_integ
        drag_work = -delta_ener - g_noncon_integ

        v = eci_velocities[i]
        r = eci_positions[i]
        atm_rot = np.array([0, 0, 72.9211e-6])
        v_rel = np.linalg.norm(v - np.cross(atm_rot, r))
        density = drag_work / (get_satellite_info(sat_name)['cd'] * get_satellite_info(sat_name)['cross_section'] * v_rel**3)

        row_data = {
            'UTC': time,
            'x': eci_positions[i][0], 'y': eci_positions[i][1], 'z': eci_positions[i][2],
            'xv': eci_velocities[i][0], 'yv': eci_velocities[i][1], 'zv': eci_velocities[i][2],
            'EDR Density': density, 'Drag Work': drag_work, 'Delta Energy': delta_ener,
            'g_con_integ': g_con_integ, 'g_noncon_integ': g_noncon_integ, 'delta_v': delta_v
        }

        for model in models_to_query:
            if model and globals().get(f"query_{model.lower()}"):
                model_func = globals()[f"query_{model.lower()}"]
                row_data[model] = model_func(position=eci_positions[i], datetime=time)
        rows_list.append(row_data)

    density_inversion_df = pd.DataFrame(rows_list)
    yyyy_mm_dd = datetime.datetime.strftime(ephemeris_df.index[0], "%Y-%m-%d")
    output_dir = save_folder or f"output/EDR/Data/{sat_name}/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"EDR_{sat_name}_{yyyy_mm_dd}_density_inversion.csv")
    print(f"file saved to {output_path}")
    density_inversion_df.to_csv(output_path, index=False)

    return density_inversion_df

def density_inversion_edr(sat_name, ephemeris_df, models_to_query=[None], freq='1S', save_folder=None):
    acc_df = compute_accelerations(sat_name, ephemeris_df, freq, save_folder)
    return perform_density_inversion(sat_name, ephemeris_df, acc_df, freq, save_folder, models_to_query)

def precomp_accs_to_density():
    #If the precomputed accelerations are already available, this function can be used to convert them to densities
    satellites = ["GRACE-FO", "CHAMP"]
    accelerations_basepath = "output/EDR/Data"
    for satellite in satellites:
        #folder containing the precomputed accelerations for many different storms
        accelerations_folder = f"{accelerations_basepath}/{satellite}/"
        for storm_file in os.listdir(accelerations_folder):
            print(f"Processing {storm_file}...")
            acc_df = pd.read_csv(os.path.join(accelerations_folder, storm_file))
            storm_date = storm_file.split("_")[-1].split(".")[0]
            print(f"Storm Date: {storm_date}")
            if satellite == "GRACE-FO":
                satellite = "GRACE-FO-A"

if __name__ == "__main__":
    pass
    # #Get density from a single set of precomputed accelerations
    # storm_date= "2023-05-06"
    # satellite="GRACE-FO-A"
    # ephemeris_df = sp3_ephem_to_df(satellite, storm_date)
    # acc_df = pd.read_csv("output/EDR/Data/GRACE-FO/new_precomp_accs_2023-05-06.csv")
    # density_inversion_df = perform_density_inversion(satellite, ephemeris_df, acc_df, models_to_query=[None], freq='1S')

    # #Get Density from SP3 Ephemeris
    # ephem_date_str = "2023-05-06"
    # sp3_ephem_gfo = sp3_ephem_to_df("GRACE-FO-A", ephem_date_str)
    # sp3_ephem_gfo['UTC'] = pd.to_datetime(sp3_ephem_gfo['UTC'])
    # sp3_ephem_gfo.set_index('UTC', inplace=True)
    # sp3_ephem_gfo = sp3_ephem_gfo[(sp3_ephem_gfo.index >= '2023-05-06 00:00:00') & (sp3_ephem_gfo.index <= '2023-05-06 19:00:00')]
    # sp3_ephem_gfo['UTC'] = sp3_ephem_gfo.index
    # edr_density_df, drag_works = density_inversion_edr("GRACE-FO", sp3_ephem_gfo, models_to_query=[None], freq='1S')
    