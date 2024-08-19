import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

# download_orekit_data_curdir("misc")
setup_orekit_curdir("misc/orekit-data.zip")
vm = orekit.initVM()

import os
from ..tools.utilities import project_acc_into_HCL, get_satellite_info, interpolate_positions, calculate_acceleration
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df
from ..tools.orekit_tools import state2acceleration, query_jb08, query_dtm2000, query_nrlmsise00
import numpy as np
import datetime
from tqdm import tqdm
import pandas as pd
from orekit.pyhelpers import setup_orekit_curdir, datetime_to_absolutedate
from ..tools.GFODataReadTools import get_gfo_inertial_accelerations
from ..tools.SWIndices import get_kp_ap_dst_f107
from .Plotting.PODDensityPlotting import plot_all_storms_scatter ,plot_densities_and_indices, reldens_sat_megaplot, get_arglat_from_df, density_compare_scatter

def density_inversion(sat_name, ephemeris_df, x_acc_col, y_acc_col, z_acc_col, force_model_config, nc_accs=False, models_to_query=['JB08'], density_freq='15S'):
    #nc_accs refers to whether the accelereation time series is only non conservative or not (set to true for accelerometer data and false for POD data)
    sat_info = get_satellite_info(sat_name)
    settings = {
        'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass'],
        'density_freq': density_freq
    }

    available_models = ['JB08', 'DTM2000', 'NRLMSISE00', None]
    for model in models_to_query:
        assert model in available_models

    assert 'UTC' in ephemeris_df.columns

    ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    ephemeris_df.set_index('UTC', inplace=True)
    ephemeris_df = ephemeris_df.asfreq(settings['density_freq'])

    #drop NA rows
    ephemeris_df.dropna(inplace=True)

    columns = [
        'UTC', 'x', 'y', 'z', 'xv', 'yv', 'zv', 'accx', 'accy', 'accz',
        'nc_accx', 'nc_accy', 'nc_accz', 'Computed Density', *(models_to_query)
    ]

    # Initialize the DataFrame with the specified columns and correct dtypes
    density_inversion_df = pd.DataFrame(columns=columns)
    rows_list = []

    for i in tqdm(range(1, len(ephemeris_df)), desc='Processing Density Inversion'):
        time = ephemeris_df.index[i]
        vel = np.array([ephemeris_df['xv'].iloc[i], ephemeris_df['yv'].iloc[i], ephemeris_df['zv'].iloc[i]])
        state_vector = np.array([ephemeris_df['x'].iloc[i], ephemeris_df['y'].iloc[i], ephemeris_df['z'].iloc[i], vel[0], vel[1], vel[2]])
        if not nc_accs: 
            #except drag acceleration
            all_accelerations = state2acceleration(state_vector, time, 
                                                            settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], 
                                                            **force_model_config)
            all_accelerations_sum = np.sum(list(all_accelerations.values()), axis=0)
            observed_acc = np.array([ephemeris_df['accx'].iloc[i], ephemeris_df['accy'].iloc[i], ephemeris_df['accz'].iloc[i]])

            nc_accelerations = all_accelerations_sum - observed_acc

            nc_accx, nc_accy, nc_accz = nc_accelerations[0], nc_accelerations[1], nc_accelerations[2]
    
        else:
            #now just compute radiation pressure since our observed only contain non conservative (SRP + ERP)
            #assuming a_nonconservative = a_rp + a_drag, where a_rp = a_srp + a_erp
            rp_fm_config = {
            'knocke_erp': True,
            'SRP': True}

            rp_accelerations = state2acceleration(state_vector, time, 
                                                            settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], 
                                                            **rp_fm_config)
            print(f"rp_accelerations: {rp_accelerations}")
            rp_accelerations_sum = np.sum(list(rp_accelerations.values()), axis=0)
            observed_acc = np.array([ephemeris_df[x_acc_col].iloc[i], ephemeris_df[y_acc_col].iloc[i], ephemeris_df[z_acc_col].iloc[i]])

            nc_accelerations = rp_accelerations_sum - observed_acc

            nc_accx, nc_accy, nc_accz = nc_accelerations[0], nc_accelerations[1], nc_accelerations[2]

        _, _, nc_acc_l = project_acc_into_HCL(nc_accx, nc_accy, nc_accz, 
                                            ephemeris_df['x'].iloc[i], ephemeris_df['y'].iloc[i], ephemeris_df['z'].iloc[i],
                                             ephemeris_df['xv'].iloc[i], ephemeris_df['yv'].iloc[i], ephemeris_df['zv'].iloc[i])

        r = np.array([ephemeris_df['x'].iloc[i], ephemeris_df['y'].iloc[i], ephemeris_df['z'].iloc[i]])
        v = np.array([ephemeris_df['xv'].iloc[i], ephemeris_df['yv'].iloc[i], ephemeris_df['zv'].iloc[i]])
        atm_rot = np.array([0, 0, 72.9211e-6])
        v_rel = v - np.cross(atm_rot, r)

        rho = -2 * (nc_acc_l / (settings['cd'] * settings['cross_section'])) * (settings['mass'] /  np.abs(np.linalg.norm(v_rel))**2)

        row_data = {
            'UTC': time, 'x': r[0], 'y': r[1], 'z': r[2], 'xv': v[0], 'yv': v[1], 'zv': v[2],
            'nc_accx': nc_accx, 'nc_accy': nc_accy, 'nc_accz': nc_accz, 'Computed Density': rho
        }

        for model in models_to_query:
            if model is not None and globals().get(f"query_{model.lower()}"):
                model_func = globals()[f"query_{model.lower()}"]
                row_data[model] = model_func(position=r, datetime=time)

        rows_list.append(row_data)

    if rows_list:
        new_rows_df = pd.DataFrame(rows_list)
        density_inversion_df = pd.concat([density_inversion_df, new_rows_df], ignore_index=True)

    return density_inversion_df

if __name__ == "__main__":
    pass
    ## Example of how to run density inversion for a single storm
    ## Beware that this will take a long time to run if you process the entire storm on your laptop
    ## I recommend slicing the storm data to a smaller time frame for testing purposes
    #Load ephemeris data
    sp3_ephem_champ = sp3_ephem_to_df("CHAMP","2005-05-07")
    #slice to keep only first 1000 rows
    sp3_ephem_champ = sp3_ephem_champ.iloc[:1000] #This should take around 15/20 minutes on a single core
    #specify the force model configuration
    force_model_config = {
    '90x90gravity': True, '3BP': True, 'solid_tides': True,
    'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
    #interpolate the ephemeris data to desired resolution (0.01S)
    interp_ephemeris_df = interpolate_positions(sp3_ephem_champ, '0.01S')
    #Numerically differentiate the interpolated ephemeris data to get acceleration
    velacc_ephem = calculate_acceleration(interp_ephemeris_df, '0.01S', filter_window_length=21, filter_polyorder=7)
    #Perform density inversion
    density_df = density_inversion("CHAMP", velacc_ephem, 'accx', 'accy', 'accz', force_model_config)
    plot_densities_and_indices([density_df], moving_avg_minutes=23, sat_name="CHAMP")
