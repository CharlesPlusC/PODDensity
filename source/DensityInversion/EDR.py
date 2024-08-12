import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.utils import Constants

import os
from ..tools.utilities import utc_to_mjd, get_satellite_info, get_satellite_info, EME2000_to_ITRF, ecef_to_lla
from ..tools.sp3_2_ephemeris import sp3_ephem_to_df
from ..tools.orekit_tools import query_jb08, query_dtm2000, query_nrlmsise00, state2acceleration
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.special import lpmn

mu = Constants.WGS84_EARTH_MU
R_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
J2 =Constants.EGM96_EARTH_C20

def U_J2(r, phi, lambda_):
    theta = np.pi / 2 - phi  # Convert latitude to colatitude
    # J2 potential term calculation, excluding the monopole term
    V_j2 = -mu / r * J2 * (R_earth / r)**2 * (3 * np.cos(theta)**2 - 1) / 2
    return V_j2

def associated_legendre_polynomials(degree, order, x):
    # Precompute all needed Legendre polynomials at once
    P = {}
    for n in range(degree + 1):
        for m in range(min(n, order) + 1):
            P_nm, _ = lpmn(m, n, x)
            P[(n, m)] = P_nm[0][-1]
    return P

def compute_gravitational_potential(r, phi, lambda_, degree, order, date):
    provider = GravityFieldFactory.getUnnormalizedProvider(degree, order)
    harmonics = provider.onDate(date)
    V_dev = np.zeros_like(phi)
    
    mu_over_r = mu / r
    r_ratio = R_earth / r
    sin_phi = np.sin(phi)

    max_order = max(order, degree) + 1
    C_nm = np.zeros((degree + 1, max_order))
    S_nm = np.zeros((degree + 1, max_order))

    for n in range(degree + 1):
        for m in range(min(n, order) + 1):
            C_nm[n, m] = harmonics.getUnnormalizedCnm(n, m)
            S_nm[n, m] = harmonics.getUnnormalizedSnm(n, m)

    # Compute all necessary Legendre polynomials once
    P = associated_legendre_polynomials(degree, order, sin_phi)

    for n in range(1, degree + 1):
        r_ratio_n = r_ratio ** n
        for m in range(min(n, order) + 1):
            P_nm = P[(n, m)]
            sectorial_term = P_nm * (C_nm[n, m] * np.cos(m * lambda_) + S_nm[n, m] * np.sin(m * lambda_))
            V_dev += mu_over_r * r_ratio_n * sectorial_term

    return -V_dev
    
def calculate_orbital_energy(sat_name, force_model_config, gravity_degree=80, gravity_order=80, other_accs=False):
    folder_save = "output/DensityInversion/EDR/Data"
    datenow = datetime.datetime.now().strftime("%Y-%m-%d")

    sat_info = get_satellite_info(sat_name)
    settings = {'cr': sat_info['cr'], 'cd': sat_info['cd'], 'cross_section': sat_info['cross_section'], 'mass': sat_info['mass']}   
    ephemeris_df = sp3_ephem_to_df(sat_name, date = '2023-05-05') # Selected storm
    ephemeris_df['UTC'] = pd.to_datetime(ephemeris_df['UTC'])
    first_day = ephemeris_df['UTC'].iloc[0]
    three_days_later = first_day + datetime.timedelta(hours=24)
    ephemeris_df = ephemeris_df[(ephemeris_df['UTC'] >= first_day) & (ephemeris_df['UTC'] <= three_days_later)]
    ephemeris_df['MJD'] = [utc_to_mjd(dt) for dt in ephemeris_df['UTC']]
    start_date_utc = ephemeris_df['UTC'].iloc[0]
    end_date_utc = ephemeris_df['UTC'].iloc[-1]
    delta_t = ephemeris_df['MJD'].iloc[1] - ephemeris_df['MJD'].iloc[0] # constant time steps

    energy_ephemeris_df = ephemeris_df.copy()
    x_ecef, y_ecef, z_ecef, xv_ecef, yv_ecef, zv_ecef = ([] for _ in range(6))
    for _, row in energy_ephemeris_df.iterrows():
        pos = [row['x'], row['y'], row['z']]
        vel = [row['xv'], row['yv'], row['zv']]
        mjd = [row['MJD']]
        itrs_pos, itrs_vel = EME2000_to_ITRF(np.array([pos]), np.array([vel]), pd.Series(mjd))

        x_ecef.append(itrs_pos[0][0])
        y_ecef.append(itrs_pos[0][1])
        z_ecef.append(itrs_pos[0][2])
        xv_ecef.append(itrs_vel[0][0])
        yv_ecef.append(itrs_vel[0][1])
        zv_ecef.append(itrs_vel[0][2])

    energy_ephemeris_df['x_ecef'] = x_ecef
    energy_ephemeris_df['y_ecef'] = y_ecef
    energy_ephemeris_df['z_ecef'] = z_ecef
    energy_ephemeris_df['xv_ecef'] = xv_ecef
    energy_ephemeris_df['yv_ecef'] = yv_ecef
    energy_ephemeris_df['zv_ecef'] = zv_ecef

    converted_coords = np.vectorize(ecef_to_lla, signature='(),(),()->(),(),()')(energy_ephemeris_df['x_ecef'], energy_ephemeris_df['y_ecef'], energy_ephemeris_df['z_ecef'])
    energy_ephemeris_df[['lat', 'lon', 'alt']] = np.column_stack(converted_coords)

    kinetic_energy = (np.linalg.norm([energy_ephemeris_df['xv'], energy_ephemeris_df['yv'], energy_ephemeris_df['zv']], axis=0))**2/2
    energy_ephemeris_df['kinetic_energy'] = kinetic_energy

    monopole_potential = mu / np.linalg.norm([energy_ephemeris_df['x'], energy_ephemeris_df['y'], energy_ephemeris_df['z']], axis=0)
    energy_ephemeris_df['monopole'] = monopole_potential

    U_non_sphericals = []
    U_j2s = []
    work_done_by_other_accs = []

    for _, row in tqdm(energy_ephemeris_df.iterrows(), total=energy_ephemeris_df.shape[0]):
        phi_rad = np.radians(row['lat'])
        lambda_rad = np.radians(row['lon'])
        #make r the norm of the x,y,z coordinates
        r = np.linalg.norm([row['x'], row['y'], row['z']])
        degree = gravity_degree
        order = gravity_order
        date = datetime_to_absolutedate(row['UTC'])

        # Compute non-spherical and J2 potential
        U_non_spher = compute_gravitational_potential(r, phi_rad, lambda_rad, degree, order, date)
        U_j2 = U_J2(r, phi_rad, lambda_rad)
        U_j2s.append(U_j2)
        U_non_sphericals.append(U_non_spher)

        if other_accs:
        # Compute the work done by other non-conservative forces (NOT drag!)
            state_vector = np.array([row['x'], row['y'], row['z'], row['xv'], row['yv'], row['zv']])
            other_acceleration = state2acceleration(state_vector, row['UTC'], settings['cr'], settings['cd'], settings['cross_section'], settings['mass'], **force_model_config)
            other_acceleration = np.sum(list(other_acceleration.values()), axis=0)
            velocity_vector = state_vector[3:]
            work = np.dot(other_acceleration, velocity_vector) * delta_t
            work_done_by_other_accs.append(work)
        else:
            work_done_by_other_accs.append(0)

    #total energies
    energy_total_HOT = kinetic_energy  - monopole_potential + U_non_sphericals
    energy_total_HOT_plus = kinetic_energy  - monopole_potential + U_non_sphericals + work_done_by_other_accs
    HOT_total_diff = energy_total_HOT[0] - energy_total_HOT
    HOT_plus_total_diff = energy_total_HOT_plus[0] - energy_total_HOT_plus
    energy_ephemeris_df['U_j2'] = U_j2s
    energy_ephemeris_df['U_non_spherical'] = U_non_sphericals
    energy_ephemeris_df['HOT_total_diff'] = HOT_total_diff
    energy_ephemeris_df['HOT_plus_total_diff'] = HOT_plus_total_diff
    energy_df = energy_ephemeris_df[['MJD','x', 'y', 'z', 'xv', 'yv', 'zv', 'lat', 'lon', 'alt', 'kinetic_energy', 'monopole', 'U_j2', 'U_non_spherical', 'HOT_total_diff', 'HOT_plus_total_diff']]
    energy_df.to_csv(os.path.join(folder_save, f"{sat_name}_energy_components_{start_date_utc}_{end_date_utc}_{datenow}.csv"), index=False)
    return energy_df

def compute_rho_eff(EDR, velocity, positions, CD, A_ref, mass, dt):
    # Convert inputs to numpy arrays if they aren't already
    positions = np.array(positions)
    velocity = np.array(velocity)
    atm_rot = np.array([0, 0, 72.9211e-6])
    
    # Initialize the output array for rho_eff
    rho_effs = []
    
    # Iterate through each position vector in the positions array
    for i, position in enumerate(positions):
        # Calculate the cross product r x atm_rot
        r_cross_atm_rot = np.cross(position, atm_rot)
        
        # Calculate v_rel for this position
        v_rel = velocity - r_cross_atm_rot
        
        # Calculate the magnitude of v_rel
        v_rel_mag = np.linalg.norm(v_rel)
        
        # Calculate the integral part (since we're given dt and v_rel is constant over dt)
        integral_v_rel3 = dt * v_rel_mag**3
        
        # Solve for rho_eff for this position
        rho_eff = (2 * mass * EDR) / (CD * A_ref * integral_v_rel3)
        rho_effs.append(rho_eff*10)
    
    return np.array(rho_eff)

def Density_from_EDR(sat_name, energy_ephemeris_df, query_models=False):
    MJD_EPOCH = datetime.datetime(1858, 11, 17)
    EDR_df = pd.DataFrame()

    start_date_mjd = energy_ephemeris_df['MJD'].iloc[0]
    start_date_utc = MJD_EPOCH + datetime.timedelta(days=start_date_mjd)

    end_date_mjd = energy_ephemeris_df['MJD'].iloc[-1]
    end_date_utc = MJD_EPOCH + datetime.timedelta(days=end_date_mjd)

    if query_models:
        energy_ephemeris_df['jb08_rho'] = None
        energy_ephemeris_df['dtm2000_rho'] = None
        energy_ephemeris_df['nrlmsise00_rho'] = None

    if 'UTC' not in energy_ephemeris_df.columns:
        energy_ephemeris_df['UTC'] = [start_date_utc + pd.Timedelta(seconds=30) * i for i in range(len(energy_ephemeris_df))]

    EDR = energy_ephemeris_df['HOT_total_diff']
    velocity = np.vstack([energy_ephemeris_df['xv'], energy_ephemeris_df['yv'], energy_ephemeris_df['zv']]).T
    position = np.vstack([energy_ephemeris_df['x'], energy_ephemeris_df['y'], energy_ephemeris_df['z']]).T
    CD = 3.2
    A_ref = 1.004
    mass = 600.2
    dt = 30
    rho_eff = compute_rho_eff(EDR, velocity, position, CD, A_ref, mass, dt)

    EDR_df['MJD'] = energy_ephemeris_df['MJD']
    EDR_df['UTC'] = energy_ephemeris_df['UTC']
    EDR_df['EDR'] = EDR
    EDR_df['rho_eff'] = rho_eff

    if query_models:
        for i in tqdm(range(len(position)), desc='Querying Density Models'):
            pos = position[i]
            t = energy_ephemeris_df['UTC'].iloc[i]
            EDR_df.at[i, 'jb08_rho'] = query_jb08(pos, t)
            # EDR_df.at[i, 'dtm2000_rho'] = query_dtm2000(pos, t)
            # EDR_df.at[i, 'nrlmsise00_rho'] = query_nrlmsise00(pos, t)

    datenow = datetime.datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join("output/DensityInversion/EDR/Data", f"EDR_{sat_name}__{start_date_utc}_{end_date_utc}_{datenow}.csv")
    EDR_df.to_csv(output_path, index=False)
    
    return EDR_df

def main():
    force_model_config = {'3BP': True, 'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True}
    # sat_names_to_test = ["GRACE-FO-A", "TerraSAR-X", "CHAMP", "GRACE-FO-B", "TanDEM-X"]
    sat_names_to_test = ["GRACE-FO-A"]

    for sat_name in sat_names_to_test:
        # orbital_energy_df_degord80 = calculate_orbital_energy(sat_name, force_model_config=force_model_config, gravity_degree=80, gravity_order=80, other_accs=False)
        orbital_energy_df_degord80 = pd.read_csv("output/DensityInversion/EDR/Data/GRACE-FO-A_energy_components_2023-05-05 18:00:12_2023-05-06 18:00:12_2024-05-10.csv")
        plt.plot(orbital_energy_df_degord80['MJD'], orbital_energy_df_degord80['HOT_total_diff'], label='HOT_total_diff_degree_80')
        plt.xlabel('Modified Julian Date')
        plt.ylabel('Δ Specific Energy (J/kg)')
        plt.title(f"{sat_name}: HOT_total_diff")
        plt.legend()
        plt.grid(True)
        # plt.savefig(f"output/DensityInversion/EDR/Plots/EDR_tseries/{sat_name}_OrbitalEnergy_{orbital_energy_df_degord80['MJD'].iloc[0]}_{orbital_energy_df_degord80['MJD'].iloc[-1]}.png")
        plt.show()

        # density_df = Density_from_EDR(sat_name, orbital_energy_df_degord80, query_models=True) #To compute the data run this line
        density_df = pd.read_csv("output/DensityInversion/EDR/Data/EDR_GRACE-FO-A__2023-05-05 18:00:12_2023-05-06 18:00:12_2024-05-13.csv") #To use the precomputed data run this line
        
        plt.figure(figsize=(6, 3))
        plt.plot(density_df['MJD'], (density_df['rho_eff']*10).rolling(window=200, center=True).mean(), label='Rolling Average 200', linewidth=1) # the time steps are 30 seconds so 200 steps corresponds to ~one orbit
        plt.plot(density_df['MJD'], density_df['jb08_rho'], label='JB08 Density')
        plt.plot(density_df['MJD'], density_df['dtm2000_rho'], label='DTM2000 Density')
        plt.plot(density_df['MJD'], density_df['nrlmsise00_rho'], label='NRLMSISE00 Density')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Modified Julian Date')
        plt.ylabel('Effective Density (kg/m^3)')
        plt.title(f"{sat_name}: Effective Density")
        plt.grid(True)
        
        # Save and show plot
        datenow = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = density_df['MJD'].iloc[0]
        end_date = density_df['MJD'].iloc[-1]
        plt.savefig(f"output/DensityInversion/EDR/Plots/Density/{sat_name}_EDR_{start_date}_{end_date}_tstamp{datenow}.png")
        plt.show()

if __name__ == "__main__":
    main()