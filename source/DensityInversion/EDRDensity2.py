import orekit
from orekit.pyhelpers import setup_orekit_curdir
# orekit.pyhelpers.download_orekit_data_curdir() #run this if you don't have the orekit-data.zip file to download it
vm = orekit.initVM()
setup_orekit_curdir("orekit-data.zip")

import pandas as pd
import matplotlib.pyplot as plt

from org.orekit.frames import FramesFactory
from orekit.pyhelpers import datetime_to_absolutedate
import numpy as np
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from orekit.pyhelpers import setup_orekit_curdir,setup_orekit_curdir, absolutedate_to_datetime, datetime_to_absolutedate

import numpy as np

from org.orekit.utils import Constants
from org.orekit.utils import PVCoordinates
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions
import numpy as np
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel

eciFrame = FramesFactory.getEME2000()
r_earth_const = Constants.WGS84_EARTH_EQUATORIAL_RADIUS #m
omega_const = Constants.WGS84_EARTH_ANGULAR_VELOCITY #rad/s, mean motion of earth's rotation
mu_earth = Constants.WGS84_EARTH_MU #m^3/s^2

# --- Simple ballistic‑coefficient estimate for GRACE‑FO -------------
#   B = M / (C_d * A)  ≈ 600.2 kg / (2.2 × 1.0 m²)
mass = 600.2      # kg
area = 1.04      # m^2
C_D   = 2.2
BALLISTIC_COEFF = C_D * (area / mass)
from org.hipparchus.geometry.euclidean.threed import Vector3D

def Numpy2Vector3D(numpyArray):
    return Vector3D(float(numpyArray[0]),float(numpyArray[1]),float(numpyArray[2]))

def PV2Numpy(PV_coord):
    position_PV = PV_coord.getPosition() #m
    velocity_PV = PV_coord.getVelocity() #m/s
    pos_vel_numpy = np.array([position_PV.getX(), position_PV.getY(), position_PV.getZ(), velocity_PV.getX(), velocity_PV.getY(), velocity_PV.getZ()])
    return pos_vel_numpy

def compute_nonSpherical_gravitational_potential(pos_ecef, this_datetime):
    date = datetime_to_absolutedate(this_datetime)
    ecefFrame = FramesFactory.getITRF(IERSConventions.IERS_2010,True)
    harmonics_provider = GravityFieldFactory.getNormalizedProvider(120, 120)
    gravityfield = HolmesFeatherstoneAttractionModel(ecefFrame, harmonics_provider)
    non_sperical_grav_pot = gravityfield.nonCentralPart(date, Numpy2Vector3D(pos_ecef), Constants.WGS84_EARTH_MU)
    return non_sperical_grav_pot

def compute_orbit_energy(pos_vel_array, datetime_array, indices):
    orbit_energy_array = np.zeros(len(indices))
    nonspherical_grav_pot_array = np.zeros(len(indices))
    orbit_energy_term1_array = np.zeros(len(indices))
    eci_KE_PE_total_array = np.zeros((len(indices), 3))
    for i, idx in enumerate(indices):
        pos_vel_eci = pos_vel_array[idx]
        this_datetime = datetime_array[idx]
        date = datetime_to_absolutedate(this_datetime)
        ecefFrame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        pvCoordinates_ECI = PVCoordinates(Numpy2Vector3D(pos_vel_eci[:3]), Numpy2Vector3D(pos_vel_eci[3:]))
        satellite_pvCoordinates_ecef = eciFrame.getTransformTo(ecefFrame, date).transformPVCoordinates(pvCoordinates_ECI)
        pos_vel_ecef = PV2Numpy(satellite_pvCoordinates_ecef)
        non_sperical_grav_pot = compute_nonSpherical_gravitational_potential(pos_vel_ecef[:3], this_datetime)
        v_sqrd = np.linalg.norm(pos_vel_ecef[3:])**2
        orbit_energy = v_sqrd / 2 - Constants.WGS84_EARTH_ANGULAR_VELOCITY**2 * (pos_vel_ecef[0]**2 + pos_vel_ecef[1]**2) / 2 - Constants.WGS84_EARTH_MU / np.linalg.norm(pos_vel_ecef[:3]) - non_sperical_grav_pot
        orbit_energy_term1 = v_sqrd / 2 - Constants.WGS84_EARTH_ANGULAR_VELOCITY**2 * (pos_vel_ecef[0]**2 + pos_vel_ecef[1]**2) / 2 - Constants.WGS84_EARTH_MU / np.linalg.norm(pos_vel_ecef[:3])
        orbit_energy_array[i] = orbit_energy
        nonspherical_grav_pot_array[i] = non_sperical_grav_pot
        orbit_energy_term1_array[i] = orbit_energy_term1
        KE_eci = (np.linalg.norm(pos_vel_eci[3:])**2) / 2
        PE_ECI = -Constants.WGS84_EARTH_MU / np.linalg.norm(pos_vel_eci[:3])
        total_energy_eci = KE_eci + PE_ECI
        eci_KE_PE_total_array[i] = [KE_eci, PE_ECI, total_energy_eci]
    return orbit_energy_array, nonspherical_grav_pot_array, orbit_energy_term1_array, eci_KE_PE_total_array

import numpy as np
from tqdm.auto import tqdm

def compute_EDR(X_sol_array, X_sol_datetime_array, perigees,
                fitspan=2, window_size=1):
    """
    Computes the energy dissipation rate (EDR) on a perigee-to-perigee basis,
    optionally over multiple perigees (fitspan), and with an optional smoothing
    window around each perigee.

    Parameters
    ----------
    X_sol_array : (N,6) array
        ECI position & velocity at each timestep.
    X_sol_datetime_array : list of datetime
        Times corresponding to each row of X_sol_array.
    perigees : list of datetime
        The times of each perigee (from determinePerigees).
    fitspan : int
        Number of perigee-to-perigee arcs to span (default 1).
    window_size : int
        Number of neighbor points *around* each perigee to use for smoothing.
        The actual half-window is window_size//2 on each side.
    Returns
    -------
    edr_list : list of float
        Smoothed total EDR for each arc.
    """

    times = np.array(X_sol_datetime_array, dtype='datetime64[s]')
    n_states = len(times)

    perigee_indices = []
    for p in perigees:
        delta_secs = np.abs((times - np.datetime64(p, 's')).astype(float))
        perigee_indices.append(int(np.argmin(delta_secs)))

    half_win = window_size // 2
    needed = set()
    for i in range(len(perigee_indices) - fitspan):
        i0 = perigee_indices[i]
        i1 = perigee_indices[i + fitspan]
        for k in range(-half_win, half_win + 1):
            si = i0 + k
            ei = i1 + k
            if 0 <= si < n_states:
                needed.add(si)
            if 0 <= ei < n_states:
                needed.add(ei)

    idx_list = sorted(needed)
    energy_vals, _, _, _ = compute_orbit_energy(
        X_sol_array, X_sol_datetime_array, idx_list
    )
    energy_map = dict(zip(idx_list, energy_vals))

    edr_list = []

    for i in range(len(perigee_indices) - fitspan):
        i0 = perigee_indices[i]
        i1 = perigee_indices[i + fitspan]
        dt_center = float((times[i1] - times[i0]).astype(float))
        if dt_center == 0:
            edr_list.append(0.0)
            continue

        small_edrs = []
        for k in range(-half_win, half_win + 1):
            si = i0 + k
            ei = i1 + k
            if si in energy_map and ei in energy_map:
                dt = float((times[ei] - times[si]).astype(float))
                if dt > 0:
                    Es = energy_map[si]
                    Ee = energy_map[ei]
                    small_edrs.append(-(Ee - Es) / dt)

        edr_cons = float(np.median(small_edrs)) if small_edrs else 0.0
        edr_list.append(edr_cons)

    return edr_list

def compute_rel_vel_mag(pos_vel):
    #no winds, co-rotating atmosphere
    omega = np.array([0, 0, 7.2921159e-5])  # Earth rotation rate [rad/s]
    return np.linalg.norm(pos_vel[:, 3:] - np.cross(omega, pos_vel[:, :3]), axis=1)


# -------------------------------------------------------------------
# Effective density for one perigee-to-perigee arc from EDR
# -------------------------------------------------------------------
def effective_density_arc(edr_value, pos_vel_arc, dt_total,
                           ballistic_coeff=BALLISTIC_COEFF):
    """
    Compute effective density for one perigee-to-perigee arc using

        ρ_eff = 2·EDR·Δt / [(C_D·A/M) · ∫ v_rel² |v| dt]

    Parameters
    ----------
    edr_value : float
        Energy dissipation rate for the arc.
    pos_vel_arc : (M,6) ndarray
        ECI position/velocity samples for the same arc.
    dt_total : float
        Total duration of the arc in s.
    ballistic_coeff : float
        C_D · A / M  (m² kg⁻¹).

    Returns
    -------
    rho_eff : float
        Effective density (kg m⁻³).
    """
    # v_rel = |v − ω×r| assuming a co-rotating atmosphere
    v_rel  = compute_rel_vel_mag(pos_vel_arc)               # shape (M,)
    speeds = np.linalg.norm(pos_vel_arc[:, 3:], axis=1)     # |v|  (M,)

    # integrand = v_rel² |v|
    integrand = v_rel**2 * speeds
    dt = dt_total / (len(v_rel) - 1)
    integral = np.sum(integrand * dt)                       # ∫ v²|v| dt

    if integral == 0 or np.isnan(integral):
        return np.nan

    rho_eff = (2 * edr_value * dt_total) / (ballistic_coeff * integral)
    return rho_eff

def compute_effective_density(EDR):
    start, end = EDR.EDR_datetime_range
    Δt = (end - start).total_seconds()         # entire orbit
    v_rel  = EDR.all_rel_vel_mag               # shape (N,)
    speeds = np.linalg.norm(EDR.all_pos_vel_states[:, 3:], axis=1)  # shape (N,)

    N  = len(v_rel)
    dt = Δt / (N - 1)                      

    integrand = v_rel**2 * speeds
    integral  = np.sum(integrand * dt)        # ∫ v²|v| dt

    numerator   = 2 * EDR.EDR_value * Δt
    denominator = EDR.ballistic_coeff * integral

    return numerator / denominator

def determinePerigees(X_sol_array, X_sol_datetime_array):
    from datetime import timedelta
    X_sol_datetime_array = np.array(X_sol_datetime_array)
    radial_velocity = np.sum(X_sol_array[:, :3] * X_sol_array[:, 3:], axis=1) / np.linalg.norm(X_sol_array[:, :3], axis=1)
    perigee_indices = np.where(np.diff(np.sign(radial_velocity)) > 0)[0]
    candidate_times = X_sol_datetime_array[perigee_indices]
    if len(candidate_times) == 0:
        return candidate_times
    accepted = [candidate_times[0]]
    target_interval = 90
    i = 1
    while i < len(candidate_times):
        gap = (candidate_times[i] - accepted[-1]).total_seconds() / 60.0
        if gap < 80:
            i += 1
            continue
        temp = []
        while i < len(candidate_times):
            gap = (candidate_times[i] - accepted[-1]).total_seconds() / 60.0
            if gap <= 110:
                if gap >= 80:
                    temp.append(candidate_times[i])
                i += 1
            else:
                break
        if temp:
            best_candidate = min(temp, key=lambda t: abs(((t - accepted[-1]).total_seconds() / 60.0) - target_interval))
            accepted.append(best_candidate)
        else:
            accepted.append(candidate_times[i])
            i += 1
    return np.array(accepted)

# -------------------------------------------------------------------
# Stand‑alone test on one storm CSV
# -------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = "/Users/charlesc/Documents/GitHub/POD-Density-Inversion/output/PODDensityInversion/Data/StormAnalysis/GRACE-FO/GRACE-FO-A_2024-05-10_density_inversion_withEDR.csv"

    # ---------- Load ephemeris & densities --------------------------
    df = pd.read_csv(csv_path, parse_dates=['UTC'])

    pos_vel_cols = ['x','y','z','xv','yv','zv']
    X_sol_array  = df[pos_vel_cols].values.astype(float)
    times_list   = df['UTC'].tolist()

    # ---------- Perigees & EDR per orbit ---------------------------
    perigee_times = determinePerigees(X_sol_array, times_list)
    edr_values = compute_EDR(X_sol_array, times_list, perigee_times, fitspan=1, window_size=1)
    # ---------- Effective density per orbit ------------------------
    orbit_mid_times = []
    rho_edr = []
    for k in range(len(perigee_times) - 1):
        # indices for this arc
        t0, t1 = perigee_times[k], perigee_times[k+1]
        mask = (df['UTC'] >= t0) & (df['UTC'] < t1)
        arc_states = X_sol_array[mask.values]
        dt_total = (t1 - t0).total_seconds()
        rho_eff = effective_density_arc(edr_values[k], arc_states, dt_total)
        rho_edr.append(rho_eff)
        orbit_mid_times.append(t0 + (t1 - t0)/2)

    # ---------- Plot ACC, NRLMSISE‑00, EDR --------------------------
    plt.figure(figsize=(10,5))
    plt.plot(df['UTC'], df['AccelerometerDensity'], label='Accelerometer', color='black')
    plt.plot(df['UTC'], df['NRLMSISE-00'], label='NRLMSISE‑00', color='green')

    plt.plot(orbit_mid_times, rho_edr, 'o-', label='EDR‑derived density', color='red')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Density (kg m⁻³)')
    plt.legend()
    plt.title('GRACE‑FO Storm: Accelerometer vs NRLMSISE‑00 vs EDR')
    plt.tight_layout()
    plt.show()