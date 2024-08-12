import numpy as np
from datetime import datetime, timedelta, timezone
from pyproj import Transformer
from astropy.time import Time
from typing import Tuple, List
import requests
import json
import pandas as pd
import orekit
from orekit.pyhelpers import setup_orekit_curdir, datetime_to_absolutedate
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
# orekit.pyhelpers.download_orekit_data_curdir()
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")

from org.orekit.frames import FramesFactory, ITRFVersion
from org.orekit.utils import PVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import PVCoordinates, IERSConventions
from org.orekit.orbits import KeplerianOrbit, CartesianOrbit
from org.orekit.forces import ForceModel
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import Constants
MU = Constants.WGS84_EARTH_MU
from math import sin, cos, sqrt, radians

def geocentric_distance(latitude_degrees, altitude):
    phi = radians(latitude_degrees)
    a = 6378137  # semi-major axis in meters
    b = 6356752  # semi-minor axis in meters

    R = sqrt((a**2 * cos(phi))**2 + (b**2 * sin(phi))**2) / sqrt((a * cos(phi))**2 + (b * sin(phi))**2)
    return R + altitude

def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to Latitude, Longitude, Altitude (LLA).

    Parameters
    ----------
    x : List[float]
        x coordinates in km.
    y : List[float]
        y coordinates in km.
    z : List[float]
        z coordinates in km.

    Returns
    -------
    tuple
        Latitudes in degrees, longitudes in degrees, and altitudes in km.
    """
    # Convert input coordinates to meters
    x_m, y_m, z_m = x * 1000, y * 1000, z * 1000
    
    # Create a transformer for converting between ECEF and LLA
    transformer = Transformer.from_crs(
        "EPSG:4978", # WGS-84 (ECEF)
        "EPSG:4326", # WGS-84 (LLA)
        always_xy=True # Specify to always return (X, Y, Z) ordering
    )

    # Convert coordinates
    lon, lat, alt_m = transformer.transform(x_m, y_m, z_m)

    # Convert altitude to kilometers
    alt_km = alt_m / 1000

    #convert altitude to distance from the center of the earth
    R = geocentric_distance(lat, alt_km)

    return lat, lon, R

def jd_to_utc(jd: float) -> datetime:
    """
    Convert Julian Date to UTC time tag (datetime object) using Astropy.

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    datetime
        UTC time tag.
    """
    #convert jd to astropy time object
    time = Time(jd, format='jd', scale='utc')
    #convert astropy time object to datetime object
    utc = time.datetime
    return utc

def HCL_diff(eph1: np.ndarray, eph2: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate the Height, Cross-Track, and Along-Track differences at each time step between two ephemerides.

    Parameters
    ----------
    eph1 : np.ndarray
        List or array of state vectors for a satellite.
    eph2 : np.ndarray
        List or array of state vectors for another satellite.

    Returns
    -------
    tuple
        Three lists, each containing the height, cross-track, and along-track differences at each time step.
    """
    H_diffs = []
    C_diffs = []
    L_diffs = []

    # Determine the minimum length to avoid IndexError
    min_length = min(len(eph1), len(eph2))

    for i in range(min_length):
        r1 = np.array(eph1[i][0:3])
        r2 = np.array(eph2[i][0:3])
        
        v1 = np.array(eph1[i][3:6])
        v2 = np.array(eph2[i][3:6])
        
        unit_radial = r1/np.linalg.norm(r1)
        unit_cross_track = np.cross(r1, v1)/np.linalg.norm(np.cross(r1, v1))
        unit_along_track = np.cross(unit_radial, unit_cross_track)

        unit_vectors = np.array([unit_radial, unit_cross_track, unit_along_track])

        r_diff = r1 - r2
        r_diff_HCL = np.matmul(unit_vectors, r_diff)

        h_diff = r_diff_HCL[0]
        c_diff = r_diff_HCL[1]
        l_diff = r_diff_HCL[2]

        H_diffs.append(h_diff)
        C_diffs.append(c_diff)
        L_diffs.append(l_diff)

    return H_diffs, C_diffs, L_diffs

def pos_vel_from_orekit_ephem(ephemeris, initial_date, end_date, step):
    times = []
    state_vectors = []  # Store position and velocity vectors

    # Determine the direction of time flow (forward or backward)
    if initial_date.compareTo(end_date) < 0:
        time_flow_forward = True
        step = abs(step)  # Ensure step is positive for forward propagation
    else:
        time_flow_forward = False
        step = -abs(step)  # Ensure step is negative for backward propagation

    current_date = initial_date
    while (time_flow_forward and current_date.compareTo(end_date) <= 0) or \
          (not time_flow_forward and current_date.compareTo(end_date) >= 0):
        state = ephemeris.propagate(current_date)
        position = state.getPVCoordinates().getPosition().toArray()
        velocity = state.getPVCoordinates().getVelocity().toArray()
        state_vector = np.concatenate([position, velocity])  # Combine position and velocity

        times.append(current_date.durationFrom(initial_date))
        state_vectors.append(state_vector)
        current_date = current_date.shiftedBy(step)

    return times, state_vectors


def extract_acceleration(state_vector_data, TLE_epochDate, SATELLITE_MASS, forceModel, rtn=False):
    """
    Extracts acceleration data using a specified force model and optionally computes RTN components.

    :param state_vector_data: Single state vector or Dictionary containing state vectors and corresponding times.
    :param TLE_epochDate: The initial epoch date for TLE.
    :param SATELLITE_MASS: The mass of the satellite.
    :param forceModel: The instance of the force model to be used.
    :param rtn: Boolean flag to compute RTN components.
    :return: List of acceleration vectors and optionally RTN components.
    """
    # Function to compute unit vector
    def unit_vector(vector):
        norm = vector.getNorm()
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        return Vector3D(vector.getX() / norm, vector.getY() / norm, vector.getZ() / norm)

    # Check if state_vector_data is a dictionary or a single state vector
    if not isinstance(state_vector_data, dict):
        # Assume state_vector_data is a single state vector
        state_vector_data = {'single_state': ([0], [state_vector_data])}

    # Extract states and times for the ephemeris
    states_and_times = state_vector_data[next(iter(state_vector_data))]
    times, states = states_and_times

    # Extract accelerations and RTN components
    accelerations = []
    rtn_components = [] if rtn else None
    for i, pv in enumerate(states):
        duration = times[i]  # Duration in seconds
        if duration == 0:
            current_date = TLE_epochDate
        else:
            current_date = TLE_epochDate.shiftedBy(duration)

        position = Vector3D(float(pv[0]), float(pv[1]), float(pv[2]))
        velocity = Vector3D(float(pv[3]), float(pv[4]), float(pv[5]))
        pvCoordinates = PVCoordinates(position, velocity)

        orbit = CartesianOrbit(pvCoordinates, FramesFactory.getEME2000(), current_date, Constants.WGS84_EARTH_MU)
        state = SpacecraftState(orbit, float(SATELLITE_MASS))

        parameters = ForceModel.cast_(forceModel).getParameters()
        acc = forceModel.acceleration(state, parameters)
        accelerations.append(acc)

        if rtn:
            # Compute RTN components
            radial_unit_vector = unit_vector(position)
            normal_unit_vector = unit_vector(Vector3D.crossProduct(position, velocity))
            transverse_unit_vector = unit_vector(Vector3D.crossProduct(normal_unit_vector, radial_unit_vector))

            radial_component = Vector3D.dotProduct(acc, radial_unit_vector)
            transverse_component = Vector3D.dotProduct(acc, transverse_unit_vector)
            normal_component = Vector3D.dotProduct(acc, normal_unit_vector)
            rtns = [radial_component, transverse_component, normal_component]
            rtn_components.append(rtns)

    return (accelerations, rtn_components) if rtn else accelerations

def pv_to_kep(pvCoordinates, frame, current_date, mu=Constants.WGS84_EARTH_MU):
    keplerian_orbit = KeplerianOrbit(pvCoordinates, frame, current_date, mu)
    a = keplerian_orbit.getA()
    e = keplerian_orbit.getE()
    i = keplerian_orbit.getI()
    omega = keplerian_orbit.getPerigeeArgument()
    raan = keplerian_orbit.getRightAscensionOfAscendingNode()
    v = keplerian_orbit.getTrueAnomaly()
    return (a, e, np.rad2deg(i), np.rad2deg(omega), np.rad2deg(raan), np.rad2deg(v))

def keplerian_elements_from_orekit_ephem(ephemeris, initial_date, end_date, step, mu):
    times = []
    keplerian_elements = []

    current_date = initial_date
    while current_date.compareTo(end_date) <= 0:
        state = ephemeris.propagate(current_date)
        pvCoordinates = state.getPVCoordinates()
        frame = state.getFrame()

        keplerian_elements.append(pv_to_kep(pvCoordinates, frame, current_date, mu))

        times.append(current_date.durationFrom(initial_date))
        current_date = current_date.shiftedBy(step)

    return times, keplerian_elements

def hcl_acc_from_sc_state(spacecraftState, acc_vec):
    """
    Calculate the HCL (Radial, Transverse, Normal) components of the given acc_vec.

    Parameters:
    spacecraftState (SpacecraftState): The state of the spacecraft.
    acc_vec (list or tuple): The acc vector components in ECEF frame.

    Returns:
    tuple: The Radial, Transverse, and Normal components of the acc vector in HCL frame.
    """
    # Get the ECI frame
    eci = FramesFactory.getEME2000()

    # Transform the ERP vector from ECEF to ECI frame
    ecef = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    transform = ecef.getTransformTo(eci, spacecraftState.getDate())
    erp_vec_ecef_pv = PVCoordinates(Vector3D(float(acc_vec[0]), float(acc_vec[1]), float(acc_vec[2])))
    erp_vec_eci_pv = transform.transformPVCoordinates(erp_vec_ecef_pv)
    erp_vec_eci = erp_vec_eci_pv.getPosition()

    # Calculate the ECI position and velocity vectors
    pv_eci = spacecraftState.getPVCoordinates(eci)
    position_eci = pv_eci.getPosition()
    velocity_eci = pv_eci.getVelocity()

    # Calculate the RTN (HCL) unit vectors
    radial_unit_vector = position_eci.normalize()
    normal_unit_vector = Vector3D.crossProduct(position_eci, velocity_eci).normalize()
    transverse_unit_vector = Vector3D.crossProduct(normal_unit_vector, radial_unit_vector)

    # Project the ERP vector onto the RTN axes
    radial_component = Vector3D.dotProduct(erp_vec_eci, radial_unit_vector)
    transverse_component = Vector3D.dotProduct(erp_vec_eci, transverse_unit_vector)
    normal_component = Vector3D.dotProduct(erp_vec_eci, normal_unit_vector)

    return radial_component, transverse_component, normal_component

def yyyy_mm_dd_hh_mm_ss_to_jd(year: int, month: int, day: int, hour: int, minute: int, second: int, milisecond: int) -> float:
    """
    Convert year, month, day, hour, minute, second to Julian Date.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month.
    day : int
        Day.
    hour : int
        Hour.
    minute : int
        Minute.
    second : int
        Second.
    milisecond : int
        Millisecond.

    Returns
    -------
    float
        Julian Date.
    """
    dt_obj = datetime(year, month, day, hour, minute, second, milisecond*1000)
    jd = Time(dt_obj).jd
    return jd

def doy_to_dom_month(year: int, doy: int) -> Tuple[int, int]:
    """
    Convert day of year to day of month and month number.

    Parameters
    ----------
    year : int
        Year.
    doy : int
        Day of year.

    Returns
    -------
    tuple
        Day of month and month number.
    """
    d = datetime(year, 1, 1) + timedelta(doy - 1)
    day_of_month = d.day
    month = d.month
    return day_of_month, month

def jd_to_utc(jd: float) -> datetime:
    """
    Convert Julian Date to UTC time tag (datetime object) using Astropy.

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    datetime
        UTC time tag.
    """
    #convert jd to astropy time object
    time = Time(jd, format='jd', scale='utc', precision=9)
    #convert astropy time object to datetime object
    utc = time.datetime
    return utc

def utc_to_mjd(utc_time: datetime) -> float:
    """
    Convert UTC time (datetime object) to Modified Julian Date using Astropy,
    rounding to the nearest full second to avoid timing errors.

    Parameters
    ----------
    utc_time : datetime
        UTC time tag.

    Returns
    -------
    float
        Modified Julian Date.
    """
    # Round the input datetime to the nearest full second
    if utc_time.microsecond >= 500000:
        rounded_utc_time = utc_time + timedelta(seconds=1)
        rounded_utc_time = rounded_utc_time.replace(microsecond=0)
    else:
        rounded_utc_time = utc_time.replace(microsecond=0)

    #the time is rounded otherwise some floating point errors can occur in the microseconds
    # Convert the rounded datetime object to Astropy Time object
    time = Time(utc_time, format='datetime', scale='utc', precision=8)

    # Convert to Modified Julian Date
    mjd = time.mjd
    return mjd

def gps_time_to_utc(gps_time, GPS_EPOCH):
    # NOTE: Specific to GRACE-FO GPS time...
    utc_time = GPS_EPOCH + timedelta(seconds=gps_time)
    return utc_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Format to nearest millisecond

def std_dev_from_lower_triangular(lower_triangular_data):
    cov_matrix = np.zeros((6, 6))
    row, col = np.tril_indices(6)
    cov_matrix[row, col] = lower_triangular_data
    cov_matrix = cov_matrix + cov_matrix.T - np.diag(cov_matrix.diagonal())
    std_dev = np.sqrt(np.diag(cov_matrix))
    return std_dev

def keys_to_string(d):
    """
    Convert the keys of a dictionary to a string with each key on a new line.
    """
    return '\n'.join(d.keys())

def SP3_to_EME2000(itrs_pos, itrs_vel, mjds):
    # Orekit Frames
    frame_CTS = FramesFactory.getITRF(ITRFVersion.ITRF_2014, IERSConventions.IERS_2010, False)
    frame_EME2000 = FramesFactory.getEME2000()

    # Prepare output arrays
    eme2000_pos = np.empty_like(itrs_pos)
    eme2000_vel = np.empty_like(itrs_vel)

    # Iterate over each row of position, velocity, and corresponding MJD
    for i in range(len(itrs_pos)):
        # Convert MJD to Julian Date and then to UTC datetime
        mjd = mjds.iloc[i]
        jd = mjd + 2400000.5
        days_since_epoch = jd - 2400000.5
        base_date = datetime(1858, 11, 17)

        # Round the days to the nearest second before creating the timedelta
        # NOTE: this is to avoid floating point errors when creating the timedelta
        seconds_since_epoch = (days_since_epoch * 86400)
        seconds_since_epoch = round(days_since_epoch * 86400)
        dt = base_date + timedelta(seconds=seconds_since_epoch)
        dt = dt.replace(tzinfo=timezone.utc)

        # Convert datetime to AbsoluteDate
        absolute_date = datetime_to_absolutedate(dt)

        # Convert inputs to Orekit's Vector3D and PVCoordinates (and convert from km to m)
        itrs_pos_vector = Vector3D(float(itrs_pos[i, 0] * 1000), float(itrs_pos[i, 1] * 1000), float(itrs_pos[i, 2] * 1000))
        itrs_vel_vector = Vector3D(float(itrs_vel[i, 0] * 1000), float(itrs_vel[i, 1] * 1000), float(itrs_vel[i, 2] * 1000))
        pv_itrs = PVCoordinates(itrs_pos_vector, itrs_vel_vector)

        # Transform Coordinates
        cts_to_eme2000 = frame_CTS.getTransformTo(frame_EME2000, absolute_date)
        pveci = cts_to_eme2000.transformPVCoordinates(pv_itrs)

        # Extract position and velocity from transformed coordinates
        eme2000_pos[i] = [pveci.getPosition().getX(), pveci.getPosition().getY(), pveci.getPosition().getZ()]
        eme2000_vel[i] = [pveci.getVelocity().getX(), pveci.getVelocity().getY(), pveci.getVelocity().getZ()]

        # Convert back from m to km 
        eme2000_pos[i] = eme2000_pos[i] / 1000
        eme2000_vel[i] = eme2000_vel[i] / 1000

    return eme2000_pos, eme2000_vel

def EME2000_to_ITRF(eme2000_pos, eme2000_vel, mjds):
    # Orekit Frames
    frame_CTS = FramesFactory.getITRF(ITRFVersion.ITRF_2014, IERSConventions.IERS_2010, False)
    frame_EME2000 = FramesFactory.getEME2000()

    # Prepare output arrays
    itrs_pos = np.empty_like(eme2000_pos)
    itrs_vel = np.empty_like(eme2000_vel)

    # Iterate over each row of position, velocity, and corresponding MJD
    for i in range(len(eme2000_pos)):
        # Convert MJD to Julian Date and then to UTC datetime
        mjd = mjds.iloc[i]
        jd = mjd + 2400000.5
        days_since_epoch = jd - 2400000.5
        base_date = datetime(1858, 11, 17)
        seconds_since_epoch = round(days_since_epoch * 86400)
        dt = base_date + timedelta(seconds=seconds_since_epoch)
        dt = dt.replace(tzinfo=timezone.utc)

        # Convert datetime to AbsoluteDate
        absolute_date = datetime_to_absolutedate(dt)

        # Convert inputs to Orekit's Vector3D and PVCoordinates (and convert from km to m)
        eme2000_pos_vector = Vector3D(float(eme2000_pos[i, 0] * 1000), float(eme2000_pos[i, 1] * 1000), float(eme2000_pos[i, 2] * 1000))
        eme2000_vel_vector = Vector3D(float(eme2000_vel[i, 0] * 1000), float(eme2000_vel[i, 1] * 1000), float(eme2000_vel[i, 2] * 1000))
        pv_eme2000 = PVCoordinates(eme2000_pos_vector, eme2000_vel_vector)

        # Transform Coordinates
        eme2000_to_cts = frame_EME2000.getTransformTo(frame_CTS, absolute_date)
        pv_itrs = eme2000_to_cts.transformPVCoordinates(pv_eme2000)

        # Extract position and velocity from transformed coordinates
        itrs_pos[i] = [pv_itrs.getPosition().getX(), pv_itrs.getPosition().getY(), pv_itrs.getPosition().getZ()]
        itrs_vel[i] = [pv_itrs.getVelocity().getX(), pv_itrs.getVelocity().getY(), pv_itrs.getVelocity().getZ()]

        # Convert back from m to km
        itrs_pos[i] = itrs_pos[i] / 1000
        itrs_vel[i] = itrs_vel[i] / 1000

    return itrs_pos, itrs_vel

    # Function to download file and return a java.io.File object
def download_file_url(url, local_filename):
    from java.io import File
    #mianly used to download the  solfsmy and dtc files for JB2008
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return File(local_filename)

def calculate_cross_correlation_matrix(covariance_matrices):
    """
    Calculate cross-correlation matrices for a list of covariance matrices.

    Args:
    covariance_matrices (list of np.array): List of covariance matrices.

    Returns:
    List of np.array: List of cross-correlation matrices corresponding to each covariance matrix.
    """
    correlation_matrices = []
    for cov_matrix in covariance_matrices:
        # Ensure the matrix is a numpy array
        cov_matrix = np.array(cov_matrix)

        # Diagonal elements (variances)
        variances = np.diag(cov_matrix)

        # Standard deviations (sqrt of variances)
        std_devs = np.sqrt(variances)

        # Initialize correlation matrix
        corr_matrix = np.zeros_like(cov_matrix)

        # Calculate correlation matrix
        for i in range(len(cov_matrix)):
            for j in range(len(cov_matrix)):
                corr_matrix[i, j] = cov_matrix[i, j] / (std_devs[i] * std_devs[j])

        correlation_matrices.append(corr_matrix)

    return correlation_matrices

def get_satellite_info(satellite_name, file_path='misc/sat_list.json'):
    with open(file_path, 'r') as file:
        sat_data = json.load(file)

    if satellite_name in sat_data:
        info = sat_data[satellite_name]
        return {k: info[k] for k in ['mass', 'cross_section', 'cd', 'cr']}
    else:
        return "Satellite not found in the list."

def posvel_to_sma(x, y, z, u, v, w):
    """
    Calculate the semi-major axis from position and velocity vectors.

    :param x, y, z: Position coordinates in meters
    :param u, v, w: Velocity components in m/s
    :return: Semi-major axis in meters
    """
    # Standard gravitational parameter for Earth in m^3/s^2 (converted from km^3/s^2)
    mu = MU

    # Position and velocity vectors
    r_vector = np.array([x, y, z])
    v_vector = np.array([u, v, w])

    # Magnitudes of r and v
    r = np.linalg.norm(r_vector)
    v = np.linalg.norm(v_vector)

    # Specific mechanical energy
    epsilon = v**2 / 2 - mu / r

    # Semi-major axis
    a = -mu / (2 * epsilon)

    return a

def interpolate_positions(df, fine_freq):
    df = df.drop_duplicates(subset='UTC').set_index('UTC')
    df = df.sort_index()
    start_time, end_time = df.index.min(), df.index.max()

    df_resampled = pd.DataFrame(index=pd.date_range(start=start_time, end=end_time, freq=fine_freq))
    columns = ['x', 'y', 'z', 'xv', 'yv', 'zv']
    df_interpolated = pd.DataFrame(index=df_resampled.index, columns=columns)

    for pos_col, vel_col in zip(['x', 'y', 'z'], ['xv', 'yv', 'zv']):
        times = df.index.astype(int) / 10**9
        new_times = df_resampled.index.astype(int) / 10**9

        df_interpolated[pos_col] = CubicSpline(times, df[pos_col])(new_times)
        df_interpolated[vel_col] = CubicSpline(times, df[vel_col])(new_times)

    df_interpolated.reset_index(inplace=True)
    df_interpolated.rename(columns={'index': 'UTC'}, inplace=True)

    return df_interpolated

def calculate_acceleration(df_interpolated, fine_freq, filter_window_length, filter_polyorder):
    fine_freq_seconds = pd.to_timedelta(fine_freq).total_seconds()
    for vel_col, acc_col in zip(['xv', 'yv', 'zv'], ['accx', 'accy', 'accz']):
        velocities = df_interpolated[vel_col].to_numpy()
        if filter_window_length > len(velocities):
            filter_window_length = len(velocities) | 1  # Ensure the window length is odd
        df_interpolated[vel_col] = savgol_filter(velocities, filter_window_length, filter_polyorder)
        df_interpolated[acc_col] = np.gradient(df_interpolated[vel_col], fine_freq_seconds)

    return df_interpolated

def project_acc_into_HCL(x_acc, y_acc, z_acc, x, y, z, xv, yv, zv):
        
        r = np.array([x, y, z])
        v = np.array([xv, yv, zv])

        unit_radial = r/np.linalg.norm(r)
        unit_cross_track = np.cross(r, v)/np.linalg.norm(np.cross(r, v))
        unit_along_track = np.cross(unit_radial, unit_cross_track)

        #put the three unit vectors into a matrix
        unit_vectors = np.array([unit_radial, unit_cross_track, unit_along_track])

        acc = np.array([x_acc, y_acc, z_acc])

        #now project the acceleration vector into the HCL frame
        acc_HCL = np.matmul(unit_vectors, acc)
        h_acc = acc_HCL[0]
        c_acc = acc_HCL[1]
        l_acc = acc_HCL[2]
        return h_acc, c_acc, l_acc