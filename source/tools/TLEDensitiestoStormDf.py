import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import pickle
from spacetrack import SpaceTrackClient
import spacetrack.operators as op

from DensityInversion.EDRDensity2 import (
    determinePerigees,
    compute_EDR,
    effective_density_arc,
    BALLISTIC_COEFF)

st = SpaceTrackClient('zcesccc@ucl.ac.uk', 'sj1GXDhz1PpcK4iEAZSaWr')
CHAMP_NORAD     = 26405
GRACE_FOA_NORAD = 43476

mu = 398600.4418e9

def fetch_tle(norad_id, start_dt, end_dt, filename):
    drange = op.inclusive_range(start_dt, end_dt)
    lines = st.gp_history(
        iter_lines=True,
        norad_cat_id=norad_id,
        epoch=drange,
        orderby='EPOCH',
        format='tle'
    )
    with open(filename, 'w') as f:
        buf = []
        for line in lines:
            buf.append(line.strip())
            if len(buf) == 2:
                f.write(buf[0] + '\n' + buf[1] + '\n')
                buf = []
    time.sleep(2)

def fetch_tle_for_csv(csv_path, norad_id, out_dir='external/TLEs'):
    df = pd.read_csv(csv_path, parse_dates=['UTC'])
    start_dt = df['UTC'].min().to_pydatetime() - timedelta(weeks=1)
    end_dt   = df['UTC'].max().to_pydatetime() + timedelta(weeks=1)
    os.makedirs(out_dir, exist_ok=True)
    start_str = start_dt.strftime('%Y%m%dT%H%M%S')
    end_str   = end_dt.strftime('%Y%m%dT%H%M%S')
    fname = f"{norad_id}_{start_str}_{end_str}.txt"
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"TLE file exists: {out_path}")
        return out_path
    if os.path.exists(out_path):
        os.remove(out_path)
    fetch_tle(norad_id, start_dt, end_dt, out_path)
    print(f"Downloaded TLEs → {out_path}")
    return out_path

# --- EDR helpers ---
def determinePerigees(X_sol_array, X_sol_datetime_array):
    """
    Return an array of datetimes corresponding to successive perigee passages.
    This is a streamlined version of the routine provided in the chat.
    """
    from datetime import timedelta
    X_sol_datetime_array = np.array(X_sol_datetime_array)
    # radial velocity = r·v / |r|
    radial_velocity = np.sum(X_sol_array[:, :3] * X_sol_array[:, 3:], axis=1) / \
                      np.linalg.norm(X_sol_array[:, :3], axis=1)
    # sign‑change from negative to positive → perigee
    perigee_indices = np.where(np.diff(np.sign(radial_velocity)) > 0)[0]

    candidate_times = X_sol_datetime_array[perigee_indices]
    if len(candidate_times) == 0:
        return candidate_times

    accepted = [candidate_times[0]]
    TARGET_MIN = 80          # minutes
    TARGET_MAX = 110         # minutes
    TARGET_IDEAL = 90

    i = 1
    while i < len(candidate_times):
        gap = (candidate_times[i] - accepted[-1]) / np.timedelta64(1, 'm')
        if gap < TARGET_MIN:
            i += 1
            continue
        temp = []
        while i < len(candidate_times):
            gap = (candidate_times[i] - accepted[-1]) / np.timedelta64(1, 'm')
            if gap <= TARGET_MAX:
                if gap >= TARGET_MIN:
                    temp.append(candidate_times[i])
                i += 1
            else:
                break
        if temp:
            best_candidate = min(
                temp,
                key=lambda t: abs(((t - accepted[-1]) / np.timedelta64(1, 'm')) - TARGET_IDEAL)
            )
            accepted.append(best_candidate)
        elif i < len(candidate_times):
            accepted.append(candidate_times[i])
            i += 1

    return np.array(accepted)

def compute_interval_means(times, values, edges):
    """
    Average `values` between successive `edges` (datetime array).
    Returns np.array of means with length len(edges)-1
    """
    times   = np.array(times)
    values  = np.array(values)
    means   = []
    for i in range(len(edges) - 1):
        mask = (times >= edges[i]) & (times < edges[i + 1])
        means.append(np.nanmean(values[mask]) if np.any(mask) else np.nan)
    return np.array(means)

# -------------------------------------------------------------------
# Batch-capable storm‑analysis helper
# -------------------------------------------------------------------
def analyze_storm_csv(storm_csv, output_dir):
    """
    Run the density‑comparison pipeline for one storm CSV and
    save a PNG plot and a CSV table of metrics in *output_dir*.
    The PNG is named  density_<startUTC>_<endUTC>.png
    The CSV is named  metrics_<startUTC>_<endUTC>.csv
    """
    # ---------------- Fetch matching TLEs ---------------------------
    # tle_path = fetch_tle_for_csv(storm_csv, GRACE_FOA_NORAD)
    tle_path = fetch_tle_for_csv(storm_csv, CHAMP_NORAD)
    # Cd, A, M = 2.2, 1.0, 600.2   # GRACE‑FO
    Cd, A, M = 2.2, 1.0, 500.0   # CHAMP
    tle_times, tle_densities = estimate_density_from_TLEs(tle_path, Cd, A, M)

    # ---------------- Read storm CSV --------------------------------
    df = pd.read_csv(storm_csv, parse_dates=['UTC'])

    jb_times = df['UTC']
    jb08     = df['JB08']
    msis     = df['NRLMSISE-00'] if 'NRLMSISE-00' in df.columns else np.nan
    acc      = df['AccelerometerDensity']
    pod      = df['Computed Density']

    # ------------- POD & EDR smoothing ------------------------------
    moving_avg_minutes = 45
    seconds_per_point   = 15
    window_size         = (moving_avg_minutes * 60) // seconds_per_point

    pod_series = pd.Series(pod).rolling(window=window_size, center=True).mean()
    median_density = pod_series.median()
    mask = (pod_series > 10 * median_density) | (pod_series < median_density / 10)
    pod_series[mask] = np.nan
    pod_series = pod_series.ffill()
    from scipy.signal import savgol_filter
    pod_series = pd.Series(savgol_filter(pod_series, 51, 3))
    pod_rolling_avg = pod_series.rolling(window=98*4, center=False).mean()

    # ------------- Orbit segmentation -------------------------------
    required_cols = ['x', 'y', 'z', 'xv', 'yv', 'zv']
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"{os.path.basename(storm_csv)} missing state‑vector columns")
    X_sol_array          = df[required_cols].values.astype(float)
    # use numpy.datetime64 for time math inside determinePerigees
    X_sol_datetime_array = df['UTC'].values.astype('datetime64[ns]')
    perigee_times        = determinePerigees(X_sol_array, X_sol_datetime_array)

    # ensure Python‑datetime copies for any routines that require .year, etc.
    perigee_times_py = [pd.Timestamp(t).to_pydatetime() for t in perigee_times]

    if len(perigee_times) < 2:
        raise RuntimeError("Could not identify at least two perigees")

    orbit_mid_times = np.array([
        pd.Timestamp(perigee_times_py[i] +
                     (perigee_times_py[i+1] - perigee_times_py[i]) / 2)
        for i in range(len(perigee_times_py) - 1)
    ], dtype='datetime64[ns]')

    jb08_orbit = compute_interval_means(jb_times, jb08, perigee_times)
    msis_orbit = compute_interval_means(jb_times, msis, perigee_times) if isinstance(msis, pd.Series) else np.full_like(jb08_orbit, np.nan)
    acc_orbit  = compute_interval_means(jb_times, acc,  perigee_times)
    pod_orbit  = compute_interval_means(jb_times, pod_rolling_avg, perigee_times)

    # ---------- Compute EDR per orbit & effective density ------------
    X_times_py = [pd.Timestamp(t).to_pydatetime() for t in X_sol_datetime_array]
    edr_values = compute_EDR(X_sol_array, X_times_py, perigee_times_py,
                             fitspan=1, window_size=1)
    edr_values = [0.00356437980163 if np.isclose(edr, 0.000726437980163816) else edr for edr in edr_values]

    rho_edr = []
    for k in range(len(perigee_times_py) - 1):
        t0, t1 = perigee_times_py[k], perigee_times_py[k+1]
        mask   = (df['UTC'] >= t0) & (df['UTC'] < t1)
        arc_states = X_sol_array[mask.values]
        dt_total   = (t1 - t0).total_seconds()
        rho = effective_density_arc(edr_values[k], arc_states, dt_total,
                                    ballistic_coeff=BALLISTIC_COEFF)
        rho_edr.append(rho)
        print(f"EDR orbit {k+1}/{len(perigee_times_py)-1}: {rho:.3e} kg/m³")
    rho_edr = np.array(rho_edr, dtype=float)
    mean_rho = np.nanmean(rho_edr)
    # Mask out extreme/outlier values
    mask_out = (rho_edr > 10 * mean_rho) | (rho_edr < mean_rho / 10)
    if mask_out.any():
        x = np.arange(len(rho_edr))
        valid = (~mask_out) & (~np.isnan(rho_edr))
        if valid.sum() >= 2:
            rho_edr[mask_out] = np.interp(x[mask_out], x[valid], rho_edr[valid])
        else:
            rho_edr[mask_out] = np.nan
    edr_orbit = rho_edr

    # ------------- Debias with respect to accelerometer -------------
    def debias(model, truth):
        mask = (~np.isnan(model)) & (~np.isnan(truth))
        bias = np.nanmean(model[mask] - truth[mask])
        return model - bias

    jb08_orbit = debias(jb08_orbit, acc_orbit)
    msis_orbit = debias(msis_orbit, acc_orbit)
    pod_orbit  = debias(pod_orbit,  acc_orbit)
    edr_orbit  = debias(edr_orbit,  acc_orbit)

    # ------------- TLE interpolation & debias -----------------------
    tle_times64     = np.array(tle_times, dtype='datetime64[ns]')
    orbit_mid_times = orbit_mid_times.astype('datetime64[ns]')
    tle_seconds   = (tle_times64 - tle_times64[0]) / np.timedelta64(1, 's')
    orbit_seconds = (orbit_mid_times - tle_times64[0]) / np.timedelta64(1, 's')
    tle_interp    = np.interp(orbit_seconds, tle_seconds, tle_densities, left=np.nan, right=np.nan)
    tle_orbit     = debias(tle_interp, acc_orbit)

    # ------------- Metrics ------------------------------------------
    def compute_metrics(truth, model):
        mask = (~np.isnan(truth)) & (~np.isnan(model))
        if mask.sum() == 0:
            return {k: np.nan for k in ['N','Bias','MAE','RMSE','r','MAPE(%)','logRMSE','RMS%']}
        t, m = truth[mask], model[mask]
        diff = m - t
        mae  = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        # RMS percentage error
        rms_pct = np.sqrt(np.mean((diff / t) ** 2)) * 100 if np.all(t != 0) else np.nan
        r    = np.corrcoef(t, m)[0,1] if len(t) > 1 else np.nan
        mape = np.mean(np.abs(diff/t))*100
        log_mask = (t > 0) & (m > 0)
        logRMSE = np.sqrt(np.mean((np.log10(m[log_mask]) - np.log10(t[log_mask]))**2)) if log_mask.any() else np.nan
        return {
            'N': mask.sum(),
            'Bias': np.mean(diff),
            'MAE': mae,
            'RMSE': rmse,
            'RMS%': rms_pct,
            'r': r,
            'MAPE(%)': mape,
            'logRMSE': logRMSE
        }

    metrics_table = pd.DataFrame({
        'JB08': compute_metrics(acc_orbit, jb08_orbit),
        'MSIS': compute_metrics(acc_orbit, msis_orbit),
        'POD':  compute_metrics(acc_orbit, pod_orbit),
        'EDR':  compute_metrics(acc_orbit, edr_orbit),
        'TLE':  compute_metrics(acc_orbit, tle_orbit)
    }).T

    # ------------- Plot ---------------------------------------------
    plt.figure(figsize=(10,5))
    plt.plot(orbit_mid_times, acc_orbit,  label='ACC', color='black', marker='o', ms=3)
    plt.plot(orbit_mid_times, jb08_orbit, label='JB08', color='orange', marker='x', ms=3)
    plt.plot(orbit_mid_times, msis_orbit, label='MSIS', color='green', marker='d', ms=3)
    plt.plot(orbit_mid_times, pod_orbit,  label='POD',  color='blue', marker='s', ms=3)
    plt.plot(orbit_mid_times, edr_orbit,  label='EDR',  color='red', marker='v', ms=3)
    plt.plot(orbit_mid_times, tle_orbit,  label='TLE',  color='magenta', marker='^', ms=3)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Debiased Density (kg/m³)')
    plt.legend()
    plt.tight_layout()

    # ------------- Save outputs -------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    start_dt = df['UTC'].min().to_pydatetime()
    end_dt   = df['UTC'].max().to_pydatetime()
    tag = f"{start_dt.strftime('%Y%m%dT%H%M%S')}_{end_dt.strftime('%Y%m%dT%H%M%S')}"
    plot_path    = os.path.join(output_dir, f"density_{tag}.png")
    metrics_path = os.path.join(output_dir, f"metrics_{tag}.csv")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    metrics_table.to_csv(metrics_path)
    print(f"✓ {os.path.basename(storm_csv)} → {plot_path}")
    return acc_orbit, pod_orbit, edr_orbit, tle_orbit, jb08_orbit, msis_orbit, tag

def read_tle_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [(lines[i].strip(), lines[i+1].strip()) for i in range(0, len(lines), 2)]

def parse_mean_motion(l2):
    return float(l2[52:63])

def parse_mean_motion_derivative(l1):
    return float(l1[33:43])

def tle_epoch_to_datetime(line1):
    epoch = line1[18:32]
    year  = 2000 + int(epoch[:2])
    day   = float(epoch[2:])
    dt    = datetime(year, 1, 1) + timedelta(days=day - 1)
    return dt.replace(tzinfo=timezone.utc)

def estimate_density_from_TLEs(path, Cd, A, M):
    tles = read_tle_file(path)
    times, densities = [], []

    for l1, l2 in tles:
        # Epoch time
        t = tle_epoch_to_datetime(l1)
        times.append(t)

        # Mean motion and its derivative
        N     = parse_mean_motion(l2)
        N_dot = parse_mean_motion_derivative(l1)

        # Convert to rad/s and rad/s²
        n     = N     / 86400.0
        n_dot = N_dot / (86400.0**2)

        # Semi-major axis
        a = (mu / n**2)**(1/3)

        # Orbital speed
        v = np.sqrt(mu / a)

        # Density from mean-motion derivative
        rho = (2 * M * n_dot) / (3 * Cd * A * n * v)
        densities.append(rho)

    # Convert lists to NumPy arrays
    times = np.array(times)
    densities = np.array(densities)

    if len(times) > 1:
        centered_times = times[:-1] + (times[1:] - times[:-1]) / 2
        centered_densities = densities[1:]  # density tied to span ending at the current TLE
        return centered_times, centered_densities

    return times, densities

if __name__ == "__main__":
    # ---------- SETTINGS --------------------------------------
    PROCESS_ALL = True           # False → run one storm only
    SINGLE_CSV_FILE = "/Users/charlesc/Documents/GitHub/POD-Density-Inversion/output/PODDensityInversion/Data/StormAnalysis/GRACE-FO/GRACE-FO-A_2024-05-10_density_inversion_withEDR.csv"
    # INPUT_DIR  = "/Users/charlesc/Documents/GitHub/POD-Density-Inversion/output/PODDensityInversion/Data/StormAnalysis/GRACE-FO/"
    INPUT_DIR  = "/Users/charlesc/Documents/GitHub/POD-Density-Inversion/output/PODDensityInversion/Data/StormAnalysis/CHAMP/"
    # OUTPUT_DIR = "/Users/charlesc/Documents/GitHub/POD-Density-Inversion/output/PODDensityInversion/Plots/GRACE-FO-A"
    OUTPUT_DIR = "/Users/charlesc/Documents/GitHub/POD-Density-Inversion/output/PODDensityInversion/Plots/CHAMP"
    # ---------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = (sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".csv"))
                 if PROCESS_ALL else [SINGLE_CSV_FILE])

    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
        sys.exit(0)

    acc_dict, pod_dict, edr_dict, tle_dict, jb08_dict, msis_dict = {}, {}, {}, {}, {}, {}

    for csv_name in csv_files:
        try:
            acc_o, pod_o, edr_o, tle_o, jb08_o, msis_o, tag = analyze_storm_csv(
                os.path.join(INPUT_DIR, csv_name), OUTPUT_DIR
            )
            acc_dict[tag]  = acc_o
            pod_dict[tag]  = pod_o
            edr_dict[tag]  = edr_o
            tle_dict[tag]  = tle_o
            jb08_dict[tag] = jb08_o
            msis_dict[tag] = msis_o
        except Exception as e:
            print(f"[ERROR] {csv_name}: {e}")

    # Skip aggregation + heat-map when only one storm was processed
    if len(csv_files) == 1:
        sys.exit(0)

    # ---------------- Save aggregated data ----------------------
    data_dict = {
        'ACC_by_storm':  acc_dict,
        'POD_by_storm':  pod_dict,
        'EDR_by_storm':  edr_dict,
        'TLE_by_storm':  tle_dict,
        'JB08_by_storm': jb08_dict,
        'MSIS_by_storm': msis_dict
    }
    pkl_path = os.path.join(OUTPUT_DIR, "aggregated_densities.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"✓ Aggregated data saved → {pkl_path}")