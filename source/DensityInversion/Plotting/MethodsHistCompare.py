"""
plot_orbit_rms_histograms.py

Creates a 5×2 grid of histograms: each row is one method (JB08, MSIS, POD, EDR, TLE),
each column is one satellite (GRACE‐FO‐A, CHAMP). Histograms show per‐orbit RMS% error:
    RMS% = |(model_density – ACC_density) / ACC_density| * 100
computed directly from the “ACC” and per‐orbit model arrays in aggregated_densities.pkl.
"""

import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1.  Paths and methods
# -------------------------------------------------------------------
ROOT = pathlib.Path(
    "/Users/charlesc/Documents/GitHub/POD-Density-Inversion/output/PODDensityInversion/Plots"
)
SAT_FOLDERS = {
    "GRACE-FO-A": ROOT / "GRACE-FO-A",
    "CHAMP":      ROOT / "CHAMP",
}
METHODS = ["JB08", "MSIS", "POD", "EDR", "TLE"]       # plotting order
COLOURS  = {
    "JB08": "orange",
    "MSIS": "forestgreen",
    "POD":  "royalblue",
    "EDR":  "crimson",
    "TLE":  "magenta"
}

# -------------------------------------------------------------------
# 2.  Collect per‐orbit RMS% error for each method and satellite
# -------------------------------------------------------------------
def load_orbit_rms(folder):
    """
    Reads aggregated_densities.pkl in `folder`, expects keys:
      - 'ACC': np.ndarray of ACC densities per orbit
      - '{method}': np.ndarray of model densities per orbit for each method
    Returns a dict mapping each method → 1D array of per‐orbit RMS% errors.
    """
    pkl_path = folder / "aggregated_densities.pkl"
    data = pickle.load(open(pkl_path, "rb"))

    acc_dict = data.get("ACC_by_storm", {})
    out = {m: [] for m in METHODS}

    for m in METHODS:
        method_dict = data.get(f"{m}_by_storm", {})
        for storm_id, acc_arr in acc_dict.items():
            if acc_arr is None or len(acc_arr) == 0:
                continue
            model_arr = method_dict.get(storm_id)
            if model_arr is None or len(model_arr) != len(acc_arr):
                continue
            mask = (acc_arr != 0) & np.isfinite(acc_arr) & np.isfinite(model_arr)
            if not np.any(mask):
                continue
            rms_vals = np.abs((model_arr[mask] - acc_arr[mask]) / acc_arr[mask]) * 100
            out[m].extend(rms_vals.tolist())
        # convert list to numpy array (or empty if no values)
        out[m] = np.array(out[m]) if out[m] else np.array([])

    return out

# Load RMS% per orbit for each satellite
rms = {sat: load_orbit_rms(path) for sat, path in SAT_FOLDERS.items()}

# -------------------------------------------------------------------
# 3.  Build a global bin set so all histograms share the same edges
# -------------------------------------------------------------------
# Concatenate all per‐orbit RMS values across methods and satellites
all_values = np.concatenate([
    np.concatenate(list(d.values())) for d in rms.values()
])
# Filter out non‐finite
valid_values = all_values[np.isfinite(all_values)]
if valid_values.size == 0:
    raise RuntimeError("No per‐orbit RMS values found to plot histograms.")
# Use 99.9th percentile as upper bound
max_bin = np.nanpercentile(valid_values, 99.9)
bin_width = 5  # percentage‐point bins
bins = np.arange(0, max_bin + bin_width, bin_width)

# -------------------------------------------------------------------
# 4.  Plot  (rows = methods, cols = satellites) ----------------------
# -------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=len(METHODS), ncols=len(SAT_FOLDERS),
    figsize=(7, 1.3 * len(METHODS)),
    sharex=True, sharey=True, tight_layout=True,
    gridspec_kw={'hspace': 0.08, 'wspace': 0.08}
)

sat_names = list(SAT_FOLDERS.keys())

for row_idx, method in enumerate(METHODS):
    for col_idx, sat in enumerate(sat_names):
        ax = axes[row_idx, col_idx] if axes.ndim == 2 else axes[row_idx]
        vals = rms[sat][method]

        ax.hist(
            vals,
            bins=bins,
            color=COLOURS[method],
            edgecolor="black",
            alpha=0.6
        )

        # print median RMS and sample count inside the subplot
        if vals.size > 0:
            median_val = np.nanmedian(vals)
            count = len(vals)
            ax.text(
                0.97, 0.95,
                # f"median = {median_val:.1f}% \n mean = {np.nanmean(vals):.1f}% \nn = {count}",
                f"n = {count}",
                ha="right", va="top",
                transform=ax.transAxes,
                fontsize=11, weight="bold"
            )

        # aesthetics
        if row_idx == 0:
            ax.set_title(sat, fontsize=11, fontweight='bold', pad=1)
        if col_idx == 0 or col_idx == 1:
            # Add method name as large transparent background text
            ax.text(
                0.5, 0.5, method,
                transform=ax.transAxes,
                fontsize=35,
                color=COLOURS[method],
                alpha=0.3,
                ha='center',
                va='center',
                rotation=0,
            )
        ax.grid(True, linestyle=":", alpha=0.4)

        #force x axis to go 0-100
        # ax.set_xlim(0, max_bin)
        ax.set_xlim(0, 100)

# Label x‐axes
axes[-1, 0].set_xlabel("RMS% Error")
axes[-1, 1].set_xlabel("RMS% Error")

plt.savefig(
    ROOT / "rms_histograms_orbitwise.png",
    dpi=300, bbox_inches="tight"
)
# plt.show()