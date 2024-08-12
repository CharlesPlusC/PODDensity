import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm
from pandas.tseries import offsets
import orekit
from orekit.pyhelpers import setup_orekit_curdir
# download_orekit_data_curdir("misc")
vm = orekit.initVM()
setup_orekit_curdir("misc/orekit-data.zip")
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.utils import PVCoordinates
from orekit.pyhelpers import datetime_to_absolutedate
from source.tools.utilities import project_acc_into_HCL, pv_to_kep, interpolate_positions, calculate_acceleration
from source.tools.SWIndices import get_kp_ap_dst_f107, read_ae, read_sym, read_imf
from org.orekit.frames import FramesFactory
import os
import numpy as np
from scipy.signal import savgol_filter

def get_arglat_from_df(densitydf_df):
    frame = FramesFactory.getEME2000()
    use_column = 'UTC' in densitydf_df.columns

    for index, row in densitydf_df.iterrows():
        x = row['x']
        y = row['y']
        z = row['z']
        xv = row['xv']
        yv = row['yv']
        zv = row['zv']
        
        utc = row['UTC'] if use_column else index

        position = Vector3D(float(x), float(y), float(z))
        velocity = Vector3D(float(xv), float(yv), float(zv))
        pvCoordinates = PVCoordinates(position, velocity)
        time = datetime_to_absolutedate(utc)
        kep_els = pv_to_kep(pvCoordinates, frame, time)
        arglat = kep_els[3] + kep_els[5]
        densitydf_df.at[index, 'arglat'] = arglat

    return densitydf_df

def plot_densities_and_indices(data_frames, moving_avg_minutes, sat_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.signal import savgol_filter
    from datetime import datetime, timedelta

    sns.set_style(style="whitegrid")

    custom_palette = ["#FF6347", "#3CB371", "#1E90FF"]

    _, kp_3hrly, hourly_dst = get_kp_ap_dst_f107()
    
    kp_3hrly['DateTime'] = pd.to_datetime(kp_3hrly['DateTime']).dt.tz_localize('UTC')
    hourly_dst['DateTime'] = pd.to_datetime(hourly_dst['DateTime']).dt.tz_localize('UTC')

    for density_df in data_frames:
        if 'UTC' in density_df.columns:
            density_df['UTC'] = pd.to_datetime(density_df['UTC'], utc=True)
            density_df.set_index('UTC', inplace=True)
        if density_df.index.tz is None:
            density_df.index = density_df.index.tz_localize('UTC')
        else:
            density_df.index = density_df.index.tz_convert('UTC')
        density_df = density_df[~density_df.index.duplicated(keep='first')]
    
    start_time = min(df.index.min() for df in data_frames)
    end_time = max(df.index.max() for df in data_frames)

    kp_3hrly = kp_3hrly[(kp_3hrly['DateTime'] >= start_time) & (kp_3hrly['DateTime'] <= end_time)]
    kp_3hrly = kp_3hrly.sort_values(by='DateTime')
    hourly_dst = hourly_dst.sort_values(by='DateTime')
    max_kp_time = kp_3hrly.loc[kp_3hrly['Kp'].idxmax(), 'DateTime']
    analysis_start_time = max_kp_time - timedelta(hours=24)
    analysis_end_time = max_kp_time + timedelta(hours=32)

    kp_3hrly_analysis = kp_3hrly[(kp_3hrly['DateTime'] >= analysis_start_time) & (kp_3hrly['DateTime'] <= analysis_end_time)]
    hourly_dst_analysis = hourly_dst[(hourly_dst['DateTime'] >= analysis_start_time) & (hourly_dst['DateTime'] <= analysis_end_time)]

    storm_category = determine_storm_category(kp_3hrly_analysis['Kp'].max())
    storm_number = -int(storm_category[1:]) if storm_category != "Below G1" else 0

    start_date_str = analysis_start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_date_str = analysis_end_time.strftime('%Y-%m-%d %H:%M:%S')
    ae = read_ae(start_date_str, end_date_str)
    sym = read_sym(start_date_str, end_date_str)
    imf = read_imf(start_date_str, end_date_str)

    def process_dataframe(df, time_col):
        if df is not None:
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df = df[(df[time_col] >= analysis_start_time) & (df[time_col] <= analysis_end_time)]
            df = df[~df[time_col].duplicated(keep='first')]
            df.set_index(time_col, inplace=True)
        return df

    ae = process_dataframe(ae, 'Datetime')
    sym = process_dataframe(sym, 'Datetime')
    imf = process_dataframe(imf, 'DateTime')

    # Invert SYM values before further processing
    if sym is not None:
        sym['minute_value'] = -sym['minute_value']

    for i, density_df in enumerate(data_frames):
        seconds_per_point = 15
        window_size = (moving_avg_minutes * 60) // seconds_per_point
        #remove all values above 1e-11 and below -1e-11
        median_density = density_df['Computed Density'].median()
        density_df['Computed Density'] = density_df['Computed Density'].apply(lambda x: 1e-11 if x > 1e-11 else (median_density if x < -1e-11 else x))
        density_df['Computed Density'] = density_df['Computed Density'].rolling(window=window_size, center=True).mean()
        median_density = density_df['Computed Density'].median()
        density_df['Computed Density'] = density_df['Computed Density'].apply(lambda x: median_density if x > 10 * median_density or x < median_density / 10 else x)
        density_df['Computed Density'] = savgol_filter(density_df['Computed Density'], 51, 3)
        density_df = density_df[(density_df.index >= analysis_start_time) & (density_df.index <= analysis_end_time)]

    nrows = 2 + (1 if ae is not None else 0) + (1 if sym is not None else 0) + (1 if imf is not None else 0)
    height_ratios = [3, 1] + ([1] if ae is not None else []) + ([1] if sym is not None else []) + ([1] if imf is not None else [])

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 8 + nrows), gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.4})

    if 'JB08' in data_frames[0]:
        sns.lineplot(ax=axs[0], data=data_frames[0], x=data_frames[0].index, y='JB08', label='JB08 Density', color=custom_palette[0], linewidth=1)
    if 'DTM2000' in data_frames[0]:
        sns.lineplot(ax=axs[0], data=data_frames[0], x=data_frames[0].index, y='DTM2000', label='DTM2000 Density', color=custom_palette[1], linewidth=1)
    if 'NRLMSISE00' in data_frames[0]:
        sns.lineplot(ax=axs[0], data=data_frames[0], x=data_frames[0].index, y='NRLMSISE00', label='NRLMSISE00 Density', color=custom_palette[2], linewidth=1)

    for i, density_df in enumerate(data_frames):
        if 'Computed Density' in density_df:
            sns.lineplot(ax=axs[0], data=density_df, x=density_df.index, y='Computed Density', label='Computed Density', color="xkcd:hot pink", linewidth=0.5)

    day, month, year = analysis_start_time.day, analysis_start_time.month, analysis_start_time.year
    axs[0].set_title(f'Model vs. Estimated: {sat_name} \n{day}-{month}-{year}', fontsize=12)
    axs[0].set_xlabel('Time (UTC)', fontsize=12)
    axs[0].set_ylabel('Density (log scale)', fontsize=12)
    axs[0].legend(loc='upper right', frameon=True)
    axs[0].set_yscale('log')
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].set_xlim(analysis_start_time, analysis_end_time)

    ax_right_top = axs[1]
    ax_kp = ax_right_top.twinx()

    for i in range(len(hourly_dst_analysis) - 1):
        ax_right_top.hlines(hourly_dst_analysis['Value'].iloc[i], hourly_dst_analysis['DateTime'].iloc[i], hourly_dst_analysis['DateTime'].iloc[i + 1], colors='xkcd:violet', linewidth=2)
    ax_right_top.hlines(hourly_dst_analysis['Value'].iloc[-1], hourly_dst_analysis['DateTime'].iloc[-1], analysis_end_time, colors='xkcd:violet', linewidth=2)
    
    for i in range(len(kp_3hrly_analysis) - 1):
        ax_kp.hlines(kp_3hrly_analysis['Kp'].iloc[i], kp_3hrly_analysis['DateTime'].iloc[i], kp_3hrly_analysis['DateTime'].iloc[i + 1], colors='xkcd:hot pink', linewidth=2)
    ax_kp.hlines(kp_3hrly_analysis['Kp'].iloc[-1], kp_3hrly_analysis['DateTime'].iloc[-1], analysis_end_time, colors='xkcd:hot pink', linewidth=2)

    ax_right_top.set_ylabel('Dst (nT)', color='xkcd:violet')
    ax_right_top.yaxis.label.set_color('xkcd:violet')
    ax_right_top.tick_params(axis='y', colors='xkcd:violet')
    ax_right_top.invert_yaxis()

    ax_kp.set_ylabel('Kp', color='xkcd:hot pink')
    ax_kp.yaxis.label.set_color('xkcd:hot pink')
    ax_kp.set_ylim(0, 9)
    ax_kp.tick_params(axis='y', colors='xkcd:hot pink')
    ax_kp.set_yticks(np.arange(0, 10, 3))
    ax_right_top.set_xlim(analysis_start_time, analysis_end_time)

    idx = 2

    if sym is not None:
        sns.lineplot(ax=axs[idx], data=sym, x=sym.index, y='minute_value', label='SYM Index', color='xkcd:violet', linewidth=1, errorbar=None)
        axs[idx].set_xlim(analysis_start_time, analysis_end_time)
        axs[idx].set_title('SYM Index')
        axs[idx].set_xlabel('Time (UTC)')
        axs[idx].set_ylabel('SYM (nT)')
        axs[idx].grid(True, linestyle='-', linewidth=0.5)
        idx += 1

    if ae is not None:
        sns.lineplot(ax=axs[idx], data=ae, x=ae.index, y='minute_value', label='AE Index', color='xkcd:orange', linewidth=1)
        axs[idx].set_xlim(analysis_start_time, analysis_end_time)
        axs[idx].set_title('AE Index')
        axs[idx].set_xlabel('Time (UTC)')
        axs[idx].set_ylabel('AE (nT)')
        axs[idx].grid(True, linestyle='-', linewidth=0.5)
        idx += 1

    if imf is not None:
        sns.lineplot(ax=axs[idx], data=imf, x=imf.index, y='Bz', label='Bz Component', color='xkcd:blue', linewidth=1)
        axs[idx].set_xlim(analysis_start_time, analysis_end_time)
        axs[idx].set_title('IMF Bz Component at L1')
        axs[idx].set_xlabel('Time (UTC)')
        axs[idx].set_ylabel('Bz (nT)')
        axs[idx].grid(True, linestyle='-', linewidth=0.5)

    # Adjust Dst and SYM axes to have the same min and max values
    min_dst_sym = min(ax_right_top.get_ylim()[0], axs[2].get_ylim()[0])
    max_dst_sym = max(ax_right_top.get_ylim()[1], axs[2].get_ylim()[1])
    ax_right_top.set_ylim(min_dst_sym, max_dst_sym)
    axs[2].set_ylim(min_dst_sym, max_dst_sym)

    plt.tight_layout() 
    plt.savefig(f'output/DensityInversion/PODDensityInversion/Plots/{sat_name}/tseries_indices_{day}_{month}_{year}.png', dpi=600)
    plt.close()

def density_compare_scatter(density_df, moving_avg_window, sat_name):
    
    save_path = f'output/DensityInversion/PODDensityInversion/Plots/{sat_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Convert moving average minutes to the number of points based on data frequency
    if 'UTC' in density_df.columns:
        density_df['UTC'] = pd.to_datetime(density_df['UTC'], utc=True)
        density_df.set_index('UTC', inplace=True)
    density_df.index = density_df.index.tz_localize('UTC') if density_df.index.tz is None else density_df.index.tz_convert('UTC')

    # Calculate moving average for the Computed Density
    freq_in_seconds = pd.to_timedelta(pd.infer_freq(density_df.index)).seconds
    window_size = (moving_avg_window * 60) // freq_in_seconds
    
    median_density = density_df['Computed Density'].median()
    IQR = density_df['Computed Density'].quantile(0.75) - density_df['Computed Density'].quantile(0.25)
    lower_bound = median_density - 5 * IQR
    upper_bound = median_density + 5 * IQR
    density_df.loc[:, 'Computed Density'] = density_df['Computed Density'].apply(lambda x: median_density if x < lower_bound or x > upper_bound else x)
    density_df['Computed Density'] = density_df['Computed Density'].rolling(window=window_size, min_periods=1, center=True).mean()
    density_df['Computed Density'] = savgol_filter(density_df['Computed Density'], 51, 3)

    # Model names to compare    
    print(f"columns: {density_df.columns}")
    model_names = ['JB08', 'DTM2000', 'NRLMSISE00']

    for model in model_names:
        plot_data = density_df.dropna(subset=['Computed Density', model])
        plot_data = plot_data[plot_data['Computed Density'] > 0]  # Ensure positive values for log scale
        
        f, ax = plt.subplots(figsize=(6, 6))

        # Draw a combo histogram and scatterplot with density contours
        sns.scatterplot(x=plot_data[model], y=plot_data['Computed Density'], s=5, color=".15", ax=ax)
        sns.histplot(x=plot_data[model], y=plot_data['Computed Density'], bins=50, pthresh=.1, cmap="rocket", cbar=True, ax=ax)
        sns.kdeplot(x=plot_data[model], y=plot_data['Computed Density'], levels=4, color="xkcd:white", linewidths=1, ax=ax)
        #log the x and y 
        ax.set_xscale('log')
        ax.set_yscale('log')
        # #add a line of y=x
        # ax.plot([1e-13, 1e-11], [1e-13, 1e-11], color='black', linestyle='--')
        # #constrain the axes to be between 1e-13 and 1e-11 and of same length
        # ax.set_xlim(1e-13, 3e-12)
        # ax.set_ylim(1e-13, 3e-12)
        ax.set_title(f'Comparison of {model} vs. Computed Density')
        ax.set_xlabel('Model Density')
        ax.set_ylabel('Computed Density')
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        plt.show()
        # plot_filename = f'comparison_{model.replace(" ", "_")}.png'
        # plt.savefig(os.path.join(save_path, plot_filename))
        plt.close()

        # Line plot of density over time for both the model and the computed density
        plt.figure(figsize=(11, 7))
        plt.plot(plot_data.index, plot_data['Computed Density'], label='Computed Density')
        plt.plot(plot_data.index, plot_data[model], label=model)
        plt.title(f'{model} vs. Computed Density Over Time')
        plt.xlabel('Epoch (UTC)')
        plt.ylabel('Density')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()
        # plot_filename = f'comparison_{model.replace(" ", "_")}_time.png'
        # plt.savefig(os.path.join(save_path, plot_filename))
        plt.close()

def determine_storm_category(kp_max):
    if kp_max < 5:
        return "Below G1"
    elif kp_max < 6:
        return "G1"
    elif kp_max < 7:
        return "G2"
    elif kp_max < 8:
        return "G3"
    elif kp_max < 9:
        return "G4"
    else:
        return "G5"

def reldens_sat_megaplot(base_dir, sat_name, moving_avg_minutes):
    storm_analysis_dir = os.path.join(base_dir, sat_name)
    if not os.path.exists(storm_analysis_dir):
        print(f"No data directory found for {sat_name}")
        return
    
    _, kp_3hrly, hourly_dst = get_kp_ap_dst_f107()
    kp_3hrly['DateTime'] = pd.to_datetime(kp_3hrly['DateTime']).dt.tz_localize('UTC')
    hourly_dst['DateTime'] = pd.to_datetime(hourly_dst['DateTime']).dt.tz_localize('UTC')

    storm_data = []
    unique_dates = set()

    for storm_file in sorted(os.listdir(storm_analysis_dir)):
        storm_file_path = os.path.join(storm_analysis_dir, storm_file)
        if os.path.isfile(storm_file_path):
            df = pd.read_csv(storm_file_path)
            df['UTC'] = pd.to_datetime(df['UTC'], utc=True)
            df.set_index('UTC', inplace=True)
            df.index = df.index.tz_convert('UTC')

            start_time = df.index.min()
            if start_time.strftime("%Y-%m-%d") in unique_dates:
                continue
            unique_dates.add(start_time.strftime("%Y-%m-%d"))

            df = get_arglat_from_df(df)

            density_types = ['Computed Density']
            for density_type in density_types:
                if density_type in df.columns:
                    print(f"plotting: {start_time}")
                    df = df[df[density_type] > -1e-11]
                    df = df[df[density_type] < 8e-12]
                    window_size = (moving_avg_minutes * 60) // 30
                    df[density_type] = df[density_type].rolling(window=window_size, min_periods=1, center=True).mean()
                    median_density = df[density_type].median()
                    df[density_type] = df[density_type].apply(lambda x: median_density if x > 10 * median_density or x < median_density / 10 else x)
                    df[density_type] = savgol_filter(df[density_type], 51, 3)

            kp_filtered = kp_3hrly[(kp_3hrly['DateTime'] >= start_time) & (kp_3hrly['DateTime'] <= start_time + datetime.timedelta(days=3))]
            max_kp_time = kp_filtered.loc[kp_filtered['Kp'].idxmax()]['DateTime'] if not kp_filtered.empty else start_time

            storm_category = determine_storm_category(kp_filtered['Kp'].max())
            storm_number = -int(storm_category[1:]) if storm_category != "Below G1" else 0

            # Adjust the plotting times based on the max Kp time
            adjusted_start_time = max_kp_time - datetime.timedelta(hours=12)
            adjusted_end_time = max_kp_time + datetime.timedelta(hours=32)

            storm_data.append((df, adjusted_start_time, adjusted_end_time, storm_category, storm_number))

    storm_data.sort(key=lambda x: x[4], reverse=True)

    num_storms = len(storm_data)
    ncols = 4
    nrows = (num_storms + ncols - 1) // ncols  # This ensures we don't have any extra blank rows
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.2 * ncols, 1 * nrows), dpi=600)
    axes = axes.flatten()

    # Hide unused axes if the number of plots isn't a perfect multiple of nrows * ncols
    for i in range(len(storm_data), len(axes)):
        axes[i].set_visible(False)

    cmap = 'nipy_spectral'

    for i, (df, adjusted_start_time, adjusted_end_time, storm_category, storm_number) in enumerate(storm_data):
        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            first_x, first_y, first_z = df.iloc[0][['x', 'y', 'z']]
            altitude = ((first_x**2 + first_y**2 + first_z**2)**0.5 - 6378137) / 1000
        else:
            altitude = 0  # Default to 0 if x, y, z are not available

        plot_df = df[(df.index >= adjusted_start_time) & (df.index <= adjusted_end_time)]
        
        local_min_density = plot_df['Computed Density'].min()
        local_max_density = plot_df['Computed Density'].max()

        relative_densities = (plot_df['Computed Density'] - local_min_density) / (local_max_density - local_min_density)
        
        sc = axes[i].scatter(plot_df.index, plot_df['arglat'], c=relative_densities, cmap=cmap, alpha=0.7, edgecolor='none', s=5)
        axes[i].set_title(f'{adjusted_start_time.strftime("%Y-%m-%d")}, {storm_category}, {altitude:.0f}km', fontsize=10)
        axes[i].set_ylabel(' ')
        axes[i].set_xlabel(' ')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.subplots_adjust(left=0.055, bottom=0.012, right=0.905, top=0.967, wspace=0.2, hspace=0.288)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalized Computed Density', rotation=270, labelpad=15)
    plt.savefig(f'output/DensityInversion/PODDensityInversion/Plots/{sat_name}_relative_density_subplot_arrays.png', dpi=300, bbox_inches='tight')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def plot_all_storms_scatter(base_dir, sat_name, moving_avg_minutes=45):
    def aggregate_density_data(storm_analysis_dir, moving_avg_minutes):
        aggregated_data = []

        for storm_file in sorted(os.listdir(storm_analysis_dir)):
            storm_file_path = os.path.join(storm_analysis_dir, storm_file)
            if os.path.isfile(storm_file_path):
                df = pd.read_csv(storm_file_path)
                df['UTC'] = pd.to_datetime(df['UTC'], utc=True)
                df.set_index('UTC', inplace=True)
                df.index = df.index.tz_convert('UTC')

                model_columns = ['JB08', 'DTM2000', 'NRLMSISE00']
                if (df[model_columns] < 1e-14).any(axis=None):
                    continue

                density_types = ['Computed Density']
                for density_type in density_types:
                    if density_type in df.columns:
                        window_size = (moving_avg_minutes * 60) // 30
                        df[density_type] = df[density_type].rolling(window=window_size, min_periods=1, center=True).mean()
                        median_density = df[density_type].median()
                        df[density_type] = df[density_type].apply(lambda x: median_density if x > 10 * median_density or x < median_density / 10 else x)
                        df[density_type] = savgol_filter(df[density_type], 51, 3)

                if (df['Computed Density'] < 1e-14).any():
                    continue

                aggregated_data.append(df)

        if aggregated_data: 
            return pd.concat(aggregated_data)
        else:
            return pd.DataFrame()

    storm_analysis_dir = os.path.join(base_dir, sat_name)
    if not os.path.exists(storm_analysis_dir):
        print(f"No data directory found for {sat_name}")
        return

    aggregated_df = aggregate_density_data(storm_analysis_dir, moving_avg_minutes)
    
    if aggregated_df.empty:
        print("No aggregated data available.")
        return

    aggregated_df.index = aggregated_df.index.tz_localize('UTC') if aggregated_df.index.tz is None else aggregated_df.index.tz_convert('UTC')

    freq_in_seconds = 30
    window_size = (moving_avg_minutes * 60) // freq_in_seconds
    
    median_density = aggregated_df['Computed Density'].median()
    IQR = aggregated_df['Computed Density'].quantile(0.75) - aggregated_df['Computed Density'].quantile(0.25)
    lower_bound = median_density - 5 * IQR
    upper_bound = median_density + 5 * IQR
    aggregated_df.loc[:, 'Computed Density'] = aggregated_df['Computed Density'].apply(lambda x: median_density if x < lower_bound or x > upper_bound else x)
    aggregated_df['Computed Density'] = aggregated_df['Computed Density'].rolling(window=window_size, min_periods=1, center=True).mean()
    aggregated_df['Computed Density'] = savgol_filter(aggregated_df['Computed Density'], 51, 3)

    model_names = ['JB08', 'DTM2000', 'NRLMSISE00']
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf

    for model in model_names:
        model_data = aggregated_df.dropna(subset=['Computed Density', model])
        model_data = model_data[model_data['Computed Density'] > 1e-14]
        model_data = model_data[model_data[model] > 1e-14]

        x_min = min(x_min, model_data[model].min())
        x_max = max(x_max, model_data[model].max())
        y_min = min(y_min, model_data['Computed Density'].min())
        y_max = max(y_max, model_data['Computed Density'].max())

    f, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    cmap = "rocket"

    for i, model in enumerate(model_names):
        plot_data = aggregated_df.dropna(subset=['Computed Density', model])
        
        sns.scatterplot(x=plot_data[model], y=plot_data['Computed Density'], s=5, color=".15", ax=axs[i])
        hist = sns.histplot(x=plot_data[model], y=plot_data['Computed Density'], bins=150, pthresh=0.05, cmap=cmap, ax=axs[i], cbar=False)

        # Calculate the histogram data manually
        hist_data, x_edges, y_edges = np.histogram2d(plot_data[model], plot_data['Computed Density'], bins=150)
        norm = plt.Normalize(vmin=0, vmax=hist_data.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = f.colorbar(sm, ax=axs[i])
        cbar.set_label('Number of Points')
        
        sns.kdeplot(x=plot_data[model], y=plot_data['Computed Density'], levels=4, color="xkcd:white", linewidths=1, ax=axs[i], bw_adjust=0.8, thresh=0.2)

        X = plot_data[model].values.reshape(-1, 1)
        y = plot_data['Computed Density'].values
        reg = LinearRegression().fit(X, y)
        r2 = r2_score(y, reg.predict(X))

        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_title(f'{model} (R²={r2:.2f})')
        axs[i].set_xlabel('Model Density')
        axs[i].set_ylabel('Computed Density')
        axs[i].grid(color='black', linestyle='-', linewidth=0.5)

    plt.setp(axs, xlim=(x_min, x_max), ylim=(y_min, y_max))
    plt.suptitle(f'{sat_name}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'output/DensityInversion/PODDensityInversion/Plots/{sat_name}_allstorm_density_scatter_plots.png', dpi=600, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    
    # # Base directory for storm analysis
    base_dir = "output/DensityInversion/PODDensityInversion/Data/StormAnalysis/"
    # # List of satellite names
    sat_names = ["CHAMP"] #"GRACE-FO-A", "TerraSAR-X", "CHAMP"

    for sat_name in sat_names:
        storm_analysis_dir = os.path.join(base_dir, sat_name)
        
    #     # Check if the directory exists before listing files
        if os.path.exists(storm_analysis_dir):
            for storm_file in os.listdir(storm_analysis_dir):
    #             # Form the full path to the storm file
                storm_file_path = os.path.join(storm_analysis_dir, storm_file)
                
                # Check if it's actually a file
                if os.path.isfile(storm_file_path):
                    storm_df = pd.read_csv(storm_file_path) 
    #                 # density_compare_scatter(storm_df, 45, sat_name)
                    plot_densities_and_indices([storm_df], 23, sat_name)

    # base_dir = "output/DensityInversion/PODDensityInversion/Data/StormAnalysis/"
    # sat_names = [ "TerraSAR-X", "GRACE-FO-A"] #
    # for sat_name in sat_names:
        # plot_all_storms_scatter(base_dir, sat_name, moving_avg_minutes=90)
        # reldens_sat_megaplot(base_dir, sat_name, moving_avg_minutes=90)
        