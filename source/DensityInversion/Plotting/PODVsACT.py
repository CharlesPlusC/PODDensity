import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_mape_histograms_from_csv(mape_subset_csv_path, output_plot_path):
    # Load the MAPE subsets
    mape_subsets = pd.read_csv(mape_subset_csv_path)

    # Define models and colors for the histograms
    models = [
        ("MAPE Computed", '#1E90FF', "POD-Accelerometry"),
        ("MAPE JB08", '#FF1493', "JB08"),
        ("MAPE DTM2000", '#9932CC', "DTM2000"),
        ("MAPE NRLMSISE-00", '#FFD700', "NRLMSISE-00"),
        ("MAPE EDR", '#32CD32', "EDR"),
    ]

    # Create a figure with subplots
    fig, axes = plt.subplots(len(models), 1, figsize=(8, 10), sharex=True)
    bins = np.arange(0, 150, 10)

    for ax, (column, color, label) in zip(axes, models):
        # Filter out NaN values
        values = mape_subsets[column].dropna()

        # Calculate statistics
        median_value = np.median(values)
        std_dev = np.std(values)
        total_count = len(values)

        # Plot histogram
        ax.hist(values, bins=bins, color=color, alpha=0.6, edgecolor="black")
        ax.set_title(label, loc='left', fontsize=12, fontweight='bold')
        ax.set_ylabel("Frequency")

        # Display statistics as text
        ax.text(
            0.95, 0.85,
            f"n = {total_count}\nMedian: {median_value:.2f}\nStd Dev: {std_dev:.2f}",
            transform=ax.transAxes, ha="right", va="center", fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        # Set y-axis limits
        ax.set_ylim(0, 15)
        ax.set_xlim(0, 150)

    # Set x-axis label on the last plot
    axes[-1].set_xlabel("MAPE (%)", fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_plot_path, dpi=600)
    plt.show()


# Path to the saved MAPE subsets and output plot
mape_subset_csv_path = "output/PaperFigures/MAPE_histograms/MAPE_Subsets_histogram.csv"
output_plot_path = "output/PaperFigures/MAPE_histograms/Storms_MAPE_histogram.png"

# Plot the histograms
plot_mape_histograms_from_csv(mape_subset_csv_path, output_plot_path)