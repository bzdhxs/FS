import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import logging

from core.constants import PLOT_DPI, PLOT_FONT

logger = logging.getLogger(__name__)


def get_wavelength_geometry(band_name):
    """Calculate physical wavelength from band name.

    Formula: 350 + (index - 1) * 4
    """
    try:
        index = int(re.findall(r'\d+', band_name)[0])
        return 350 + (index - 1) * 4
    except Exception:
        return 0


def plot_selected_features(original_data_path, selected_data_path, show=True):
    """Plot spectrum with selected features marked.

    Args:
        original_data_path: Path to original data (for background spectrum)
        selected_data_path: Path to selected features (to mark selected points)
        show: Whether to display plot window
    """
    # Check files
    if not os.path.exists(original_data_path) or not os.path.exists(selected_data_path):
        logger.error("File not found")
        return

    # Extract algorithm name from filename
    filename = os.path.basename(selected_data_path)
    try:
        algo_name = os.path.splitext(filename)[0].split('-')[-1]
    except Exception:
        algo_name = "Selection"

    plot_title = f"{algo_name} Feature Selection Result"

    # Load data
    df_orig = pd.read_csv(original_data_path)
    df_sel = pd.read_csv(selected_data_path)

    # Extract band columns
    all_bands = [c for c in df_orig.columns if c.startswith('b')]
    selected_bands = [c for c in df_sel.columns if c.startswith('b')]
    num_selected = len(selected_bands)

    # Calculate mean spectrum and normalize
    mean_spectrum_all = df_orig[all_bands].mean(axis=0) / 10000.0

    # Prepare coordinates
    x_all_loc = [get_wavelength_geometry(b) for b in all_bands]
    y_all_val = mean_spectrum_all.values

    x_sel_loc = [get_wavelength_geometry(b) for b in selected_bands]
    y_sel_val = mean_spectrum_all[selected_bands].values

    # Set plot style locally (avoid global pollution)
    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.rcParams.update({
            "font.family": 'serif',
            "font.serif": [PLOT_FONT],
            "mathtext.fontset": 'stix',
            "axes.unicode_minus": False
        })

        plt.figure(figsize=(12, 6))

        # Plot background spectrum
        plt.plot(x_all_loc, y_all_val, color='#1f77b4', linewidth=1.5, alpha=0.6, label='Mean Spectrum')

        # Plot selected features
        label_text = f'Selected Features (Count: {num_selected})'
        plt.scatter(x_sel_loc, y_sel_val, color='red', s=20, zorder=5, label=label_text)

        # X-axis settings
        plt.xticks(
            ticks=x_sel_loc,
            labels=selected_bands,
            rotation=45,
            ha='right',
            fontsize=10,
            fontname=PLOT_FONT
        )
        plt.xlabel('Band', fontsize=14, fontweight='bold', fontname=PLOT_FONT)

        if x_all_loc:
            plt.xlim(min(x_all_loc) - 10, max(x_all_loc) + 10)

        # Y-axis and title
        plt.yticks(rotation=0, fontsize=10, fontname=PLOT_FONT)
        plt.ylabel('Reflectance', fontsize=14, fontweight='bold', fontname=PLOT_FONT)
        plt.title(plot_title, fontsize=16, fontweight='bold', pad=15, fontname=PLOT_FONT)

        # Legend
        legend = plt.legend(loc='upper left', frameon=True, fontsize=11)
        for text in legend.get_texts():
            text.set_fontname(PLOT_FONT)

        plt.grid(True, linestyle='--', alpha=0.5)

        # Save
        save_dir = os.path.dirname(selected_data_path)
        output_filename = os.path.join(save_dir, f"{algo_name}_selection_plot.png")

        plt.tight_layout()
        plt.savefig(output_filename, dpi=PLOT_DPI)
        logger.info(f"Plot saved: {output_filename}")

        if show:
            plt.show()
        plt.close()


# 测试入口
if __name__ == "__main__":
    print("Testing visualizer module...")