import os
import logging
import matplotlib.pyplot as plt
import numpy as np

from core.constants import PLOT_DPI, PLOT_FONT, SPECTRUM_PLOT_SIZE, SPECTRUM_PLOT_DPI

logger = logging.getLogger(__name__)


def plot_regression_results(y_train, p_train, y_test, p_test,
                            metrics_train, metrics_test, save_path):
    """Plot predicted vs measured scatter plot (SCI style).

    Args:
        y_train: Training true values
        p_train: Training predicted values
        y_test: Test true values
        p_test: Test predicted values
        metrics_train: Training metrics dict
        metrics_test: Test metrics dict
        save_path: Path to save the plot
    """
    # Use style context to avoid global pollution
    try:
        style_name = 'seaborn-v0_8-whitegrid'
        plt.style.library[style_name]  # Check if available
    except KeyError:
        style_name = 'seaborn-whitegrid'

    with plt.style.context(style_name):
        plt.rcParams.update({
            "font.family": 'serif',
            "font.serif": [PLOT_FONT],
            "mathtext.fontset": 'stix',
            "axes.unicode_minus": False,
            "xtick.direction": 'in',
            "ytick.direction": 'in',
        })

        # Create figure
        fig, ax = plt.subplots(figsize=SPECTRUM_PLOT_SIZE, dpi=SPECTRUM_PLOT_DPI)

        # Data preparation
        all_data = np.concatenate([y_train, p_train, y_test, p_test])
        data_min = all_data.min()
        data_max = all_data.max()

        margin = (data_max - data_min) * 0.05
        axis_min = data_min - margin
        axis_max = data_max + margin

        # Plot 1:1 reference line
        ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--',
                linewidth=1.5, alpha=0.6, label="1:1 Line")

        # Scatter plots
        ax.scatter(y_train, p_train, c='#1f77b4', alpha=0.7, s=50,
                   edgecolors='k', linewidth=0.5,
                   label=f"Train ($R^2$={metrics_train['R2']:.3f}, RMSE={metrics_train['RMSE']:.3f})")

        ax.scatter(y_test, p_test, c='#d62728', marker='^', alpha=0.8, s=60,
                   edgecolors='k', linewidth=0.5,
                   label=f"Test ($R^2$={metrics_test['R2']:.3f}, RMSE={metrics_test['RMSE']:.3f})")

        # Axis settings
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel("Measured Value (TS)", fontsize=14, fontweight='bold', fontname=PLOT_FONT)
        ax.set_ylabel("Predicted Value (TS)", fontsize=14, fontweight='bold', fontname=PLOT_FONT)

        ax.tick_params(axis='both', which='major', labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(PLOT_FONT)

        legend = ax.legend(loc='upper left', frameon=True, fontsize=11,
                          framealpha=0.9, edgecolor='gray')
        for text in legend.get_texts():
            text.set_fontname(PLOT_FONT)

        ax.grid(True, linestyle='--', alpha=0.4)

        # Save
        plt.tight_layout()
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"Plot saved: {os.path.basename(save_path)}")