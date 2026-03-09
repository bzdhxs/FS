"""Global constants shared across algorithms and models.

This module contains only cross-algorithm shared constants that rarely change.
Algorithm-specific parameters (epoch, pop_size, penalty) should be defined
in their respective algorithm files.
"""

# Binary threshold for metaheuristic algorithms
BINARY_THRESHOLD = 0.5

# Fitness penalty value when no features are selected
FITNESS_PENALTY_DEFAULT = 99999.0

# Maximum PLS components for fitness function
MAX_PLS_COMPONENTS = 5

# Internal validation set size for feature selection
INTERNAL_VAL_SIZE = 0.3

# Global random state for reproducibility
DEFAULT_RANDOM_STATE = 42


# Visualization constants
PLOT_DPI = 300
PLOT_FONT = 'Times New Roman'
PLOT_FONT_SIZE = 12
SPECTRUM_PLOT_DPI = 500
SPECTRUM_PLOT_SIZE = (5, 5)

# Wavelength calculation constants
WAVELENGTH_START = 350  # nm
WAVELENGTH_STEP = 4     # nm per band
