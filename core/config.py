"""Configuration management using dataclasses and YAML.

This module provides a type-safe configuration system that loads from YAML files
and supports command-line overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml


@dataclass
class AppConfig:
    """Application configuration with defaults and type hints."""

    # Algorithm and model selection
    algo_name: str = "HHO"
    model_name: str = "RF"

    # Data settings
    resource_dir: str = "resource"
    data_file: str = "dataSet.csv"
    target_col: str = "TS"
    band_start: int = 14
    band_end: int = 164

    # Processing options
    test_size: float = 0.3
    show_plots: bool = False

    # Output settings
    base_log_dir: str = "log"

    # Optional parameter overrides
    algo_params: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            AppConfig instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @property
    def band_range(self) -> Tuple[int, int]:
        """Get band range as tuple."""
        return (self.band_start, self.band_end)

    @property
    def original_data_path(self) -> Path:
        """Get full path to original data file."""
        return Path(self.resource_dir) / self.data_file

    def get_output_dir(self, timestamp: str) -> Path:
        """Get output directory path for a run.

        Args:
            timestamp: Timestamp string (e.g., "20260228_143052")

        Returns:
            Path to output directory
        """
        dir_name = f"{timestamp}_{self.algo_name}_{self.model_name}"
        return Path(self.base_log_dir) / dir_name

    def get_train_path(self, output_dir: Path) -> Path:
        """Get path to training data file."""
        return output_dir / "train_data.csv"

    def get_test_path(self, output_dir: Path) -> Path:
        """Get path to test data file."""
        return output_dir / "test_data.csv"

    def get_selection_path(self, output_dir: Path) -> Path:
        """Get path to selected features file."""
        return output_dir / f"selected_features-{self.algo_name}.csv"

    def validate(self):
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.band_start >= self.band_end:
            raise ValueError(f"band_start ({self.band_start}) must be < band_end ({self.band_end})")

        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")

        if not self.original_data_path.exists():
            raise ValueError(f"Data file not found: {self.original_data_path}")
