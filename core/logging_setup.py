"""Logging configuration for the application.

Provides structured logging with separate log files for different modules
and organized output directory structure.
"""

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logger(output_dir: Path, module_name: str = "main", name: str = "FS_SSC") -> logging.Logger:
    """Setup logger with file and console handlers.

    Args:
        output_dir: Base output directory
        module_name: Module name for log file (e.g., "main", "feature_selection", "modeling")
        name: Logger name

    Returns:
        Configured logger instance
    """
    # Create logs subdirectory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{name}.{module_name}")
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler - save to logs/{module_name}.log
    log_file = log_dir / f"{module_name}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Formatter (仅用于文件日志)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Console handler — 使用 RichHandler 美化
    console_handler = RichHandler(
        level=logging.INFO,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
        log_time_format='%H:%M:%S',
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def create_output_structure(output_dir: Path):
    """Create organized output directory structure.

    Args:
        output_dir: Base output directory

    Creates:
        output_dir/
        ├── logs/           # Log files
        ├── data/           # Data files (train, test, selected features)
        ├── plots/          # Visualization files
        └── results/        # Result files (predictions, metrics)
    """
    subdirs = ['logs', 'data', 'plots', 'results']
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
