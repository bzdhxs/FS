"""FS_SSC Main Entry Point.

Orchestrates the feature selection and modeling pipeline using plugin-based
algorithm and model discovery. No code changes needed when adding new algorithms.

Usage:
    python main.py                    # Use default config.yaml
    python main.py --config exp.yaml  # Use custom config
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import AppConfig
from core.logging_setup import setup_logger, create_output_structure
from core.registry import get_algorithm, get_model, list_algorithms, list_models
from utils.data_processor import DataProcessor
from visualizer import feature_selection_visualizer as fsv

# Trigger plugin auto-discovery by importing packages
import feature_selection  # noqa: F401
import model  # noqa: F401


def export_config_json(cfg: AppConfig, output_dir: Path, timestamp: str, train_size: int, test_size: int):
    """Export experiment configuration to JSON.

    Args:
        cfg: Application configuration
        output_dir: Output directory
        timestamp: Experiment timestamp
        train_size: Number of training samples
        test_size: Number of test samples
    """
    config_data = {
        "experiment": {
            "id": f"{timestamp}_{cfg.algo_name}_{cfg.model_name}",
            "timestamp": datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
            "algorithm": cfg.algo_name,
            "model": cfg.model_name
        },
        "algorithm_params": cfg.algo_params if cfg.algo_params else {},
        "model_params": cfg.model_params if cfg.model_params else {},
        "data_config": {
            "source_file": cfg.data_file,
            "target_column": cfg.target_col,
            "band_range": list(cfg.band_range),
            "test_size": cfg.test_size,
            "n_train": train_size,
            "n_test": test_size,
            "n_features_total": cfg.band_end - cfg.band_start
        }
    }

    config_file = output_dir / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)


def export_summary_json(output_dir: Path, timestamp: str, cfg: AppConfig, 
                       selection_result, model_result, selection_time: float, modeling_time: float):
    """Export experiment summary to JSON.

    Args:
        output_dir: Output directory
        timestamp: Experiment timestamp
        cfg: Application configuration
        selection_result: Feature selection result
        model_result: Model training result
        selection_time: Feature selection time in seconds
        modeling_time: Model training time in seconds
    """
    summary_data = {
        "experiment_id": f"{timestamp}_{cfg.algo_name}_{cfg.model_name}",
        "feature_selection": {
            "n_selected": len(selection_result.selected_features),
            "selection_time_sec": round(selection_time, 2),
            "selected_indices": selection_result.selected_indices if hasattr(selection_result, 'selected_indices') else []
        },
        "model_training": {
            "training_time_sec": round(modeling_time, 2),
            "best_hyperparams": model_result['best_params']
        },
        "performance": {
            "train": {
                "R2": round(model_result['train_metrics']['R2'], 4),
                "RMSE": round(model_result['train_metrics']['RMSE'], 4),
                "MAE": round(model_result['train_metrics']['MAE'], 4),
                "RPD": round(model_result['train_metrics']['RPD'], 4)
            },
            "test": {
                "R2": round(model_result['test_metrics']['R2'], 4),
                "RMSE": round(model_result['test_metrics']['RMSE'], 4),
                "MAE": round(model_result['test_metrics']['MAE'], 4),
                "RPD": round(model_result['test_metrics']['RPD'], 4)
            }
        }
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="FS_SSC: Feature Selection for Soil Salt Content")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config_path = args.config
    if os.path.exists(config_path):
        cfg = AppConfig.from_yaml(config_path)
    else:
        cfg = AppConfig()

    # Create output directory with structured subdirectories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = cfg.get_output_dir(timestamp)
    create_output_structure(output_dir)

    # Setup logger
    logger = setup_logger(output_dir, module_name="main")
    logger.info(f"Task started: {timestamp}_{cfg.algo_name}_{cfg.model_name}")
    logger.info(f"Available algorithms: {list_algorithms()}")
    logger.info(f"Available models: {list_models()}")

    # Validate algorithm and model exist
    try:
        AlgoClass = get_algorithm(cfg.algo_name)
        ModelClass = get_model(cfg.model_name)
    except KeyError as e:
        logger.error(str(e))
        return

    # =========================================================
    # Step 1: Data Preprocessing
    # =========================================================
    logger.info("[Step 1] Data preprocessing...")
    processor = DataProcessor(logger)

    try:
        train_csv, test_csv = processor.load_and_preprocess(
            original_data_path=str(cfg.original_data_path),
            target_col=cfg.target_col,
            output_dir=str(output_dir),
            test_size=cfg.test_size
        )
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return

    # Export config JSON
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    export_config_json(cfg, output_dir, timestamp, len(train_df), len(test_df))
    logger.info("Configuration exported to config.json")

    # =========================================================
    # Step 2: Feature Engineering
    # =========================================================
    logger.info(f"[Step 2] Feature processing ({cfg.algo_name})...")

    selector = AlgoClass(
        target_col=cfg.target_col,
        band_range=cfg.band_range,
        logger=logger,
        **cfg.algo_params
    )

    # Save selected features to data/ subdirectory
    data_dir = output_dir / 'data'
    selection_path = str(data_dir / f'selected_features_{cfg.algo_name}.csv')
    is_pca_mode = selector.mode == "extraction"

    selection_start = time.time()
    if is_pca_mode:
        logger.info("   Mode: Feature Extraction")
        pca_train = str(data_dir / 'train_pca.csv')
        pca_test = str(data_dir / 'test_pca.csv')

        result = selector.run_selection(
            input_path=train_csv,
            output_path=pca_train,
            test_path=test_csv,
            test_output_path=pca_test,
            **cfg.algo_params
        )
        selected_feats = result.selected_features
        model_train_path = pca_train
        model_test_path = pca_test
    else:
        logger.info("   Mode: Feature Selection")
        result = selector.run_selection(
            input_path=train_csv,
            output_path=selection_path,
            **cfg.algo_params
        )
        selected_feats = result.selected_features
        model_train_path = train_csv
        model_test_path = test_csv
    
    selection_time = time.time() - selection_start
    logger.info(f"Feature processing done | Features: {len(selected_feats)} | Time: {selection_time:.2f}s")

    # =========================================================
    # Step 3: Visualization
    # =========================================================
    logger.info("[Step 3] Visualization...")
    if not is_pca_mode:
        try:
            fsv.plot_selected_features(
                original_data_path=train_csv,
                selected_data_path=selection_path,
                show=cfg.show_plots
            )
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    else:
        logger.info("   [Skip] PCA mode - skipping band visualization.")

    # =========================================================
    # Step 4: Modeling
    # =========================================================
    logger.info(f"[Step 4] Modeling ({cfg.model_name})...")

    model_instance = ModelClass(logger=logger, **cfg.model_params)

    modeling_start = time.time()
    model_result = model_instance.run_modeling(
        train_path=model_train_path,
        test_path=model_test_path,
        selected_features=selected_feats,
        target_col=cfg.target_col,
        output_dir=str(output_dir)
    )
    modeling_time = time.time() - modeling_start

    # =========================================================
    # Step 5: Export Summary & Report
    # =========================================================
    # Export summary JSON
    export_summary_json(output_dir, timestamp, cfg, result, model_result, selection_time, modeling_time)
    logger.info("Summary exported to summary.json")

    logger.info("=" * 40)
    logger.info("FINAL REPORT")
    logger.info("=" * 40)

    train_metrics = model_result['train_metrics']
    test_metrics = model_result['test_metrics']
    logger.info(f"   [Train] R2={train_metrics['R2']:.4f}, RMSE={train_metrics['RMSE']:.4f}")
    logger.info(f"   [Test ] R2={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.4f}")
    logger.info(f"   [Time ] Selection: {selection_time:.2f}s, Modeling: {modeling_time:.2f}s")
    logger.info("=" * 40)
    logger.info("Run Completed.")


if __name__ == "__main__":
    main()
