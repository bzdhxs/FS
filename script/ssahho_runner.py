"""SSAHHO 实验共享工具函数。

run_ssahho_comparison.py、run_ssahho_ablation.py、run_algo_comparison.py 共用此模块。
提供：
- run_single()：单次运行（特征选择 + 指定模型建模）
- append_result()：立即追加结果到 CSV（支持中断恢复）
- summarize_results()：生成 summary.csv 和 band_frequency.csv
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.registry import get_algorithm, get_model
from utils.data_processor import DataProcessor


# ── 常量 ─────────────────────────────────────────────────
TARGET_COL = "TS"
BAND_START = 14
BAND_END   = 164
TEST_SIZE  = 0.3
MODEL_NAME = "PLS"
CV_SEED    = 42   # 内层 KFold 固定种子


def run_single(
    variant_label: str,
    algo_name: str,
    algo_seed: int,
    split_seed: int,
    data_path: str,
    output_dir: Path,
    epoch: int,
    pop_size: int,
    penalty: float = 0.1,
    gamma: float   = 0.1,
    n_top: int     = 30,   # 每种变换保留的 Top N 特征，候选池 = 4 * n_top
    model_name: str = "PLS",  # 回归模型名称，支持 PLS / SVM / RF
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """单次运行：特征选择 + 指定模型建模。

    Args:
        variant_label: 变体标签，如 "SSAHHO" / "HHO"
        algo_name:     注册算法名，如 "SSAHHO" / "SSAHHO_HHO"
        algo_seed:     算法随机种子（控制种群初始化和进化随机性）
        split_seed:    数据划分随机种子
        data_path:     原始数据 CSV 路径
        output_dir:    本次运行的详细文件输出目录
        epoch:         迭代轮数
        pop_size:      种群大小
        penalty:       稀疏性惩罚系数
        gamma:         冗余惩罚系数
        model_name:    回归模型名称（PLS / SVM / RF）
        logger:        日志器

    Returns:
        dict，包含：
            variant, algo_seed, split_seed, cv_seed, model_name,
            train_r2, train_rmse, test_r2, test_rmse,
            n_selected, selected_feature_names,
            elapsed_sec, status, error_msg
    """
    if logger is None:
        logger = logging.getLogger("ssahho_runner")

    # 文件名前缀（含 variant + split_seed + algo_seed，避免不同变体重复 seed 冲突）
    # 替换 / 为 _ 防止 variant_label 含路径分隔符时 tempfile 报错
    safe_label  = variant_label.replace("/", "_")
    file_prefix = f"{safe_label}_split{split_seed}_seed{algo_seed}"

    base_result = {
        "variant":              variant_label,
        "algo_seed":            algo_seed,
        "split_seed":           split_seed,
        "cv_seed":              CV_SEED,
        "model_name":           model_name,
        "train_r2":             np.nan,
        "train_rmse":           np.nan,
        "test_r2":              np.nan,
        "test_rmse":            np.nan,
        "n_selected":           0,
        "selected_feature_names": "",
        "elapsed_sec":          0.0,
        "status":               "fail",
        "error_msg":            "",
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"ssahho_{file_prefix}_"))
    t0 = time.time()

    try:
        # 1. 数据划分（固定 split_seed）
        processor = DataProcessor(logger)
        train_csv, test_csv = processor.load_and_preprocess(
            original_data_path=data_path,
            target_col=TARGET_COL,
            output_dir=str(tmp_dir),
            test_size=TEST_SIZE,
            random_state=split_seed,
        )

        # 2. 特征选择
        AlgoClass = get_algorithm(algo_name)
        selector  = AlgoClass(
            target_col=TARGET_COL,
            band_range=(BAND_START, BAND_END),
            logger=logger,
            epoch=epoch,
            pop_size=pop_size,
            penalty=penalty,
            gamma=gamma,
            n_top=n_top,
        )

        selection_path = str(tmp_dir / f"selected_{algo_name}.csv")
        result = selector.run_selection(
            input_path=train_csv,
            output_path=selection_path,
            algo_seed=algo_seed,
            cv_seed=CV_SEED,
            save_candidate_summary=False,  # 实验阶段不重复保存
        )
        elapsed = time.time() - t0

        selected_feats = result.selected_features
        n_selected     = len(selected_feats)

        if n_selected == 0:
            base_result.update({
                "elapsed_sec": elapsed,
                "status":      "fail",
                "error_msg":   "no features selected",
            })
            return base_result

        # 3. 回归建模（支持 PLS / SVM / RF）
        ModelClass    = get_model(model_name)
        model_instance = ModelClass(
            logger=logger,
            n_trials=100,
            cv_folds=5,
        )
        model_result = model_instance.run_modeling(
            train_path=train_csv,
            test_path=test_csv,
            selected_features=selected_feats,
            target_col=TARGET_COL,
            output_dir=str(tmp_dir),
        )

        tm  = model_result["test_metrics"]
        trm = model_result["train_metrics"]

        # 4. 复制详细文件到 output_dir
        _copy_detail_files(tmp_dir, output_dir, file_prefix)

        base_result.update({
            "train_r2":              trm["R2"],
            "train_rmse":            trm["RMSE"],
            "test_r2":               tm["R2"],
            "test_rmse":             tm["RMSE"],
            "n_selected":            n_selected,
            "selected_feature_names": ",".join(selected_feats),
            "elapsed_sec":           elapsed,
            "status":                "success",
            "error_msg":             "",
        })

    except Exception as e:
        base_result.update({
            "elapsed_sec": time.time() - t0,
            "status":      "fail",
            "error_msg":   str(e),
        })
        logger.warning(f"run_single 失败 [{file_prefix}]: {e}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return base_result


def _copy_detail_files(tmp_dir: Path, output_dir: Path, file_prefix: str) -> None:
    """将单次运行产生的详细文件复制到实验输出目录。

    文件名格式：run_{file_prefix}.csv
    """
    detail_files = {
        "fitness_history.csv":         "fitness_history",
        "selected_count_history.csv":  "selected_count_history",
        "selected_features_detail.csv": "selected_features_detail",
        "rabbit_summary.csv":          "rabbit_summary",
    }
    for src_name, sub_dir in detail_files.items():
        src = tmp_dir / src_name
        if src.exists():
            dst_dir = output_dir / sub_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"run_{file_prefix}.csv"
            shutil.copy2(src, dst)


def append_result(results_csv: Path, row: Dict[str, Any]) -> None:
    """每次运行后立即追加一行到 results.csv，支持中断恢复。"""
    df_new = pd.DataFrame([row])
    if results_csv.exists():
        df_new.to_csv(results_csv, mode="a", header=False, index=False)
    else:
        df_new.to_csv(results_csv, index=False)


def summarize_results(
    results_csv: Path,
    summary_csv: Path,
    freq_csv: Path,
    variants: List[Dict],
) -> None:
    """读取 results_csv，生成 summary.csv 和 band_frequency.csv。

    band_frequency.csv 的 key = (transform, band_name)，
    但 SSAHHO 的 selected_feature_names 只存 band_name（简版），
    因此频次统计按 band_name 做，transform 信息在 selected_features_detail/ 中。
    """
    df_all = pd.read_csv(results_csv)

    # ── summary.csv ──────────────────────────────────────
    summary_rows = []
    for v in variants:
        label = v["label"]
        df_v  = df_all[(df_all["variant"] == label) & (df_all["status"] == "success")]
        if df_v.empty:
            continue
        summary_rows.append({
            "variant":       label,
            "n_runs":        len(df_v),
            "Train_R2_mean": round(df_v["train_r2"].mean(), 4),
            "Train_R2_std":  round(df_v["train_r2"].std(),  4),
            "Train_RMSE_mean": round(df_v["train_rmse"].mean(), 4),
            "Train_RMSE_std":  round(df_v["train_rmse"].std(),  4),
            "Test_R2_mean":  round(df_v["test_r2"].mean(), 4),
            "Test_R2_std":   round(df_v["test_r2"].std(),  4),
            "Test_RMSE_mean": round(df_v["test_rmse"].mean(), 4),
            "Test_RMSE_std":  round(df_v["test_rmse"].std(),  4),
            "n_feat_mean":   round(df_v["n_selected"].mean(), 1),
            "n_feat_std":    round(df_v["n_selected"].std(),  1),
            "time_mean_sec": round(df_v["elapsed_sec"].mean(), 1),
            "fail_count":    int((df_all["variant"] == label).sum() - len(df_v)),
        })
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    # ── band_frequency.csv ───────────────────────────────
    freq_rows = []
    for v in variants:
        label = v["label"]
        df_v  = df_all[(df_all["variant"] == label) & (df_all["status"] == "success")]
        n_runs = len(df_v)
        if n_runs == 0:
            continue
        all_bands: List[str] = []
        for feats_str in df_v["selected_feature_names"].dropna():
            if feats_str:
                all_bands.extend(feats_str.split(","))
        freq = Counter(all_bands)
        for band, count in sorted(freq.items(), key=lambda x: -x[1]):
            freq_rows.append({
                "variant":   label,
                "band_name": band,
                "count":     count,
                "frequency": round(count / n_runs, 3),
            })
    pd.DataFrame(freq_rows).to_csv(freq_csv, index=False)
