"""
AB-HHO 消融对比实验脚本

使用固定的 resource/train.csv 和 resource/test.csv，
对以下变体各运行 N_RUNS 次，统计 Test R²、RMSE、MAE、选中波段数的均值±标准差。

变体：
  HHO        — 原始基线
  ABHHO_I1   — 仅改进一（Tent 初始化 + 非线性逃逸能量）
  ABHHO_I2   — 仅改进二（自适应二值化 + 稀疏修复）
  ABHHO_I3   — 仅改进三（NRMSE 适应度）
  ABHHO      — 三个改进全部启用

用法：
  python scripts/experiments/run_abhho.py
  python scripts/experiments/run_abhho.py --runs 10 --epoch 50
"""

import os
import sys
import time
import argparse
import logging
import tempfile
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import feature_selection  # noqa: F401 触发插件自动发现
import model              # noqa: F401

from core.registry import get_algorithm, get_model
from core.constants import DEFAULT_RANDOM_STATE

# =============================================================================
# 实验配置
# =============================================================================

N_RUNS   = 30
EPOCH    = 200
POP_SIZE = 50

TRAIN_CSV = str(PROJECT_ROOT / "resource" / "train.csv")
TEST_CSV  = str(PROJECT_ROOT / "resource" / "test.csv")
TARGET_COL = "TS"
BAND_START = 14
BAND_END   = 164
MODEL_NAME = "PLS"

# AB-HHO 超参数
ABHHO_PARAMS = dict(
    epoch=EPOCH,
    pop_size=POP_SIZE,
    a=0.9,
    gamma=2.0,
    alpha_min=2.0,
    alpha_max=8.0,
    r_max=0.5,
)

VARIANTS = [
    {"label": "HHO",      "algo": "HHO",      "extra": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"label": "ABHHO_I1", "algo": "ABHHO_I1", "extra": ABHHO_PARAMS},
    {"label": "ABHHO_I2", "algo": "ABHHO_I2", "extra": ABHHO_PARAMS},
    {"label": "ABHHO_I3", "algo": "ABHHO_I3", "extra": ABHHO_PARAMS},
    {"label": "ABHHO",    "algo": "ABHHO",    "extra": ABHHO_PARAMS},
]


# =============================================================================
# 单次运行
# =============================================================================

def run_once(algo_name, algo_params, tmp_dir, logger):
    """
    运行一次特征选择 + PLS 建模管道。
    直接使用固定的 train/test CSV，不重新划分数据。

    Returns
    -------
    dict: train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae,
          n_selected, selected_features, elapsed_sec
    """
    AlgoClass = get_algorithm(algo_name)
    selector = AlgoClass(
        target_col=TARGET_COL,
        band_range=(BAND_START, BAND_END),
        logger=logger,
        **algo_params,
    )

    selection_path = str(tmp_dir / f"selected_{algo_name}.csv")
    t0 = time.time()
    result = selector.run_selection(
        input_path=TRAIN_CSV,
        output_path=selection_path,
    )
    elapsed = time.time() - t0

    selected_feats = result.selected_features
    n_selected = len(selected_feats)

    if n_selected == 0:
        return {
            "train_r2": np.nan, "train_rmse": np.nan, "train_mae": np.nan,
            "test_r2": np.nan,  "test_rmse": np.nan,  "test_mae": np.nan,
            "n_selected": 0, "selected_features": [], "elapsed_sec": elapsed,
        }

    # PLS 建模
    ModelClass = get_model(MODEL_NAME)
    model_instance = ModelClass(
        logger=logger,
        n_trials=200,
        cv_folds=5,
    )
    model_result = model_instance.run_modeling(
        train_path=TRAIN_CSV,
        test_path=TEST_CSV,
        selected_features=selected_feats,
        target_col=TARGET_COL,
        output_dir=str(tmp_dir),
    )

    tm  = model_result["test_metrics"]
    trm = model_result["train_metrics"]
    return {
        "train_r2":          trm["R2"],
        "train_rmse":        trm["RMSE"],
        "train_mae":         trm["MAE"],
        "test_r2":           tm["R2"],
        "test_rmse":         tm["RMSE"],
        "test_mae":          tm["MAE"],
        "n_selected":        n_selected,
        "selected_features": selected_feats,
        "elapsed_sec":       elapsed,
    }


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="AB-HHO 消融对比实验")
    parser.add_argument("--runs",  type=int, default=N_RUNS,  help=f"每个变体运行次数（默认 {N_RUNS}）")
    parser.add_argument("--epoch", type=int, default=EPOCH,   help=f"迭代轮数（默认 {EPOCH}）")
    return parser.parse_args()


def main():
    args = parse_args()
    n_runs = args.runs
    epoch  = args.epoch

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "log" / f"abhho_ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_csv = out_dir / "results.csv"

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    silent_logger = logging.getLogger("abhho_silent")
    silent_logger.setLevel(logging.WARNING)

    print(f"\n{'='*60}")
    print(f"  AB-HHO 消融对比实验")
    print(f"  变体数：{len(VARIANTS)}  每变体运行：{n_runs} 次  Epoch：{epoch}")
    print(f"  Train: {TRAIN_CSV}")
    print(f"  Test:  {TEST_CSV}")
    print(f"  结果目录：{out_dir}")
    print(f"{'='*60}\n")

    all_rows = []

    for v_idx, variant in enumerate(VARIANTS):
        label     = variant["label"]
        algo_name = variant["algo"]
        extra     = dict(variant["extra"])
        extra["epoch"] = epoch  # 允许命令行覆盖 epoch

        print(f"[{v_idx+1}/{len(VARIANTS)}] {label:12s}", end="", flush=True)
        variant_rows = []

        for run_i in range(n_runs):
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"abhho_{label}_run{run_i}_"))
            try:
                metrics = run_once(
                    algo_name=algo_name,
                    algo_params=extra,
                    tmp_dir=tmp_dir,
                    logger=silent_logger,
                )
            except Exception as e:
                print(f"\n  [警告] {label} run {run_i} 失败: {e}")
                metrics = {
                    "train_r2": np.nan, "train_rmse": np.nan, "train_mae": np.nan,
                    "test_r2":  np.nan, "test_rmse":  np.nan, "test_mae":  np.nan,
                    "n_selected": 0, "selected_features": [], "elapsed_sec": 0,
                }
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            row = {
                "variant":           label,
                "run":               run_i,
                "train_r2":          metrics["train_r2"],
                "train_rmse":        metrics["train_rmse"],
                "train_mae":         metrics["train_mae"],
                "test_r2":           metrics["test_r2"],
                "test_rmse":         metrics["test_rmse"],
                "test_mae":          metrics["test_mae"],
                "n_selected":        metrics["n_selected"],
                "selected_features": ";".join(metrics.get("selected_features", [])),
                "elapsed_sec":       metrics["elapsed_sec"],
            }
            variant_rows.append(row)
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(result_csv, index=False)
            print(".", end="", flush=True)

        df_v = pd.DataFrame(variant_rows)
        print(
            f"  Train R²={df_v['train_r2'].mean():.4f}±{df_v['train_r2'].std():.4f}"
            f"  Test R²={df_v['test_r2'].mean():.4f}±{df_v['test_r2'].std():.4f}"
            f"  RMSE={df_v['test_rmse'].mean():.4f}"
            f"  n_feat={df_v['n_selected'].mean():.1f}"
        )

    # -------------------------------------------------------------------------
    # 汇总统计
    # -------------------------------------------------------------------------
    df_all = pd.DataFrame(all_rows)

    summary_rows = []
    for variant in VARIANTS:
        label = variant["label"]
        df_v  = df_all[df_all["variant"] == label]
        summary_rows.append({
            "variant":         label,
            "Train_R2":        f"{df_v['train_r2'].mean():.4f}±{df_v['train_r2'].std():.4f}",
            "Train_RMSE":      f"{df_v['train_rmse'].mean():.4f}±{df_v['train_rmse'].std():.4f}",
            "Train_MAE":       f"{df_v['train_mae'].mean():.4f}±{df_v['train_mae'].std():.4f}",
            "Test_R2":         f"{df_v['test_r2'].mean():.4f}±{df_v['test_r2'].std():.4f}",
            "Test_RMSE":       f"{df_v['test_rmse'].mean():.4f}±{df_v['test_rmse'].std():.4f}",
            "Test_MAE":        f"{df_v['test_mae'].mean():.4f}±{df_v['test_mae'].std():.4f}",
            "n_feat":          f"{df_v['n_selected'].mean():.1f}±{df_v['n_selected'].std():.1f}",
            "time_sec":        f"{df_v['elapsed_sec'].mean():.1f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    # 波段频次
    band_freq_rows = []
    for variant in VARIANTS:
        label = variant["label"]
        df_v  = df_all[df_all["variant"] == label]
        all_bands = []
        for feats_str in df_v["selected_features"].dropna():
            if feats_str:
                all_bands.extend(feats_str.split(";"))
        freq = Counter(all_bands)
        for band, count in sorted(freq.items(), key=lambda x: -x[1]):
            band_freq_rows.append({
                "variant":   label,
                "band":      band,
                "count":     count,
                "frequency": round(count / len(df_v), 3),
            })
    pd.DataFrame(band_freq_rows).to_csv(out_dir / "band_frequency.csv", index=False)

    # 控制台打印汇总
    print(f"\n{'='*90}")
    print("  AB-HHO 消融实验汇总")
    print(f"{'='*90}")
    print(f"{'变体':<12} {'Train R²':<22} {'Test R²':<22} {'Test RMSE':<22} {'n_feat':<14} {'time(s)'}")
    print("-" * 90)
    for row in summary_rows:
        print(f"{row['variant']:<12} {row['Train_R2']:<22} {row['Test_R2']:<22} "
              f"{row['Test_RMSE']:<22} {row['n_feat']:<14} {row['time_sec']}")

    # 高频波段
    df_bf = pd.DataFrame(band_freq_rows)
    print(f"\n{'='*90}")
    print("  各变体高频波段（出现频率 ≥ 50%）")
    print(f"{'='*90}")
    for variant in VARIANTS:
        label  = variant["label"]
        df_vbf = df_bf[(df_bf["variant"] == label) & (df_bf["frequency"] >= 0.5)]
        bands_str = ", ".join(
            f"{r['band']}({r['frequency']*100:.0f}%)"
            for _, r in df_vbf.iterrows()
        ) if not df_vbf.empty else "无"
        print(f"  {label:<12}: {bands_str}")

    print(f"\n详细结果：{result_csv}")
    print(f"汇总结果：{summary_csv}\n")


if __name__ == "__main__":
    main()
