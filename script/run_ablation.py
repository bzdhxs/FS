"""
消融对比实验脚本

对以下 5 个变体各运行 N_RUNS 次，统计 Test R²、RMSE、MAE、选中波段数的均值±标准差。

变体：
  HHO       — 原始基线
  MSHHO_I1  — 仅改进一（Tent+OBL 初始化）
  MSHHO_I2  — 仅改进二（Cauchy 变异扰动）
  MSHHO_I3  — 仅改进三（乘法冗余惩罚）
  MSHHO     — 三个改进全部启用

用法：
  python script/run_ablation.py
  python script/run_ablation.py --runs 10       # 快速验证
  python script/run_ablation.py --epoch 50      # 减少迭代轮数加速
"""

import os
import sys
import time
import json
import argparse
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 触发插件自动发现
import feature_selection  # noqa: F401
import model              # noqa: F401

from core.registry import get_algorithm, get_model
from core.constants import DEFAULT_RANDOM_STATE
from utils.data_processor import DataProcessor


# =============================================================================
# 实验配置
# =============================================================================

N_RUNS = 40          # 每个变体运行次数
EPOCH = 50           # HHO 迭代轮数（收敛只需 ~23 轮，50 轮足够）
POP_SIZE = 300       # HHO 种群大小（大种群提升搜索多样性）
GAMMA = 0.1          # 冗余惩罚系数（降低至 0.1 解决过度稀疏问题）
PENALTY = 0.2        # 稀疏性惩罚系数

# 5 个实验变体
# MSHHO 系列使用独立的 epoch/pop_size（收敛需 ~27 轮，80 轮留余量）
VARIANTS = [
    {"label": "HHO",      "algo": "HHO",      "extra": {}},
    {"label": "MSHHO_I1", "algo": "MSHHO_I1", "extra": {"epoch": 80, "pop_size": 200}},
    {"label": "MSHHO_I2", "algo": "MSHHO_I2", "extra": {"epoch": 80, "pop_size": 200}},
    {"label": "MSHHO_I3", "algo": "MSHHO_I3", "extra": {"epoch": 80, "pop_size": 200, "gamma_redundancy": GAMMA}},
    {"label": "MSHHO",    "algo": "MSHHO",    "extra": {"epoch": 80, "pop_size": 200, "gamma_redundancy": GAMMA}},
]

# 数据配置（与 config.yaml 保持一致）
DATA_FILE   = str(PROJECT_ROOT / "resource" / "dataSet_sg_fd.csv")  # S-G 一阶导数预处理后的数据
TARGET_COL  = "TS"
BAND_START  = 14
BAND_END    = 164
TEST_SIZE   = 0.3
MODEL_NAME  = "PLS"


# =============================================================================
# 工具函数
# =============================================================================

def _build_rf_params_range(n_features):
    """根据选中特征数动态调整 RF 超参数搜索范围，防止小特征集过拟合。"""
    # 特征越少，树的数量和深度上限越保守
    n_est_max  = min(200, max(20, n_features * 10))
    depth_max  = min(10,  max(3,  n_features))
    return {
        "n_estimators":      (10, n_est_max),
        "max_depth":         (2,  depth_max),
        "min_samples_split": (2,  10),
        "min_samples_leaf":  (1,  4),
        "max_features":      ['sqrt', 'log2', 0.3, 0.5, 0.7],
    }


# =============================================================================
# 单次运行
# =============================================================================

def run_once(algo_name, algo_params, random_state, tmp_dir, logger):
    """
    运行一次完整的特征选择 + RF 建模管道。

    Parameters
    ----------
    algo_name : str
        算法注册名
    algo_params : dict
        算法参数
    random_state : int
        随机种子（控制数据划分的随机性）
    tmp_dir : Path
        临时输出目录
    logger : logging.Logger

    Returns
    -------
    dict  包含 test_r2, test_rmse, test_mae, n_selected, elapsed_sec
    """
    # --- 数据预处理（每次用不同 random_state 划分，模拟多次独立实验）---
    processor = DataProcessor(logger)
    train_csv, test_csv = processor.load_and_preprocess(
        original_data_path=DATA_FILE,
        target_col=TARGET_COL,
        output_dir=str(tmp_dir),
        test_size=TEST_SIZE,
        random_state=random_state,
    )

    # --- 特征选择 ---
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
        input_path=train_csv,
        output_path=selection_path,
    )
    elapsed = time.time() - t0

    selected_feats = result.selected_features
    n_selected = len(selected_feats)

    if n_selected == 0:
        # 无特征被选中，返回惩罚值
        return {"train_r2": np.nan, "train_rmse": np.nan, "train_mae": np.nan,
                "test_r2": -999, "test_rmse": 999, "test_mae": 999,
                "n_selected": 0, "selected_features": [], "elapsed_sec": elapsed}

    # --- RF 建模（动态超参数范围，根据选中特征数自适应调整）---
    params_range = _build_rf_params_range(n_selected)
    ModelClass = get_model(MODEL_NAME)
    model_instance = ModelClass(
        logger=logger,
        n_trials=100,       # 消融实验适当减少 trial 数加快速度
        cv_folds=5,
        params_range=params_range,
    )
    model_result = model_instance.run_modeling(
        train_path=train_csv,
        test_path=test_csv,
        selected_features=selected_feats,
        target_col=TARGET_COL,
        output_dir=str(tmp_dir),
    )

    tm = model_result["test_metrics"]
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
    parser = argparse.ArgumentParser(description="MS-HHO 消融对比实验")
    parser.add_argument("--runs",  type=int, default=N_RUNS,  help=f"每个变体运行次数（默认 {N_RUNS}）")
    parser.add_argument("--epoch", type=int, default=EPOCH,   help=f"迭代轮数（默认 {EPOCH}）")
    return parser.parse_args()


def main():
    args = parse_args()
    n_runs = args.runs
    epoch  = args.epoch

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "script" / f"ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 结果 CSV（每次运行后立即追加，支持中断恢复）
    result_csv = out_dir / "ablation_results.csv"

    # 日志
    logging.basicConfig(
        level=logging.WARNING,          # 抑制算法内部日志，只显示进度
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    silent_logger = logging.getLogger("ablation_silent")
    silent_logger.setLevel(logging.WARNING)

    print(f"\n{'='*60}")
    print(f"  MS-HHO 消融对比实验")
    print(f"  变体数：{len(VARIANTS)}  每变体运行：{n_runs} 次  Epoch：{epoch}")
    print(f"  结果目录：{out_dir}")
    print(f"{'='*60}\n")

    all_rows = []

    for v_idx, variant in enumerate(VARIANTS):
        label     = variant["label"]
        algo_name = variant["algo"]
        extra     = variant["extra"]

        # 构造完整算法参数
        algo_params = {
            "epoch":    epoch,
            "pop_size": POP_SIZE,
            "penalty":  PENALTY,
            **extra,
        }

        print(f"[{v_idx+1}/{len(VARIANTS)}] {label:12s}", end="", flush=True)
        variant_rows = []

        for run_i in range(n_runs):
            # 数据划分固定随机种子，30次复用同一 train/test 划分
            # 算法内部（种群初始化、进化）的随机性由 numpy 全局状态自然产生
            random_state = 42

            # 临时目录（每次运行后清理）
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"ablation_{label}_run{run_i}_"))

            try:
                metrics = run_once(
                    algo_name=algo_name,
                    algo_params=algo_params,
                    random_state=random_state,
                    tmp_dir=tmp_dir,
                    logger=silent_logger,
                )
            except Exception as e:
                print(f"\n  [警告] {label} run {run_i} 失败: {e}")
                metrics = {
                    "train_r2": np.nan, "train_rmse": np.nan, "train_mae": np.nan,
                    "test_r2": np.nan, "test_rmse": np.nan, "test_mae": np.nan,
                    "n_selected": 0, "elapsed_sec": 0,
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

            # 立即写入 CSV（支持中断恢复）
            pd.DataFrame(all_rows).to_csv(result_csv, index=False)

            # 进度点
            print(".", end="", flush=True)

        # 本变体统计（同时显示训练集和测试集）
        df_v = pd.DataFrame(variant_rows)
        tr2_mean  = df_v["train_r2"].mean()
        tr2_std   = df_v["train_r2"].std()
        r2_mean   = df_v["test_r2"].mean()
        r2_std    = df_v["test_r2"].std()
        rmse_mean = df_v["test_rmse"].mean()
        n_mean    = df_v["n_selected"].mean()
        print(f"  Train R²={tr2_mean:.4f}±{tr2_std:.4f}  Test R²={r2_mean:.4f}±{r2_std:.4f}  RMSE={rmse_mean:.4f}  n_feat={n_mean:.1f}")

    # ==========================================================================
    # 汇总统计
    # ==========================================================================
    df_all = pd.DataFrame(all_rows)

    summary_rows = []
    for variant in VARIANTS:
        label = variant["label"]
        df_v = df_all[df_all["variant"] == label]
        summary_rows.append({
            "variant":          label,
            "Train_R2_mean":    round(df_v["train_r2"].mean(), 4),
            "Train_R2_std":     round(df_v["train_r2"].std(),  4),
            "Train_RMSE_mean":  round(df_v["train_rmse"].mean(), 4),
            "Train_RMSE_std":   round(df_v["train_rmse"].std(),  4),
            "Train_MAE_mean":   round(df_v["train_mae"].mean(), 4),
            "Train_MAE_std":    round(df_v["train_mae"].std(),  4),
            "Test_R2_mean":     round(df_v["test_r2"].mean(), 4),
            "Test_R2_std":      round(df_v["test_r2"].std(),  4),
            "Test_RMSE_mean":   round(df_v["test_rmse"].mean(), 4),
            "Test_RMSE_std":    round(df_v["test_rmse"].std(),  4),
            "Test_MAE_mean":    round(df_v["test_mae"].mean(), 4),
            "Test_MAE_std":     round(df_v["test_mae"].std(),  4),
            "n_feat_mean":      round(df_v["n_selected"].mean(), 1),
            "n_feat_std":       round(df_v["n_selected"].std(),  1),
            "time_mean_sec":    round(df_v["elapsed_sec"].mean(), 1),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "ablation_summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    # 每个变体的波段频次统计，单独保存一个 CSV
    band_freq_csv = out_dir / "ablation_band_frequency.csv"
    band_freq_rows = []
    for variant in VARIANTS:
        label = variant["label"]
        df_v = df_all[df_all["variant"] == label]
        # 展开所有运行的 selected_features（分号分隔）
        all_bands = []
        for feats_str in df_v["selected_features"].dropna():
            if feats_str:
                all_bands.extend(feats_str.split(";"))
        from collections import Counter
        freq = Counter(all_bands)
        for band, count in sorted(freq.items(), key=lambda x: -x[1]):
            band_freq_rows.append({
                "variant":    label,
                "band":       band,
                "count":      count,
                "frequency":  round(count / len(df_v), 3),  # 出现频率（占总运行次数比例）
            })
    pd.DataFrame(band_freq_rows).to_csv(band_freq_csv, index=False)

    # 控制台打印汇总表
    print(f"\n{'='*80}")
    print("  消融实验汇总（训练集 / 测试集）")
    print(f"{'='*80}")
    col_w = [12, 18, 18, 18, 18, 8]
    header = (
        f"{'变体':<{col_w[0]}}"
        f"{'Train R²(mean±std)':<{col_w[1]}}"
        f"{'Test R²(mean±std)':<{col_w[2]}}"
        f"{'Train RMSE(mean±std)':<{col_w[3]}}"
        f"{'Test RMSE(mean±std)':<{col_w[4]}}"
        f"{'n_feat':<{col_w[5]}}"
    )
    print(header)
    print("-" * sum(col_w))
    for row in summary_rows:
        tr2_str   = f"{row['Train_R2_mean']}±{row['Train_R2_std']}"
        r2_str    = f"{row['Test_R2_mean']}±{row['Test_R2_std']}"
        trmse_str = f"{row['Train_RMSE_mean']}±{row['Train_RMSE_std']}"
        rmse_str  = f"{row['Test_RMSE_mean']}±{row['Test_RMSE_std']}"
        print(
            f"{row['variant']:<{col_w[0]}}"
            f"{tr2_str:<{col_w[1]}}"
            f"{r2_str:<{col_w[2]}}"
            f"{trmse_str:<{col_w[3]}}"
            f"{rmse_str:<{col_w[4]}}"
            f"{row['n_feat_mean']:<{col_w[5]}}"
        )

    # 控制台打印各变体高频波段（出现频率 >= 50%）
    print(f"\n{'='*80}")
    print("  各变体高频波段（出现频率 ≥ 50%）")
    print(f"{'='*80}")
    df_bf = pd.DataFrame(band_freq_rows)
    for variant in VARIANTS:
        label = variant["label"]
        df_v_bf = df_bf[(df_bf["variant"] == label) & (df_bf["frequency"] >= 0.5)]
        bands_str = ", ".join(
            f"{r['band']}({r['count']}次/{r['frequency']*100:.0f}%)"
            for _, r in df_v_bf.iterrows()
        ) if not df_v_bf.empty else "无"
        print(f"  {label:<12}: {bands_str}")

    print(f"\n详细结果：{result_csv}")
    print(f"汇总结果：{summary_csv}")
    print(f"波段频次：{band_freq_csv}\n")


if __name__ == "__main__":
    main()
