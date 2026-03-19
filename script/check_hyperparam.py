"""
超参数敏感性分析脚本

对 HHO 和 MSHHO 的关键超参数做单变量扫描，
每组参数运行 N_RUNS_PER_POINT 次取均值，绘制双 Y 轴敏感性折线图。

用法：
  python script/check_hyperparam.py
  python script/check_hyperparam.py --runs 2 --epoch 10   # 快速冒烟测试
"""

import os
import sys
import argparse
import importlib.util
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import feature_selection  # noqa: F401
import model              # noqa: F401

# 动态加载 run_ablation.py 复用 run_once()
_spec = importlib.util.spec_from_file_location(
    "run_ablation",
    str(PROJECT_ROOT / "script" / "run_ablation.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_once = _mod.run_once

# =============================================================================
# 实验配置
# =============================================================================

N_RUNS_PER_POINT = 10   # 每个参数值运行次数
BASE_EPOCH = 200         # 基准迭代轮数（可通过命令行覆盖）

# 基准参数（固定其他参数时使用）
BASE_PARAMS = {
    "HHO":   {"epoch": BASE_EPOCH, "pop_size": 250, "penalty": 0.2},
    "MSHHO": {"epoch": BASE_EPOCH, "pop_size": 50,  "penalty": 0.2, "gamma_redundancy": 0.3},
}

# 单变量扫描网格
PARAM_GRIDS = {
    "HHO": [
        {"param": "pop_size", "values": [20, 50, 100, 250]},
        {"param": "epoch",    "values": [100, 200, 300]},
        {"param": "penalty",  "values": [0.1, 0.2, 0.3, 0.5]},
    ],
    "MSHHO": [
        {"param": "pop_size",         "values": [20, 50, 100]},
        {"param": "epoch",            "values": [100, 200, 300]},
        {"param": "penalty",          "values": [0.1, 0.2, 0.3, 0.5]},
        {"param": "gamma_redundancy", "values": [0.1, 0.15, 0.3, 0.5]},
    ],
}

# 参数中文标签
PARAM_LABELS = {
    "pop_size":         "种群大小",
    "epoch":            "迭代轮数",
    "penalty":          "稀疏惩罚系数 β",
    "gamma_redundancy": "冗余惩罚系数 γ",
}

# matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 核心函数
# =============================================================================

def run_param_sweep(algo_name, param_name, param_values, base_params, n_runs, logger):
    """
    固定其他参数，对 param_name 做单变量扫描。

    Returns
    -------
    pd.DataFrame
        列：param_value, r2_mean, r2_std, rmse_mean, rmse_std, n_feat_mean, n_feat_std
    """
    rows = []
    for val in param_values:
        # 构造本次参数（覆盖基准中的 param_name）
        params = {**base_params, param_name: val}

        r2_list     = []
        rmse_list   = []
        n_feat_list = []

        for run_i in range(n_runs):
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"hp_{algo_name}_{param_name}{val}_r{run_i}_"))
            try:
                metrics = run_once(
                    algo_name=algo_name,
                    algo_params=params,
                    random_state=42,
                    tmp_dir=tmp_dir,
                    logger=logger,
                )
                r2_list.append(metrics["test_r2"])
                rmse_list.append(metrics["test_rmse"])
                n_feat_list.append(metrics["n_selected"])
            except Exception as e:
                logger.warning(f"run failed ({algo_name} {param_name}={val} run{run_i}): {e}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        rows.append({
            "param_value":  val,
            "r2_mean":      round(float(np.nanmean(r2_list)),     4) if r2_list else np.nan,
            "r2_std":       round(float(np.nanstd(r2_list)),      4) if r2_list else np.nan,
            "rmse_mean":    round(float(np.nanmean(rmse_list)),   4) if rmse_list else np.nan,
            "rmse_std":     round(float(np.nanstd(rmse_list)),    4) if rmse_list else np.nan,
            "n_feat_mean":  round(float(np.nanmean(n_feat_list)), 2) if n_feat_list else np.nan,
            "n_feat_std":   round(float(np.nanstd(n_feat_list)),  2) if n_feat_list else np.nan,
            "n_valid_runs": len(r2_list),
        })

    return pd.DataFrame(rows)


def plot_sensitivity(sweep_results, algo_name, out_path):
    """
    绘制超参数敏感性图（每个参数一个子图，双 Y 轴）。

    Parameters
    ----------
    sweep_results : dict
        {param_name: DataFrame}
    algo_name : str
    out_path : Path
    """
    n_params = len(sweep_results)
    fig, axes = plt.subplots(1, n_params, figsize=(5.5 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    for ax, (param_name, df) in zip(axes, sweep_results.items()):
        x = df["param_value"].values
        x_str = [str(v) for v in x]  # x 轴标签（支持非均匀间距）
        x_pos = np.arange(len(x))

        # 左轴：Test R²
        color_r2 = '#2196F3'
        ax.errorbar(x_pos, df["r2_mean"], yerr=df["r2_std"],
                    marker='o', color=color_r2, linewidth=1.8,
                    capsize=4, label='Test R²')
        ax.set_ylabel('Test R²', color=color_r2, fontsize=11)
        ax.tick_params(axis='y', labelcolor=color_r2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_str, fontsize=10)
        ax.set_xlabel(PARAM_LABELS.get(param_name, param_name), fontsize=11)

        # 右轴：n_feat
        color_nf = '#FF9800'
        ax2 = ax.twinx()
        ax2.errorbar(x_pos, df["n_feat_mean"], yerr=df["n_feat_std"],
                     marker='s', color=color_nf, linewidth=1.8,
                     linestyle='--', capsize=4, label='选中波段数')
        ax2.set_ylabel('选中波段数', color=color_nf, fontsize=11)
        ax2.tick_params(axis='y', labelcolor=color_nf)

        # 标注最优点（R² 最高）
        best_idx = int(np.nanargmax(df["r2_mean"].values))
        ax.axvline(x=best_idx, color='gray', linestyle=':', alpha=0.6)
        ax.annotate(
            f'最优\n{x_str[best_idx]}',
            xy=(best_idx, df["r2_mean"].iloc[best_idx]),
            xytext=(best_idx + 0.3, df["r2_mean"].iloc[best_idx]),
            fontsize=8, color='gray',
        )

        ax.set_title(f'{PARAM_LABELS.get(param_name, param_name)}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.35)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower right')

    fig.suptitle(f'{algo_name} 超参数敏感性分析', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  敏感性图已保存：{out_path}")


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MS-HHO 超参数敏感性分析")
    parser.add_argument("--runs",  type=int, default=N_RUNS_PER_POINT,
                        help=f"每个参数值运行次数（默认 {N_RUNS_PER_POINT}）")
    parser.add_argument("--epoch", type=int, default=BASE_EPOCH,
                        help=f"基准迭代轮数（默认 {BASE_EPOCH}）")
    return parser.parse_args()


def main():
    args = parse_args()
    n_runs = args.runs
    epoch  = args.epoch

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "script" / f"hyperparam_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 静默日志
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    silent_logger = logging.getLogger("hyperparam_silent")
    silent_logger.setLevel(logging.WARNING)

    # 统计总运行次数
    total_points = sum(
        sum(len(g["values"]) for g in grids)
        for grids in PARAM_GRIDS.values()
    )
    print(f"\n{'='*60}")
    print(f"  超参数敏感性分析")
    print(f"  参数点数：{total_points}  每点运行：{n_runs} 次  基准 Epoch：{epoch}")
    print(f"  预计总运行次数：{total_points * n_runs}")
    print(f"  结果目录：{out_dir}")
    print(f"{'='*60}\n")

    all_best_params = {}

    for algo_name, grids in PARAM_GRIDS.items():
        print(f"[{algo_name}] 开始扫描...")
        base_params = {**BASE_PARAMS[algo_name], "epoch": epoch}
        sweep_results = {}
        best_params   = {**base_params}

        for grid in grids:
            param_name   = grid["param"]
            param_values = grid["values"]
            label        = PARAM_LABELS.get(param_name, param_name)

            print(f"  扫描 {label}: {param_values}", flush=True)
            df = run_param_sweep(
                algo_name, param_name, param_values,
                base_params, n_runs, silent_logger
            )
            sweep_results[param_name] = df

            # 保存单参数扫描 CSV
            csv_path = out_dir / f"sweep_{algo_name}_{param_name}.csv"
            df.to_csv(csv_path, index=False)

            # 最优参数（Test R² 最高）
            best_row = df.loc[df["r2_mean"].idxmax()]
            best_val = best_row["param_value"]
            best_params[param_name] = best_val
            print(
                f"    最优 {label} = {best_val}  "
                f"(R²={best_row['r2_mean']:.4f}±{best_row['r2_std']:.4f}  "
                f"n_feat={best_row['n_feat_mean']:.1f})"
            )

        # 绘制敏感性图
        plot_sensitivity(sweep_results, algo_name, out_dir / f"sensitivity_{algo_name}.png")

        all_best_params[algo_name] = best_params
        print(f"  [{algo_name}] 推荐参数：{best_params}\n")

    # 汇总打印
    print(f"\n{'='*60}")
    print("  超参数推荐汇总")
    print(f"{'='*60}")
    for algo_name, params in all_best_params.items():
        print(f"  {algo_name}:")
        for k, v in params.items():
            print(f"    {k} = {v}")

    # 保存推荐参数 JSON
    import json
    rec_path = out_dir / "recommended_params.json"
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(all_best_params, f, ensure_ascii=False, indent=2)
    print(f"\n推荐参数已保存：{rec_path}")
    print(f"结果目录：{out_dir}\n")


if __name__ == "__main__":
    main()
