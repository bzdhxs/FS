"""
收敛性验证脚本

对 HHO 和 MSHHO 各运行 N_RUNS 次，提取每次的适应度收敛曲线，
判断是否在 EPOCH 轮内收敛，并绘制收敛曲线图。

用法：
  python script/check_convergence.py
  python script/check_convergence.py --runs 2 --epoch 10   # 快速冒烟测试
"""

import os
import sys
import argparse
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

import feature_selection  # noqa: F401 触发插件自动发现
import model              # noqa: F401

from core.registry import get_algorithm, get_model
from utils.data_processor import DataProcessor

# =============================================================================
# 实验配置
# =============================================================================

N_RUNS  = 5     # 每个算法运行次数
EPOCH   = 200   # 迭代轮数

# 收敛判断参数
CONV_WINDOW    = 20    # 滑动窗口大小（轮）
CONV_THRESHOLD = 1e-4  # 极差阈值

# 数据配置
DATA_FILE  = str(PROJECT_ROOT / "resource" / "dataSet.csv")
TARGET_COL = "TS"
BAND_START = 14
BAND_END   = 164
TEST_SIZE  = 0.3

# 待分析的算法及其参数
ALGOS = [
    {
        "label":  "HHO",
        "algo":   "HHO",
        "params": {"epoch": EPOCH, "pop_size": 250, "penalty": 0.2},
    },
    {
        "label":  "MSHHO",
        "algo":   "MSHHO",
        "params": {"epoch": EPOCH, "pop_size": 50, "penalty": 0.2, "gamma_redundancy": 0.3},
    },
]

# matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 核心函数
# =============================================================================

def run_once_with_history(algo_name, algo_params, tmp_dir, logger):
    """
    运行一次特征选择，返回全局最优适应度收敛曲线。

    Returns
    -------
    np.ndarray, shape (epoch,)
        每轮的全局最优适应度值
    """
    # 数据预处理（固定 random_state=42，保证每次用同一划分）
    processor = DataProcessor(logger)
    train_csv, _ = processor.load_and_preprocess(
        original_data_path=DATA_FILE,
        target_col=TARGET_COL,
        output_dir=str(tmp_dir),
        test_size=TEST_SIZE,
        random_state=42,
    )

    # 实例化 selector
    AlgoClass = get_algorithm(algo_name)
    selector = AlgoClass(
        target_col=TARGET_COL,
        band_range=(BAND_START, BAND_END),
        logger=logger,
        **algo_params,
    )

    # 运行特征选择
    selection_path = str(tmp_dir / f"selected_{algo_name}.csv")
    selector.run_selection(input_path=train_csv, output_path=selection_path)

    # 提取收敛曲线（依赖 _last_optimizer 在 run_selection 末尾被赋值）
    history = selector._last_optimizer.history.list_global_best_fit
    return np.array(history, dtype=float)


def check_convergence(fitness_curve, window=CONV_WINDOW, threshold=CONV_THRESHOLD):
    """
    判断收敛：滑动窗口内 fitness 极差 < threshold。

    Returns
    -------
    converged : bool
    converge_epoch : int or None  首次满足条件的轮次（1-indexed）
    final_fitness : float
    """
    n = len(fitness_curve)
    for i in range(window, n + 1):
        if np.ptp(fitness_curve[i - window: i]) < threshold:
            return True, i, float(fitness_curve[-1])
    return False, None, float(fitness_curve[-1])


def plot_convergence(all_curves_dict, out_path):
    """
    绘制收敛曲线图。

    Parameters
    ----------
    all_curves_dict : dict
        {"HHO": ndarray(n_runs, epoch), "MSHHO": ndarray(n_runs, epoch)}
    out_path : Path
    """
    n_algos = len(all_curves_dict)
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5), sharey=False)
    if n_algos == 1:
        axes = [axes]

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    for ax, (algo_name, curves), color in zip(axes, all_curves_dict.items(), colors):
        # curves: shape (n_runs, epoch)
        mean_curve = np.mean(curves, axis=0)
        std_curve  = np.std(curves, axis=0)
        epochs     = np.arange(1, curves.shape[1] + 1)

        # 各次原始曲线（半透明）
        for i, c in enumerate(curves):
            ax.plot(epochs, c, alpha=0.25, linewidth=0.8, color=color,
                    label='单次运行' if i == 0 else '')

        # 均值曲线
        ax.plot(epochs, mean_curve, linewidth=2.2, color=color, label='均值')

        # ±std 带
        ax.fill_between(epochs,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.15, color=color, label='±1 std')

        ax.set_title(f'{algo_name} 适应度收敛曲线', fontsize=13)
        ax.set_xlabel('迭代轮次', fontsize=11)
        ax.set_ylabel('适应度（越小越好）', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"收敛曲线图已保存：{out_path}")


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MS-HHO 收敛性验证")
    parser.add_argument("--runs",  type=int, default=N_RUNS,  help=f"每个算法运行次数（默认 {N_RUNS}）")
    parser.add_argument("--epoch", type=int, default=EPOCH,   help=f"迭代轮数（默认 {EPOCH}）")
    return parser.parse_args()


def main():
    args = parse_args()
    n_runs = args.runs
    epoch  = args.epoch

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "script" / f"convergence_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 静默日志（抑制算法内部输出）
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    silent_logger = logging.getLogger("convergence_silent")
    silent_logger.setLevel(logging.WARNING)

    print(f"\n{'='*60}")
    print(f"  收敛性验证实验")
    print(f"  算法数：{len(ALGOS)}  每算法运行：{n_runs} 次  Epoch：{epoch}")
    print(f"  收敛判断：最后 {CONV_WINDOW} 轮极差 < {CONV_THRESHOLD}")
    print(f"  结果目录：{out_dir}")
    print(f"{'='*60}\n")

    all_curves_dict = {}
    stat_rows = []

    for algo_cfg in ALGOS:
        label     = algo_cfg["label"]
        algo_name = algo_cfg["algo"]
        params    = {**algo_cfg["params"], "epoch": epoch}

        print(f"[{label}] 运行 {n_runs} 次...", flush=True)
        curves       = []
        conv_epochs  = []
        final_fits   = []

        for run_i in range(n_runs):
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"conv_{label}_run{run_i}_"))
            try:
                curve = run_once_with_history(algo_name, params, tmp_dir, silent_logger)
                curves.append(curve)

                converged, conv_ep, final_fit = check_convergence(curve)
                conv_epochs.append(conv_ep if converged else epoch)
                final_fits.append(final_fit)

                status = f"收敛@{conv_ep}轮" if converged else "未收敛"
                print(f"  run {run_i:2d}: 最终 fitness={final_fit:.6f}  {status}")
            except Exception as e:
                print(f"  run {run_i:2d}: 失败 — {e}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if not curves:
            print(f"  [警告] {label} 所有运行均失败，跳过\n")
            continue

        curves_arr = np.array(curves)  # (n_runs, epoch)
        all_curves_dict[label] = curves_arr

        conv_rate = sum(1 for e in conv_epochs if e < epoch) / len(conv_epochs)
        stat_rows.append({
            "algo":                label,
            "n_runs":              len(curves),
            "converge_rate":       round(conv_rate, 3),
            "converge_epoch_mean": round(float(np.mean(conv_epochs)), 1),
            "converge_epoch_std":  round(float(np.std(conv_epochs)), 1),
            "final_fitness_mean":  round(float(np.mean(final_fits)), 6),
            "final_fitness_std":   round(float(np.std(final_fits)), 6),
            "fitness_min":         round(float(np.min(final_fits)), 6),
            "fitness_max":         round(float(np.max(final_fits)), 6),
        })

        print(f"  收敛率={conv_rate*100:.0f}%  "
              f"平均收敛轮次={np.mean(conv_epochs):.1f}±{np.std(conv_epochs):.1f}  "
              f"最终 fitness={np.mean(final_fits):.6f}±{np.std(final_fits):.6f}\n")

    # 保存统计 CSV
    stats_csv = out_dir / "convergence_stats.csv"
    pd.DataFrame(stat_rows).to_csv(stats_csv, index=False)

    # 保存原始收敛曲线数据
    for label, curves_arr in all_curves_dict.items():
        df_curves = pd.DataFrame(
            curves_arr,
            columns=[f"epoch_{i+1}" for i in range(curves_arr.shape[1])]
        )
        df_curves.insert(0, "run", range(len(curves_arr)))
        df_curves.to_csv(out_dir / f"curves_{label}.csv", index=False)

    # 绘图
    if all_curves_dict:
        plot_convergence(all_curves_dict, out_dir / "convergence_curves.png")

    # 控制台汇总
    print(f"\n{'='*60}")
    print("  收敛性汇总")
    print(f"{'='*60}")
    print(f"{'算法':<10} {'收敛率':>8} {'收敛轮次(mean±std)':>22} {'最终fitness(mean±std)':>26}")
    print("-" * 70)
    for row in stat_rows:
        print(
            f"{row['algo']:<10}"
            f"{row['converge_rate']*100:>7.0f}%"
            f"  {row['converge_epoch_mean']:>6.1f}±{row['converge_epoch_std']:<6.1f}"
            f"  {row['final_fitness_mean']:.6f}±{row['final_fitness_std']:.6f}"
        )

    print(f"\n统计结果：{stats_csv}")
    print(f"收敛曲线：{out_dir / 'convergence_curves.png'}\n")


if __name__ == "__main__":
    main()
