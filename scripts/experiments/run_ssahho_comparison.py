"""SSAHHO 主对比实验：HHO vs SSAHHO（30 次）。

用法：
  python script/run_ssahho_comparison.py
  python script/run_ssahho_comparison.py --runs 3 --epoch 10 --pop 10   # 冒烟测试
  python script/run_ssahho_comparison.py --runs 5 --epoch 30 --pop 20   # 小规模稳定性
  python script/run_ssahho_comparison.py --runs 10 --epoch 80 --pop 40  # 主实验预演
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 触发插件自动发现
import feature_selection  # noqa: F401
import model              # noqa: F401

from script.ssahho_runner import append_result, run_single, summarize_results

# ══════════════════════════════════════════════════════════
# 实验配置
# ══════════════════════════════════════════════════════════

SPLIT_SEED = 42
ALGO_SEEDS = list(range(30))
N_RUNS     = 30
EPOCH      = 120
POP_SIZE   = 60
PENALTY    = 0.1
GAMMA      = 0.1

DATA_FILE = str(PROJECT_ROOT / "resource" / "dataSet.csv")

VARIANTS = [
    {"label": "HHO",        "algo": "SSAHHO_HHO"},
    {"label": "+SCI+SDGE",  "algo": "SSAHHO_SCI_SDGE"},
]


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="SSAHHO 主对比实验")
    parser.add_argument("--runs",    type=int,   default=N_RUNS,   help=f"每个变体运行次数（默认 {N_RUNS}）")
    parser.add_argument("--epoch",   type=int,   default=EPOCH,    help=f"迭代轮数（默认 {EPOCH}）")
    parser.add_argument("--pop",     type=int,   default=POP_SIZE, help=f"种群大小（默认 {POP_SIZE}）")
    parser.add_argument("--penalty", type=float, default=PENALTY,  help=f"稀疏性惩罚（默认 {PENALTY}）")
    parser.add_argument("--gamma",   type=float, default=GAMMA,    help=f"冗余惩罚（默认 {GAMMA}）")
    parser.add_argument("--n_top",   type=int,   default=30,       help="每种变换保留 Top N 特征，候选池=4*n_top（默认 30）")
    return parser.parse_args()


def main():
    args   = parse_args()
    n_runs = args.runs
    epoch  = args.epoch
    pop    = args.pop
    n_top  = args.n_top

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = PROJECT_ROOT / "log" / f"comparison_ssahho_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"
    summary_csv = out_dir / "summary.csv"
    freq_csv    = out_dir / "band_frequency.csv"

    # 日志（抑制算法内部输出，只显示进度）
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    silent_logger = logging.getLogger("comparison_silent")
    silent_logger.setLevel(logging.WARNING)

    print(f"\n{'='*60}")
    print(f"  SSAHHO 主对比实验")
    print(f"  变体数：{len(VARIANTS)}  每变体运行：{n_runs} 次")
    print(f"  Epoch={epoch}  Pop={pop}  split_seed={SPLIT_SEED}  n_top={n_top}（候选池={4*n_top}维）")
    print(f"  结果目录：{out_dir}")
    print(f"{'='*60}\n")

    for v_idx, variant in enumerate(VARIANTS):
        label     = variant["label"]
        algo_name = variant["algo"]

        print(f"[{v_idx+1}/{len(VARIANTS)}] {label:12s}", end="", flush=True)

        for run_i in range(n_runs):
            algo_seed = ALGO_SEEDS[run_i] if run_i < len(ALGO_SEEDS) else run_i

            row = run_single(
                variant_label=label,
                algo_name=algo_name,
                algo_seed=algo_seed,
                split_seed=SPLIT_SEED,
                data_path=DATA_FILE,
                output_dir=out_dir,
                epoch=epoch,
                pop_size=pop,
                penalty=args.penalty,
                gamma=args.gamma,
                n_top=n_top,
                logger=silent_logger,
            )
            append_result(results_csv, row)

            # 进度点（失败用 x）
            print("." if row["status"] == "success" else "x", end="", flush=True)

        # 本变体实时统计
        df_all = pd.read_csv(results_csv)
        df_v   = df_all[(df_all["variant"] == label) & (df_all["status"] == "success")]
        if not df_v.empty:
            print(
                f"  Test R²={df_v['test_r2'].mean():.4f}±{df_v['test_r2'].std():.4f}"
                f"  n_feat={df_v['n_selected'].mean():.1f}"
            )
        else:
            print("  全部失败")

    # 汇总
    summarize_results(results_csv, summary_csv, freq_csv, VARIANTS)
    _print_summary(summary_csv, freq_csv, VARIANTS)

    print(f"\n详细结果：{results_csv}")
    print(f"汇总结果：{summary_csv}")
    print(f"波段频次：{freq_csv}\n")


def _print_summary(summary_csv: Path, freq_csv: Path, variants: list) -> None:
    """控制台打印汇总表和高频波段。"""
    df_s = pd.read_csv(summary_csv)

    print(f"\n{'='*70}")
    print("  对比实验汇总")
    print(f"{'='*70}")
    print(f"{'变体':<14} {'Test R²(mean±std)':<22} {'RMSE(mean±std)':<20} {'n_feat':<8} {'失败'}")
    print("-" * 70)
    for _, row in df_s.iterrows():
        r2_str   = f"{row['Test_R2_mean']}±{row['Test_R2_std']}"
        rmse_str = f"{row['Test_RMSE_mean']}±{row['Test_RMSE_std']}"
        print(f"{row['variant']:<14} {r2_str:<22} {rmse_str:<20} {row['n_feat_mean']:<8} {row['fail_count']}")

    print(f"\n{'='*70}")
    print("  高频波段（出现频率 ≥ 50%）")
    print(f"{'='*70}")
    df_f = pd.read_csv(freq_csv)
    for v in variants:
        label  = v["label"]
        df_vf  = df_f[(df_f["variant"] == label) & (df_f["frequency"] >= 0.5)]
        bands  = ", ".join(
            f"{r['band_name']}({r['frequency']*100:.0f}%)"
            for _, r in df_vf.iterrows()
        ) if not df_vf.empty else "无"
        print(f"  {label:<12}: {bands}")


if __name__ == "__main__":
    main()
