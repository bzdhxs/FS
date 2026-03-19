"""penalty 扫描实验：λ=0.1/0.2/0.3，每组 n=20，仅跑主线变体 SSAHHO_SCI_SDGE。

用法：
  python script/run_penalty_scan.py
  python script/run_penalty_scan.py --runs 3 --epoch 10 --pop 10   # 冒烟测试
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import feature_selection  # noqa: F401
import model              # noqa: F401

from script.ssahho_runner import append_result, run_single

# ══════════════════════════════════════════════════════════
# 实验配置
# ══════════════════════════════════════════════════════════

SPLIT_SEED     = 42
N_RUNS         = 20
EPOCH          = 120
POP_SIZE       = 60
PENALTY_VALUES = [0.1, 0.2, 0.3]
ALGO_NAME      = "SSAHHO_SCI_SDGE"

DATA_FILE = str(PROJECT_ROOT / "resource" / "dataSet.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="penalty 扫描实验")
    parser.add_argument("--runs",  type=int, default=N_RUNS,   help=f"每组运行次数（默认 {N_RUNS}）")
    parser.add_argument("--epoch", type=int, default=EPOCH,    help=f"迭代轮数（默认 {EPOCH}）")
    parser.add_argument("--pop",   type=int, default=POP_SIZE, help=f"种群大小（默认 {POP_SIZE}）")
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = PROJECT_ROOT / "log" / f"penalty_scan_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    silent_logger = logging.getLogger("penalty_scan_silent")
    silent_logger.setLevel(logging.WARNING)

    print(f"\n{'='*60}")
    print(f"  penalty 扫描实验（fitness = (1-R²_cv) + λ·|S|/D）")
    print(f"  λ 候选值：{PENALTY_VALUES}")
    print(f"  每组运行：{args.runs} 次  Epoch={args.epoch}  Pop={args.pop}")
    print(f"  结果目录：{out_dir}")
    print(f"{'='*60}\n")

    for penalty in PENALTY_VALUES:
        label = f"λ={penalty}"
        print(f"[λ={penalty}]", end="", flush=True)

        for run_i in range(args.runs):
            row = run_single(
                variant_label=label,
                algo_name=ALGO_NAME,
                algo_seed=run_i,
                split_seed=SPLIT_SEED,
                data_path=DATA_FILE,
                output_dir=out_dir,
                epoch=args.epoch,
                pop_size=args.pop,
                penalty=penalty,
                gamma=0.0,   # 新 fitness 不使用 gamma，传 0 保持接口兼容
                logger=silent_logger,
            )
            append_result(results_csv, row)
            print("." if row["status"] == "success" else "x", end="", flush=True)

        # 实时统计
        df_all = pd.read_csv(results_csv)
        df_v   = df_all[(df_all["variant"] == label) & (df_all["status"] == "success")]
        if not df_v.empty:
            print(
                f"  Test R²={df_v['test_r2'].mean():.4f}±{df_v['test_r2'].std():.4f}"
                f"  n_feat={df_v['n_selected'].mean():.1f}±{df_v['n_selected'].std():.1f}"
            )
        else:
            print("  全部失败")

    # ── 汇总与统计检验 ────────────────────────────────────
    _print_summary(results_csv, PENALTY_VALUES)
    print(f"\n详细结果：{results_csv}\n")


def _print_summary(results_csv: Path, penalty_values: list) -> None:
    df = pd.read_csv(results_csv)
    df = df[df["status"] == "success"]

    print(f"\n{'='*70}")
    print("  penalty 扫描汇总")
    print(f"{'='*70}")
    print(f"{'λ':<10} {'Test R²(mean±std)':<24} {'n_feat(mean±std)':<20} {'n'}")
    print("-" * 70)

    groups = {}
    for p in penalty_values:
        label = f"λ={p}"
        sub = df[df["variant"] == label]
        groups[label] = sub["test_r2"].values
        n_feat_mean = sub["n_selected"].mean()
        n_feat_std  = sub["n_selected"].std()
        r2_mean     = sub["test_r2"].mean()
        r2_std      = sub["test_r2"].std()
        print(f"{label:<10} {r2_mean:.4f}±{r2_std:.4f}          {n_feat_mean:.1f}±{n_feat_std:.1f}            {len(sub)}")

    # Wilcoxon 两两比较
    labels = [f"λ={p}" for p in penalty_values]
    print(f"\n{'='*70}")
    print("  Wilcoxon 两两比较（单侧，H1: 行 > 列）")
    print(f"{'='*70}")
    print(f"{'':>12}", end="")
    for lb in labels:
        print(f"{lb:>14}", end="")
    print()
    for lb_a in labels:
        print(f"{lb_a:>12}", end="")
        for lb_b in labels:
            if lb_a == lb_b:
                print(f"{'—':>14}", end="")
            else:
                a = groups[lb_a]
                b = groups[lb_b]
                if len(a) > 0 and len(b) > 0:
                    _, p = stats.wilcoxon(a, b, alternative="greater")
                    sig = "*" if p < 0.05 else "ns"
                    print(f"{p:.3f}{sig:>3}", end="         ")
                else:
                    print(f"{'N/A':>14}", end="")
        print()


if __name__ == "__main__":
    main()
