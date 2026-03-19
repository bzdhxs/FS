"""SSAHHO Repeated Hold-out 补充实验。

10 个不同 split_seed × 5 次算法运行 = 50 次结果。
验证 HHO vs +SCI+SDGE 的结论不依赖单次数据划分。

用法：
  python script/run_ssahho_repeated_holdout.py
  python script/run_ssahho_repeated_holdout.py --algo_runs 3 --epoch 10 --pop 10  # 快速验证
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import feature_selection  # noqa: F401
import model              # noqa: F401

from script.ssahho_runner import append_result, run_single, summarize_results

# ══════════════════════════════════════════════════════════
# 实验配置
# ══════════════════════════════════════════════════════════

SPLIT_SEEDS = list(range(10))   # 10 个不同数据划分
ALGO_SEEDS  = list(range(5))    # 每个划分跑 5 次算法
EPOCH       = 120
POP_SIZE    = 60
PENALTY     = 0.1
GAMMA       = 0.1

DATA_FILE = str(PROJECT_ROOT / "resource" / "dataSet.csv")

VARIANTS = [
    {"label": "HHO",       "algo": "SSAHHO_HHO"},
    {"label": "+SCI+SDGE", "algo": "SSAHHO_SCI_SDGE"},
]


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="SSAHHO Repeated Hold-out 补充实验")
    parser.add_argument("--split_runs", type=int, default=len(SPLIT_SEEDS),
                        help=f"数据划分次数（默认 {len(SPLIT_SEEDS)}）")
    parser.add_argument("--algo_runs",  type=int, default=len(ALGO_SEEDS),
                        help=f"每个划分的算法运行次数（默认 {len(ALGO_SEEDS)}）")
    parser.add_argument("--epoch",   type=int,   default=EPOCH,   help=f"迭代轮数（默认 {EPOCH}）")
    parser.add_argument("--pop",     type=int,   default=POP_SIZE, help=f"种群大小（默认 {POP_SIZE}）")
    return parser.parse_args()


def main():
    args        = parse_args()
    n_splits    = args.split_runs
    n_algo_runs = args.algo_runs
    epoch       = args.epoch
    pop         = args.pop

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = PROJECT_ROOT / "log" / f"repeated_holdout_ssahho_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"
    summary_csv = out_dir / "summary.csv"
    freq_csv    = out_dir / "band_frequency.csv"

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    silent_logger = logging.getLogger("rh_silent")
    silent_logger.setLevel(logging.WARNING)

    total_runs = len(VARIANTS) * n_splits * n_algo_runs
    print(f"\n{'='*65}")
    print(f"  SSAHHO Repeated Hold-out 补充实验")
    print(f"  划分数={n_splits}  每划分算法运行={n_algo_runs}  总运行={total_runs}")
    print(f"  Epoch={epoch}  Pop={pop}")
    print(f"  结果目录：{out_dir}")
    print(f"{'='*65}\n")

    for v_idx, variant in enumerate(VARIANTS):
        label     = variant["label"]
        algo_name = variant["algo"]
        print(f"[{v_idx+1}/{len(VARIANTS)}] {label:12s}")

        for s_i in range(n_splits):
            split_seed = SPLIT_SEEDS[s_i] if s_i < len(SPLIT_SEEDS) else s_i
            print(f"  split_seed={split_seed:2d} ", end="", flush=True)

            for a_i in range(n_algo_runs):
                algo_seed = ALGO_SEEDS[a_i] if a_i < len(ALGO_SEEDS) else a_i

                row = run_single(
                    variant_label=label,
                    algo_name=algo_name,
                    algo_seed=algo_seed,
                    split_seed=split_seed,
                    data_path=DATA_FILE,
                    output_dir=out_dir,
                    epoch=epoch,
                    pop_size=pop,
                    penalty=PENALTY,
                    gamma=GAMMA,
                    logger=silent_logger,
                )
                append_result(results_csv, row)
                print("." if row["status"] == "success" else "x", end="", flush=True)

            print()  # 换行

    # ── 汇总 ──────────────────────────────────────────────
    summarize_results(results_csv, summary_csv, freq_csv, VARIANTS)
    _print_summary(results_csv, VARIANTS)

    print(f"\n详细结果：{results_csv}")
    print(f"汇总结果：{summary_csv}\n")


def _print_summary(results_csv: Path, variants: list) -> None:
    """按 split_seed 分组打印，并做整体统计检验。"""
    df = pd.read_csv(results_csv)
    df = df[df["status"] == "success"]

    print(f"\n{'='*65}")
    print("  Repeated Hold-out 汇总")
    print(f"{'='*65}")

    # 每个 split_seed 的均值
    print(f"\n{'split_seed':<12} {'HHO R²':<14} {'+SCI+SDGE R²':<14} {'ΔR²'}")
    print("-" * 55)
    delta_list = []
    for ss in sorted(df["split_seed"].unique()):
        df_ss = df[df["split_seed"] == ss]
        hho_r2  = df_ss[df_ss["variant"] == "HHO"]["test_r2"].mean()
        sdge_r2 = df_ss[df_ss["variant"] == "+SCI+SDGE"]["test_r2"].mean()
        delta   = sdge_r2 - hho_r2
        delta_list.append(delta)
        print(f"{ss:<12} {hho_r2:.4f}         {sdge_r2:.4f}         {delta:+.4f}")

    # 整体统计
    hho_all  = df[df["variant"] == "HHO"]["test_r2"].values
    sdge_all = df[df["variant"] == "+SCI+SDGE"]["test_r2"].values
    _, p = stats.wilcoxon(sdge_all, hho_all, alternative="greater")
    d = (sdge_all.mean() - hho_all.mean()) / np.sqrt(
        (sdge_all.std(ddof=1)**2 + hho_all.std(ddof=1)**2) / 2
    )

    print(f"\n整体（{len(hho_all)} 次）:")
    print(f"  HHO:       {hho_all.mean():.4f} ± {hho_all.std():.4f}")
    print(f"  +SCI+SDGE: {sdge_all.mean():.4f} ± {sdge_all.std():.4f}")
    print(f"  Wilcoxon p={p:.4f}{'*' if p<0.05 else ' ns'}  Cohen d={d:.3f}")
    print(f"  ΔR² 跨划分: mean={np.mean(delta_list):+.4f}  "
          f"std={np.std(delta_list):.4f}  "
          f"正向比例={sum(1 for x in delta_list if x>0)}/{len(delta_list)}")


if __name__ == "__main__":
    main()
