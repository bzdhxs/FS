"""多算法多数据集对比实验：HHO / SSA / SMA × PLS / SVM / RF × raw / enriched。

用法：
  python script/run_algo_comparison.py
  python script/run_algo_comparison.py --runs 2 --epoch 10 --pop 10   # 冒烟测试
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

# 触发插件自动发现（含新增的 SSA / SMA）
import feature_selection  # noqa: F401
import model              # noqa: F401

from script.ssahho_runner import append_result, run_single

# ══════════════════════════════════════════════════════════
# 实验配置
# ══════════════════════════════════════════════════════════

SPLIT_SEED = 42
N_RUNS     = 20
EPOCH      = 120
POP_SIZE   = 60
PENALTY    = 0.1
GAMMA      = 0.1
N_TOP      = 30   # enriched 模式：每种变换保留 Top-30，候选池 = 4×30 = 120 维

ALGOS  = ["HHO", "SSA", "SMA"]
MODELS = ["PLS", "SVM", "RF"]

# raw 模式：n_top=0 表示不做变换扩充，直接用原始波段（BaseMealpySelector 天然支持）
# enriched 模式：n_top=N_TOP，通过 build_fusion_candidates 构建 4 种变换融合候选池
DATASET_MODES = [
    {"label": "raw",      "n_top": 0},
    {"label": "enriched", "n_top": N_TOP},
]

DATA_FILE = str(PROJECT_ROOT / "resource" / "dataSet.csv")


# ══════════════════════════════════════════════════════════
# 参数解析
# ══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="多算法多数据集对比实验")
    parser.add_argument("--runs",    type=int,   default=N_RUNS,   help=f"每组运行次数（默认 {N_RUNS}）")
    parser.add_argument("--epoch",   type=int,   default=EPOCH,    help=f"迭代轮数（默认 {EPOCH}）")
    parser.add_argument("--pop",     type=int,   default=POP_SIZE, help=f"种群大小（默认 {POP_SIZE}）")
    parser.add_argument("--penalty", type=float, default=PENALTY,  help=f"稀疏性惩罚（默认 {PENALTY}）")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════

def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = PROJECT_ROOT / "log" / f"algo_comparison_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    silent_logger = logging.getLogger("algo_comparison_silent")
    silent_logger.setLevel(logging.WARNING)

    total = len(DATASET_MODES) * len(ALGOS) * len(MODELS) * args.runs
    print(f"\n{'='*65}")
    print(f"  多算法多数据集对比实验")
    print(f"  算法：{ALGOS}  模型：{MODELS}")
    print(f"  数据集：{[d['label'] for d in DATASET_MODES]}")
    print(f"  每组运行：{args.runs} 次  Epoch={args.epoch}  Pop={args.pop}")
    print(f"  总运行次数：{total}")
    print(f"  结果目录：{out_dir}")
    print(f"{'='*65}\n")

    combo_idx = 0
    total_combos = len(DATASET_MODES) * len(ALGOS) * len(MODELS)

    for ds in DATASET_MODES:
        ds_label = ds["label"]
        n_top    = ds["n_top"]

        for algo_name in ALGOS:
            for model_name in MODELS:
                combo_idx += 1
                variant_label = f"{ds_label}/{algo_name}/{model_name}"
                print(f"[{combo_idx}/{total_combos}] {variant_label}", end="", flush=True)

                for run_i in range(args.runs):
                    row = run_single(
                        variant_label=variant_label,
                        algo_name=algo_name,
                        algo_seed=run_i,
                        split_seed=SPLIT_SEED,
                        data_path=DATA_FILE,
                        output_dir=out_dir,
                        epoch=args.epoch,
                        pop_size=args.pop,
                        penalty=args.penalty,
                        gamma=GAMMA,
                        n_top=n_top,
                        model_name=model_name,
                        logger=silent_logger,
                    )
                    # 补充 dataset 和 algo 字段，方便后续分组
                    row["dataset"] = ds_label
                    row["algo"]    = algo_name
                    append_result(results_csv, row)
                    print("." if row["status"] == "success" else "x", end="", flush=True)

                # 实时统计
                df_all = pd.read_csv(results_csv)
                df_v   = df_all[
                    (df_all["variant"] == variant_label) & (df_all["status"] == "success")
                ]
                if not df_v.empty:
                    print(
                        f"  R²={df_v['test_r2'].mean():.4f}±{df_v['test_r2'].std():.4f}"
                        f"  n_feat={df_v['n_selected'].mean():.1f}"
                    )
                else:
                    print("  全部失败")

    # 生成汇总
    _save_summary(results_csv, out_dir)
    print(f"\n详细结果：{results_csv}")
    print(f"汇总结果：{out_dir / 'summary.csv'}")
    print(f"Wilcoxon：{out_dir / 'wilcoxon_table.csv'}\n")


# ══════════════════════════════════════════════════════════
# 汇总与统计检验
# ══════════════════════════════════════════════════════════

def _save_summary(results_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_csv)
    df_ok = df[df["status"] == "success"].copy()

    # ── summary.csv：按 dataset × algo × model 分组 ──────
    summary_rows = []
    for ds_label in df_ok["dataset"].unique():
        for algo in ALGOS:
            for mdl in MODELS:
                variant = f"{ds_label}/{algo}/{mdl}"
                sub = df_ok[df_ok["variant"] == variant]
                if sub.empty:
                    continue
                summary_rows.append({
                    "dataset":        ds_label,
                    "algo":           algo,
                    "model":          mdl,
                    "n_runs":         len(sub),
                    "Test_R2_mean":   round(sub["test_r2"].mean(),   4),
                    "Test_R2_std":    round(sub["test_r2"].std(),    4),
                    "Test_RMSE_mean": round(sub["test_rmse"].mean(), 4),
                    "Test_RMSE_std":  round(sub["test_rmse"].std(),  4),
                    "n_feat_mean":    round(sub["n_selected"].mean(), 1),
                    "n_feat_std":     round(sub["n_selected"].std(),  1),
                    "fail_count":     int((df["variant"] == variant).sum() - len(sub)),
                })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "summary.csv", index=False)

    # 控制台打印
    print(f"\n{'='*75}")
    print("  汇总（按数据集分组）")
    print(f"{'='*75}")
    for ds_label in df_ok["dataset"].unique():
        print(f"\n  [{ds_label}]")
        print(f"  {'算法':<6} {'模型':<6} {'Test R²(mean±std)':<22} {'RMSE(mean±std)':<20} {'n_feat'}")
        print("  " + "-" * 65)
        sub_s = df_summary[df_summary["dataset"] == ds_label]
        for _, row in sub_s.iterrows():
            r2_str   = f"{row['Test_R2_mean']}±{row['Test_R2_std']}"
            rmse_str = f"{row['Test_RMSE_mean']}±{row['Test_RMSE_std']}"
            print(f"  {row['algo']:<6} {row['model']:<6} {r2_str:<22} {rmse_str:<20} {row['n_feat_mean']}")

    # ── wilcoxon_table.csv：同一数据集内算法两两比较（最佳模型下） ──
    wilcoxon_rows = []
    for ds_label in df_ok["dataset"].unique():
        for mdl in MODELS:
            for algo_a in ALGOS:
                for algo_b in ALGOS:
                    if algo_a >= algo_b:
                        continue
                    va = f"{ds_label}/{algo_a}/{mdl}"
                    vb = f"{ds_label}/{algo_b}/{mdl}"
                    a_vals = df_ok[df_ok["variant"] == va]["test_r2"].values
                    b_vals = df_ok[df_ok["variant"] == vb]["test_r2"].values
                    if len(a_vals) < 5 or len(b_vals) < 5:
                        continue
                    n = min(len(a_vals), len(b_vals))
                    try:
                        _, p = stats.wilcoxon(a_vals[:n], b_vals[:n])
                        wilcoxon_rows.append({
                            "dataset": ds_label,
                            "model":   mdl,
                            "algo_a":  algo_a,
                            "algo_b":  algo_b,
                            "mean_a":  round(float(a_vals.mean()), 4),
                            "mean_b":  round(float(b_vals.mean()), 4),
                            "p_value": round(float(p), 4),
                            "sig":     "*" if p < 0.05 else "ns",
                        })
                    except Exception:
                        pass

    pd.DataFrame(wilcoxon_rows).to_csv(out_dir / "wilcoxon_table.csv", index=False)


if __name__ == "__main__":
    main()
