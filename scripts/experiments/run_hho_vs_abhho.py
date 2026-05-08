"""
HHO vs AB-HHO 对比实验 + 消融实验

依赖：先运行 run_transform_comparison.py，从其 summary.csv 中自动选取
      Test R² 最高的光谱变换作为输入特征。

实验一：HHO vs ABHHO（30次重复，PLS / SVM / RF）
实验二：消融实验（20次重复，PLS / SVM / RF）
  变体：HHO / ABHHO_I1 / ABHHO_I2 / ABHHO_I3 / ABHHO

用法：
  python scripts/experiments/run_hho_vs_abhho.py --transform_result <summary.csv路径>
  python scripts/experiments/run_hho_vs_abhho.py --transform_result <path> --runs1 30 --runs2 20
  python scripts/experiments/run_hho_vs_abhho.py --transform FOD   # 手动指定变换，跳过自动选取
"""

import argparse
import logging
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import feature_selection  # noqa: F401
import model              # noqa: F401

from core.registry import get_algorithm, get_model

# =============================================================================
# 配置
# =============================================================================

TRAIN_CSV  = str(PROJECT_ROOT / "resource" / "train.csv")
TEST_CSV   = str(PROJECT_ROOT / "resource" / "test.csv")
TARGET_COL = "TS"
BAND_START = 14
BAND_END   = 164

EPOCH    = 200
POP_SIZE = 100

SG_WINDOW    = 11
SG_POLYORDER = 2

ABHHO_PARAMS = dict(
    epoch=EPOCH, pop_size=POP_SIZE,
    a=0.9, gamma=2.0, alpha_min=2.0, alpha_max=8.0, r_max=0.15,
)

# 各变换有效波段范围（裁边）
TRANSFORM_BAND_RANGE = {
    "Raw": (BAND_START, BAND_END),
    "SNV": (BAND_START, BAND_END),
    "FOD": (18, 160),
    "CR":  (22, 156),
}

MODELS = ["PLS", "SVM", "RF"]

# 实验一变体
COMPARISON_VARIANTS = [
    {"label": "HHO",   "algo": "HHO",   "extra": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"label": "ABHHO", "algo": "ABHHO", "extra": ABHHO_PARAMS},
]

# 实验二变体
ABLATION_VARIANTS = [
    {"label": "HHO",      "algo": "HHO",      "extra": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"label": "ABHHO_I1", "algo": "ABHHO_I1", "extra": ABHHO_PARAMS},
    {"label": "ABHHO_I2", "algo": "ABHHO_I2", "extra": ABHHO_PARAMS},
    {"label": "ABHHO_I3", "algo": "ABHHO_I3", "extra": ABHHO_PARAMS},
    {"label": "ABHHO",    "algo": "ABHHO",    "extra": ABHHO_PARAMS},
]


# =============================================================================
# 光谱变换
# =============================================================================

def apply_sg(X):
    return savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLYORDER, deriv=0, axis=1)

def apply_snv(X):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    return (X - m) / (s + 1e-10)

def apply_fod(X):
    return savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLYORDER, deriv=1, axis=1)

def apply_cr(X):
    result = np.ones_like(X)
    xs = np.arange(X.shape[1], dtype=float)
    for i in range(len(X)):
        hull = [0]
        y = X[i].copy()
        for k in range(1, len(xs)):
            while len(hull) >= 2:
                x1,y1 = xs[hull[-2]],y[hull[-2]]
                x2,y2 = xs[hull[-1]],y[hull[-1]]
                x3,y3 = xs[k],y[k]
                if (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1) >= 0:
                    hull.pop()
                else:
                    break
            hull.append(k)
        cont = np.interp(xs, xs[hull], y[hull])
        result[i] = y / (cont + 1e-10)
    return result

def get_transformed(t_name, X_train_raw, X_test_raw):
    X_tr_sg = apply_sg(X_train_raw)
    X_te_sg = apply_sg(X_test_raw)
    if t_name == "Raw":
        X_tr, X_te = X_tr_sg, X_te_sg
    elif t_name == "SNV":
        X_tr, X_te = apply_snv(X_tr_sg), apply_snv(X_te_sg)
    elif t_name == "FOD":
        X_tr, X_te = apply_fod(X_tr_sg), apply_fod(X_te_sg)
    elif t_name == "CR":
        X_tr, X_te = apply_cr(X_tr_sg), apply_cr(X_te_sg)
    else:
        raise ValueError(f"Unknown transform: {t_name}")

    b_start, b_end = TRANSFORM_BAND_RANGE[t_name]
    idx_s = b_start - BAND_START
    idx_e = b_end   - BAND_START
    band_cols = [f"b{i}" for i in range(b_start, b_end)]
    return X_tr[:, idx_s:idx_e], X_te[:, idx_s:idx_e], band_cols


# =============================================================================
# 单次运行
# =============================================================================

def run_once(algo_name, algo_params, model_name,
             X_tr, y_train, X_te, y_test,
             band_cols, tmp_dir, logger):
    """特征选择 + 单个模型建模，返回指标 dict。"""

    # 写临时训练 CSV
    df_tr = pd.DataFrame(X_tr, columns=band_cols)
    df_tr[TARGET_COL] = y_train
    train_tmp = str(tmp_dir / "train_tmp.csv")
    df_tr.to_csv(train_tmp, index=False)
    selection_out = str(tmp_dir / "selected.csv")

    # 特征选择
    AlgoClass = get_algorithm(algo_name)
    selector = AlgoClass(
        target_col=TARGET_COL,
        band_range=(int(band_cols[0][1:]), int(band_cols[-1][1:]) + 1),
        logger=logger,
        **algo_params,
    )
    t0 = time.time()
    result = selector.run_selection(input_path=train_tmp, output_path=selection_out)
    elapsed = time.time() - t0

    selected_feats = result.selected_features
    n_selected = len(selected_feats)

    if n_selected == 0:
        return {
            "train_r2": np.nan, "train_rmse": np.nan, "train_mae": np.nan,
            "test_r2":  np.nan, "test_rmse":  np.nan, "test_mae":  np.nan,
            "n_selected": 0, "selected_features": [], "elapsed_sec": elapsed,
        }

    feat_idx  = [band_cols.index(f) for f in selected_feats]
    X_tr_sel  = X_tr[:, feat_idx]
    X_te_sel  = X_te[:, feat_idx]

    # 建模
    ModelClass = get_model(model_name)
    mdl = ModelClass(logger=logger)
    pred_train, pred_test, _ = mdl.train_and_predict(X_tr_sel, y_train, X_te_sel)

    return {
        "train_r2":          r2_score(y_train, pred_train),
        "train_rmse":        np.sqrt(mean_squared_error(y_train, pred_train)),
        "train_mae":         mean_absolute_error(y_train, pred_train),
        "test_r2":           r2_score(y_test, pred_test),
        "test_rmse":         np.sqrt(mean_squared_error(y_test, pred_test)),
        "test_mae":          mean_absolute_error(y_test, pred_test),
        "n_selected":        n_selected,
        "selected_features": selected_feats,
        "elapsed_sec":       elapsed,
    }


# =============================================================================
# 实验主体（通用）
# =============================================================================

def run_experiment(exp_name, variants, n_runs, models,
                   X_tr, y_train, X_te, y_test, band_cols,
                   out_dir, logger):
    """
    通用实验循环：variants × models × n_runs。
    每次运行后立即追加写 CSV，支持中断恢复。
    """
    result_csv = out_dir / f"{exp_name}_results.csv"
    all_rows   = []

    total = len(variants) * len(models)
    done  = 0

    for variant in variants:
        label     = variant["label"]
        algo_name = variant["algo"]
        algo_params = dict(variant["extra"])

        for model_name in models:
            done += 1
            tag = f"{label}/{model_name}"
            print(f"  [{done}/{total}] {tag:20s}", end="  ", flush=True)

            for run_i in range(n_runs):
                tmp_dir = Path(tempfile.mkdtemp(prefix=f"{label}_{model_name}_r{run_i}_"))
                try:
                    metrics = run_once(
                        algo_name, algo_params, model_name,
                        X_tr, y_train, X_te, y_test,
                        band_cols, tmp_dir, logger,
                    )
                except Exception as e:
                    print(f"\n    [警告] {tag} run {run_i} 失败: {e}")
                    metrics = {
                        "train_r2": np.nan, "train_rmse": np.nan, "train_mae": np.nan,
                        "test_r2":  np.nan, "test_rmse":  np.nan, "test_mae":  np.nan,
                        "n_selected": 0, "selected_features": [], "elapsed_sec": 0,
                    }
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                row = {
                    "variant":           label,
                    "model":             model_name,
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
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(result_csv, index=False)
                print(".", end="", flush=True)

            # 本 variant+model 统计
            df_vm = pd.DataFrame([r for r in all_rows
                                   if r["variant"] == label and r["model"] == model_name])
            print(f"  Test R²={df_vm['test_r2'].mean():.4f}±{df_vm['test_r2'].std():.4f}"
                  f"  RMSE={df_vm['test_rmse'].mean():.4f}"
                  f"  n={df_vm['n_selected'].mean():.1f}")

    # 汇总
    df_all = pd.DataFrame(all_rows)
    summary_rows = []
    for variant in variants:
        label = variant["label"]
        for model_name in models:
            df_vm = df_all[(df_all["variant"] == label) & (df_all["model"] == model_name)]
            summary_rows.append({
                "variant":    label,
                "model":      model_name,
                "Train_R2":   f"{df_vm['train_r2'].mean():.4f}±{df_vm['train_r2'].std():.4f}",
                "Train_RMSE": f"{df_vm['train_rmse'].mean():.4f}±{df_vm['train_rmse'].std():.4f}",
                "Train_MAE":  f"{df_vm['train_mae'].mean():.4f}±{df_vm['train_mae'].std():.4f}",
                "Test_R2":    f"{df_vm['test_r2'].mean():.4f}±{df_vm['test_r2'].std():.4f}",
                "Test_RMSE":  f"{df_vm['test_rmse'].mean():.4f}±{df_vm['test_rmse'].std():.4f}",
                "Test_MAE":   f"{df_vm['test_mae'].mean():.4f}±{df_vm['test_mae'].std():.4f}",
                "n_feat":     f"{df_vm['n_selected'].mean():.1f}±{df_vm['n_selected'].std():.1f}",
                "time_sec":   f"{df_vm['elapsed_sec'].mean():.1f}",
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / f"{exp_name}_summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    # 波段频次
    band_freq_rows = []
    for variant in variants:
        label = variant["label"]
        for model_name in models:
            df_vm = df_all[(df_all["variant"] == label) & (df_all["model"] == model_name)]
            all_bands = []
            for s in df_vm["selected_features"].dropna():
                if s:
                    all_bands.extend(s.split(";"))
            for band, cnt in sorted(Counter(all_bands).items(), key=lambda x: -x[1]):
                band_freq_rows.append({
                    "variant": label, "model": model_name, "band": band,
                    "count": cnt, "frequency": round(cnt / len(df_vm), 3),
                })
    pd.DataFrame(band_freq_rows).to_csv(
        out_dir / f"{exp_name}_band_frequency.csv", index=False)

    # 控制台打印汇总
    print(f"\n{'='*95}")
    print(f"  {exp_name} 汇总")
    print(f"{'='*95}")
    print(f"{'变体':<12} {'模型':<6} {'Train R²':<22} {'Test R²':<22} {'Test RMSE':<20} {'n_feat'}")
    print("-" * 95)
    for r in summary_rows:
        print(f"{r['variant']:<12} {r['model']:<6} {r['Train_R2']:<22} "
              f"{r['Test_R2']:<22} {r['Test_RMSE']:<20} {r['n_feat']}")

    return df_summary


# =============================================================================
# 主流程
# =============================================================================

def pick_best_transform(summary_csv_path: str) -> str:
    """从 transform_comparison 的 summary.csv 中选 Test R² 均值最高的变换。"""
    df = pd.read_csv(summary_csv_path)
    # Test_R2 列格式为 "0.xxxx±0.xxxx"，取均值部分
    df["test_r2_mean"] = df["Test_R2"].apply(lambda x: float(str(x).split("±")[0]))
    best = df.loc[df["test_r2_mean"].idxmax(), "transform"]
    print(f"\n  最优变换：{best}  "
          f"(Test R²={df.loc[df['test_r2_mean'].idxmax(), 'Test_R2']})")
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="HHO vs AB-HHO 对比 + 消融实验")
    parser.add_argument("--transform_result", type=str, default=None,
                        help="transform_comparison 的 summary.csv 路径，用于自动选最优变换")
    parser.add_argument("--transform", type=str, default=None,
                        help="手动指定变换（Raw/SNV/FOD/CR），优先于 --transform_result")
    parser.add_argument("--runs1", type=int, default=30, help="实验一重复次数（默认30）")
    parser.add_argument("--runs2", type=int, default=20, help="实验二重复次数（默认20）")
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--skip_exp1", action="store_true", help="跳过实验一，只跑消融")
    parser.add_argument("--skip_exp2", action="store_true", help="跳过实验二，只跑对比")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 确定最优变换 ─────────────────────────────────────────────────────
    if args.transform:
        best_transform = args.transform
        print(f"\n手动指定变换：{best_transform}")
    elif args.transform_result:
        best_transform = pick_best_transform(args.transform_result)
    else:
        raise ValueError("请提供 --transform_result 或 --transform 参数")

    # ── 更新 epoch ────────────────────────────────────────────────────────
    ABHHO_PARAMS["epoch"] = args.epoch
    for v in COMPARISON_VARIANTS + ABLATION_VARIANTS:
        if "epoch" in v["extra"]:
            v["extra"]["epoch"] = args.epoch

    # ── 输出目录 ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "log" / \
              f"hho_vs_abhho_{best_transform}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 读取数据 + 变换 ───────────────────────────────────────────────────
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    all_band_cols = [f"b{i}" for i in range(BAND_START, BAND_END)]

    X_train_raw = df_train[all_band_cols].values.astype(float)
    X_test_raw  = df_test[all_band_cols].values.astype(float)
    y_train     = df_train[TARGET_COL].values
    y_test      = df_test[TARGET_COL].values

    X_tr, X_te, band_cols = get_transformed(best_transform, X_train_raw, X_test_raw)

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    silent = logging.getLogger("hho_vs_abhho")
    silent.setLevel(logging.WARNING)

    print(f"\n{'='*65}")
    print(f"  HHO vs AB-HHO 实验")
    print(f"  变换：{best_transform}  波段数：{len(band_cols)}")
    print(f"  实验一：{args.runs1} 次  实验二：{args.runs2} 次  Epoch：{args.epoch}")
    print(f"  下游模型：{MODELS}")
    print(f"  结果目录：{out_dir}")
    print(f"{'='*65}")

    # ── 实验一：HHO vs ABHHO ──────────────────────────────────────────────
    if not args.skip_exp1:
        print(f"\n{'─'*65}")
        print(f"  实验一：HHO vs ABHHO（{args.runs1} 次重复）")
        print(f"{'─'*65}")
        run_experiment(
            exp_name="exp1_comparison",
            variants=COMPARISON_VARIANTS,
            n_runs=args.runs1,
            models=MODELS,
            X_tr=X_tr, y_train=y_train,
            X_te=X_te, y_test=y_test,
            band_cols=band_cols,
            out_dir=out_dir,
            logger=silent,
        )

    # ── 实验二：消融实验 ──────────────────────────────────────────────────
    if not args.skip_exp2:
        print(f"\n{'─'*65}")
        print(f"  实验二：消融实验（{args.runs2} 次重复）")
        print(f"{'─'*65}")
        run_experiment(
            exp_name="exp2_ablation",
            variants=ABLATION_VARIANTS,
            n_runs=args.runs2,
            models=MODELS,
            X_tr=X_tr, y_train=y_train,
            X_te=X_te, y_test=y_test,
            band_cols=band_cols,
            out_dir=out_dir,
            logger=silent,
        )

    print(f"\n全部完成，结果目录：{out_dir}\n")


if __name__ == "__main__":
    main()
