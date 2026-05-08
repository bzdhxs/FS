"""
光谱变换 × AB-HHO 对比实验

对 Raw / SNV / FOD / CR 四种变换分别运行 AB-HHO 特征选择，
下游用 PLS 评估精度，重复 30 次，统计均值±标准差。

变换在脚本内完成，不保存中间 CSV。

用法：
  python scripts/experiments/run_transform_comparison.py
  python scripts/experiments/run_transform_comparison.py --runs 5 --epoch 20
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
from core.constants import DEFAULT_RANDOM_STATE

# =============================================================================
# 配置
# =============================================================================

TRAIN_CSV  = str(PROJECT_ROOT / "resource" / "train.csv")
TEST_CSV   = str(PROJECT_ROOT / "resource" / "test.csv")
TARGET_COL = "TS"
BAND_START = 14
BAND_END   = 164   # 不含，即 b14–b163

N_RUNS   = 30
EPOCH    = 200
POP_SIZE = 50

SG_WINDOW    = 11
SG_POLYORDER = 2

ABHHO_PARAMS = dict(
    epoch=EPOCH,
    pop_size=POP_SIZE,
    a=0.9,
    gamma=2.0,
    alpha_min=2.0,
    alpha_max=8.0,
    r_max=0.5,
)

# 各变换有效波段范围（裁边后）
TRANSFORM_BAND_RANGE = {
    "Raw": (BAND_START, BAND_END),        # b14–b163，150维
    "SNV": (BAND_START, BAND_END),        # b14–b163，150维，无边缘效应
    "FOD": (18,         160),             # b18–b159，142维，两端各去4个
    "CR":  (22,         156),             # b22–b155，134维，两端各去8个
}


# =============================================================================
# 光谱变换
# =============================================================================

def apply_sg(X: np.ndarray) -> np.ndarray:
    return savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLYORDER,
                         deriv=0, axis=1)


def apply_snv(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-10)


def apply_fod(X: np.ndarray) -> np.ndarray:
    return savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLYORDER,
                         deriv=1, axis=1)


def apply_cr(X: np.ndarray) -> np.ndarray:
    n_samples, n_bands = X.shape
    result = np.ones_like(X)
    xs = np.arange(n_bands, dtype=float)
    for i in range(n_samples):
        cont = _upper_convex_hull(xs, X[i].copy())
        result[i] = X[i] / (cont + 1e-10)
    return result


def _upper_convex_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    hull = [0]
    for i in range(1, len(x)):
        while len(hull) >= 2:
            x1, y1 = x[hull[-2]], y[hull[-2]]
            x2, y2 = x[hull[-1]], y[hull[-1]]
            x3, y3 = x[i],        y[i]
            if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(x, x[hull], y[hull])


def get_transformed(t_name: str,
                    X_train_raw: np.ndarray,
                    X_test_raw: np.ndarray,
                    all_band_cols: list):
    """
    对原始数据做 SG 平滑 + 指定变换，并按 TRANSFORM_BAND_RANGE 裁边。

    Returns
    -------
    X_tr, X_te : 变换+裁边后的特征矩阵
    band_cols  : 对应的波段列名列表
    """
    # SG 平滑（在全波段上做，再裁边）
    X_tr_sg = apply_sg(X_train_raw)
    X_te_sg = apply_sg(X_test_raw)

    # 变换
    if t_name == "Raw":
        X_tr, X_te = X_tr_sg, X_te_sg
    elif t_name == "SNV":
        X_tr = apply_snv(X_tr_sg)
        X_te = apply_snv(X_te_sg)
    elif t_name == "FOD":
        X_tr = apply_fod(X_tr_sg)
        X_te = apply_fod(X_te_sg)
    elif t_name == "CR":
        X_tr = apply_cr(X_tr_sg)
        X_te = apply_cr(X_te_sg)
    else:
        raise ValueError(f"Unknown transform: {t_name}")

    # 裁边：取有效波段范围
    b_start, b_end = TRANSFORM_BAND_RANGE[t_name]
    # all_band_cols 对应 b14–b163，索引偏移 = b_start - BAND_START
    idx_start = b_start - BAND_START
    idx_end   = b_end   - BAND_START
    band_cols = [f"b{i}" for i in range(b_start, b_end)]

    return X_tr[:, idx_start:idx_end], X_te[:, idx_start:idx_end], band_cols


# =============================================================================
# 单次运行
# =============================================================================

def run_once(t_name: str,
             X_tr: np.ndarray, y_train: np.ndarray,
             X_te: np.ndarray, y_test: np.ndarray,
             band_cols: list, tmp_dir: Path, logger) -> dict:
    """
    一次完整的 AB-HHO 特征选择 + PLS 建模。

    特征选择在内存中完成（写临时 CSV 供 selector 读取），
    建模直接用 train_and_predict 绕过 run_modeling 的 CSV 依赖。
    """
    n_bands = len(band_cols)

    # ── 写临时训练 CSV（selector 需要文件路径）────────────────────────────
    df_tr = pd.DataFrame(X_tr, columns=band_cols)
    df_tr[TARGET_COL] = y_train
    train_tmp = str(tmp_dir / "train_tmp.csv")
    df_tr.to_csv(train_tmp, index=False)

    selection_out = str(tmp_dir / "selected.csv")

    # ── AB-HHO 特征选择 ──────────────────────────────────────────────────
    AlgoClass = get_algorithm("ABHHO")
    selector = AlgoClass(
        target_col=TARGET_COL,
        band_range=(int(band_cols[0][1:]), int(band_cols[-1][1:]) + 1),
        logger=logger,
        **ABHHO_PARAMS,
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

    # ── 取选中特征的列索引 ────────────────────────────────────────────────
    feat_idx = [band_cols.index(f) for f in selected_feats]
    X_tr_sel = X_tr[:, feat_idx]
    X_te_sel = X_te[:, feat_idx]

    # ── PLS 建模（直接调用 train_and_predict）────────────────────────────
    ModelClass = get_model("PLS")
    pls = ModelClass(logger=logger)
    pred_train, pred_test, _ = pls.train_and_predict(X_tr_sel, y_train, X_te_sel)

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
# 主流程
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="光谱变换 × AB-HHO 对比实验")
    parser.add_argument("--runs",  type=int, default=N_RUNS)
    parser.add_argument("--epoch", type=int, default=EPOCH)
    return parser.parse_args()


def main():
    args = parse_args()
    n_runs = args.runs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "log" / f"transform_comparison_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_csv = out_dir / "results.csv"

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    silent = logging.getLogger("transform_exp")
    silent.setLevel(logging.WARNING)

    # ── 读取原始数据（一次性）────────────────────────────────────────────
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    all_band_cols = [f"b{i}" for i in range(BAND_START, BAND_END)]

    X_train_raw = df_train[all_band_cols].values.astype(float)
    X_test_raw  = df_test[all_band_cols].values.astype(float)
    y_train     = df_train[TARGET_COL].values
    y_test      = df_test[TARGET_COL].values

    transforms = ["Raw", "SNV", "FOD", "CR"]

    print(f"\n{'='*65}")
    print(f"  光谱变换 × AB-HHO 对比实验")
    print(f"  变换：{transforms}  重复：{n_runs} 次  Epoch：{args.epoch}")
    print(f"{'='*65}\n")

    ABHHO_PARAMS["epoch"] = args.epoch
    all_rows = []

    for t_name in transforms:
        # 预计算变换（所有 run 共用同一份变换数据）
        X_tr, X_te, band_cols = get_transformed(
            t_name, X_train_raw, X_test_raw, all_band_cols)

        print(f"[{t_name:4s}] 波段数={len(band_cols)}", end="  ", flush=True)

        for run_i in range(n_runs):
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"tc_{t_name}_run{run_i}_"))
            try:
                metrics = run_once(t_name, X_tr, y_train, X_te, y_test,
                                   band_cols, tmp_dir, silent)
            except Exception as e:
                print(f"\n  [警告] {t_name} run {run_i} 失败: {e}")
                metrics = {
                    "train_r2": np.nan, "train_rmse": np.nan, "train_mae": np.nan,
                    "test_r2":  np.nan, "test_rmse":  np.nan, "test_mae":  np.nan,
                    "n_selected": 0, "selected_features": [], "elapsed_sec": 0,
                }
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            row = {
                "transform":         t_name,
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

        # 本变换统计
        df_t = pd.DataFrame([r for r in all_rows if r["transform"] == t_name])
        print(
            f"\n       Train R²={df_t['train_r2'].mean():.4f}±{df_t['train_r2'].std():.4f}"
            f"  Test R²={df_t['test_r2'].mean():.4f}±{df_t['test_r2'].std():.4f}"
            f"  RMSE={df_t['test_rmse'].mean():.4f}"
            f"  n_feat={df_t['n_selected'].mean():.1f}"
        )

    # ── 汇总 ─────────────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_rows)
    summary_rows = []
    for t_name in transforms:
        df_t = df_all[df_all["transform"] == t_name]
        summary_rows.append({
            "transform":   t_name,
            "Train_R2":    f"{df_t['train_r2'].mean():.4f}±{df_t['train_r2'].std():.4f}",
            "Train_RMSE":  f"{df_t['train_rmse'].mean():.4f}±{df_t['train_rmse'].std():.4f}",
            "Train_MAE":   f"{df_t['train_mae'].mean():.4f}±{df_t['train_mae'].std():.4f}",
            "Test_R2":     f"{df_t['test_r2'].mean():.4f}±{df_t['test_r2'].std():.4f}",
            "Test_RMSE":   f"{df_t['test_rmse'].mean():.4f}±{df_t['test_rmse'].std():.4f}",
            "Test_MAE":    f"{df_t['test_mae'].mean():.4f}±{df_t['test_mae'].std():.4f}",
            "n_feat":      f"{df_t['n_selected'].mean():.1f}±{df_t['n_selected'].std():.1f}",
            "time_sec":    f"{df_t['elapsed_sec'].mean():.1f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "summary.csv", index=False)

    # 波段频次
    band_freq_rows = []
    for t_name in transforms:
        df_t = df_all[df_all["transform"] == t_name]
        all_bands = []
        for s in df_t["selected_features"].dropna():
            if s:
                all_bands.extend(s.split(";"))
        for band, cnt in sorted(Counter(all_bands).items(), key=lambda x: -x[1]):
            band_freq_rows.append({
                "transform": t_name, "band": band,
                "count": cnt, "frequency": round(cnt / len(df_t), 3),
            })
    pd.DataFrame(band_freq_rows).to_csv(out_dir / "band_frequency.csv", index=False)

    # 控制台汇总
    print(f"\n{'='*85}")
    print("  汇总结果")
    print(f"{'='*85}")
    print(f"{'变换':<6} {'Train R²':<22} {'Test R²':<22} {'Test RMSE':<20} {'n_feat':<14} time(s)")
    print("-" * 85)
    for r in summary_rows:
        print(f"{r['transform']:<6} {r['Train_R2']:<22} {r['Test_R2']:<22} "
              f"{r['Test_RMSE']:<20} {r['n_feat']:<14} {r['time_sec']}")

    # 高频波段
    df_bf = pd.DataFrame(band_freq_rows)
    print(f"\n{'='*85}")
    print("  各变换高频波段（出现频率 ≥ 50%）")
    print(f"{'='*85}")
    for t_name in transforms:
        df_vbf = df_bf[(df_bf["transform"] == t_name) & (df_bf["frequency"] >= 0.5)]
        bands_str = ", ".join(
            f"{r['band']}({r['frequency']*100:.0f}%)"
            for _, r in df_vbf.iterrows()
        ) if not df_vbf.empty else "无"
        print(f"  {t_name:<6}: {bands_str}")

    print(f"\n详细结果：{result_csv}")
    print(f"汇总结果：{out_dir / 'summary.csv'}\n")


if __name__ == "__main__":
    main()
