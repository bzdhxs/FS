"""
阶段一：光谱融合数据集构建

流程：
  1. 读取 resource/dataSet.csv（68样本，b14~b163，150波段）
  2. SG 平滑（window=11, polyorder=2）
  3. 生成三种变换：
       Raw : SG平滑后直接用，150维
       FOD : 差分公式 (A_{i+1}-A_i)/(λ_{i+1}-λ_i)，删全零列
       CR  : 上凸包归一化，去首尾各1列
  4. 每种变换分别 MinMaxScaler（fit on 全量，仅用于保存；
     实验脚本里 train/test 划分后会重新 fit on train）
  5. 横向拼接，列名加前缀：raw_bXX / fod_bXX / cr_bXX
  6. 保存到 resource/fusion/fusion_dataset.csv
     同时保存 resource/fusion/fusion_meta.json（各变换列范围）

用法：
  python scripts/preprocessing/build_fusion_dataset.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

DATA_FILE  = PROJECT_ROOT / "resource" / "dataSet.csv"
OUT_DIR    = PROJECT_ROOT / "resource" / "fusion"
TARGET_COL = "TS"
BAND_START = 14
BAND_END   = 164          # 不含，即 b14~b163，共150波段
SG_WINDOW  = 11
SG_POLY    = 2


# ══════════════════════════════════════════════════════════════════════════════
# 光谱变换
# ══════════════════════════════════════════════════════════════════════════════

def apply_sg(X: np.ndarray) -> np.ndarray:
    return savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLY,
                         deriv=0, axis=1)


def apply_fod(X: np.ndarray, band_names: list):
    """
    差分一阶导数：FOD_i = (A_{i+1} - A_i) / (λ_{i+1} - λ_i)
    波段间隔固定为4nm（项目常量 WAVELENGTH_STEP=4）。
    最后一列无后继，用前向差分补全（与前一列相同）。
    删除全零列（所有样本值均为0的列）。
    返回 (变换后矩阵, 保留的列名列表)
    """
    delta = 4.0  # nm，波段间隔
    n, d = X.shape
    fod = np.zeros((n, d))
    fod[:, :-1] = (X[:, 1:] - X[:, :-1]) / delta
    fod[:, -1]  = fod[:, -2]   # 末列用前向差分补全

    # 删全零列
    nonzero_mask = ~np.all(np.abs(fod) < 1e-12, axis=0)
    fod_clean    = fod[:, nonzero_mask]
    kept_names   = [band_names[i] for i in range(d) if nonzero_mask[i]]

    removed = d - fod_clean.shape[1]
    print(f"  FOD: {d}维 → 删除 {removed} 个全零列 → {fod_clean.shape[1]}维")
    return fod_clean, kept_names


def apply_cr(X: np.ndarray, band_names: list):
    """
    连续统去除：CR_i = X_i / 上凸包插值
    去掉首尾各1列（凸包端点比值恒为1，无区分度）。
    返回 (变换后矩阵, 保留的列名列表)
    """
    n, d = X.shape
    result = np.ones((n, d))
    xs = np.arange(d, dtype=float)
    for i in range(n):
        cont = _upper_convex_hull(xs, X[i].copy())
        result[i] = X[i] / (cont + 1e-10)

    # 去首尾各1列
    cr_clean   = result[:, 1:-1]
    kept_names = band_names[1:-1]
    print(f"  CR : {d}维 → 去首尾各1列 → {cr_clean.shape[1]}维")
    return cr_clean, kept_names


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


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 读取原始数据 ───────────────────────────────────────────────────────
    df = pd.read_csv(DATA_FILE)
    band_cols = [f"b{i}" for i in range(BAND_START, BAND_END)]
    X_raw = df[band_cols].values.astype(float)
    y     = df[TARGET_COL].values
    print(f"原始数据：{X_raw.shape[0]} 样本  {X_raw.shape[1]} 波段")

    # ── 2. SG 平滑 ────────────────────────────────────────────────────────────
    X_sg = apply_sg(X_raw)
    print(f"SG平滑完成（window={SG_WINDOW}, polyorder={SG_POLY}）")

    # ── 3. 三种变换 ───────────────────────────────────────────────────────────
    print("\n各变换处理：")

    # Raw
    X_raw_t    = X_sg.copy()
    raw_names  = band_cols.copy()
    print(f"  Raw: {len(raw_names)}维（直接使用SG平滑结果）")

    # FOD
    X_fod, fod_names = apply_fod(X_sg, band_cols)

    # CR
    X_cr, cr_names = apply_cr(X_sg, band_cols)

    # ── 4. 分别 MinMaxScaler ──────────────────────────────────────────────────
    scaler_raw = MinMaxScaler()
    scaler_fod = MinMaxScaler()
    scaler_cr  = MinMaxScaler()

    X_raw_scaled = scaler_raw.fit_transform(X_raw_t)
    X_fod_scaled = scaler_fod.fit_transform(X_fod)
    X_cr_scaled  = scaler_cr.fit_transform(X_cr)

    # ── 5. 拼接，列名加前缀 ───────────────────────────────────────────────────
    raw_col_names = [f"raw_{b}" for b in raw_names]
    fod_col_names = [f"fod_{b}" for b in fod_names]
    cr_col_names  = [f"cr_{b}"  for b in cr_names]

    all_col_names = raw_col_names + fod_col_names + cr_col_names
    X_fusion = np.hstack([X_raw_scaled, X_fod_scaled, X_cr_scaled])

    total_dims = X_fusion.shape[1]
    print(f"\n拼接结果：{len(raw_col_names)} + {len(fod_col_names)} + "
          f"{len(cr_col_names)} = {total_dims} 维")

    # ── 6. 保存 fusion_dataset.csv ────────────────────────────────────────────
    df_fusion = pd.DataFrame(X_fusion, columns=all_col_names)
    df_fusion.insert(0, TARGET_COL, y)

    # 保留原始元信息列
    for col in ["id", "Lon", "Lat", "EC"]:
        if col in df.columns:
            df_fusion.insert(0, col, df[col].values)

    out_csv = OUT_DIR / "fusion_dataset.csv"
    df_fusion.to_csv(out_csv, index=False)
    print(f"\n融合数据集已保存：{out_csv}")
    print(f"  shape: {df_fusion.shape}")

    # ── 7. 保存 fusion_meta.json ──────────────────────────────────────────────
    meta = {
        "total_dims":    total_dims,
        "raw_dims":      len(raw_col_names),
        "fod_dims":      len(fod_col_names),
        "cr_dims":       len(cr_col_names),
        "raw_cols":      raw_col_names,
        "fod_cols":      fod_col_names,
        "cr_cols":       cr_col_names,
        "raw_col_range": [0, len(raw_col_names)],
        "fod_col_range": [len(raw_col_names), len(raw_col_names) + len(fod_col_names)],
        "cr_col_range":  [len(raw_col_names) + len(fod_col_names), total_dims],
        "target_col":    TARGET_COL,
        "sg_window":     SG_WINDOW,
        "sg_polyorder":  SG_POLY,
        "note": (
            "MinMaxScaler fit on 全量数据（仅用于保存）。"
            "实验脚本中 train/test 划分后需重新 fit_transform(train) / transform(test)。"
            "fusion_dataset.csv 中特征列已归一化，直接读取后按 raw_col_range 等切分即可。"
        )
    }

    out_meta = OUT_DIR / "fusion_meta.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"元信息已保存：{out_meta}")

    # ── 8. 控制台汇总 ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Raw : {len(raw_col_names):3d} 维  列：{raw_col_names[0]} ~ {raw_col_names[-1]}")
    print(f"  FOD : {len(fod_col_names):3d} 维  列：{fod_col_names[0]} ~ {fod_col_names[-1]}")
    print(f"  CR  : {len(cr_col_names):3d} 维  列：{cr_col_names[0]} ~ {cr_col_names[-1]}")
    print(f"  合计: {total_dims:3d} 维")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
