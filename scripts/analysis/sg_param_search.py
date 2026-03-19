"""
SG滤波参数搜索脚本
- 用训练集交叉验证，以 PLS 的 RMSECV 为指标，网格搜索最优 window_length 和 polyorder
- 用最优参数对 train.csv / test.csv 做 SG 平滑，保存处理后数据
- 输出详细 Markdown 日志到 log/sg_param_search/
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 配置 ──────────────────────────────────────────────────────────────────────
TRAIN_CSV        = "resource/train.csv"
TEST_CSV         = "resource/test.csv"
TARGET_COL       = "TS"
BAND_START       = 1
BAND_END         = 164
WL_START         = 350   # nm，b1 对应波长
WL_STEP          = 4     # nm/band
CV_FOLDS         = 5
MAX_PLS_COMP     = 10

WINDOW_CANDIDATES   = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25]
POLYORDER_CANDIDATES = [1, 2, 3, 4]

RUN_TIME  = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR   = f"log/sg_param_search/{RUN_TIME}"
FIG_PATH  = f"{LOG_DIR}/sg_param_search.png"
MD_PATH   = f"{LOG_DIR}/report.md"
CSV_PATH  = f"{LOG_DIR}/all_results.csv"
TRAIN_OUT = f"{LOG_DIR}/train_sg.csv"
TEST_OUT  = f"{LOG_DIR}/test_sg.csv"

os.makedirs(LOG_DIR, exist_ok=True)

# ── 数据加载 ──────────────────────────────────────────────────────────────────
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)
band_cols = [f"b{i}" for i in range(BAND_START, BAND_END + 1)]
X_raw = df_train[band_cols].values.astype(float)
y     = df_train[TARGET_COL].values.astype(float)

print(f"训练集: {X_raw.shape[0]} 样本, {X_raw.shape[1]} 波段")
print(f"测试集: {df_test.shape[0]} 样本")

# ── 工具函数 ──────────────────────────────────────────────────────────────────
def apply_sg(X, window_length, polyorder, deriv=0):
    return savgol_filter(X, window_length=window_length,
                         polyorder=polyorder, deriv=deriv,
                         delta=WL_STEP, axis=1)

def best_pls_rmsecv(X, y, max_comp, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_rmsecv, best_n = np.inf, 1
    for n in range(1, max_comp + 1):
        pls = PLSRegression(n_components=n, max_iter=500)
        errors = []
        for tr, val in kf.split(X):
            pls.fit(X[tr], y[tr])
            pred = pls.predict(X[val]).ravel()
            errors.append(mean_squared_error(y[val], pred))
        rmsecv = np.sqrt(np.mean(errors))
        if rmsecv < best_rmsecv:
            best_rmsecv, best_n = rmsecv, n
    return best_rmsecv, best_n

# ── 网格搜索 ──────────────────────────────────────────────────────────────────
results = []
total = sum(1 for wl in WINDOW_CANDIDATES for po in POLYORDER_CANDIDATES if po < wl)
print(f"\n开始网格搜索，共 {total} 组参数...\n")
print(f"{'window':>8} {'polyorder':>10} {'best_n':>8} {'RMSECV':>10}")
print("-" * 42)

for wl in WINDOW_CANDIDATES:
    for po in POLYORDER_CANDIDATES:
        if po >= wl:
            continue
        X_sg = apply_sg(X_raw, wl, po, deriv=0)
        rmsecv, best_n = best_pls_rmsecv(X_sg, y, MAX_PLS_COMP, CV_FOLDS)
        results.append({"window_length": wl, "polyorder": po,
                         "best_n_components": best_n, "RMSECV": rmsecv})
        print(f"{wl:>8} {po:>10} {best_n:>8} {rmsecv:>10.4f}")

results_df = pd.DataFrame(results).sort_values("RMSECV").reset_index(drop=True)
results_df.to_csv(CSV_PATH, index=False)

best = results_df.iloc[0]
best_w = int(best.window_length)
best_p = int(best.polyorder)
best_n = int(best.best_n_components)
best_rmsecv = best.RMSECV

print(f"\n最优参数: window_length={best_w}, polyorder={best_p}, RMSECV={best_rmsecv:.4f}")

# ── 用最优参数处理并保存数据集 ────────────────────────────────────────────────
for df_in, out_path, label in [
    (df_train, TRAIN_OUT, "训练集"),
    (df_test,  TEST_OUT,  "测试集"),
]:
    df_out = df_in.copy()
    X_in   = df_in[band_cols].values.astype(float)
    X_smoothed = apply_sg(X_in, best_w, best_p, deriv=0)
    df_out[band_cols] = X_smoothed
    df_out.to_csv(out_path, index=False)
    print(f"{label} SG处理完成 → {out_path}")

# ── 可视化 ────────────────────────────────────────────────────────────────────
pivot = results_df.pivot(index="polyorder", columns="window_length", values="RMSECV")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 热力图
im = axes[0].imshow(pivot.values, aspect='auto', cmap='RdYlGn_r', origin='lower')
axes[0].set_xticks(range(len(pivot.columns)))
axes[0].set_xticklabels(pivot.columns)
axes[0].set_yticks(range(len(pivot.index)))
axes[0].set_yticklabels(pivot.index)
axes[0].set_xlabel("window_length")
axes[0].set_ylabel("polyorder")
axes[0].set_title("SG参数 RMSECV 热力图（越绿越好）")
plt.colorbar(im, ax=axes[0])
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            axes[0].text(j, i, f"{val:.3f}", ha='center', va='center',
                         fontsize=7, color='black')

# 折线图
for po, grp in results_df.groupby("polyorder"):
    grp_s = grp.sort_values("window_length")
    axes[1].plot(grp_s["window_length"], grp_s["RMSECV"],
                 marker='o', label=f"polyorder={po}")
axes[1].set_xlabel("window_length")
axes[1].set_ylabel("RMSECV")
axes[1].set_title("不同 polyorder 下 RMSECV 随窗口变化")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].scatter([best_w], [best_rmsecv], color='red', zorder=5, s=80)
axes[1].annotate(f"  w={best_w}, p={best_p}",
                 (best_w, best_rmsecv), fontsize=9, color='red')

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: {FIG_PATH}")

# ── 生成 Markdown 日志 ────────────────────────────────────────────────────────
top5 = results_df.head(5)
all_table_rows = "\n".join(
    f"| {int(r.window_length)} | {int(r.polyorder)} | {int(r.best_n_components)} | {r.RMSECV:.4f} |"
    for _, r in results_df.iterrows()
)

md = f"""# SG 滤波参数搜索报告

**运行时间：** {RUN_TIME}

---

## 1. 脚本功能说明

本脚本用于为高光谱数据的 Savitzky-Golay (SG) 平滑滤波选取最优参数，流程如下：

1. **加载训练集** `{TRAIN_CSV}`（{X_raw.shape[0]} 个样本，{X_raw.shape[1]} 个波段，目标变量 `{TARGET_COL}`）
2. **网格搜索**：对 `window_length` × `polyorder` 的所有合法组合（共 {total} 组），分别：
   - 对训练集光谱做 SG 平滑（`deriv=0`，仅平滑不求导）
   - 用 PLS 回归 + {CV_FOLDS} 折交叉验证，搜索最优成分数（1~{MAX_PLS_COMP}），记录最小 RMSECV
3. **选出最优参数**：以 RMSECV 最小为准则
4. **应用最优参数**：对训练集和测试集做 SG 平滑，保存处理后数据
5. **输出**：热力图 + 折线图、完整结果 CSV、本报告

### 参数搜索范围

| 参数 | 候选值 |
|------|--------|
| `window_length` | {WINDOW_CANDIDATES} |
| `polyorder` | {POLYORDER_CANDIDATES} |
| 约束 | `polyorder < window_length` |

### 评价指标

$$\\text{{RMSECV}} = \\sqrt{{\\frac{{1}}{{k}} \\sum_{{i=1}}^{{k}} \\text{{MSE}}_i}}$$

其中 $k={CV_FOLDS}$，每折使用 PLS 回归，成分数在 1~{MAX_PLS_COMP} 内自动选取最优。

---

## 2. 完整搜索结果

| window_length | polyorder | best_n_components | RMSECV |
|:---:|:---:|:---:|:---:|
{all_table_rows}

---

## 3. Top 5 参数组合

| 排名 | window_length | polyorder | best_n_components | RMSECV |
|:---:|:---:|:---:|:---:|:---:|
""" + "\n".join(
    f"| {i+1} | {int(r.window_length)} | {int(r.polyorder)} | {int(r.best_n_components)} | {r.RMSECV:.4f} |"
    for i, (_, r) in enumerate(top5.iterrows())
) + f"""

---

## 4. 最优参数

| 参数 | 值 |
|------|----|
| `window_length` | **{best_w}** |
| `polyorder` | **{best_p}** |
| 最优 PLS 成分数 | {best_n} |
| RMSECV | **{best_rmsecv:.4f}** |

---

## 5. 可视化

![SG参数搜索热力图与折线图](./sg_param_search.png)

- **左图（热力图）**：颜色越绿表示 RMSECV 越低，参数组合越优
- **右图（折线图）**：各 polyorder 下 RMSECV 随 window_length 的变化趋势，红点为最优组合

---

## 6. 数据处理结果

使用最优参数 `window_length={best_w}, polyorder={best_p}` 对原始数据集进行 SG 平滑处理：

| 数据集 | 原始文件 | 处理后文件 | 样本数 |
|--------|----------|------------|--------|
| 训练集 | `{TRAIN_CSV}` | `{TRAIN_OUT}` | {df_train.shape[0]} |
| 测试集 | `{TEST_CSV}` | `{TEST_OUT}` | {df_test.shape[0]} |

处理方式：对 b1~b164 全部波段做 SG 平滑（`deriv=0`），非波段列（id, Lon, Lat, TS, EC）保持不变。

---

## 7. 输出文件清单

| 文件 | 说明 |
|------|------|
| `report.md` | 本报告 |
| `all_results.csv` | 所有参数组合的完整搜索结果 |
| `sg_param_search.png` | 热力图 + 折线图 |
| `train_sg.csv` | SG平滑后的训练集 |
| `test_sg.csv` | SG平滑后的测试集 |
"""

with open(MD_PATH, "w", encoding="utf-8") as f:
    f.write(md)

print(f"\nMarkdown 报告已保存: {MD_PATH}")
print(f"日志目录: {LOG_DIR}")
