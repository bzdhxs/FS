"""
实验结果可视化脚本
生成以下图表：
  1. 实验一：HHO vs ABHHO - Test R² / RMSE 箱线图（按模型分组）
  2. 实验一：HHO vs ABHHO - 特征数箱线图
  3. 实验二：消融实验 - Test R² 柱状图（均值±标准差）
  4. 实验二：消融实验 - 特征数柱状图
  5. 实验一：各模型最优组合的 Test R² 散点对比（每次run的结果）

用法：
  python scripts/experiments/plot_results.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────────
EXP_DIR = Path(__file__).resolve().parent / "hho_vs_abhho_CR_20260321_033321"
OUT_DIR = EXP_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

EXP1 = EXP_DIR / "exp1_comparison_results.csv"
EXP2 = EXP_DIR / "exp2_ablation_results.csv"

# ── 样式 ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLORS = {
    "HHO":      "#4C72B0",
    "ABHHO":    "#DD8452",
    "ABHHO_I1": "#55A868",
    "ABHHO_I2": "#C44E52",
    "ABHHO_I3": "#8172B2",
}
MODELS = ["PLS", "SVM", "RF"]


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def savefig(name):
    path = OUT_DIR / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path.name}")


# =============================================================================
# 图1：实验一 Test R² 箱线图（3个子图，每个模型一个）
# =============================================================================
def plot_exp1_boxplot_r2(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    fig.suptitle("Exp1: HHO vs AB-HHO — Test R² (30 runs)", fontsize=13, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        data = [sub[sub["variant"] == v]["test_r2"].values for v in ["HHO", "ABHHO"]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", linewidth=2))
        for patch, v in zip(bp["boxes"], ["HHO", "ABHHO"]):
            patch.set_facecolor(COLORS[v])
            patch.set_alpha(0.85)
        ax.set_title(model, fontsize=11)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["HHO", "AB-HHO"])
        ax.set_ylabel("Test R²" if model == "PLS" else "")

    plt.tight_layout()
    savefig("exp1_test_r2_boxplot.png")


# =============================================================================
# 图2：实验一 Test RMSE 箱线图
# =============================================================================
def plot_exp1_boxplot_rmse(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    fig.suptitle("Exp1: HHO vs AB-HHO — Test RMSE (30 runs)", fontsize=13, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        data = [sub[sub["variant"] == v]["test_rmse"].values for v in ["HHO", "ABHHO"]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", linewidth=2))
        for patch, v in zip(bp["boxes"], ["HHO", "ABHHO"]):
            patch.set_facecolor(COLORS[v])
            patch.set_alpha(0.85)
        ax.set_title(model, fontsize=11)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["HHO", "AB-HHO"])
        ax.set_ylabel("Test RMSE" if model == "PLS" else "")

    plt.tight_layout()
    savefig("exp1_test_rmse_boxplot.png")


# =============================================================================
# 图3：实验一 特征数箱线图
# =============================================================================
def plot_exp1_nfeat(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    fig.suptitle("Exp1: HHO vs AB-HHO — # Selected Features (30 runs)", fontsize=13, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        data = [sub[sub["variant"] == v]["n_selected"].values for v in ["HHO", "ABHHO"]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", linewidth=2))
        for patch, v in zip(bp["boxes"], ["HHO", "ABHHO"]):
            patch.set_facecolor(COLORS[v])
            patch.set_alpha(0.85)
        ax.set_title(model, fontsize=11)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["HHO", "AB-HHO"])
        ax.set_ylabel("# Features" if model == "PLS" else "")

    plt.tight_layout()
    savefig("exp1_nfeat_boxplot.png")


# =============================================================================
# 图4：实验一 每次run的 Test R² 散点对比（HHO vs ABHHO，按模型）
# =============================================================================
def plot_exp1_scatter(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Exp1: Per-run Test R² — HHO vs AB-HHO", fontsize=13, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        hho   = df[(df["variant"] == "HHO")   & (df["model"] == model)]["test_r2"].values
        abhho = df[(df["variant"] == "ABHHO") & (df["model"] == model)]["test_r2"].values
        n = min(len(hho), len(abhho))
        ax.scatter(hho[:n], abhho[:n], alpha=0.7, edgecolors="white",
                   linewidths=0.5, s=50, color=COLORS["ABHHO"])
        lims = [min(hho.min(), abhho.min()) - 0.02,
                max(hho.max(), abhho.max()) + 0.02]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="y=x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("HHO Test R²")
        ax.set_ylabel("AB-HHO Test R²" if model == "PLS" else "")
        ax.set_title(model, fontsize=11)
        # 标注 AB-HHO 胜出比例
        win_rate = (abhho[:n] > hho[:n]).mean()
        ax.text(0.05, 0.92, f"AB-HHO wins: {win_rate:.0%}",
                transform=ax.transAxes, fontsize=9, color=COLORS["ABHHO"])

    plt.tight_layout()
    savefig("exp1_scatter_r2.png")


# =============================================================================
# 图5：实验二 消融 Test R² 柱状图（均值±std，按模型分组）
# =============================================================================
def plot_exp2_bar_r2(df):
    variants = ["HHO", "ABHHO_I1", "ABHHO_I2", "ABHHO_I3", "ABHHO"]
    labels   = ["HHO", "I1", "I2", "I3", "AB-HHO"]
    x = np.arange(len(MODELS))
    width = 0.15
    offsets = np.linspace(-(len(variants)-1)/2, (len(variants)-1)/2, len(variants)) * width

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Exp2: Ablation — Test R² (mean ± std, 20 runs)", fontsize=13, fontweight="bold")

    for i, (v, lbl) in enumerate(zip(variants, labels)):
        means, stds = [], []
        for model in MODELS:
            sub = df[(df["variant"] == v) & (df["model"] == model)]["test_r2"]
            means.append(sub.mean())
            stds.append(sub.std())
        bars = ax.bar(x + offsets[i], means, width, yerr=stds,
                      label=lbl, color=COLORS[v], alpha=0.85,
                      capsize=3, error_kw=dict(linewidth=1))

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Test R²")
    ax.legend(title="Variant", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    savefig("exp2_ablation_r2_bar.png")


# =============================================================================
# 图6：实验二 消融特征数柱状图
# =============================================================================
def plot_exp2_bar_nfeat(df):
    variants = ["HHO", "ABHHO_I1", "ABHHO_I2", "ABHHO_I3", "ABHHO"]
    labels   = ["HHO", "I1", "I2", "I3", "AB-HHO"]
    x = np.arange(len(MODELS))
    width = 0.15
    offsets = np.linspace(-(len(variants)-1)/2, (len(variants)-1)/2, len(variants)) * width

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Exp2: Ablation — # Selected Features (mean ± std, 20 runs)", fontsize=13, fontweight="bold")

    for i, (v, lbl) in enumerate(zip(variants, labels)):
        means, stds = [], []
        for model in MODELS:
            sub = df[(df["variant"] == v) & (df["model"] == model)]["n_selected"]
            means.append(sub.mean())
            stds.append(sub.std())
        ax.bar(x + offsets[i], means, width, yerr=stds,
               label=lbl, color=COLORS[v], alpha=0.85,
               capsize=3, error_kw=dict(linewidth=1))

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("# Selected Features")
    ax.legend(title="Variant", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    savefig("exp2_ablation_nfeat_bar.png")


# =============================================================================
# 图7：实验二 消融 Test RMSE 柱状图
# =============================================================================
def plot_exp2_bar_rmse(df):
    variants = ["HHO", "ABHHO_I1", "ABHHO_I2", "ABHHO_I3", "ABHHO"]
    labels   = ["HHO", "I1", "I2", "I3", "AB-HHO"]
    x = np.arange(len(MODELS))
    width = 0.15
    offsets = np.linspace(-(len(variants)-1)/2, (len(variants)-1)/2, len(variants)) * width

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Exp2: Ablation — Test RMSE (mean ± std, 20 runs)", fontsize=13, fontweight="bold")

    for i, (v, lbl) in enumerate(zip(variants, labels)):
        means, stds = [], []
        for model in MODELS:
            sub = df[(df["variant"] == v) & (df["model"] == model)]["test_rmse"]
            means.append(sub.mean())
            stds.append(sub.std())
        ax.bar(x + offsets[i], means, width, yerr=stds,
               label=lbl, color=COLORS[v], alpha=0.85,
               capsize=3, error_kw=dict(linewidth=1))

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Test RMSE")
    ax.legend(title="Variant", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    savefig("exp2_ablation_rmse_bar.png")


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    print("Loading data...")
    df1 = pd.read_csv(EXP1)
    df2 = pd.read_csv(EXP2)

    print("Generating plots...")
    plot_exp1_boxplot_r2(df1)
    plot_exp1_boxplot_rmse(df1)
    plot_exp1_nfeat(df1)
    plot_exp1_scatter(df1)
    plot_exp2_bar_r2(df2)
    plot_exp2_bar_nfeat(df2)
    plot_exp2_bar_rmse(df2)

    print(f"\nAll figures saved to: {OUT_DIR}")
