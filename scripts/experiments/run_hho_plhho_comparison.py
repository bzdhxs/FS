"""HHO vs PL-HHO 对比实验（融合数据集版）

数据集：resource/fusion/fusion_dataset.csv
        Raw(150) + FOD(~148) + CR(148) 三种变换拼接，共约446维

Fitness：fit = 0.5 * RMSE_cv + 0.5 * (n_selected / D)
         10折交叉验证，对标 Tan2025 论文

实验结构：HHO / PLHHO × PLSR / SVM / RF × 10次独立运行

结果输出到 log/hho_vs_plhho_fusion_<timestamp>/
  results.csv    — 每次运行明细
  summary.csv    — 均值±标准差汇总
  wilcoxon.csv   — Wilcoxon 显著性检验
  figures/       — 可视化图表

用法：
  python scripts/experiments/run_hho_plhho_comparison.py
  python scripts/experiments/run_hho_plhho_comparison.py --runs 2 --epoch 10 --pop 20 --fast
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import feature_selection  # noqa: F401
import model              # noqa: F401

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils.data_split import regression_stratified_split
from core.constants import DEFAULT_RANDOM_STATE, MAX_PLS_COMPONENTS, BINARY_THRESHOLD

# 波长映射常量（b14 对应 350+(14-1)*4 = 402nm）
WAVELENGTH_START = 350   # nm
WAVELENGTH_STEP  = 4     # nm/band

# ══════════════════════════════════════════════════════════════════════════════
# 实验配置
# ══════════════════════════════════════════════════════════════════════════════

FUSION_FILE = str(PROJECT_ROOT / "resource" / "fusion" / "fusion_dataset.csv")
TARGET_COL  = "TS"
META_COLS   = ["id", "Lon", "Lat", "TS", "EC"]

N_RUNS   = 10
EPOCH    = 200
POP_SIZE = 50
# fitness 权重：0.7×(RMSE/y_std) + 软约束惩罚（目标唯一波段数区间 30~40）
ALPHA          = 0.7    # RMSE 归一化项权重
PENALTY_WEIGHT = 0.3    # 软约束惩罚权重
TARGET_MIN     = 30     # 目标唯一波段数下界
TARGET_MAX     = 40     # 目标唯一波段数上界
# 5折交叉验证（fitness 内部，加速）
CV_FOLDS = 5

ALGOS  = ["HHO", "PLHHO"]
MODELS = ["PLSR", "SVM", "RF"]

# Optuna 调参轮数
OPTUNA_TRIALS = {"PLSR": 50, "SVM": 200, "RF": 200}

COLORS = {
    "HHO":   "#4C72B0",
    "PLHHO": "#DD8452",
}

# ══════════════════════════════════════════════════════════════════════════════
# 参数解析
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="HHO vs PL-HHO 融合数据集对比实验")
    p.add_argument("--runs",  type=int, default=N_RUNS)
    p.add_argument("--epoch", type=int, default=EPOCH)
    p.add_argument("--pop",   type=int, default=POP_SIZE)
    p.add_argument("--fast",  action="store_true", help="Optuna trials 缩减为 10")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """读取融合数据集，返回特征矩阵和目标变量。
    注意：fusion_dataset.csv 中特征已经过全量 MinMaxScaler，
    实验中 train/test 划分后需重新 fit_transform(train)/transform(test)。
    """
    df = pd.read_csv(FUSION_FILE)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].values.astype(float)
    y = df[TARGET_COL].values
    return X, y, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# Fitness 函数（PLSR RMSE 归一化 + 软约束目标区间）
# ══════════════════════════════════════════════════════════════════════════════

def make_fitness(X: np.ndarray, y: np.ndarray, cv_seed: int, y_std: float,
                 feat_cols: list):
    """
    fit = 0.7 × (RMSE_plsr / y_std) + soft_penalty

    soft_penalty 基于唯一物理波段数（raw_b50/fod_b50/cr_b50 算同一个波段）：
      n_unique < TARGET_MIN : 0.3 × (TARGET_MIN - n_unique) / TARGET_MIN
      n_unique > TARGET_MAX : 0.3 × (n_unique - TARGET_MAX) / n_unique_max
      TARGET_MIN ≤ n_unique ≤ TARGET_MAX : 0
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=cv_seed)
    n_unique_max = len({c.split("_", 1)[1] if "_" in c else c for c in feat_cols})

    def _count_unique(sel):
        return len({feat_cols[i].split("_", 1)[1] if "_" in feat_cols[i]
                    else feat_cols[i] for i in sel})

    def fitness_fn(solution):
        sel = np.where(solution > BINARY_THRESHOLD)[0]
        if len(sel) == 0:
            return 99999.0
        try:
            n_unique  = _count_unique(sel)
            n_comp    = min(len(sel), MAX_PLS_COMPONENTS)
            rmse_list = []
            for tr, vl in kf.split(X):
                sc   = MinMaxScaler()
                X_tr = sc.fit_transform(X[tr][:, sel])
                X_vl = sc.transform(X[vl][:, sel])
                m    = PLSRegression(n_components=n_comp)
                m.fit(X_tr, y[tr])
                pred = m.predict(X_vl).flatten()
                rmse_list.append(np.sqrt(mean_squared_error(y[vl], pred)))
            rmse      = np.mean(rmse_list)
            rmse_norm = rmse / y_std

            # 软约束（基于唯一物理波段数）
            if n_unique < TARGET_MIN:
                penalty = PENALTY_WEIGHT * (TARGET_MIN - n_unique) / TARGET_MIN
            elif n_unique > TARGET_MAX:
                penalty = PENALTY_WEIGHT * (n_unique - TARGET_MAX) / n_unique_max
            else:
                penalty = 0.0

            return ALPHA * rmse_norm + penalty
        except Exception:
            return 99999.0

    return fitness_fn


# ══════════════════════════════════════════════════════════════════════════════
# 特征选择
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_selection(algo_name: str, X: np.ndarray, y: np.ndarray,
                          epoch: int, pop_size: int, run_seed: int,
                          y_std: float, feat_cols: list) -> np.ndarray:
    from mealpy import FloatVar

    fitness_fn = make_fitness(X, y, cv_seed=run_seed, y_std=y_std, feat_cols=feat_cols)
    D = X.shape[1]

    problem = {
        "obj_func": fitness_fn,
        "bounds":   FloatVar(lb=[0] * D, ub=[1] * D),
        "minmax":   "min",
        "log_to":   None,
    }

    if algo_name == "HHO":
        from mealpy.swarm_based.HHO import OriginalHHO
        opt = OriginalHHO(epoch=epoch, pop_size=pop_size)

    elif algo_name == "PLHHO":
        from improve.PLHHO import PLHarrisHawks
        opt = PLHarrisHawks(
            epoch=epoch, pop_size=pop_size,
            gamma=2.0, n_periods=1, cauchy_c0=0.2,
        )
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    agent   = opt.solve(problem)
    sel_idx = np.where(agent.solution > BINARY_THRESHOLD)[0]
    return sel_idx


# ══════════════════════════════════════════════════════════════════════════════
# 下游模型（Optuna 调参）
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model_name: str, X_tr: np.ndarray, y_tr: np.ndarray,
                X_te: np.ndarray, n_trials: int, cv_seed: int):
    cv = KFold(n_splits=5, shuffle=True, random_state=cv_seed)

    if model_name == "PLSR":
        max_comp = min(X_tr.shape[1], X_tr.shape[0] - 1, 20)
        max_comp = max(max_comp, 1)

        def obj(trial):
            nc = trial.suggest_int("n_components", 1, max_comp)
            s  = cross_val_score(PLSRegression(n_components=nc), X_tr, y_tr,
                                 cv=cv, scoring="neg_root_mean_squared_error")
            return -s.mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=cv_seed))
        study.optimize(obj, n_trials=min(n_trials, max_comp), show_progress_bar=False)
        bp    = study.best_params
        final = PLSRegression(n_components=bp["n_components"])
        final.fit(X_tr, y_tr)
        return final.predict(X_tr).flatten(), final.predict(X_te).flatten(), bp

    elif model_name == "SVM":
        def obj(trial):
            C       = trial.suggest_float("C",       1e-2, 1e3, log=True)
            gamma   = trial.suggest_float("gamma",   1e-4, 1e1, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
            kernel  = trial.suggest_categorical("kernel", ["rbf", "linear"])
            pipe = Pipeline([("sc", StandardScaler()),
                             ("svr", SVR(C=C, gamma=gamma,
                                        epsilon=epsilon, kernel=kernel))])
            s = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                scoring="neg_root_mean_squared_error")
            return -s.mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=cv_seed))
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        bp    = study.best_params
        final = Pipeline([("sc", StandardScaler()),
                          ("svr", SVR(C=bp["C"], gamma=bp["gamma"],
                                     epsilon=bp["epsilon"], kernel=bp["kernel"]))])
        final.fit(X_tr, y_tr)
        return final.predict(X_tr).flatten(), final.predict(X_te).flatten(), bp

    elif model_name == "RF":
        def obj(trial):
            rf = RandomForestRegressor(
                n_estimators  = trial.suggest_int("n_estimators",     20, 300),
                max_depth     = trial.suggest_int("max_depth",         2,  15),
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf  = trial.suggest_int("min_samples_leaf",  1,  5),
                max_features  = trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
                random_state=DEFAULT_RANDOM_STATE, n_jobs=-1,
            )
            s = cross_val_score(rf, X_tr, y_tr, cv=cv,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
            return -s.mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=cv_seed))
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        bp    = study.best_params
        final = RandomForestRegressor(**bp, random_state=DEFAULT_RANDOM_STATE, n_jobs=-1)
        final.fit(X_tr, y_tr)
        return final.predict(X_tr).flatten(), final.predict(X_te).flatten(), bp

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ══════════════════════════════════════════════════════════════════════════════
# 单次运行
# ══════════════════════════════════════════════════════════════════════════════

def run_once(algo: str, model_name: str, X: np.ndarray, y: np.ndarray,
             feat_cols: list, run_id: int, epoch: int, pop_size: int,
             n_trials: int, logger: logging.Logger) -> dict | None:

    seed = DEFAULT_RANDOM_STATE + run_id * 7

    # 数据划分
    X_tr, X_te, y_tr, y_te = regression_stratified_split(
        X, y, test_size=0.3, n_bins=5, random_state=seed
    )

    # y_std 只用训练集计算，避免测试集泄露
    y_std_tr = float(y_tr.std()) or 1.0

    # 特征选择（在训练集上做，传入 y_std_tr 和 feat_cols）
    sel_idx = run_feature_selection(algo, X_tr, y_tr, epoch, pop_size,
                                    run_seed=seed, y_std=y_std_tr,
                                    feat_cols=feat_cols)
    n_feat  = len(sel_idx)

    if n_feat == 0:
        logger.warning(f"  [{algo}/{model_name}] run {run_id}: 0 features, skip.")
        return None

    # 唯一物理波段数
    sel_cols = [feat_cols[i] for i in sel_idx]
    n_bands  = len({c.split("_", 1)[1] if "_" in c else c for c in sel_cols})
    n_raw    = sum(1 for c in sel_cols if c.startswith("raw_"))
    n_fod    = sum(1 for c in sel_cols if c.startswith("fod_"))
    n_cr     = sum(1 for c in sel_cols if c.startswith("cr_"))

    # 重新归一化（fit on train only）
    sc       = MinMaxScaler()
    X_tr_sel = sc.fit_transform(X_tr[:, sel_idx])
    X_te_sel = sc.transform(X_te[:, sel_idx])

    # 下游模型
    y_pred_tr, y_pred_te, best_params = train_model(
        model_name, X_tr_sel, y_tr, X_te_sel, n_trials, cv_seed=seed)

    # 训练集指标
    train_r2   = r2_score(y_tr, y_pred_tr)
    train_rmse = np.sqrt(mean_squared_error(y_tr, y_pred_tr))
    train_mae  = mean_absolute_error(y_tr, y_pred_tr)

    # 测试集指标
    test_r2   = r2_score(y_te, y_pred_te)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
    test_mae  = mean_absolute_error(y_te, y_pred_te)

    logger.info(
        f"  [{algo:6s}/{model_name:4s}] run {run_id:2d} | "
        f"bands={n_bands:3d}(raw={n_raw} fod={n_fod} cr={n_cr} cols={n_feat}) | "
        f"Train R²={train_r2:.4f} RMSE={train_rmse:.4f} | "
        f"Test  R²={test_r2:.4f} RMSE={test_rmse:.4f}"
    )

    return {
        "algo": algo, "model": model_name, "run": run_id,
        "n_bands":     n_bands,
        "n_features":  n_feat,
        "n_raw": n_raw, "n_fod": n_fod, "n_cr": n_cr,
        "train_r2":    train_r2,   "train_rmse": train_rmse, "train_mae": train_mae,
        "test_r2":     test_r2,    "test_rmse":  test_rmse,  "test_mae":  test_mae,
        "best_params": str(best_params),
        "selected_cols": ",".join(sel_cols),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(df: pd.DataFrame, out_dir: Path, n_runs: int):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # train/test 指标列名
    metrics_test  = ["test_r2",  "test_rmse",  "test_mae"]
    metrics_train = ["train_r2", "train_rmse", "train_mae"]
    m_labels = {
        "test_r2":   "Test R²",   "test_rmse":  "Test RMSE",  "test_mae":  "Test MAE",
        "train_r2":  "Train R²",  "train_rmse": "Train RMSE", "train_mae": "Train MAE",
    }

    # ── 图1：测试集箱线图 3×3（指标 × 模型）─────────────────────────────
    fig, axes = plt.subplots(3, len(MODELS), figsize=(14, 10), constrained_layout=True)
    fig.suptitle("HHO vs PL-HHO  ×  PLSR / SVM / RF  —  测试集\n（融合数据集 Raw+FOD+CR）",
                 fontsize=13, fontweight="bold")
    for row, metric in enumerate(metrics_test):
        for col, mname in enumerate(MODELS):
            ax = axes[row][col]
            data = [df[(df["algo"] == a) & (df["model"] == mname)][metric].dropna().values
                    for a in ALGOS]
            bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", linewidth=2))
            for patch, algo in zip(bp["boxes"], ALGOS):
                patch.set_facecolor(COLORS[algo]); patch.set_alpha(0.75)
            ax.set_title(mname, fontsize=11)
            ax.set_xticks([1, 2]); ax.set_xticklabels(ALGOS, fontsize=10)
            if col == 0:
                ax.set_ylabel(m_labels[metric], fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
    patches = [mpatches.Patch(color=COLORS[a], alpha=0.75, label=a) for a in ALGOS]
    fig.legend(handles=patches, loc="upper right", fontsize=10)
    fig.savefig(fig_dir / "fig1_test_boxplot.png", dpi=200)
    plt.close(fig)

    # ── 图2：训练集箱线图 3×3 ────────────────────────────────────────────
    fig, axes = plt.subplots(3, len(MODELS), figsize=(14, 10), constrained_layout=True)
    fig.suptitle("HHO vs PL-HHO  ×  PLSR / SVM / RF  —  训练集\n（融合数据集 Raw+FOD+CR）",
                 fontsize=13, fontweight="bold")
    for row, metric in enumerate(metrics_train):
        for col, mname in enumerate(MODELS):
            ax = axes[row][col]
            data = [df[(df["algo"] == a) & (df["model"] == mname)][metric].dropna().values
                    for a in ALGOS]
            bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", linewidth=2))
            for patch, algo in zip(bp["boxes"], ALGOS):
                patch.set_facecolor(COLORS[algo]); patch.set_alpha(0.75)
            ax.set_title(mname, fontsize=11)
            ax.set_xticks([1, 2]); ax.set_xticklabels(ALGOS, fontsize=10)
            if col == 0:
                ax.set_ylabel(m_labels[metric], fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.legend(handles=patches, loc="upper right", fontsize=10)
    fig.savefig(fig_dir / "fig2_train_boxplot.png", dpi=200)
    plt.close(fig)

    # ── 图3：Train vs Test R² 对比柱状图（每模型一组）────────────────────
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 5), sharey=False)
    fig.suptitle("Train vs Test R²  均值±标准差", fontsize=13)
    for col, mname in enumerate(MODELS):
        ax = axes[col]
        x  = np.arange(len(ALGOS))
        w  = 0.35
        for i, (split, metric) in enumerate([("Train", "train_r2"), ("Test", "test_r2")]):
            means = [df[(df["algo"] == a) & (df["model"] == mname)][metric].mean()
                     for a in ALGOS]
            stds  = [df[(df["algo"] == a) & (df["model"] == mname)][metric].std()
                     for a in ALGOS]
            bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=stds,
                          label=split, alpha=0.8, capsize=4,
                          color=["#4C72B0", "#DD8452"][i],
                          error_kw=dict(elinewidth=1.2))
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(mname, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(ALGOS, fontsize=10)
        ax.set_ylabel("R²", fontsize=10)
        ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_train_test_r2.png", dpi=200)
    plt.close(fig)

    # ── 图4：Test 均值柱状图（RMSE / MAE）────────────────────────────────
    for metric in ["test_rmse", "test_mae"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        x, w = np.arange(len(MODELS)), 0.35
        for i, algo in enumerate(ALGOS):
            means = [df[(df["algo"] == algo) & (df["model"] == m)][metric].mean()
                     for m in MODELS]
            stds  = [df[(df["algo"] == algo) & (df["model"] == m)][metric].std()
                     for m in MODELS]
            bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=stds,
                          label=algo, color=COLORS[algo], alpha=0.8,
                          capsize=4, error_kw=dict(elinewidth=1.2))
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(stds) * 0.05,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=11)
        ax.set_ylabel(m_labels[metric], fontsize=11)
        ax.set_title(f"{m_labels[metric]}  均值±标准差（{n_runs}次）", fontsize=12)
        ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig4_bar_{metric}.png", dpi=200)
        plt.close(fig)

    # ── 图5：特征数分布直方图 ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo in ALGOS:
        vals = df[df["algo"] == algo]["n_features"].dropna()
        ax.hist(vals, bins=15, alpha=0.6, label=algo,
                color=COLORS[algo], edgecolor="white")
    ax.set_xlabel("选中特征数", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.set_title("特征选择数量分布", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_feature_count.png", dpi=200)
    plt.close(fig)

    # ── 图6：Test R² 各次运行趋势 ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 4), sharey=False)
    fig.suptitle("Test R²  各次运行趋势", fontsize=13)
    for col, mname in enumerate(MODELS):
        ax = axes[col]
        for algo in ALGOS:
            sub = df[(df["algo"] == algo) & (df["model"] == mname)].sort_values("run")
            ax.plot(sub["run"], sub["test_r2"], marker="o", label=algo,
                    color=COLORS[algo], linewidth=1.5, markersize=4)
        ax.set_title(mname, fontsize=11); ax.set_xlabel("Run", fontsize=9)
        ax.set_ylabel("Test R²", fontsize=9)
        ax.legend(fontsize=8); ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_test_r2_trend.png", dpi=200)
    plt.close(fig)

    print(f"  图表已保存至 {fig_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 频次统计 + 波长映射
# ══════════════════════════════════════════════════════════════════════════════

def col_to_band_index(col_name: str) -> int:
    """从列名提取原始波段编号，如 raw_b45 / fod_b45 / cr_b45 → 45"""
    return int(col_name.split("_b")[1])


def col_to_wavelength(col_name: str) -> float:
    """波段编号 → 波长(nm)，b1 对应 350nm，步长4nm"""
    band_idx = col_to_band_index(col_name)
    return WAVELENGTH_START + (band_idx - 1) * WAVELENGTH_STEP


def col_to_transform(col_name: str) -> str:
    """从列名提取变换类型：raw / fod / cr"""
    return col_name.split("_b")[0]


def compute_band_frequency(df: pd.DataFrame, feat_cols: list,
                            out_dir: Path, n_runs: int):
    """
    统计每个波段（原始编号）在10次运行中被选中的频次。
    分别统计 HHO 和 PLHHO，输出：
      - band_frequency.csv：每个波段的频次（按变换分列）
      - fig7_band_frequency.png：频次柱状图（x轴=波长nm）
      - fig8_band_wavelength.png：选中波段在光谱上的分布热图
    """
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 只取有 selected_cols 的行（特征选择结果）
    # 每个 algo 取所有 model 的 run（因为特征选择与 model 无关，同一 run 结果相同）
    # 实际上同一 algo+run 的不同 model 特征选择结果相同，取第一个 model 去重
    freq_rows = []

    for algo in ALGOS:
        sub = df[(df["algo"] == algo) & (df["model"] == MODELS[0])].dropna(
            subset=["selected_cols"])
        if sub.empty:
            continue

        # 统计每列被选中次数
        col_counts = {c: 0 for c in feat_cols}
        total_runs = len(sub)
        for _, row in sub.iterrows():
            if pd.isna(row["selected_cols"]) or row["selected_cols"] == "":
                continue
            for col in row["selected_cols"].split(","):
                col = col.strip()
                if col in col_counts:
                    col_counts[col] += 1

        # 转为波段级频次（按变换分列）
        band_dict = {}
        for col, cnt in col_counts.items():
            b   = col_to_band_index(col)
            t   = col_to_transform(col)
            wl  = col_to_wavelength(col)
            key = b
            if key not in band_dict:
                band_dict[key] = {"band": b, "wavelength_nm": wl,
                                  "raw": 0, "fod": 0, "cr": 0, "total": 0}
            band_dict[key][t]       += cnt
            band_dict[key]["total"] += cnt

        band_df = pd.DataFrame(band_dict.values()).sort_values("band")
        band_df["algo"]      = algo
        band_df["n_runs"]    = total_runs
        band_df["freq_pct"]  = (band_df["total"] / (total_runs * 3) * 100).round(1)
        freq_rows.append(band_df)

    if not freq_rows:
        return

    freq_df = pd.concat(freq_rows, ignore_index=True)
    freq_df.to_csv(out_dir / "band_frequency.csv", index=False)

    # ── 图7：频次柱状图（x=波长，y=被选次数，按变换堆叠）────────────────
    fig, axes = plt.subplots(len(ALGOS), 1, figsize=(16, 8), sharey=False)
    if len(ALGOS) == 1:
        axes = [axes]
    transform_colors = {"raw": "#4C72B0", "fod": "#DD8452", "cr": "#55A868"}

    for ax, algo in zip(axes, ALGOS):
        sub = freq_df[freq_df["algo"] == algo]
        if sub.empty:
            continue
        wl = sub["wavelength_nm"].values
        bottom = np.zeros(len(sub))
        for t in ["raw", "fod", "cr"]:
            vals = sub[t].values
            ax.bar(wl, vals, bottom=bottom, width=3.5,
                   color=transform_colors[t], alpha=0.8, label=t.upper())
            bottom += vals
        ax.set_title(f"{algo}  波段选择频次（{sub['n_runs'].iloc[0]}次运行）",
                     fontsize=11)
        ax.set_xlabel("波长 (nm)", fontsize=10)
        ax.set_ylabel("被选次数", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        # 标注高频波段（频次 > n_runs*2）
        threshold = sub["n_runs"].iloc[0] * 2
        for _, r in sub[sub["total"] > threshold].iterrows():
            ax.annotate(f"{int(r['wavelength_nm'])}nm",
                        xy=(r["wavelength_nm"], r["total"]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color="red")

    fig.suptitle("各波段被选频次（Raw/FOD/CR 堆叠）", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_band_frequency.png", dpi=200)
    plt.close(fig)

    # ── 图8：HHO vs PLHHO 频次对比折线图 ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    for algo in ALGOS:
        sub = freq_df[freq_df["algo"] == algo].sort_values("wavelength_nm")
        if sub.empty:
            continue
        ax.plot(sub["wavelength_nm"], sub["total"],
                label=algo, color=COLORS[algo], linewidth=1.5, alpha=0.8)
        ax.fill_between(sub["wavelength_nm"], sub["total"],
                        alpha=0.15, color=COLORS[algo])
    ax.set_xlabel("波长 (nm)", fontsize=11)
    ax.set_ylabel("被选总次数", fontsize=11)
    ax.set_title("HHO vs PL-HHO  波段选择频次对比", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_band_freq_compare.png", dpi=200)
    plt.close(fig)

    # 打印高频波段摘要
    print("\n  高频波段（被选次数 > 运行次数×1.5）：")
    for algo in ALGOS:
        sub = freq_df[freq_df["algo"] == algo]
        if sub.empty:
            continue
        threshold = sub["n_runs"].iloc[0] * 1.5
        hot = sub[sub["total"] > threshold].sort_values("total", ascending=False)
        print(f"    {algo}: {len(hot)} 个波段")
        for _, r in hot.head(10).iterrows():
            print(f"      b{int(r['band'])} ({int(r['wavelength_nm'])}nm)  "
                  f"total={int(r['total'])}  raw={int(r['raw'])} fod={int(r['fod'])} cr={int(r['cr'])}")


# ══════════════════════════════════════════════════════════════════════════════
# 汇总统计
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(df: pd.DataFrame):
    rows = []
    for algo in ALGOS:
        for mname in MODELS:
            sub = df[(df["algo"] == algo) & (df["model"] == mname)]
            if sub.empty:
                continue
            row = {"algo": algo, "model": mname, "n_runs": len(sub)}
            for metric in ["train_r2", "train_rmse", "train_mae",
                           "test_r2",  "test_rmse",  "test_mae",
                           "n_bands"]:
                v = sub[metric].dropna()
                row[f"{metric}_mean"] = round(v.mean(), 4)
                row[f"{metric}_std"]  = round(v.std(),  4)
                row[f"{metric}_best"] = round(
                    v.max() if metric.endswith("r2") else v.min(), 4)
            rows.append(row)

    summary = pd.DataFrame(rows)

    # Wilcoxon 检验（PLHHO > HHO，Test R²）
    wilcoxon_rows = []
    for mname in MODELS:
        a = df[(df["algo"] == "HHO")   & (df["model"] == mname)]["test_r2"].dropna().values
        b = df[(df["algo"] == "PLHHO") & (df["model"] == mname)]["test_r2"].dropna().values
        if len(a) >= 5 and len(b) >= 5:
            try:
                stat, pval = stats.wilcoxon(b, a, alternative="greater")
                wilcoxon_rows.append({
                    "model": mname,
                    "statistic": round(stat, 4),
                    "p_value":   round(pval, 4),
                    "significant": pval < 0.05,
                })
            except Exception:
                pass

    return summary, pd.DataFrame(wilcoxon_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args     = parse_args()
    n_runs   = args.runs
    epoch    = args.epoch
    pop_size = args.pop
    trials   = {k: 10 for k in OPTUNA_TRIALS} if args.fast else OPTUNA_TRIALS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = PROJECT_ROOT / "log" / f"hho_vs_plhho_fusion_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "run.log", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("HHO vs PL-HHO 对比实验（融合数据集 Raw+FOD+CR）")
    logger.info(f"  算法：{ALGOS}  模型：{MODELS}")
    logger.info(f"  每组运行：{n_runs}次  Epoch={epoch}  Pop={pop_size}")
    logger.info(f"  Fitness：{ALPHA}·(RMSE/y_std) + soft_penalty[{TARGET_MIN},{TARGET_MAX}]  CV={CV_FOLDS}折")
    logger.info(f"  Optuna trials：{trials}")
    logger.info(f"  结果目录：{out_dir}")
    logger.info("=" * 70)

    X, y, feat_cols = load_data()
    logger.info(f"融合数据集加载完成：{X.shape[0]}样本  {X.shape[1]}维特征")

    results_csv = out_dir / "results.csv"
    all_rows: list[dict] = []
    total = len(ALGOS) * len(MODELS) * n_runs
    done  = 0

    for algo in ALGOS:
        for mname in MODELS:
            logger.info(f"\n▶ {algo} × {mname}  ({n_runs} runs)")
            for run_id in range(n_runs):
                row = run_once(
                    algo=algo, model_name=mname,
                    X=X, y=y, feat_cols=feat_cols,
                    run_id=run_id,
                    epoch=epoch, pop_size=pop_size,
                    n_trials=trials[mname],
                    logger=logger,
                )
                if row is not None:
                    all_rows.append(row)
                    pd.DataFrame(all_rows).to_csv(results_csv, index=False)
                done += 1
                logger.info(f"  进度：{done}/{total}")

    df = pd.read_csv(results_csv)

    summary, wilcoxon = build_summary(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    wilcoxon.to_csv(out_dir / "wilcoxon.csv", index=False)

    plot_results(df, out_dir, n_runs)

    # 频次统计 + 波长映射
    compute_band_frequency(df, feat_cols, out_dir, n_runs)

    # 控制台汇总
    logger.info("\n" + "=" * 80)
    logger.info("汇总结果（均值±标准差）")
    logger.info("=" * 80)
    logger.info(f"  {'算法':6s}  {'模型':4s}  {'Train R²':>12s}  {'Train RMSE':>12s}  {'Train MAE':>10s}  "
                f"{'Test R²':>10s}  {'Test RMSE':>10s}  {'Test MAE':>10s}  {'特征数':>8s}")
    logger.info("  " + "-" * 76)
    for _, row in summary.iterrows():
        logger.info(
            f"  {row['algo']:6s}  {row['model']:4s}  "
            f"{row['train_r2_mean']:.4f}±{row['train_r2_std']:.4f}  "
            f"{row['train_rmse_mean']:.4f}±{row['train_rmse_std']:.4f}  "
            f"{row['train_mae_mean']:.4f}±{row['train_mae_std']:.4f}  "
            f"{row['test_r2_mean']:.4f}±{row['test_r2_std']:.4f}  "
            f"{row['test_rmse_mean']:.4f}±{row['test_rmse_std']:.4f}  "
            f"{row['test_mae_mean']:.4f}±{row['test_mae_std']:.4f}  "
            f"{row['n_features_mean']:.1f}±{row['n_features_std']:.1f}"
        )

    if not wilcoxon.empty:
        logger.info("\nWilcoxon 检验（PLHHO > HHO，Test R²）")
        for _, row in wilcoxon.iterrows():
            sig = "✓ 显著" if row["significant"] else "✗ 不显著"
            logger.info(f"  {row['model']:4s}  p={row['p_value']:.4f}  {sig}")

    logger.info(f"\n全部完成，结果目录：{out_dir}")


if __name__ == "__main__":
    main()
