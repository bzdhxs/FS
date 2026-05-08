"""
HHO vs AB-HHO 对比实验（融合数据集 Raw+FOD+CR）

数据源：resource/fusion/fusion_dataset.csv（448 维特征）
每次 run 内部做 stratified split，避免固定 train/test 泄露。

Fitness：0.7 * (RMSE / y_std_train) + soft_penalty[40, 80]
  - y_std 只用训练集计算
  - 特征数在 [40, 80] 内惩罚为 0，超出线性惩罚

诊断记录：每次 run 额外记录 best_fitness、raw/fod/cr 各类型选中波段数

用法：
  python scripts/experiments/run_abhho_fusion.py
  python scripts/experiments/run_abhho_fusion.py --runs 15 --epoch 200 --pop 50
  python scripts/experiments/run_abhho_fusion.py --fast   # epoch=50 pop=30 runs=3 快速验证
"""

import argparse
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import feature_selection  # noqa: F401
from core.constants import DEFAULT_RANDOM_STATE, FITNESS_PENALTY_DEFAULT, MAX_PLS_COMPONENTS
from utils.data_split import regression_stratified_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 配置
# =============================================================================

FUSION_CSV = PROJECT_ROOT / "resource" / "fusion" / "fusion_dataset.csv"
TARGET_COL = "TS"

EPOCH    = 200
POP_SIZE = 50
N_RUNS   = 15

ALGOS  = ["HHO", "ABHHO"]
MODELS = ["PLSR", "SVM", "RF"]

SOFT_LOW  = 30   # 目标唯一波段数下界（ABHHO 改进目标）
SOFT_HIGH = 40   # 目标唯一波段数上界
W_RMSE    = 0.7

OPTUNA_TRIALS = {"PLSR": 50, "SVM": 200, "RF": 200}

# =============================================================================
# 数据加载
# =============================================================================

def load_fusion_data():
    df = pd.read_csv(FUSION_CSV)
    drop_cols = [c for c in ["EC", "Lat", "Lon", "id"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    y = df[TARGET_COL].values.astype(float)
    feat_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feat_cols].values.astype(float)
    return X, y, feat_cols


# =============================================================================
# Fitness（软约束版，供特征选择器内部使用）
# =============================================================================

def count_unique_bands(sel_idx: np.ndarray, feat_cols: list) -> int:
    """将选中的特征列折叠为唯一物理波段数。
    raw_b50 / fod_b50 / cr_b50 均对应波段 50，只算 1 个。
    """
    bands = set()
    for i in sel_idx:
        col = feat_cols[i]          # e.g. "raw_b50", "fod_b14", "cr_b130"
        # 取下划线后的波段编号部分，如 "b50"
        bands.add(col.split("_", 1)[1])
    return len(bands)


def make_fitness(X: np.ndarray, y: np.ndarray, y_std: float,
                 n_dims: int, feat_cols: list):
    """返回 soft_penalty fitness 闭包，绑定训练集数据。
    soft_penalty 基于唯一物理波段数，区间 [SOFT_LOW, SOFT_HIGH]。
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    # 最大唯一波段数（用于归一化惩罚上界）
    n_unique_max = len({c.split("_", 1)[1] for c in feat_cols})

    def soft_pen(n_unique):
        if n_unique < SOFT_LOW:
            return (1 - W_RMSE) * (SOFT_LOW - n_unique) / SOFT_LOW
        if n_unique > SOFT_HIGH:
            return (1 - W_RMSE) * (n_unique - SOFT_HIGH) / n_unique_max
        return 0.0

    def fitness_fn(solution):
        sel_idx = np.where(solution > 0.5)[0]
        if len(sel_idx) == 0:
            return FITNESS_PENALTY_DEFAULT
        try:
            n_unique = count_unique_bands(sel_idx, feat_cols)
            n_comp = min(len(sel_idx), MAX_PLS_COMPONENTS)
            rmse_list = []
            for tr_idx, val_idx in kf.split(X):
                mdl = PLSRegression(n_components=n_comp)
                mdl.fit(X[tr_idx][:, sel_idx], y[tr_idx])
                pred = mdl.predict(X[val_idx][:, sel_idx]).flatten()
                rmse_list.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
            return W_RMSE * (np.mean(rmse_list) / y_std) + soft_pen(n_unique)
        except Exception:
            return FITNESS_PENALTY_DEFAULT

    return fitness_fn


# =============================================================================
# 特征选择（直接调用 mealpy 优化器，绕过 selector 的 CSV 接口）
# =============================================================================

def run_feature_selection(algo: str, X_tr: np.ndarray, y_tr: np.ndarray,
                          epoch: int, pop_size: int,
                          run_seed: int, y_std: float,
                          feat_cols: list) -> np.ndarray:
    """返回选中特征的列索引数组。"""
    from mealpy import FloatVar
    from improve.ABHHO import AdaptiveBinaryHHO
    from mealpy.swarm_based.HHO import OriginalHHO

    n_dims = X_tr.shape[1]
    fitness_fn = make_fitness(X_tr, y_tr, y_std, n_dims, feat_cols)

    problem_dict = {
        "obj_func": fitness_fn,
        "bounds":   FloatVar(lb=[0.0] * n_dims, ub=[1.0] * n_dims),
        "minmax":   "min",
        "log_to":   None,
        "seed":     run_seed,
    }

    if algo == "HHO":
        optimizer = OriginalHHO(epoch=epoch, pop_size=pop_size)
    elif algo == "ABHHO":
        optimizer = AdaptiveBinaryHHO(
            epoch=epoch, pop_size=pop_size,
            gamma=2.0, rho=0.95, elite_ratio=0.2,
            tau=0.6, beta=0.6, delta=2, stagnation_patience=8,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    agent = optimizer.solve(problem_dict)
    best_fitness = float(getattr(agent.target, "fitness", np.nan))
    sel_idx = np.where(agent.solution > 0.5)[0]
    return sel_idx, best_fitness


# =============================================================================
# 下游建模
# =============================================================================

def train_model(model_name: str, X_tr: np.ndarray, y_tr: np.ndarray,
                X_te: np.ndarray, n_trials: int, cv_seed: int):
    cv = KFold(n_splits=5, shuffle=True, random_state=cv_seed)

    if model_name == "PLSR":
        max_comp = max(1, min(X_tr.shape[1], X_tr.shape[0] - 1, 20))

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
        y_pred_tr = np.clip(final.predict(X_tr).flatten(), y_tr.min() - 3 * y_tr.std(), y_tr.max() + 3 * y_tr.std())
        y_pred_te = np.clip(final.predict(X_te).flatten(), y_tr.min() - 3 * y_tr.std(), y_tr.max() + 3 * y_tr.std())
        return y_pred_tr, y_pred_te, bp

    elif model_name == "SVM":
        def obj(trial):
            C       = trial.suggest_float("C",       1e-2, 1e3, log=True)
            gamma   = trial.suggest_float("gamma",   1e-4, 1e1, log=True)
            epsilon = trial.suggest_float("epsilon",  1e-3, 1.0, log=True)
            kernel  = trial.suggest_categorical("kernel", ["rbf", "linear"])
            s = cross_val_score(
                SVR(C=C, gamma=gamma, epsilon=epsilon, kernel=kernel),
                X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
            return -s.mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=cv_seed))
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        bp    = study.best_params
        final = SVR(**bp)
        final.fit(X_tr, y_tr)
        return final.predict(X_tr), final.predict(X_te), bp

    elif model_name == "RF":
        def obj(trial):
            params = {
                "n_estimators":    trial.suggest_int("n_estimators",    10, 300),
                "max_depth":       trial.suggest_int("max_depth",        2,  15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf",  1,  5),
                "max_features":    trial.suggest_categorical("max_features",
                                       [0.3, 0.5, 0.7, "sqrt", "log2"]),
            }
            s = cross_val_score(RandomForestRegressor(**params, random_state=cv_seed),
                                X_tr, y_tr, cv=cv,
                                scoring="neg_root_mean_squared_error")
            return -s.mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=cv_seed))
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        bp    = study.best_params
        final = RandomForestRegressor(**bp, random_state=cv_seed)
        final.fit(X_tr, y_tr)
        return final.predict(X_tr), final.predict(X_te), bp

    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# 单次 run
# =============================================================================

def run_once(algo: str, model_name: str,
             X: np.ndarray, y: np.ndarray, feat_cols: list,
             run_id: int, epoch: int, pop_size: int,
             n_trials: int, logger: logging.Logger) -> dict | None:

    seed = DEFAULT_RANDOM_STATE + run_id * 7

    # 划分（测试集不可见）
    X_tr, X_te, y_tr, y_te = regression_stratified_split(
        X, y, test_size=0.3, n_bins=5, random_state=seed
    )

    # y_std 只用训练集
    y_std_tr = float(y_tr.std(ddof=1)) or 1.0

    # 特征选择
    t0 = time.time()
    sel_idx, best_fitness = run_feature_selection(
        algo, X_tr, y_tr, epoch, pop_size, run_seed=seed, y_std=y_std_tr,
        feat_cols=feat_cols
    )
    elapsed_fs = time.time() - t0

    n_feat = len(sel_idx)
    if n_feat == 0:
        logger.warning(f"[{algo}/{model_name}] run {run_id}: 0 features selected, skip.")
        return None

    # 诊断：各变换类型列数 + 唯一物理波段数
    sel_cols  = [feat_cols[i] for i in sel_idx]
    n_raw     = sum(1 for c in sel_cols if c.startswith("raw_"))
    n_fod     = sum(1 for c in sel_cols if c.startswith("fod_"))
    n_cr      = sum(1 for c in sel_cols if c.startswith("cr_"))
    n_unique  = count_unique_bands(sel_idx, feat_cols)

    # 归一化（fit on train only）
    sc       = MinMaxScaler()
    X_tr_sel = sc.fit_transform(X_tr[:, sel_idx])
    X_te_sel = sc.transform(X_te[:, sel_idx])

    # 下游建模
    y_pred_tr, y_pred_te, best_params = train_model(
        model_name, X_tr_sel, y_tr, X_te_sel, n_trials, cv_seed=seed
    )

    # 指标
    train_r2   = r2_score(y_tr, y_pred_tr)
    train_rmse = np.sqrt(mean_squared_error(y_tr, y_pred_tr))
    train_mae  = mean_absolute_error(y_tr, y_pred_tr)
    test_r2    = r2_score(y_te, y_pred_te)
    test_rmse  = np.sqrt(mean_squared_error(y_te, y_pred_te))
    test_mae   = mean_absolute_error(y_te, y_pred_te)

    logger.info(
        f"  [{algo:6s}/{model_name:4s}] run {run_id:2d} | "
        f"bands={n_unique:3d}(raw={n_raw} fod={n_fod} cr={n_cr} cols={n_feat}) | "
        f"fitness={best_fitness:.4f} | "
        f"Train R²={train_r2:.4f} RMSE={train_rmse:.4f} | "
        f"Test  R²={test_r2:.4f} RMSE={test_rmse:.4f}"
    )

    return {
        "algo":        algo,
        "model":       model_name,
        "run":         run_id,
        "n_bands":     n_unique,   # 唯一物理波段数（主要指标）
        "n_features":  n_feat,     # 原始特征列数（含多变换重复）
        "n_raw":       n_raw,
        "n_fod":       n_fod,
        "n_cr":        n_cr,
        "best_fitness": best_fitness,
        "train_r2":    train_r2,
        "train_rmse":  train_rmse,
        "train_mae":   train_mae,
        "test_r2":     test_r2,
        "test_rmse":   test_rmse,
        "test_mae":    test_mae,
        "elapsed_fs":  elapsed_fs,
        "best_params": str(best_params),
        "selected_cols": ",".join(sel_cols),
    }


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs",  type=int, default=N_RUNS)
    p.add_argument("--epoch", type=int, default=EPOCH)
    p.add_argument("--pop",   type=int, default=POP_SIZE)
    p.add_argument("--fast",  action="store_true",
                   help="快速验证模式：epoch=50 pop=30 runs=3")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.runs, args.epoch, args.pop = 3, 50, 30

    # 输出目录
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "log" / f"hho_vs_abhho_fusion_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 日志
    log_file = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("abhho_fusion")

    # 数据
    X, y, feat_cols = load_fusion_data()
    y_std_global = float(y.std())

    logger.info("=" * 70)
    logger.info("HHO vs AB-HHO 对比实验（融合数据集 Raw+FOD+CR）")
    logger.info(f"  算法：{ALGOS}  模型：{MODELS}")
    logger.info(f"  每组运行：{args.runs}次  Epoch={args.epoch}  Pop={args.pop}")
    logger.info(f"  Fitness：{W_RMSE}·(RMSE/y_std_train) + soft_penalty[{SOFT_LOW},{SOFT_HIGH}]  CV=5折")
    logger.info(f"  Optuna trials：{OPTUNA_TRIALS}")
    logger.info(f"  结果目录：{out_dir}")
    logger.info("=" * 70)
    logger.info(f"融合数据集加载完成：{X.shape[0]}样本  {X.shape[1]}维特征  y_std={y_std_global:.4f}")

    results_csv = out_dir / "results.csv"
    all_rows: list[dict] = []
    total = len(ALGOS) * len(MODELS) * args.runs
    done  = 0

    for algo in ALGOS:
        for mname in MODELS:
            logger.info(f"\n▶ {algo} × {mname}  ({args.runs} runs)")
            for run_id in range(args.runs):
                row = run_once(
                    algo, mname, X, y, feat_cols,
                    run_id=run_id,
                    epoch=args.epoch,
                    pop_size=args.pop,
                    n_trials=OPTUNA_TRIALS[mname],
                    logger=logger,
                )
                done += 1
                if row:
                    all_rows.append(row)
                    pd.DataFrame(all_rows).to_csv(results_csv, index=False)
                logger.info(f"  进度：{done}/{total}")

    # ── 汇总统计 ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    summary_rows = []
    for algo in ALGOS:
        for mname in MODELS:
            sub = df[(df["algo"] == algo) & (df["model"] == mname)]
            if sub.empty:
                continue
            summary_rows.append({
                "algo":              algo,
                "model":             mname,
                "n_runs":            len(sub),
                "train_r2_mean":     round(sub["train_r2"].mean(), 4),
                "train_r2_std":      round(sub["train_r2"].std(),  4),
                "train_r2_best":     round(sub["train_r2"].max(),  4),
                "train_rmse_mean":   round(sub["train_rmse"].mean(), 4),
                "train_rmse_std":    round(sub["train_rmse"].std(),  4),
                "test_r2_mean":      round(sub["test_r2"].mean(), 4),
                "test_r2_std":       round(sub["test_r2"].std(),  4),
                "test_r2_best":      round(sub["test_r2"].max(),  4),
                "test_rmse_mean":    round(sub["test_rmse"].mean(), 4),
                "test_rmse_std":     round(sub["test_rmse"].std(),  4),
                "test_rmse_best":    round(sub["test_rmse"].min(),  4),
                "test_mae_mean":     round(sub["test_mae"].mean(),  4),
                "n_bands_mean":      round(sub["n_bands"].mean(), 1),   # 唯一物理波段数
                "n_bands_std":       round(sub["n_bands"].std(),  1),
                "n_features_mean":   round(sub["n_features"].mean(), 1), # 原始列数（含多变换）
                "n_raw_mean":        round(sub["n_raw"].mean(), 1),
                "n_fod_mean":        round(sub["n_fod"].mean(), 1),
                "n_cr_mean":         round(sub["n_cr"].mean(),  1),
                "best_fitness_mean": round(sub["best_fitness"].mean(), 4),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "summary.csv", index=False)

    # ── Wilcoxon 检验（HHO vs ABHHO，按 model）────────────────────────────
    wilcoxon_rows = []
    for mname in MODELS:
        hho_r2   = df[(df["algo"] == "HHO")   & (df["model"] == mname)]["test_r2"].values
        abhho_r2 = df[(df["algo"] == "ABHHO") & (df["model"] == mname)]["test_r2"].values
        if len(hho_r2) >= 5 and len(abhho_r2) >= 5:
            n = min(len(hho_r2), len(abhho_r2))
            try:
                stat, p = wilcoxon(hho_r2[:n], abhho_r2[:n])
            except Exception:
                stat, p = np.nan, np.nan
            wilcoxon_rows.append({
                "model": mname, "statistic": stat,
                "p_value": round(p, 4) if not np.isnan(p) else np.nan,
                "significant": p < 0.05 if not np.isnan(p) else False,
            })
    pd.DataFrame(wilcoxon_rows).to_csv(out_dir / "wilcoxon.csv", index=False)

    # ── 波段频次 ──────────────────────────────────────────────────────────
    freq_rows = []
    for algo in ALGOS:
        for mname in MODELS:
            sub = df[(df["algo"] == algo) & (df["model"] == mname)]
            all_bands = []
            for s in sub["selected_cols"].dropna():
                if s:
                    all_bands.extend(s.split(","))
            n_total = len(sub)
            for band, cnt in sorted(Counter(all_bands).items(), key=lambda x: -x[1]):
                freq_rows.append({
                    "algo": algo, "model": mname, "band": band,
                    "count": cnt, "frequency": round(cnt / n_total, 3),
                })
    pd.DataFrame(freq_rows).to_csv(out_dir / "band_frequency.csv", index=False)

    # ── 控制台打印汇总 ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 90)
    logger.info("汇总结果")
    logger.info("=" * 90)
    logger.info(f"{'algo':<8} {'model':<6} {'Test R²(mean±std)':<22} {'Test RMSE(mean)':<18} "
                f"{'n_bands(mean)':<15} {'raw/fod/cr cols'}")
    logger.info("-" * 90)
    for r in summary_rows:
        logger.info(
            f"{r['algo']:<8} {r['model']:<6} "
            f"{r['test_r2_mean']:.4f}±{r['test_r2_std']:.4f}        "
            f"{r['test_rmse_mean']:<18.4f}"
            f"{r['n_bands_mean']:<15.1f}"
            f"{r['n_raw_mean']:.1f}/{r['n_fod_mean']:.1f}/{r['n_cr_mean']:.1f}"
        )
    logger.info(f"\n全部完成，结果目录：{out_dir}")


if __name__ == "__main__":
    main()
