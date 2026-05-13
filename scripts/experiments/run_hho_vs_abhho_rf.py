"""
HHO vs ABHHO 对比实验（融合数据集 fod04 + fod16 + cr）

数据源：resource/fod04_fod16_cr/train.csv + test.csv（448 维特征）
  - 训练集 45 样本，测试集 23 样本
  - 特征列：fod04_b14..fod04_b163, fod16_b14..fod16_b163, cr_b14..cr_b158

Fitness：RMSE_cv / y_std + 0.15 * |S_unique| / 150
  - PLSR 5-fold CV 作为快速代理评估
  - |S_unique|：唯一物理波段数（fod04_b44/fod16_b44/cr_b44 计为 1 个）
  - D_unique = 150（总物理波段数）

下游建模：RF + Optuna 超参数优化
  - 每次运行独立 Optuna 调参，避免超参数泄露

用法：
  python scripts/experiments/run_hho_vs_abhho_rf.py
  python scripts/experiments/run_hho_vs_abhho_rf.py --runs 30 --epoch 200 --pop 50
  python scripts/experiments/run_hho_vs_abhho_rf.py --fast   # 快速验证
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.constants import DEFAULT_RANDOM_STATE, FITNESS_PENALTY_DEFAULT, MAX_PLS_COMPONENTS
from utils.data_split import regression_stratified_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 配置
# =============================================================================

TRAIN_CSV = PROJECT_ROOT / "resource" / "fod04_fod16_cr" / "train.csv"
TEST_CSV  = PROJECT_ROOT / "resource" / "fod04_fod16_cr" / "test.csv"
TARGET_COL = "TS"

EPOCH    = 200
POP_SIZE = 50
N_RUNS   = 30

LAMBDA_SPARSE = 0.03
N_UNIQUE_MAX  = 150  # 总物理波段数 b14-b163

ALGOS = ["HHO", "ABHHO"]

RF_OPTUNA_TRIALS = 200


# =============================================================================
# 数据加载
# =============================================================================

def load_data():
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    meta_cols = ["id", "Lon", "Lat", TARGET_COL]
    feat_cols = [c for c in df_train.columns if c not in meta_cols]

    X_train = df_train[feat_cols].values.astype(float)
    y_train = df_train[TARGET_COL].values.astype(float)
    X_test  = df_test[feat_cols].values.astype(float)
    y_test  = df_test[TARGET_COL].values.astype(float)

    return X_train, y_train, X_test, y_test, feat_cols


# =============================================================================
# Fitness
# =============================================================================

def count_unique_bands(sel_idx, feat_cols):
    bands = set()
    for i in sel_idx:
        col = feat_cols[i]
        bands.add(col.split("_", 1)[1])
    return len(bands)


def make_fitness(X, y, y_std, feat_cols):
    kf = KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)

    def fitness_fn(solution):
        sel_idx = np.where(solution > 0.5)[0]
        if len(sel_idx) == 0:
            return FITNESS_PENALTY_DEFAULT
        try:
            n_comp = min(len(sel_idx), MAX_PLS_COMPONENTS)
            rmse_list = []
            for tr_idx, val_idx in kf.split(X):
                mdl = PLSRegression(n_components=n_comp)
                mdl.fit(X[tr_idx][:, sel_idx], y[tr_idx])
                pred = mdl.predict(X[val_idx][:, sel_idx]).flatten()
                rmse_list.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
            rmse = np.mean(rmse_list)
            n_unique = count_unique_bands(sel_idx, feat_cols)
            return (rmse / y_std) + LAMBDA_SPARSE * (n_unique / N_UNIQUE_MAX)
        except Exception:
            return FITNESS_PENALTY_DEFAULT

    return fitness_fn


# =============================================================================
# 特征选择
# =============================================================================

def run_feature_selection(algo, X_tr, y_tr, epoch, pop_size, run_seed, y_std, feat_cols):
    from mealpy import FloatVar
    from improve.ABHHO import AdaptiveBinaryHHO
    from mealpy.swarm_based.HHO import OriginalHHO

    fitness_fn = make_fitness(X_tr, y_tr, y_std, feat_cols)

    problem_dict = {
        "obj_func": fitness_fn,
        "bounds":   FloatVar(lb=[0.0] * X_tr.shape[1], ub=[1.0] * X_tr.shape[1]),
        "minmax":   "min",
        "log_to":   None,
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
# RF + Optuna
# =============================================================================

def train_rf_optuna(X_tr, y_tr, X_te, n_trials, cv_seed):
    cv = KFold(n_splits=5, shuffle=True, random_state=cv_seed)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
            "max_depth":         trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features":      trial.suggest_categorical("max_features",
                                      [0.3, 0.5, 0.7, "sqrt", "log2"]),
        }
        scores = cross_val_score(
            RandomForestRegressor(**params, random_state=cv_seed, n_jobs=-1),
            X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error"
        )
        return -scores.mean()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cv_seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    final_model = RandomForestRegressor(**best_params, random_state=cv_seed, n_jobs=-1)
    final_model.fit(X_tr, y_tr)

    y_pred_tr = final_model.predict(X_tr)
    y_pred_te = final_model.predict(X_te)

    return y_pred_tr, y_pred_te, best_params


# =============================================================================
# 单次运行
# =============================================================================

def run_once(algo, X_train, y_train, X_test, y_test, feat_cols,
             run_id, epoch, pop_size, n_trials, logger):

    seed = DEFAULT_RANDOM_STATE + run_id * 7

    # y_std 只用训练集
    y_std_tr = float(np.std(y_train, ddof=1)) or 1.0

    # 特征选择
    t0 = time.time()
    sel_idx, best_fitness = run_feature_selection(
        algo, X_train, y_train, epoch, pop_size, run_seed=seed,
        y_std=y_std_tr, feat_cols=feat_cols
    )
    elapsed_fs = time.time() - t0

    n_feat = len(sel_idx)
    if n_feat == 0:
        logger.warning(f"[{algo}] run {run_id}: 0 features selected, skip.")
        return None

    sel_cols = [feat_cols[i] for i in sel_idx]
    n_fod04 = sum(1 for c in sel_cols if c.startswith("fod04_"))
    n_fod16 = sum(1 for c in sel_cols if c.startswith("fod16_"))
    n_cr    = sum(1 for c in sel_cols if c.startswith("cr_"))
    n_unique = count_unique_bands(sel_idx, feat_cols)

    # 归一化（fit on train only）
    sc = MinMaxScaler()
    X_tr_sel = sc.fit_transform(X_train[:, sel_idx])
    X_te_sel = sc.transform(X_test[:, sel_idx])

    # RF + Optuna
    y_pred_tr, y_pred_te, best_params = train_rf_optuna(
        X_tr_sel, y_train, X_te_sel, n_trials, cv_seed=seed
    )

    train_r2   = r2_score(y_train, y_pred_tr)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_tr))
    train_mae  = mean_absolute_error(y_train, y_pred_tr)
    test_r2    = r2_score(y_test, y_pred_te)
    test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_te))
    test_mae   = mean_absolute_error(y_test, y_pred_te)

    logger.info(
        f"  [{algo:6s}] run {run_id:2d} | "
        f"bands={n_unique:3d}(fod04={n_fod04} fod16={n_fod16} cr={n_cr} cols={n_feat}) | "
        f"fitness={best_fitness:.4f} | "
        f"Train R2={train_r2:.4f} RMSE={train_rmse:.4f} | "
        f"Test  R2={test_r2:.4f} RMSE={test_rmse:.4f}"
    )

    return {
        "algo":          algo,
        "run":           run_id,
        "n_bands":       n_unique,
        "n_features":    n_feat,
        "n_fod04":       n_fod04,
        "n_fod16":       n_fod16,
        "n_cr":          n_cr,
        "best_fitness":  best_fitness,
        "train_r2":      train_r2,
        "train_rmse":    train_rmse,
        "train_mae":     train_mae,
        "test_r2":       test_r2,
        "test_rmse":     test_rmse,
        "test_mae":      test_mae,
        "elapsed_fs":    elapsed_fs,
        "best_params":   str(best_params),
        "selected_cols": ",".join(sel_cols),
    }


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="HHO vs ABHHO 对比实验（融合数据集 + RF）")
    p.add_argument("--runs",  type=int, default=N_RUNS,  help="每组重复次数")
    p.add_argument("--epoch", type=int, default=EPOCH,    help="优化迭代轮数")
    p.add_argument("--pop",   type=int, default=POP_SIZE, help="种群大小")
    p.add_argument("--trials", type=int, default=RF_OPTUNA_TRIALS, help="RF Optuna trials")
    p.add_argument("--fast",  action="store_true", help="快速验证：epoch=50 pop=30 runs=3 trials=20")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.runs, args.epoch, args.pop, args.trials = 3, 50, 30, 20

    # 输出目录
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "log" / f"hho_vs_abhho_rf_{ts}"
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
    logger = logging.getLogger("hho_vs_abhho_rf")

    # 数据
    X_train, y_train, X_test, y_test, feat_cols = load_data()

    logger.info("=" * 70)
    logger.info("HHO vs AB-HHO 对比实验（融合数据集 fod04+fod16+cr + RF）")
    logger.info(f"  算法：{ALGOS}  模型：RF（Optuna {args.trials} trials）")
    logger.info(f"  每组运行：{args.runs}次  Epoch={args.epoch}  Pop={args.pop}")
    logger.info(f"  Fitness：RMSE_cv/y_std + {LAMBDA_SPARSE}*|S_unique|/{N_UNIQUE_MAX}")
    logger.info(f"  训练集：{X_train.shape[0]}样本  测试集：{X_test.shape[0]}样本  特征：{len(feat_cols)}维")
    logger.info(f"  结果目录：{out_dir}")
    logger.info("=" * 70)

    results_csv = out_dir / "results.csv"
    all_rows = []
    total = len(ALGOS) * args.runs
    done  = 0

    for algo in ALGOS:
        logger.info(f"\n>>> {algo}  ({args.runs} runs)")
        for run_id in range(args.runs):
            row = run_once(
                algo, X_train, y_train, X_test, y_test, feat_cols,
                run_id=run_id,
                epoch=args.epoch,
                pop_size=args.pop,
                n_trials=args.trials,
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
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        summary_rows.append({
            "algo":              algo,
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
            "test_mae_mean":     round(sub["test_mae"].mean(), 4),
            "n_bands_mean":      round(sub["n_bands"].mean(), 1),
            "n_bands_std":       round(sub["n_bands"].std(),  1),
            "n_features_mean":   round(sub["n_features"].mean(), 1),
            "n_fod04_mean":      round(sub["n_fod04"].mean(), 1),
            "n_fod16_mean":      round(sub["n_fod16"].mean(), 1),
            "n_cr_mean":         round(sub["n_cr"].mean(),  1),
            "best_fitness_mean": round(sub["best_fitness"].mean(), 4),
            "elapsed_fs_mean":   round(sub["elapsed_fs"].mean(), 1),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "summary.csv", index=False)

    # ── Wilcoxon 检验 ─────────────────────────────────────────────────────
    hho_r2   = df[df["algo"] == "HHO"]["test_r2"].values
    abhho_r2 = df[df["algo"] == "ABHHO"]["test_r2"].values
    n_min = min(len(hho_r2), len(abhho_r2))
    if n_min >= 5:
        try:
            stat, p = wilcoxon(hho_r2[:n_min], abhho_r2[:n_min])
        except Exception:
            stat, p = np.nan, np.nan
        wilcoxon_result = {
            "statistic": stat,
            "p_value": round(p, 4) if not np.isnan(p) else np.nan,
            "significant": p < 0.05 if not np.isnan(p) else False,
        }
        pd.DataFrame([wilcoxon_result]).to_csv(out_dir / "wilcoxon.csv", index=False)
        logger.info(f"\nWilcoxon test: stat={stat:.4f}, p={p:.4f}, "
                    f"significant={'Yes' if not np.isnan(p) and p < 0.05 else 'No'}")

    # ── Cohen's d ──────────────────────────────────────────────────────────
    if n_min >= 2:
        diff = abhho_r2[:n_min] - hho_r2[:n_min]
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
        logger.info(f"Cohen's d = {d:.3f}")

    # ── 波段频次 ──────────────────────────────────────────────────────────
    freq_rows = []
    for algo in ALGOS:
        sub = df[df["algo"] == algo]
        all_bands = []
        for s in sub["selected_cols"].dropna():
            if s:
                all_bands.extend(s.split(","))
        n_total = len(sub)
        for band, cnt in sorted(Counter(all_bands).items(), key=lambda x: -x[1]):
            freq_rows.append({
                "algo": algo, "band": band,
                "count": cnt, "frequency": round(cnt / n_total, 3),
            })
    pd.DataFrame(freq_rows).to_csv(out_dir / "band_frequency.csv", index=False)

    # ── 控制台打印汇总 ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("汇总结果")
    logger.info("=" * 80)
    logger.info(f"{'algo':<8} {'Test R2(mean+std)':<22} {'Test RMSE(mean)':<18} "
                f"{'n_bands(mean+std)':<20} {'fod04/fod16/cr'}")
    logger.info("-" * 80)
    for r in summary_rows:
        logger.info(
            f"{r['algo']:<8} "
            f"{r['test_r2_mean']:.4f}+{r['test_r2_std']:.4f}        "
            f"{r['test_rmse_mean']:<18.4f}"
            f"{r['n_bands_mean']:.1f}+{r['n_bands_std']:.1f}          "
            f"{r['n_fod04_mean']:.1f}/{r['n_fod16_mean']:.1f}/{r['n_cr_mean']:.1f}"
        )

    logger.info(f"\n全部完成，结果目录：{out_dir}")


if __name__ == "__main__":
    main()