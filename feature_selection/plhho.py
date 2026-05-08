"""PL-HHO (Lévy-enhanced HHO) feature selection plugin.

改进策略：
  改进一：Tent 混沌 + OBL 双重初始化
  改进二：幂律 × 余弦周期性逃逸能量
  改进三：时变柯西连续扰动 rabbit
  改进四：停滞触发翻转变异
  改进五：精英记忆衰减表 + 软引导
"""

import numpy as np
import logging
from typing import Optional, Tuple

from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from mealpy import FloatVar

from core.registry import register_algorithm
from core.constants import (
    BINARY_THRESHOLD,
    FITNESS_PENALTY_DEFAULT,
    MAX_PLS_COMPONENTS,
    DEFAULT_RANDOM_STATE,
)
from feature_selection.base import BaseFeatureSelector, SelectionResult
from improve.PLHHO import PLHarrisHawks
from utils.data_split import regression_stratified_split


@register_algorithm("PLHHO")
class PLHHOSelector(BaseFeatureSelector):
    """PL-HHO feature selector.

    Parameters
    ----------
    target_col : str
    band_range : tuple
    logger : logging.Logger, optional
    epoch : int, default=200
    pop_size : int, default=50
    gamma : float, default=2.0
    n_periods : int, default=1
    cauchy_c0 : float, default=0.3
    rho : float, default=0.95
    elite_ratio : float, default=0.2
    tau : float, default=0.6
    beta : float, default=0.6
    flip_ratio : float, default=0.1
    stagnation_patience : int, default=8
    guide_strength : float, default=0.3
    """

    default_epoch               = 200
    default_pop_size            = 50
    default_gamma               = 2.0
    default_n_periods           = 1
    default_cauchy_c0           = 0.3
    default_rho                 = 0.95
    default_elite_ratio         = 0.2
    default_tau                 = 0.6
    default_beta                = 0.6
    default_flip_ratio          = 0.1
    default_stagnation_patience = 8
    default_guide_strength      = 0.3

    def __init__(self, target_col: str, band_range: Tuple[int, int],
                 logger: Optional[logging.Logger] = None, **kwargs):
        super().__init__(target_col, band_range, logger)

        self.epoch               = kwargs.get("epoch",               self.default_epoch)
        self.pop_size            = kwargs.get("pop_size",            self.default_pop_size)
        self.gamma               = kwargs.get("gamma",               self.default_gamma)
        self.n_periods           = kwargs.get("n_periods",           self.default_n_periods)
        self.cauchy_c0           = kwargs.get("cauchy_c0",           self.default_cauchy_c0)
        self.rho                 = kwargs.get("rho",                 self.default_rho)
        self.elite_ratio         = kwargs.get("elite_ratio",         self.default_elite_ratio)
        self.tau                 = kwargs.get("tau",                 self.default_tau)
        self.beta                = kwargs.get("beta",                self.default_beta)
        self.flip_ratio          = kwargs.get("flip_ratio",          self.default_flip_ratio)
        self.stagnation_patience = kwargs.get("stagnation_patience", self.default_stagnation_patience)
        self.guide_strength      = kwargs.get("guide_strength",      self.default_guide_strength)

        self.enable_chaos_init      = kwargs.get("enable_chaos_init",      True)
        self.enable_obl             = kwargs.get("enable_obl",             True)
        self.enable_periodic_energy = kwargs.get("enable_periodic_energy", True)
        self.enable_cauchy          = kwargs.get("enable_cauchy",          True)
        self.enable_elite_memory    = kwargs.get("enable_elite_memory",    True)

    # ------------------------------------------------------------------
    # 适应度函数（PLSR RMSE 归一化 + 软约束，与实验脚本保持一致）
    # ------------------------------------------------------------------

    def _make_fitness(self, X: np.ndarray, y: np.ndarray, feat_cols: list,
                      y_std: float = None):
        """fit = 0.7·(RMSE/y_std) + soft_penalty[目标唯一波段数区间 30~40]
        soft_penalty 基于唯一物理波段数（raw_b50/fod_b50/cr_b50 算同一个波段）。
        y_std 应由调用方传入训练集的 std，避免测试集泄露。
        """
        kf    = KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        y_std = float(y_std) if y_std is not None else (float(y.std(ddof=1)) or 1.0)
        n_dims = X.shape[1]
        TARGET_MIN, TARGET_MAX = 30, 40
        ALPHA, PW = 0.7, 0.3
        # 最大唯一波段数（用于归一化惩罚上界）
        n_unique_max = len({c.split("_", 1)[1] if "_" in c else c for c in feat_cols})

        def _count_unique(sel_idx):
            return len({feat_cols[i].split("_", 1)[1] if "_" in feat_cols[i]
                        else feat_cols[i] for i in sel_idx})

        def fitness_function(solution):
            sel_idx  = np.where(solution > BINARY_THRESHOLD)[0]
            if len(sel_idx) == 0:
                return FITNESS_PENALTY_DEFAULT
            try:
                n_unique   = _count_unique(sel_idx)
                n_comp     = min(len(sel_idx), MAX_PLS_COMPONENTS)
                rmse_list  = []
                for tr_idx, val_idx in kf.split(X):
                    sc   = MinMaxScaler()
                    X_tr = sc.fit_transform(X[tr_idx][:, sel_idx])
                    X_vl = sc.transform(X[val_idx][:, sel_idx])
                    m    = PLSRegression(n_components=n_comp)
                    m.fit(X_tr, y[tr_idx])
                    pred = m.predict(X_vl).flatten()
                    rmse_list.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
                rmse_norm = np.mean(rmse_list) / y_std

                if n_unique < TARGET_MIN:
                    penalty = PW * (TARGET_MIN - n_unique) / TARGET_MIN
                elif n_unique > TARGET_MAX:
                    penalty = PW * (n_unique - TARGET_MAX) / n_unique_max
                else:
                    penalty = 0.0

                return ALPHA * rmse_norm + penalty
            except Exception:
                return FITNESS_PENALTY_DEFAULT

        return fitness_function

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run_selection(self, input_path: str, output_path: str, **kwargs) -> SelectionResult:
        self.logger.info(f"[PL-HHO] Loading data: {input_path}")
        self.logger.info(
            f"[PL-HHO] epoch={self.epoch}  pop={self.pop_size}  "
            f"gamma={self.gamma}  n_periods={self.n_periods}  "
            f"cauchy_c0={self.cauchy_c0}  guide_strength={self.guide_strength}"
        )

        # ── 数据加载与预处理 ──────────────────────────────────────────
        df, X_raw, y = self.load_data(input_path)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw)

        # y_std 只用训练集计算，避免测试集泄露
        _, _, y_tr_for_std, _ = regression_stratified_split(
            X, y, test_size=0.3, n_bins=5, random_state=DEFAULT_RANDOM_STATE
        )
        y_std_train = float(np.std(y_tr_for_std, ddof=1)) or 1.0

        # ── 构建优化器 ────────────────────────────────────────────────
        optimizer = PLHarrisHawks(
            epoch=self.epoch,
            pop_size=self.pop_size,
            gamma=self.gamma,
            n_periods=self.n_periods,
            cauchy_c0=self.cauchy_c0,
            rho=self.rho,
            elite_ratio=self.elite_ratio,
            tau=self.tau,
            beta=self.beta,
            flip_ratio=self.flip_ratio,
            stagnation_patience=self.stagnation_patience,
            guide_strength=self.guide_strength,
            enable_chaos_init=self.enable_chaos_init,
            enable_obl=self.enable_obl,
            enable_periodic_energy=self.enable_periodic_energy,
            enable_cauchy=self.enable_cauchy,
            enable_elite_memory=self.enable_elite_memory,
        )

        # ── 问题定义 ──────────────────────────────────────────────────
        problem_dict = {
            "obj_func": self._make_fitness(X, y, self.feat_cols, y_std=y_std_train),
            "bounds":   FloatVar(lb=[0] * X.shape[1], ub=[1] * X.shape[1]),
            "minmax":   "min",
            "log_to":   None,
        }

        # ── 求解 ──────────────────────────────────────────────────────
        self.logger.info("[PL-HHO] Starting optimization...")
        agent = optimizer.solve(problem_dict)
        self._last_optimizer = optimizer

        # ── 提取结果 ──────────────────────────────────────────────────
        best_pos   = agent.solution
        best_idx   = np.where(best_pos > BINARY_THRESHOLD)[0]
        best_feats = [self.feat_cols[i] for i in best_idx]
        fit_val    = getattr(agent.target, "fitness", getattr(agent, "fitness", "N/A"))

        self.logger.info(f"[PL-HHO] Best fitness : {fit_val:.6f}")
        self.logger.info(f"[PL-HHO] Selected     : {len(best_feats)} / {X.shape[1]} bands")
        self.logger.info(f"[PL-HHO] Reduction    : {(1 - len(best_feats)/X.shape[1])*100:.1f}%")

        self.save_selection_result(df, y, best_feats, output_path)

        return SelectionResult(
            selected_features=best_feats,
            selected_indices=best_idx.tolist(),
        )
