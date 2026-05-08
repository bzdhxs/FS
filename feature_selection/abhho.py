"""AB-HHO (Adaptive Binary HHO) feature selection.

三个改进点：
  改进一（搜索机制）：Tent 混沌初始化 + 非线性逃逸能量调度
  改进二（精英记忆）：衰减记忆表 + 质量加权 + 融合引导 + 停滞触发局部扰动
  改进三（评价机制）：归一化 RMSE + 特征比例的线性加权适应度函数
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from mealpy import FloatVar

from core.registry import register_algorithm
from core.constants import (
    FITNESS_PENALTY_DEFAULT,
    MAX_PLS_COMPONENTS,
    DEFAULT_RANDOM_STATE,
)
from feature_selection.base import BaseFeatureSelector, SelectionResult
from improve.ABHHO import AdaptiveBinaryHHO
from utils.data_split import regression_stratified_split


@register_algorithm("ABHHO")
class ABHHOSelector(BaseFeatureSelector):
    """
    Adaptive Binary HHO 特征选择器。

    改进一：Tent 混沌初始化 + 非线性逃逸能量（improve/ABHHO.py）
    改进二：精英记忆引导 + 停滞触发局部扰动（improve/ABHHO.py）
    改进三：归一化 RMSE + 特征比例线性加权适应度（选择器层）

    Parameters
    ----------
    a : float, default=0.9
        适应度函数精度项权重
    gamma : float, default=2.0
        非线性逃逸能量调节参数
    rho : float, default=0.95
        记忆表遗忘因子
    elite_ratio : float, default=0.2
        精英比例
    tau : float, default=0.6
        频次阈值
    beta : float, default=0.6
        精英引导启动时机（t > beta*T）
    delta : int, default=2
        局部扰动替换特征数
    stagnation_patience : int, default=8
        停滞触发阈值
    """

    default_epoch    = 200
    default_pop_size = 50

    def __init__(self, target_col, band_range, logger=None, **kwargs):
        super().__init__(target_col, band_range, logger)
        self.epoch    = kwargs.get('epoch',    self.default_epoch)
        self.pop_size = kwargs.get('pop_size', self.default_pop_size)
        self.a        = kwargs.get('a',        0.9)
        self.gamma    = kwargs.get('gamma',    2.0)
        # 改进二参数
        self.rho                 = kwargs.get('rho',                 0.95)
        self.elite_ratio         = kwargs.get('elite_ratio',         0.2)
        self.tau                 = kwargs.get('tau',                 0.6)
        self.beta                = kwargs.get('beta',                0.6)
        self.delta               = kwargs.get('delta',               2)
        self.stagnation_patience = kwargs.get('stagnation_patience', 8)
        # 消融开关
        self.enable_chaos_init       = kwargs.get('enable_chaos_init',       True)
        self.enable_nonlinear_energy = kwargs.get('enable_nonlinear_energy', True)
        self.enable_elite_memory     = kwargs.get('enable_elite_memory',     True)
        self.enable_nrmse_fitness    = kwargs.get('enable_nrmse_fitness',    True)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run_selection(self, input_path, output_path, **kwargs):
        self.logger.info(f"Loading data from: {input_path}")
        self.logger.info(
            f"Parameters: Epoch={self.epoch}, Pop={self.pop_size}, "
            f"a={self.a}, gamma={self.gamma}, rho={self.rho}, "
            f"tau={self.tau}, beta={self.beta}, delta={self.delta}"
        )

        df, X_raw, y = self.load_data(input_path)

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw)
        n_samples, n_bands = X.shape

        # y_std 只用训练集计算，避免测试集泄露
        _, _, y_tr_for_std, _ = regression_stratified_split(
            X, y, test_size=0.3, n_bins=5, random_state=DEFAULT_RANDOM_STATE
        )
        y_std = float(np.std(y_tr_for_std, ddof=1)) or 1.0

        kf = KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)

        # ------------------------------------------------------------------
        # 改进三：适应度函数
        # Fitness = 0.7 * (RMSE / y_std) + soft_penalty
        # soft_penalty 基于唯一物理波段数，区间 [30, 40]
        # ------------------------------------------------------------------
        SOFT_LOW  = 30
        SOFT_HIGH = 40
        W_RMSE    = 0.7
        feat_cols = self.feat_cols
        n_unique_max = len({c.split("_", 1)[1] if "_" in c else c for c in feat_cols})

        def _count_unique(sel_idx):
            return len({feat_cols[i].split("_", 1)[1] if "_" in feat_cols[i]
                        else feat_cols[i] for i in sel_idx})

        def soft_penalty(sel_idx):
            n_u = _count_unique(sel_idx)
            if n_u < SOFT_LOW:
                return (1 - W_RMSE) * (SOFT_LOW - n_u) / SOFT_LOW
            if n_u > SOFT_HIGH:
                return (1 - W_RMSE) * (n_u - SOFT_HIGH) / n_unique_max
            return 0.0

        enable_nrmse = self.enable_nrmse_fitness

        def fitness_function(solution):
            sel_idx = np.where(solution > 0.5)[0]

            if len(sel_idx) == 0:
                return FITNESS_PENALTY_DEFAULT

            try:
                n_comp = min(len(sel_idx), MAX_PLS_COMPONENTS)
                rmse_scores = []

                for train_idx, val_idx in kf.split(X):
                    X_tr  = X[train_idx][:, sel_idx]
                    y_tr  = y[train_idx]
                    X_val = X[val_idx][:, sel_idx]
                    y_val = y[val_idx]

                    mdl = PLSRegression(n_components=n_comp)
                    mdl.fit(X_tr, y_tr)
                    y_pred = mdl.predict(X_val).flatten()
                    rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

                rmse = np.mean(rmse_scores)

                if enable_nrmse:
                    fitness = W_RMSE * (rmse / y_std) + soft_penalty(sel_idx)
                else:
                    # 消融退化：(1-R²) + 0.2*ratio
                    from sklearn.metrics import r2_score
                    r2_list = []
                    for train_idx, val_idx in kf.split(X):
                        m = PLSRegression(n_components=n_comp)
                        m.fit(X[train_idx][:, sel_idx], y[train_idx])
                        r2_list.append(r2_score(
                            y[val_idx], m.predict(X[val_idx][:, sel_idx]).flatten()))
                    fitness = (1 - np.mean(r2_list)) + 0.2 * _count_unique(sel_idx) / n_unique_max
                return fitness
            except Exception:
                return FITNESS_PENALTY_DEFAULT

        # 创建优化器
        optimizer = AdaptiveBinaryHHO(
            epoch=self.epoch,
            pop_size=self.pop_size,
            gamma=self.gamma,
            rho=self.rho,
            elite_ratio=self.elite_ratio,
            tau=self.tau,
            beta=self.beta,
            delta=self.delta,
            stagnation_patience=self.stagnation_patience,
            enable_chaos_init=self.enable_chaos_init,
            enable_nonlinear_energy=self.enable_nonlinear_energy,
            enable_elite_memory=self.enable_elite_memory,
        )

        problem_dict = {
            "obj_func": fitness_function,
            "bounds":   FloatVar(lb=[0.0] * n_bands, ub=[1.0] * n_bands),
            "minmax":   "min",
            "log_to":   None,
        }

        self.logger.info("Starting AB-HHO optimization...")
        agent = optimizer.solve(problem_dict)
        self._last_optimizer = optimizer

        # 提取最优二值解（固定阈值 0.5）
        best_pos  = agent.solution
        best_idx  = np.where(best_pos > 0.5)[0]
        best_feats = [self.feat_cols[i] for i in best_idx]

        fit_val = getattr(agent.target, 'fitness', 'N/A')
        self.logger.info(f"Best fitness: {fit_val:.6f}")
        self.logger.info(f"Selected features: {len(best_feats)} / {n_bands}")

        self.save_selection_result(df, y, best_feats, output_path)

        return SelectionResult(
            selected_features=best_feats,
            selected_indices=best_idx.tolist()
        )


# =============================================================================
# 消融实验变体
# =============================================================================

@register_algorithm("ABHHO_I1")
class ABHHO_I1_Selector(ABHHOSelector):
    """消融变体：仅改进一（Tent 初始化 + 非线性能量）。"""
    def __init__(self, target_col, band_range, logger=None, **kwargs):
        kwargs.setdefault('enable_chaos_init',       True)
        kwargs.setdefault('enable_nonlinear_energy', True)
        kwargs.setdefault('enable_elite_memory',     False)
        kwargs.setdefault('enable_nrmse_fitness',    False)
        super().__init__(target_col, band_range, logger, **kwargs)


@register_algorithm("ABHHO_I2")
class ABHHO_I2_Selector(ABHHOSelector):
    """消融变体：仅改进二（精英记忆引导 + 局部扰动）。"""
    def __init__(self, target_col, band_range, logger=None, **kwargs):
        kwargs.setdefault('enable_chaos_init',       False)
        kwargs.setdefault('enable_nonlinear_energy', False)
        kwargs.setdefault('enable_elite_memory',     True)
        kwargs.setdefault('enable_nrmse_fitness',    False)
        super().__init__(target_col, band_range, logger, **kwargs)


@register_algorithm("ABHHO_I3")
class ABHHO_I3_Selector(ABHHOSelector):
    """消融变体：仅改进三（NRMSE 适应度）。"""
    def __init__(self, target_col, band_range, logger=None, **kwargs):
        kwargs.setdefault('enable_chaos_init',       False)
        kwargs.setdefault('enable_nonlinear_energy', False)
        kwargs.setdefault('enable_elite_memory',     False)
        kwargs.setdefault('enable_nrmse_fitness',    True)
        super().__init__(target_col, band_range, logger, **kwargs)
