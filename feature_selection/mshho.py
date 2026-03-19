"""MS-HHO (Multi-Strategy HHO) feature selection.

在标准 HHO 基础上新增改进三：乘法结构冗余惩罚适应度函数。
优化器层改进（Tent+OBL 初始化、Cauchy 扰动）由 improve/MSHHO.py 实现。
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from mealpy import FloatVar

from core.registry import register_algorithm
from core.constants import (
    BINARY_THRESHOLD,
    FITNESS_PENALTY_DEFAULT,
    MAX_PLS_COMPONENTS,
    INTERNAL_VAL_SIZE,
    DEFAULT_RANDOM_STATE,
)
from feature_selection.base import BaseMealpySelector, SelectionResult
from improve.MSHHO import MultiStrategyHHO
from utils.data_split import regression_stratified_split


@register_algorithm("MSHHO")
class MSHHOSelector(BaseMealpySelector):
    """
    Multi-Strategy HHO 特征选择器。

    改进三：乘法结构冗余惩罚适应度函数
        标准适应度（加法）：F = (1-R²) + β·ratio
        改进后（乘法）：    F = (1-R²) · (1 + β·ratio) · (1 + γ·Redundancy)

        乘法结构保证冗余惩罚是相对的，不受 R² 绝对值影响，
        无论 R² 好坏，冗余惩罚始终按比例生效。

    Parameters
    ----------
    gamma_redundancy : float, default=0.3
        冗余惩罚系数，控制冗余惩罚的强度
    """

    default_epoch = 200
    default_pop_size = 50
    default_penalty = 0.2

    def __init__(self, target_col, band_range, logger=None, **kwargs):
        super().__init__(target_col, band_range, logger, **kwargs)
        # 冗余惩罚系数，可通过 config.yaml 的 algo_params 覆盖
        self.gamma_redundancy = self.extra_params.get('gamma_redundancy', 0.3)
        # 改进开关（消融实验用）
        self.enable_chaos_init = self.extra_params.get('enable_chaos_init', True)
        self.enable_cauchy = self.extra_params.get('enable_cauchy', True)
        self.enable_redundancy = self.extra_params.get('enable_redundancy', True)

    def create_optimizer(self):
        """返回 MultiStrategyHHO 优化器实例，传入改进开关。"""
        return MultiStrategyHHO(
            epoch=self.epoch,
            pop_size=self.pop_size,
            enable_chaos_init=self.enable_chaos_init,
            enable_cauchy=self.enable_cauchy,
        )

    def _precompute_correlation(self, X_raw):
        """
        预计算波段间 Pearson 相关系数绝对值矩阵（一次性，优化前执行）。

        Parameters
        ----------
        X_raw : ndarray, shape (n_samples, n_bands)
            原始（未缩放）波段数据

        Returns
        -------
        C : ndarray, shape (n_bands, n_bands)
            相关系数绝对值矩阵，对角线为 0
        """
        # np.corrcoef 返回 (n_bands, n_bands) 的相关矩阵
        C = np.abs(np.corrcoef(X_raw.T))
        np.fill_diagonal(C, 0.0)
        return C

    def _redundancy_penalty(self, sel_idx, C):
        """
        计算选中特征集的平均冗余度。

        公式：Redundancy(S) = (1/|S|²) · Σ_{i∈S} Σ_{j∈S} C[i,j]
        值域 [0, 1]，越高表示选中波段间相关性越强（越冗余）。

        Parameters
        ----------
        sel_idx : ndarray
            选中波段的索引
        C : ndarray
            预计算的相关系数矩阵

        Returns
        -------
        float
            冗余度量值
        """
        n = len(sel_idx)
        if n <= 1:
            return 0.0
        sub = C[np.ix_(sel_idx, sel_idx)]
        return sub.sum() / (n * n)

    def run_selection(self, input_path, output_path, **kwargs):
        """
        运行 MS-HHO 特征选择，使用乘法结构冗余惩罚适应度函数。
        """
        self.logger.info(f"Loading data from: {input_path}")
        self.logger.info(
            f"Parameters: Epoch={self.epoch}, Pop={self.pop_size}, "
            f"Penalty={self.penalty}, Gamma={self.gamma_redundancy}"
        )

        # 加载并预处理数据
        df, X_raw, y = self.load_data(input_path)

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw)

        # 预计算相关矩阵（使用原始数据，避免缩放影响相关系数）
        C = self._precompute_correlation(X_raw)
        self.logger.info("Band correlation matrix precomputed.")

        # 使用 5 折交叉验证评估适应度（更抗过拟合）
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)

        gamma = self.gamma_redundancy
        penalty = self.penalty
        n_bands = X.shape[1]
        enable_redundancy = self.enable_redundancy

        # 定义乘法结构适应度函数
        def fitness_function(solution):
            sel_idx = np.where(solution > BINARY_THRESHOLD)[0]

            if len(sel_idx) == 0:
                return FITNESS_PENALTY_DEFAULT

            try:
                n_comp = min(len(sel_idx), MAX_PLS_COMPONENTS)
                r2_scores = []

                # 5 折交叉验证
                for train_idx, val_idx in kf.split(X):
                    X_train_fold = X[train_idx][:, sel_idx]
                    y_train_fold = y[train_idx]
                    X_val_fold = X[val_idx][:, sel_idx]
                    y_val_fold = y[val_idx]

                    model = PLSRegression(n_components=n_comp)
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)

                    if y_pred.ndim > 1:
                        y_pred = y_pred.flatten()

                    r2_scores.append(r2_score(y_val_fold, y_pred))

                r2 = np.mean(r2_scores)  # 取 5 折平均 R²
                ratio = len(sel_idx) / n_bands

                if enable_redundancy:
                    # 乘法结构：冗余惩罚按比例生效，不受 R² 绝对值影响
                    redundancy = self._redundancy_penalty(sel_idx, C)
                    fitness = (1 - r2) * (1 + penalty * ratio) * (1 + gamma * redundancy)
                else:
                    # 退化为标准加法结构（与原始 HHO 一致）
                    fitness = (1 - r2) + (penalty * ratio)
                return fitness
            except Exception:
                return FITNESS_PENALTY_DEFAULT

        # 创建优化器
        optimizer = self.create_optimizer()

        # 配置问题
        problem_dict = {
            "obj_func": fitness_function,
            "bounds": FloatVar(lb=[0] * n_bands, ub=[1] * n_bands),
            "minmax": "min",
            "log_to": None,
        }

        # 求解
        self.logger.info("Starting MS-HHO optimization...")
        agent = optimizer.solve(problem_dict)
        self._last_optimizer = optimizer  # 暴露 history 供外部访问收敛曲线

        # 提取结果
        best_pos = agent.solution
        best_idx = np.where(best_pos > BINARY_THRESHOLD)[0]
        best_feats = [self.feat_cols[i] for i in best_idx]

        fit_val = getattr(agent.target, 'fitness', getattr(agent, 'fitness', 'N/A'))
        self.logger.info(f"Best fitness: {fit_val}")
        self.logger.info(f"Selected features: {len(best_feats)}")

        # 保存结果
        self.save_selection_result(df, y, best_feats, output_path)

        return SelectionResult(
            selected_features=best_feats,
            selected_indices=best_idx.tolist()
        )


# =============================================================================
# 消融实验变体类
# 通过开关组合控制各改进点的启用状态
# =============================================================================

@register_algorithm("MSHHO_I1")
class MSHHO_I1_Selector(MSHHOSelector):
    """消融变体：仅启用改进一（Tent+OBL 初始化），关闭改进二和改进三。"""

    def __init__(self, target_col, band_range, logger=None, **kwargs):
        kwargs.setdefault('enable_chaos_init', True)
        kwargs.setdefault('enable_cauchy', False)
        kwargs.setdefault('enable_redundancy', False)
        super().__init__(target_col, band_range, logger, **kwargs)


@register_algorithm("MSHHO_I2")
class MSHHO_I2_Selector(MSHHOSelector):
    """消融变体：仅启用改进二（Cauchy 变异扰动），关闭改进一和改进三。"""

    def __init__(self, target_col, band_range, logger=None, **kwargs):
        kwargs.setdefault('enable_chaos_init', False)
        kwargs.setdefault('enable_cauchy', True)
        kwargs.setdefault('enable_redundancy', False)
        super().__init__(target_col, band_range, logger, **kwargs)


@register_algorithm("MSHHO_I3")
class MSHHO_I3_Selector(MSHHOSelector):
    """消融变体：仅启用改进三（乘法冗余惩罚适应度），关闭改进一和改进二。"""

    def __init__(self, target_col, band_range, logger=None, **kwargs):
        kwargs.setdefault('enable_chaos_init', False)
        kwargs.setdefault('enable_cauchy', False)
        kwargs.setdefault('enable_redundancy', True)
        super().__init__(target_col, band_range, logger, **kwargs)
