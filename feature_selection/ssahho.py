"""SSAHHO：Spectral-Structure-Aware Harris Hawks Optimization。

针对高光谱波段选择问题的四个核心缺陷设计了对应改进：
- SCI  (Spectral Coverage Initialization)：光谱覆盖感知初始化
- SDGE (Spectral Density-Guided Exploration)：光谱密度引导探索
- RSC  (Redundancy-Suppressed Capture)：冗余抑制捕获（exploitation 修正项）
- Late Refinement：后期邻域精修（first-improvement greedy）

架构说明：
    继承 BaseFeatureSelector，自己实现完整迭代循环（不依赖 mealpy），
    以便侵入迭代过程实现 SDGE/RSC/Refinement。

消融变体（通过类属性开关控制）：
    SSAHHO_HHO        - 基线：所有改进关闭
    SSAHHO_SCI        - 仅 SCI
    SSAHHO_SCI_SDGE   - SCI + SDGE
    SSAHHO_SCI_SDGE_RSC - SCI + SDGE + RSC
    SSAHHO            - 完整版（所有改进开启）
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from core.constants import (
    DEFAULT_RANDOM_STATE,
    MAX_PLS_COMPONENTS,
    WAVELENGTH_START,
    WAVELENGTH_STEP,
)
from core.registry import register_algorithm
from feature_selection.base import BaseFeatureSelector, SelectionResult
from utils.candidate_selector import (
    CandidateFeature,
    build_fusion_candidates,
    build_spectral_neighbors,
    build_spectral_windows,
    print_candidate_summary,
)
from utils.spectral_transforms import apply_all_transforms


class SSAHHOSelector(BaseFeatureSelector):
    """Spectral-Structure-Aware HHO 特征选择器（完整版）。

    继承 BaseFeatureSelector，自己实现完整 HHO 迭代循环。
    消融变体通过子类覆盖 enable_* 类属性实现。
    """

    # ── 默认超参数 ────────────────────────────────────────
    default_epoch    = 120
    default_pop_size = 60
    default_penalty  = 0.1   # 稀疏性惩罚系数（λ）
    default_gamma    = 0.1   # 冗余惩罚系数（γ）
    default_n_top    = 30    # 每种变换保留的 Top N 特征

    # ── 消融开关（子类通过类属性覆盖）────────────────────
    enable_sci        = True
    enable_sdge       = True
    enable_rsc        = True
    enable_refinement = True

    def __init__(self, target_col, band_range, logger=None, **kwargs):
        super().__init__(target_col, band_range, logger)
        self.epoch    = kwargs.get("epoch",    self.default_epoch)
        self.pop_size = kwargs.get("pop_size", self.default_pop_size)
        self.penalty  = kwargs.get("penalty",  self.default_penalty)
        self.gamma    = kwargs.get("gamma",    self.default_gamma)
        self.n_top    = kwargs.get("n_top",    self.default_n_top)

        # 内部状态（迭代时使用，供 _hho_exploitation 访问种群均值）
        self._population: Optional[np.ndarray] = None
        self._corr_matrix: Optional[np.ndarray] = None

    # ══════════════════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════════════════

    def _v_transfer(self, velocity: np.ndarray, current_binary: np.ndarray) -> np.ndarray:
        """V 形转换函数：将连续速度映射为二值位置。

        标准二值 HHO/PSO 做法：
            T(v) = |tanh(v)|
            x_new[i] = 1 - x[i]  if rand < T(v[i])
                       x[i]       otherwise

        Args:
            velocity:       连续位置向量（HHO exploration/exploitation 的输出）
            current_binary: 当前 hawk 的二值位置（0/1）

        Returns:
            新的二值位置向量
        """
        t_val   = np.abs(np.tanh(velocity))
        flip    = np.random.rand(len(velocity)) < t_val
        new_bin = current_binary.copy().astype(float)
        new_bin[flip] = 1.0 - new_bin[flip]
        return new_bin

    def _to_binary(self, x: np.ndarray) -> np.ndarray:
        """统一二值化入口，所有模块调用此方法。"""
        return (np.asarray(x) > 0.5).astype(np.uint8)

    def _levy_flight(self, dim: int, beta: float = 1.5) -> np.ndarray:
        """Lévy 飞行采样（Mantegna 算法）。"""
        sigma = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        return u / (np.abs(v) ** (1 / beta))

    # ══════════════════════════════════════════════════════
    # 初始化
    # ══════════════════════════════════════════════════════

    def _spectral_coverage_init(
        self, n_features: int, windows: List[List[int]]
    ) -> np.ndarray:
        """SCI 初始化（稀疏版）。

        目标特征数 k_init ~ Uniform(8, 15)：
        - 70% 来自覆盖窗口（每窗口最多 1 个，避免重叠）
        - 30% 从未选中特征随机补充
        """
        population = np.zeros((self.pop_size, n_features))

        for i in range(self.pop_size):
            k_init = np.random.randint(8, 16)  # [8, 15]
            selected: set = set()

            # 70% 来自覆盖窗口
            n_from_windows = max(1, int(0.7 * k_init))
            shuffled_windows = np.random.permutation(len(windows))
            for w_idx in shuffled_windows[:n_from_windows]:
                feat = np.random.choice(windows[w_idx])
                selected.add(int(feat))

            # 30% 从未选中特征随机补充
            n_from_random = k_init - len(selected)
            remaining = list(set(range(n_features)) - selected)
            if n_from_random > 0 and len(remaining) >= n_from_random:
                extra = np.random.choice(remaining, size=n_from_random, replace=False)
                selected.update(int(e) for e in extra)

            for idx in selected:
                population[i, idx] = 1.0

        return population

    def _random_init(self, n_features: int) -> np.ndarray:
        """标准随机初始化（SCI 关闭时使用）。

        每个 hawk 随机选 k ~ Uniform(8, 15) 个特征。
        """
        population = np.zeros((self.pop_size, n_features))
        for i in range(self.pop_size):
            k = np.random.randint(8, 16)
            idx = np.random.choice(n_features, size=min(k, n_features), replace=False)
            population[i, idx] = 1.0
        return population

    # ══════════════════════════════════════════════════════
    # 探索阶段
    # ══════════════════════════════════════════════════════

    def _hho_exploration(
        self,
        hawk: np.ndarray,
        rabbit: np.ndarray,
        population: np.ndarray,
    ) -> np.ndarray:
        """标准 HHO exploration（两种策略随机选一）。

        - q >= 0.5：随机 hawk 位置 + 随机跳跃
        - q <  0.5：基于 rabbit 的随机游走
        """
        q  = np.random.rand()
        r1 = np.random.rand()
        r2 = np.random.rand()

        if q >= 0.5:
            rand_hawk = population[np.random.randint(len(population))]
            new_pos = rand_hawk - r1 * np.abs(rand_hawk - 2 * r2 * hawk)
        else:
            new_pos = (
                rabbit - np.mean(population, axis=0)
                - r1 * (np.random.rand() * (1 - 0) + 0 - hawk)
            )
        return np.clip(new_pos, 0, 1)

    def _compute_window_density(
        self,
        population: np.ndarray,
        windows: List[List[int]],
    ) -> np.ndarray:
        """计算每个 window 的占用率。

        占用率 = 有多少 hawk 在此 window 中至少选了 1 个特征 / 总 hawk 数。
        """
        density = np.zeros(len(windows))
        binary_pop = self._to_binary(population)  # (pop_size, n_features)
        for i, window in enumerate(windows):
            # 每个 hawk 在此 window 中是否至少有 1 个激活特征
            density[i] = np.any(binary_pop[:, window] == 1, axis=1).mean()
        return density

    def _sdge_exploration(
        self,
        hawk: np.ndarray,
        rabbit: np.ndarray,
        windows: List[List[int]],
        density: np.ndarray,
    ) -> np.ndarray:
        """SDGE：基于窗口占用率的密度引导探索。

        按反密度权重选目标 window，在该 window 内随机选一个特征做更新。
        """
        # 反密度权重（低密度区域更容易被选中）
        weights = 1.0 / (density + 1e-6)
        weights /= weights.sum()

        target_window_idx = np.random.choice(len(windows), p=weights)
        target_feat = np.random.choice(windows[target_window_idx])

        # 在目标特征维度上向 rabbit 移动，其余维度保持标准 exploration
        new_pos = hawk.copy()
        r = np.random.rand()
        new_pos[target_feat] = rabbit[target_feat] + r * (
            rabbit[target_feat] - hawk[target_feat]
        )
        return np.clip(new_pos, 0, 1)

    # ══════════════════════════════════════════════════════
    # 开发阶段
    # ══════════════════════════════════════════════════════

    def _hho_exploitation(
        self,
        hawk: np.ndarray,
        rabbit: np.ndarray,
        E: float,
        fitness_func,
    ) -> np.ndarray:
        """标准 HHO exploitation（4 种策略）+ 可选 RSC 修正。

        RSC 是修正项，不是替换项：先做标准 exploitation，再对结果做冗余抑制。
        """
        r = np.random.rand()
        J = 2 * (1 - np.random.rand())
        D = np.abs(rabbit - hawk)

        if abs(E) >= 0.5 and r >= 0.5:
            # Soft besiege
            base_pos = rabbit - E * np.abs(J * rabbit - hawk)

        elif abs(E) < 0.5 and r >= 0.5:
            # Hard besiege
            base_pos = rabbit - E * D

        elif abs(E) >= 0.5 and r < 0.5:
            # Soft besiege with progressive rapid dives（Lévy）
            levy = self._levy_flight(len(hawk))
            Y = rabbit - E * np.abs(J * rabbit - hawk)
            Z = Y + np.random.rand(len(hawk)) * levy
            Y = np.clip(Y, 0, 1)
            Z = np.clip(Z, 0, 1)
            base_pos = Y if fitness_func(Y) < fitness_func(Z) else Z

        else:
            # Hard besiege with progressive rapid dives（Lévy）
            levy = self._levy_flight(len(hawk))
            mean_pop = np.mean(self._population, axis=0)
            Y = rabbit - E * np.abs(J * rabbit - mean_pop)
            Z = Y + np.random.rand(len(hawk)) * levy
            Y = np.clip(Y, 0, 1)
            Z = np.clip(Z, 0, 1)
            base_pos = Y if fitness_func(Y) < fitness_func(Z) else Z

        base_pos = np.clip(base_pos, 0, 1)

        # RSC 修正（如果启用）
        if self.enable_rsc and self._corr_matrix is not None:
            return self._rsc_correction(base_pos, self._corr_matrix)
        return base_pos

    def _rsc_correction(
        self,
        base_pos: np.ndarray,
        corr_matrix: np.ndarray,
    ) -> np.ndarray:
        """RSC 冗余抑制修正。

        对 exploitation 结果中已激活的高冗余特征以低概率保留。
        keep_prob = clip(1 - redundancy, 0.5, 1.0)  # 下限 0.5，防止过稀
        若修正后全部清零，回退到 base_pos。
        """
        sel_idx = np.where(self._to_binary(base_pos) == 1)[0]
        if len(sel_idx) <= 1:
            return base_pos

        corrected = base_pos.copy()
        for i, idx in enumerate(sel_idx):
            others = np.delete(sel_idx, i)
            redundancy = float(np.abs(corr_matrix[idx, others]).mean())
            keep_prob = float(np.clip(1.0 - redundancy, 0.5, 1.0))
            if np.random.rand() > keep_prob:
                corrected[idx] = 0.0

        # 防止全部清零
        if self._to_binary(corrected).sum() == 0:
            return base_pos
        return corrected

    # ══════════════════════════════════════════════════════
    # 后期精修
    # ══════════════════════════════════════════════════════

    def _late_refinement(
        self,
        solution: np.ndarray,
        fitness_func,
        neighbors: Dict[int, List[int]],
        max_trials: int = 5,
    ) -> np.ndarray:
        """Late Refinement（first-improvement greedy）。

        - 相邻 swap 基于 spectral_neighbors（真实波长邻居）
        - 只把已选特征换成未选邻居
        - first-improvement：找到第一个改善立即接受并结束本轮
        - 每轮最多 max_trials 次尝试
        """
        best_sol = solution.copy()
        best_fit = fitness_func(solution)
        sel_idx  = list(np.where(self._to_binary(solution) == 1)[0])

        trials = 0
        for idx in np.random.permutation(sel_idx):
            if trials >= max_trials:
                break
            for neighbor in neighbors.get(int(idx), []):
                if trials >= max_trials:
                    break
                # 只把已选特征换成未选邻居
                if self._to_binary(best_sol)[neighbor] == 0:
                    new_sol = best_sol.copy()
                    new_sol[idx]      = 0.0
                    new_sol[neighbor] = 1.0
                    new_fit = fitness_func(new_sol)
                    trials += 1
                    if new_fit < best_fit:
                        # first-improvement：立即接受
                        best_sol = new_sol
                        best_fit = new_fit
                        return best_sol  # 本轮只做一次改善

        return best_sol

    # ══════════════════════════════════════════════════════
    # 适应度函数
    # ══════════════════════════════════════════════════════

    # 固定边界参数（不参与调优）
    _K_MIN  = 3    # 最小特征数约束

    def _make_fitness_function(
        self,
        X_fusion: np.ndarray,
        y: np.ndarray,
        kf_splits: List[Tuple],
        corr_matrix: np.ndarray,
    ):
        """构建带 cache 的适应度函数。

        fitness = (1 - R²_cv) + λ·|S|/D + γ·redundancy + 0.3·max(0, k_min-k)/k_min

        其中：
            R²_cv      : 5折交叉验证决定系数均值
            |S|        : 选中特征数
            D          : 候选池总维数
            λ          : 稀疏惩罚系数（self.penalty）
            γ          : 冗余惩罚系数（self.gamma）
            redundancy : 选中特征间平均绝对相关系数（上三角均值）
            k_min      : 最小特征数下界（固定为 _K_MIN=3）

        n_components 约束：
            n_comp = max(1, min(k, MAX_PLS_COMPONENTS, len(train_idx)-1))
        """
        cache: Dict[tuple, float] = {}
        n_features = X_fusion.shape[1]
        k_min = self._K_MIN

        def fitness_function(solution: np.ndarray) -> float:
            sel_idx = tuple(np.where(self._to_binary(solution) == 1)[0])

            if sel_idx in cache:
                return cache[sel_idx]

            if len(sel_idx) == 0:
                cache[sel_idx] = 999.0
                return 999.0

            try:
                r2_scores = []
                for train_idx, val_idx in kf_splits:
                    n_comp = max(1, min(
                        len(sel_idx),
                        MAX_PLS_COMPONENTS,
                        len(train_idx) - 1,
                    ))
                    X_tr  = X_fusion[train_idx][:, list(sel_idx)]
                    X_val = X_fusion[val_idx][:, list(sel_idx)]
                    model = PLSRegression(n_components=n_comp)
                    model.fit(X_tr, y[train_idx])
                    y_pred = model.predict(X_val).flatten()
                    r2_scores.append(r2_score(y[val_idx], y_pred))

                r2_cv = float(np.mean(r2_scores))
                k     = len(sel_idx)
                ratio = k / n_features

                # 冗余惩罚（对角线已置 0，等价于上三角均值）
                if k > 1:
                    sub = corr_matrix[np.ix_(list(sel_idx), list(sel_idx))]
                    redundancy = float(sub.sum() / (k * (k - 1)))
                else:
                    redundancy = 0.0

                # 最小特征数软约束（固定系数 0.3，不参与调优）
                min_feat_penalty = 0.3 * max(0, k_min - k) / k_min

                result = (1 - r2_cv) + self.penalty * ratio + self.gamma * redundancy + min_feat_penalty

            except Exception:
                result = 999.0

            cache[sel_idx] = result
            return result

        return fitness_function

    # ══════════════════════════════════════════════════════
    # 主流程
    # ══════════════════════════════════════════════════════

    def run_selection(self, input_path: str, output_path: str, **kwargs) -> SelectionResult:
        """主流程：数据加载 → 多光谱融合 → 粗筛 → SSAHHO 迭代。

        kwargs 支持：
            algo_seed (int): 算法随机种子，默认 0
            cv_seed   (int): KFold 随机种子，默认 42
            save_candidate_summary (bool): 是否保存结构检查 CSV，默认 True（首次运行）
        """
        # 1. 设置随机种子
        algo_seed = int(kwargs.get("algo_seed", 0))
        cv_seed   = int(kwargs.get("cv_seed",   DEFAULT_RANDOM_STATE))
        np.random.seed(algo_seed)
        random.seed(algo_seed)

        self.logger.info(
            f"SSAHHO run_selection | algo_seed={algo_seed} cv_seed={cv_seed} "
            f"epoch={self.epoch} pop_size={self.pop_size} "
            f"SCI={self.enable_sci} SDGE={self.enable_sdge} "
            f"RSC={self.enable_rsc} Refine={self.enable_refinement}"
        )

        # 2. 加载数据
        df, X_raw, y = self.load_data(input_path)
        self.logger.info(f"数据加载完成: {X_raw.shape}, 目标变量: {self.target_col}")

        # 3. 4 种光谱变换（仅在训练集上）
        transforms = apply_all_transforms(X_raw)

        # 4. 相关性粗筛（仅在训练集上，测试集完全隔离）
        candidates, X_fusion = build_fusion_candidates(
            X_raw, y, transforms, self.feat_cols, n_top=self.n_top
        )
        self.logger.info(f"候选池构建完成: {X_fusion.shape[1]} 维")

        # 5. 结构检查（保存 candidate_summary.csv）
        save_summary = bool(kwargs.get("save_candidate_summary", True))
        if save_summary:
            windows_for_check  = build_spectral_windows(candidates)
            neighbors_for_check = build_spectral_neighbors(candidates)
            summary_path = str(Path(output_path).parent / "candidate_summary.csv")
            print_candidate_summary(
                candidates, windows_for_check, neighbors_for_check, summary_path
            )

        # 6. 预计算相关矩阵（对角线置 0）
        corr_matrix = np.abs(np.corrcoef(X_fusion.T))
        np.fill_diagonal(corr_matrix, 0.0)
        self._corr_matrix = corr_matrix

        # 7. 预计算 KFold splits（cv_seed 固定）
        kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
        kf_splits = list(kf.split(X_fusion))

        # 8. 构建 fitness function（带 cache）
        fitness_func = self._make_fitness_function(X_fusion, y, kf_splits, corr_matrix)

        # 9. 构建 windows / neighbors（按真实波长）
        windows   = build_spectral_windows(candidates)
        neighbors = build_spectral_neighbors(candidates)
        n_features = X_fusion.shape[1]

        # 10. 初始化种群
        if self.enable_sci:
            population = self._spectral_coverage_init(n_features, windows)
        else:
            population = self._random_init(n_features)

        self._population = population

        # 评估初始种群
        fitness_values = np.array([fitness_func(ind) for ind in population])
        best_idx       = int(np.argmin(fitness_values))
        rabbit         = population[best_idx].copy()
        rabbit_fitness = float(fitness_values[best_idx])

        # 11. 主迭代循环
        fitness_history:        List[float] = []
        selected_count_history: List[int]   = []

        for t in range(self.epoch):
            E0 = 2 * np.random.rand() - 1
            E  = 2 * E0 * (1 - t / self.epoch)

            # 每代更新一次窗口密度（SDGE 用）
            if self.enable_sdge:
                density = self._compute_window_density(population, windows)

            for i in range(self.pop_size):
                current_bin = self._to_binary(population[i]).astype(float)

                if abs(E) >= 1:   # Exploration
                    if self.enable_sdge:
                        velocity = self._sdge_exploration(
                            population[i], rabbit, windows, density
                        )
                    else:
                        velocity = self._hho_exploration(
                            population[i], rabbit, population
                        )
                    # Exploration 输出为连续速度，用 V 形转换映射为二值位置
                    new_pos = self._v_transfer(velocity, current_bin)
                else:             # Exploitation（RSC 在内部作为修正项）
                    # Exploitation 直接在二值空间操作，输出已是 0/1 附近的值
                    new_pos = self._hho_exploitation(
                        population[i], rabbit, E, fitness_func
                    )
                    new_pos = self._to_binary(new_pos).astype(float)

                new_fit = fitness_func(new_pos)

                # 贪心更新 hawk
                if new_fit < fitness_values[i]:
                    population[i]     = new_pos
                    fitness_values[i] = new_fit

                # 更新 rabbit
                if new_fit < rabbit_fitness:
                    rabbit         = new_pos.copy()
                    rabbit_fitness = new_fit

            # 同步种群引用（供 _hho_exploitation 访问均值）
            self._population = population

            # Late Refinement（后期阶段，t >= 0.7*epoch）
            if self.enable_refinement and t >= int(0.7 * self.epoch):
                refined     = self._late_refinement(rabbit, fitness_func, neighbors)
                refined_fit = fitness_func(refined)
                if refined_fit < rabbit_fitness:
                    rabbit         = refined
                    rabbit_fitness = refined_fit

            fitness_history.append(rabbit_fitness)
            selected_count_history.append(int(self._to_binary(rabbit).sum()))

        self.logger.info(
            f"迭代完成 | 最终 fitness={rabbit_fitness:.4f} "
            f"选中特征数={selected_count_history[-1]}"
        )

        # 12. 提取结果，映射回原始波段名
        best_sel_idx = list(np.where(self._to_binary(rabbit) == 1)[0])
        selected_candidates = [candidates[i] for i in best_sel_idx]

        # 去重（同一 band_name 可能来自不同 transform，取第一个）
        seen_bands: set = set()
        best_feats: List[str] = []
        for cf in selected_candidates:
            if cf.band_name not in seen_bands:
                best_feats.append(cf.band_name)
                seen_bands.add(cf.band_name)

        # 13. 保存 fitness_history / selected_count_history
        out_dir = Path(output_path).parent
        _save_history(fitness_history,        out_dir / "fitness_history.csv",        "fitness")
        _save_history(selected_count_history, out_dir / "selected_count_history.csv", "n_selected")

        # 14. 保存 selected_features_detail.csv
        detail_rows = [
            {
                "fusion_col": cf.fusion_col,
                "transform":  cf.transform,
                "band_name":  cf.band_name,
                "wavelength": cf.wavelength,
            }
            for cf in selected_candidates
        ]
        pd.DataFrame(detail_rows).to_csv(
            out_dir / "selected_features_detail.csv", index=False
        )

        # 15. 保存 rabbit_summary.csv
        pd.DataFrame([{
            "n_selected":            len(best_feats),
            "selected_feature_names": ",".join(best_feats),
            "selected_wavelengths":   ",".join(
                f"{cf.wavelength:.0f}" for cf in selected_candidates
            ),
            "final_fitness":          rabbit_fitness,
        }]).to_csv(out_dir / "rabbit_summary.csv", index=False)

        # 16. 调用 save_selection_result
        self.save_selection_result(df, y, best_feats, output_path)

        return SelectionResult(
            selected_features=best_feats,
            selected_indices=[self.feat_cols.index(f) for f in best_feats],
        )


# ══════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════

def _save_history(data: list, path: Path, col_name: str) -> None:
    """保存历史数据到 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"epoch": range(len(data)), col_name: data}).to_csv(path, index=False)


# ══════════════════════════════════════════════════════════
# 消融变体注册
# ══════════════════════════════════════════════════════════

@register_algorithm("SSAHHO_HHO")
class SSAHHO_HHO(SSAHHOSelector):
    """基线：标准 HHO（所有改进关闭）。"""
    enable_sci = enable_sdge = enable_rsc = enable_refinement = False


@register_algorithm("SSAHHO_SCI")
class SSAHHO_SCI(SSAHHOSelector):
    """仅 SCI。"""
    enable_sci        = True
    enable_sdge       = False
    enable_rsc        = False
    enable_refinement = False


@register_algorithm("SSAHHO_SCI_SDGE")
class SSAHHO_SCI_SDGE(SSAHHOSelector):
    """SCI + SDGE。"""
    enable_sci        = True
    enable_sdge       = True
    enable_rsc        = False
    enable_refinement = False


@register_algorithm("SSAHHO_SCI_SDGE_RSC")
class SSAHHO_SCI_SDGE_RSC(SSAHHOSelector):
    """SCI + SDGE + RSC。"""
    enable_sci        = True
    enable_sdge       = True
    enable_rsc        = True
    enable_refinement = False


@register_algorithm("SSAHHO")
class SSAHHOFull(SSAHHOSelector):
    """完整版 SSAHHO（SCI + SDGE + RSC + Late Refinement）。"""
    enable_sci        = True
    enable_sdge       = True
    enable_rsc        = True
    enable_refinement = True
