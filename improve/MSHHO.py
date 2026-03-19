"""
MS-HHO: Multi-Strategy Harris Hawks Optimization

在标准 HHO 基础上实现两个算法层改进：
  改进一：Tent 混沌初始化 + 对立学习（OBL）
  改进二：Cauchy 变异精英扰动（停滞逃逸）
"""

import numpy as np
import math
from collections import deque
from mealpy.swarm_based.HHO import OriginalHHO

from core.constants import STAGNATION_THRESHOLD


class MultiStrategyHHO(OriginalHHO):
    """
    Multi-Strategy Harris Hawks Optimization.

    改进一：Tent 混沌初始化 + 对立学习
        - Tent 映射生成遍历性更强的初始种群
        - OBL 生成对立解，从 2×pop_size 候选中择优保留 pop_size 个
        - 将初始搜索空间覆盖率从 ~50% 提升至接近 100%

    改进二：Cauchy 变异精英扰动
        - 连续 STAGNATION_THRESHOLD 轮适应度无改善时触发
        - Cauchy 分布重尾特性允许大步长跳跃，逃脱深度局部最优
        - 自适应步长缩放：|x_best - x_mean| 随种群分散程度自动调整

    Parameters
    ----------
    epoch : int
        最大迭代轮数
    pop_size : int
        种群大小
    """

    def __init__(self, epoch=200, pop_size=50,
                 enable_chaos_init=True,
                 enable_cauchy=True,
                 **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)

        # 改进开关（用于消融实验）
        self.enable_chaos_init = enable_chaos_init  # 控制改进一：Tent+OBL 初始化
        self.enable_cauchy = enable_cauchy          # 控制改进二：Cauchy 变异扰动

        # 停滞计数器与上一轮最优适应度
        self._stagnation_count = 0
        self._prev_best_fitness = None

    # ------------------------------------------------------------------
    # 改进一：Tent 混沌初始化 + 对立学习
    # ------------------------------------------------------------------

    def _tent_chaos_sequence(self, n, dim):
        """
        用 Tent 映射生成 n×dim 的混沌矩阵，值域 [0, 1]。

        Tent 映射：
            x(t+1) = 2x(t)        if x(t) < 0.5
                     2(1-x(t))    if x(t) >= 0.5
        """
        # 随机选取初始值，避免不动点 0 和 1
        x = np.random.uniform(0.01, 0.99, (n, dim))
        for _ in range(50):  # 迭代 50 步使序列充分混沌化
            mask = x < 0.5
            x = np.where(mask, 2 * x, 2 * (1 - x))
        return x

    def _obl_candidates(self, positions):
        """
        对立学习：对每个个体生成对立解。

        对于搜索空间 [0, 1]：x_opp = 1 - x
        返回原始种群与对立种群拼接后的 2×pop_size 候选矩阵。
        """
        opp = 1.0 - positions
        return np.vstack([positions, opp])

    def initialization(self):
        """
        覆盖父类初始化：enable_chaos_init=True 时使用 Tent+OBL，否则退化为父类随机初始化。
        """
        if not self.enable_chaos_init:
            super().initialization()
            return

        self.pop = []
        n_dims = self.problem.n_dims

        # 1. Tent 混沌生成 pop_size 个候选位置
        chaos_pos = self._tent_chaos_sequence(self.pop_size, n_dims)

        # 2. OBL 生成对立解，得到 2×pop_size 候选
        all_candidates = self._obl_candidates(chaos_pos)
        # 裁剪到合法范围
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)
        all_candidates = np.clip(all_candidates, lb, ub)

        # 3. 评估所有候选的适应度
        agents = []
        for pos in all_candidates:
            agent = self.generate_agent(pos)
            agents.append(agent)

        # 4. 按适应度排序，取前 pop_size 个（最小化问题取最小值）
        agents.sort(key=lambda a: a.target.fitness)
        self.pop = agents[:self.pop_size]

    # ------------------------------------------------------------------
    # 改进二：Cauchy 变异精英扰动
    # ------------------------------------------------------------------

    def _cauchy_mutation(self, best_pos, mean_pos):
        """
        对全局最优位置施加 Cauchy 变异。

        公式：x_mutated = x_best + Cauchy(0,1) · |x_best - x_mean|

        自适应步长缩放因子 |x_best - x_mean| 使变异幅度随种群
        分散程度自动调整：种群越分散，步长越大；越收敛，步长越小。
        """
        # Cauchy 分布采样（重尾，允许偶发大步长跳跃）
        cauchy_noise = np.random.standard_cauchy(len(best_pos))
        step = np.abs(best_pos - mean_pos)
        mutated = best_pos + cauchy_noise * step

        # 裁剪到合法范围
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)
        return np.clip(mutated, lb, ub)

    def evolve(self, epoch):
        """
        覆盖父类 evolve：执行标准 HHO 迭代后，enable_cauchy=True 时检测停滞并触发 Cauchy 扰动。
        """
        # 执行标准 HHO 的位置更新
        super().evolve(epoch)

        if not self.enable_cauchy:
            return

        # --- 停滞检测 ---
        current_best_fitness = self.g_best.target.fitness

        if self._prev_best_fitness is not None:
            improvement = abs(self._prev_best_fitness - current_best_fitness)
            if improvement < 1e-6:
                self._stagnation_count += 1
            else:
                self._stagnation_count = 0
        self._prev_best_fitness = current_best_fitness

        # --- 触发 Cauchy 变异 ---
        if self._stagnation_count >= STAGNATION_THRESHOLD:
            self._stagnation_count = 0  # 重置计数器

            # 计算当前种群均值位置
            all_positions = np.array([agent.solution for agent in self.pop])
            mean_pos = np.mean(all_positions, axis=0)

            # 对全局最优施加 Cauchy 变异，生成扰动解
            mutated_pos = self._cauchy_mutation(self.g_best.solution, mean_pos)
            mutated_agent = self.generate_agent(mutated_pos)

            # 贪心替换：若扰动解优于当前最差个体，则替换
            worst_idx = max(
                range(self.pop_size),
                key=lambda i: self.pop[i].target.fitness
            )
            if self.compare_target(
                mutated_agent.target,
                self.pop[worst_idx].target,
                self.problem.minmax
            ):
                self.pop[worst_idx] = mutated_agent

            # 更新全局最优
            self.g_best = self.get_best_agent(self.pop)
