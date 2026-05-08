"""
AB-HHO: Adaptive Binary Harris Hawks Optimization
优化器层改进：
  改进一：Tent 混沌初始化
  改进二（搜索机制）：非线性自适应逃逸能量调度（gamma 参数）
  改进三（精英记忆）：衰减记忆表 + 质量加权 + 融合引导 + 停滞触发局部扰动
"""

import numpy as np
from mealpy.swarm_based.HHO import OriginalHHO


class AdaptiveBinaryHHO(OriginalHHO):
    """
    Adaptive Binary HHO 优化器。

    改进一：Tent 混沌初始化
    改进二：非线性逃逸能量  E(t) = 2*E0*(1-(t/T)^gamma)
    改进三：精英记忆引导 + 停滞触发局部扰动
        - 衰减记忆表：F_j^t = rho*F_j^{t-1} + sum_{elite} w_i * B_ij
        - 融合引导：t > beta*T 时，高频特征强制保留，其余保持当前最优
        - 停滞触发：连续 stagnation_patience 轮无改善时执行局部扰动

    Parameters
    ----------
    gamma : float, default=2.0
    rho : float, default=0.95        遗忘因子
    elite_ratio : float, default=0.2  精英比例
    tau : float, default=0.6          频次阈值
    beta : float, default=0.6         精英引导启动时机（t > beta*T）
    delta : int, default=2            局部扰动替换特征数
    stagnation_patience : int, default=8
    enable_chaos_init : bool, default=True
    enable_nonlinear_energy : bool, default=True
    enable_elite_memory : bool, default=True
    """

    def __init__(self, epoch=200, pop_size=50, gamma=2.0,
                 rho=0.95, elite_ratio=0.2, tau=0.6,
                 beta=0.6, delta=2, stagnation_patience=8,
                 enable_chaos_init=True,
                 enable_nonlinear_energy=True,
                 enable_elite_memory=True,
                 **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.gamma              = gamma
        self.rho                = rho
        self.elite_ratio        = elite_ratio
        self.tau                = tau
        self.beta               = beta
        self.delta              = delta
        self.stagnation_patience = stagnation_patience
        self.enable_chaos_init       = enable_chaos_init
        self.enable_nonlinear_energy = enable_nonlinear_energy
        self.enable_elite_memory     = enable_elite_memory

        # 运行时状态（在 solve() 开始前由 _init_memory 初始化）
        self._memory      = None   # shape (n_dims,)  衰减记忆表
        self._stag_count  = 0      # 停滞计数器
        self._prev_best   = None   # 上一轮最优适应度

    # =========================================================================
    # 改进一：Tent 混沌初始化
    # =========================================================================

    def _tent_chaos_sequence(self, n, dim):
        x = np.random.uniform(0.01, 0.99, (n, dim))
        for _ in range(100):
            mask = x < 0.5
            x = np.where(mask, 2 * x, 2 * (1 - x))
            x = np.clip(x, 1e-6, 1 - 1e-6)
        return x

    def initialization(self):
        if not self.enable_chaos_init:
            super().initialization()
        else:
            self.pop = []
            lb = np.array(self.problem.lb)
            ub = np.array(self.problem.ub)
            chaos = self._tent_chaos_sequence(self.pop_size, self.problem.n_dims)
            for pos in lb + chaos * (ub - lb):
                self.pop.append(self.generate_agent(pos))

        # 初始化记忆表
        if self.enable_elite_memory:
            self._memory     = np.zeros(self.problem.n_dims)
            self._stag_count = 0
            self._prev_best  = None

    # =========================================================================
    # 改进三辅助：二值化（固定阈值 0.5，供记忆表统计用）
    # =========================================================================

    def _to_binary(self, solution):
        """连续位置 → 二值向量（固定阈值 0.5）。"""
        return (solution > 0.5).astype(float)

    # =========================================================================
    # 改进三：精英记忆更新
    # =========================================================================

    def _update_memory(self):
        """
        每轮 evolve 结束后调用。
        1. 按适应度排序，取前 elite_ratio 个精英个体
        2. 计算质量权重 w_i = (f_max - f_i + eps) / sum(...)
        3. 衰减更新：F = rho*F + sum(w_i * B_i)
        """
        fits = np.array([a.target.fitness for a in self.pop])
        n_elite = max(1, int(self.pop_size * self.elite_ratio))

        # 最小化问题：适应度越小越好，取最小的 n_elite 个
        elite_idx = np.argsort(fits)[:n_elite]
        elite_fits = fits[elite_idx]

        f_max = elite_fits.max()
        eps   = 1e-10
        raw_w = f_max - elite_fits + eps
        w     = raw_w / (raw_w.sum() + eps)   # 归一化权重

        # 加权二值化贡献
        contribution = np.zeros(self.problem.n_dims)
        for rank, idx in enumerate(elite_idx):
            b = self._to_binary(self.pop[idx].solution)
            contribution += w[rank] * b

        self._memory = self.rho * self._memory + contribution

    # =========================================================================
    # 改进三：融合引导解构造
    # =========================================================================

    def _build_guided_solution(self):
        """
        根据记忆表构造融合引导解（连续位置形式）。

        高频特征（F_j > tau * max(F)）对应维度设为 ub（强制倾向选中），
        其余维度保持当前最优解的连续位置不变。
        返回连续位置向量，供 get_target 评估。
        """
        lb  = np.array(self.problem.lb)
        ub  = np.array(self.problem.ub)
        mid = (lb + ub) / 2.0

        threshold = self.tau * self._memory.max()
        guided    = self.g_best.solution.copy()

        # 高频特征：推向上界（二值化后更可能为 1）
        high_freq_mask = self._memory > threshold
        guided[high_freq_mask] = ub[high_freq_mask]

        return guided

    # =========================================================================
    # 改进三：停滞触发局部扰动
    # =========================================================================

    def _local_perturbation(self, center_solution):
        """
        以 center_solution 为中心执行局部扰动：
        - 将连续位置二值化，得到当前特征集
        - 从已选特征中随机移除 delta 个
        - 从未选特征中随机加入 delta 个
        - 将扰动后的二值解映射回连续位置（选中→ub，未选→lb）
        返回扰动后的连续位置向量。
        """
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)

        binary   = self._to_binary(center_solution)
        selected = np.where(binary == 1)[0]
        unselected = np.where(binary == 0)[0]

        if len(selected) == 0 or len(unselected) == 0:
            return center_solution.copy()

        d = min(self.delta, len(selected), len(unselected))

        remove_idx = np.random.choice(selected,   size=d, replace=False)
        add_idx    = np.random.choice(unselected, size=d, replace=False)

        new_binary = binary.copy()
        new_binary[remove_idx] = 0
        new_binary[add_idx]    = 1

        # 二值 → 连续位置（选中推向上界，未选推向下界）
        perturbed = np.where(new_binary == 1, ub, lb)
        return perturbed

    # =========================================================================
    # 改进二 + 改进三：完整 evolve override
    # =========================================================================

    def evolve(self, epoch):
        # ── 标准 HHO 位置更新（含非线性逃逸能量）────────────────────────
        if not self.enable_nonlinear_energy:
            super().evolve(epoch)
        else:
            pop_new = []
            for idx in range(self.pop_size):
                E0 = 2 * self.generator.uniform() - 1
                E  = 2 * E0 * (1.0 - (epoch / self.epoch) ** self.gamma)
                J  = 2 * (1 - self.generator.uniform())

                if np.abs(E) >= 1:
                    if self.generator.random() >= 0.5:
                        X_rand = self.pop[self.generator.integers(0, self.pop_size)].solution.copy()
                        pos_new = X_rand - self.generator.uniform() * np.abs(
                            X_rand - 2 * self.generator.uniform() * self.pop[idx].solution)
                    else:
                        X_m = np.mean([x.solution for x in self.pop], axis=0)
                        pos_new = (self.g_best.solution - X_m) - self.generator.uniform() * (
                            np.array(self.problem.lb) + self.generator.uniform() * (
                                np.array(self.problem.ub) - np.array(self.problem.lb)))
                    pos_new = self.correct_solution(pos_new)
                    pop_new.append(self.generate_empty_agent(pos_new))
                else:
                    if self.generator.random() >= 0.5:
                        delta_X = self.g_best.solution - self.pop[idx].solution
                        if np.abs(E) >= 0.5:
                            pos_new = delta_X - E * np.abs(
                                J * self.g_best.solution - self.pop[idx].solution)
                        else:
                            pos_new = self.g_best.solution - E * np.abs(delta_X)
                        pos_new = self.correct_solution(pos_new)
                        pop_new.append(self.generate_empty_agent(pos_new))
                    else:
                        LF_D = self.get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1)
                        if np.abs(E) >= 0.5:
                            Y = self.g_best.solution - E * np.abs(
                                J * self.g_best.solution - self.pop[idx].solution)
                        else:
                            X_m = np.mean([x.solution for x in self.pop], axis=0)
                            Y = self.g_best.solution - E * np.abs(
                                J * self.g_best.solution - X_m)
                        pos_Y    = self.correct_solution(Y)
                        target_Y = self.get_target(pos_Y)
                        Z        = Y + self.generator.uniform(
                            self.problem.lb, self.problem.ub) * LF_D
                        pos_Z    = self.correct_solution(Z)
                        target_Z = self.get_target(pos_Z)
                        if self.compare_target(target_Y, self.pop[idx].target,
                                               self.problem.minmax):
                            a = self.generate_empty_agent(pos_Y)
                            a.target = target_Y
                            pop_new.append(a)
                            continue
                        if self.compare_target(target_Z, self.pop[idx].target,
                                               self.problem.minmax):
                            a = self.generate_empty_agent(pos_Z)
                            a.target = target_Z
                            pop_new.append(a)
                            continue
                        pop_new.append(self.pop[idx].copy())

            if self.mode not in self.AVAILABLE_MODES:
                for i, agent in enumerate(pop_new):
                    pop_new[i].target = self.get_target(agent.solution)
            else:
                pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax)

        # ── 改进三：精英记忆 + 融合引导 + 停滞扰动 ──────────────────────
        if not self.enable_elite_memory:
            return

        # 1. 更新记忆表
        self._update_memory()

        current_best_fitness = self.g_best.target.fitness

        # 2. 后期精英融合引导（t > beta*T）
        if epoch > self.beta * self.epoch:
            guided_pos    = self._build_guided_solution()
            guided_pos    = self.correct_solution(guided_pos)
            guided_target = self.get_target(guided_pos)

            if self.compare_target(guided_target, self.g_best.target,
                                   self.problem.minmax):
                # 融合解更优，更新全局最优
                self.g_best.solution = guided_pos
                self.g_best.target   = guided_target
                current_best_fitness = guided_target.fitness
                center_for_perturb   = guided_pos
            else:
                # 融合解不优，仍以融合解为扰动中心（携带历史信息）
                center_for_perturb = guided_pos
        else:
            center_for_perturb = self.g_best.solution

        # 3. 停滞检测
        if self._prev_best is not None:
            improvement = abs(self._prev_best - current_best_fitness)
            if improvement < 1e-8:
                self._stag_count += 1
            else:
                self._stag_count = 0
        self._prev_best = current_best_fitness

        # 4. 停滞触发局部扰动
        if self._stag_count >= self.stagnation_patience:
            self._stag_count = 0
            perturbed_pos    = self._local_perturbation(center_for_perturb)
            perturbed_pos    = self.correct_solution(perturbed_pos)
            perturbed_target = self.get_target(perturbed_pos)

            if self.compare_target(perturbed_target, self.g_best.target,
                                   self.problem.minmax):
                self.g_best.solution = perturbed_pos
                self.g_best.target   = perturbed_target
                self._prev_best      = perturbed_target.fitness
