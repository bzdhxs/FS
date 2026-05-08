"""
PL-HHO: Lévy-enhanced Harris Hawks Optimization
针对高光谱小样本特征选择任务的改进 HHO。

改进一（初始化）：Tent 混沌 + 反向学习（OBL）双重初始化
    - Tent 混沌序列保证初始种群均匀覆盖搜索空间
    - 同时生成反向解 X_obl = 1 - X，从 2N 候选中择优取 N 个
    - 不引入任何先验偏置，由 fitness 自主引导搜索方向

改进二（逃逸能量）：幂律 × 余弦周期性衰减
    - E(t) = 2·E₀·(1-(t/T)^γ)·cos(2π·k·t/T)
    - 单调幂律保证整体趋势收敛，余弦项周期性恢复探索能力

改进三（rabbit 扰动）：时变柯西连续扰动
    - 每轮对全局最优施加柯西扰动：X_rabbit += w(t)·Cauchy(0,1)
    - w(t) = cauchy_c0·(1 - t/T)，早期大步跳跃，后期自动退化

改进四（停滞响应）：翻转变异
    - 停滞时对精英个体执行维度翻转：X_j → 1 - X_j
    - 翻转比例 flip_ratio 控制，强制跳出局部最优

改进五（精英记忆）：衰减记忆表 + 软引导
    - F_j^t = rho·F_j^{t-1} + Σ w_i·B_ij
    - t > beta·T 时对高频特征施加软引导（插值而非强制置位）
    - 引导强度 guide_strength 控制，仅在引导解优于当前最优时接受
"""

import numpy as np
from mealpy.swarm_based.HHO import OriginalHHO


class PLHarrisHawks(OriginalHHO):
    """
    Lévy-enhanced HHO (PL-HHO)

    Parameters
    ----------
    epoch : int
    pop_size : int
    gamma : float        幂律能量指数（默认 2.0）
    n_periods : int      余弦周期数 k（默认 1）
    cauchy_c0 : float    柯西扰动初始幅度（默认 0.3）
    rho : float          精英记忆衰减因子（默认 0.95）
    elite_ratio : float  精英比例（默认 0.2）
    tau : float          高频特征阈值（默认 0.6）
    beta : float         精英引导启动时机 t > beta·T（默认 0.6）
    flip_ratio : float   停滞时翻转的维度比例（默认 0.1）
    stagnation_patience : int  停滞触发阈值（默认 8）
    enable_chaos_init : bool   是否启用 Tent 混沌初始化（默认 True）
    enable_obl : bool          是否启用反向学习（默认 True）
    enable_periodic_energy : bool  是否启用周期性能量（默认 True）
    enable_cauchy : bool       是否启用柯西扰动（默认 True）
    enable_elite_memory : bool 是否启用精英记忆（默认 True）
    """

    def __init__(self, epoch=200, pop_size=50,
                 gamma=2.0, n_periods=1, cauchy_c0=0.3,
                 rho=0.95, elite_ratio=0.2, tau=0.6,
                 beta=0.6, flip_ratio=0.1, stagnation_patience=8,
                 guide_strength=0.3,
                 enable_chaos_init=True,
                 enable_obl=True,
                 enable_periodic_energy=True,
                 enable_cauchy=True,
                 enable_elite_memory=True,
                 **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.gamma               = gamma
        self.n_periods           = n_periods
        self.cauchy_c0           = cauchy_c0
        self.rho                 = rho
        self.elite_ratio         = elite_ratio
        self.tau                 = tau
        self.beta                = beta
        self.flip_ratio          = flip_ratio
        self.stagnation_patience = stagnation_patience
        self.guide_strength      = guide_strength  # 软引导插值强度 [0,1]

        self.enable_chaos_init      = enable_chaos_init
        self.enable_obl             = enable_obl
        self.enable_periodic_energy = enable_periodic_energy
        self.enable_cauchy          = enable_cauchy
        self.enable_elite_memory    = enable_elite_memory

        # 运行时状态
        self._memory     = None
        self._stag_count = 0
        self._prev_best  = None

    # =========================================================================
    # 改进一：三重初始化
    # =========================================================================

    def _tent_sequence(self, n, dim):
        """生成 Tent 混沌序列，shape (n, dim)，值域 (0,1)。"""
        x = np.random.uniform(0.01, 0.99, (n, dim))
        for _ in range(100):
            x = np.where(x < 0.5, 2 * x, 2 * (1 - x))
            x = np.clip(x, 1e-6, 1 - 1e-6)
        return x

    def initialization(self):
        dim = self.problem.n_dims
        lb  = np.array(self.problem.lb)
        ub  = np.array(self.problem.ub)

        # ── 基础位置生成（Tent 混沌 or 均匀随机）────────────────────────
        if self.enable_chaos_init:
            base = self._tent_sequence(self.pop_size, dim)
        else:
            base = np.random.uniform(0, 1, (self.pop_size, dim))

        positions = lb + base * (ub - lb)

        # ── 反向学习 OBL ──────────────────────────────────────────────────
        if self.enable_obl:
            obl_positions = lb + ub - positions
            obl_positions = np.clip(obl_positions, lb, ub)
            candidates = np.vstack([positions, obl_positions])
            fits = np.array([
                self.problem.obj_func(candidates[i]) for i in range(len(candidates))
            ])
            best_idx  = np.argsort(fits)[:self.pop_size]
            positions = candidates[best_idx]

        # ── 构建种群 ──────────────────────────────────────────────────────
        self.pop = []
        for pos in positions:
            pos = np.clip(pos, lb, ub)
            self.pop.append(self.generate_agent(pos))

        # 初始化记忆表
        if self.enable_elite_memory:
            self._memory     = np.zeros(dim)
            self._stag_count = 0
            self._prev_best  = None

    # =========================================================================
    # 精英记忆辅助
    # =========================================================================

    def _to_binary(self, solution):
        return (solution > 0.5).astype(float)

    def _update_memory(self):
        fits    = np.array([a.target.fitness for a in self.pop])
        n_elite = max(1, int(self.pop_size * self.elite_ratio))
        elite_idx   = np.argsort(fits)[:n_elite]
        elite_fits  = fits[elite_idx]

        f_max = elite_fits.max()
        eps   = 1e-10
        raw_w = f_max - elite_fits + eps
        w     = raw_w / (raw_w.sum() + eps)

        contribution = np.zeros(self.problem.n_dims)
        for rank, idx in enumerate(elite_idx):
            b = self._to_binary(self.pop[idx].solution)
            contribution += w[rank] * b

        self._memory = self.rho * self._memory + contribution

    def _build_guided_solution(self):
        """软引导：对高频特征做插值而非强制置位，保留搜索自由度。"""
        ub        = np.array(self.problem.ub)
        threshold = self.tau * self._memory.max()
        guided    = self.g_best.solution.copy()
        high_freq = self._memory > threshold
        # 插值：guided = (1-α)·g_best + α·ub，α = guide_strength
        guided[high_freq] = (
            (1 - self.guide_strength) * self.g_best.solution[high_freq]
            + self.guide_strength * ub[high_freq]
        )
        return guided

    # =========================================================================
    # 改进四：翻转变异（停滞响应）
    # =========================================================================

    def _flip_mutation(self, solution):
        """随机翻转 flip_ratio 比例的维度（连续位置：选中↔未选中）。"""
        lb  = np.array(self.problem.lb)
        ub  = np.array(self.problem.ub)
        dim = len(solution)

        n_flip = max(1, int(dim * self.flip_ratio))
        flip_idx = np.random.choice(dim, size=n_flip, replace=False)

        new_sol = solution.copy()
        for j in flip_idx:
            # 翻转：若当前偏向上界（选中）则推向下界，反之亦然
            if new_sol[j] > 0.5:
                new_sol[j] = lb[j]
            else:
                new_sol[j] = ub[j]
        return new_sol

    # =========================================================================
    # 改进二 + 改进三 + 改进四 + 改进五：完整 evolve
    # =========================================================================

    def evolve(self, epoch):
        # ── 改进三：柯西扰动 rabbit ───────────────────────────────────────
        if self.enable_cauchy:
            w_t = self.cauchy_c0 * (1.0 - epoch / self.epoch)
            cauchy_noise = np.random.standard_cauchy(self.problem.n_dims)
            # 截断极端值，防止柯西重尾过度跳跃
            cauchy_noise = np.clip(cauchy_noise, -5, 5)
            rabbit_perturbed = self.g_best.solution + w_t * cauchy_noise
            rabbit_perturbed = np.clip(
                rabbit_perturbed,
                self.problem.lb,
                self.problem.ub
            )
            target_perturbed = self.get_target(rabbit_perturbed)
            if self.compare_target(target_perturbed, self.g_best.target,
                                   self.problem.minmax):
                self.g_best.solution = rabbit_perturbed
                self.g_best.target   = target_perturbed

        # ── 改进二：周期性逃逸能量 + 标准 HHO 位置更新 ───────────────────
        pop_new = []
        for idx in range(self.pop_size):
            E0 = 2 * self.generator.uniform() - 1

            if self.enable_periodic_energy:
                # E(t) = 2·E₀·(1-(t/T)^γ)·cos(2π·k·t/T)
                decay    = 1.0 - (epoch / self.epoch) ** self.gamma
                periodic = np.cos(2 * np.pi * self.n_periods * epoch / self.epoch)
                E = 2 * E0 * decay * periodic
            else:
                E = 2 * E0 * (1.0 - epoch / self.epoch)

            J = 2 * (1 - self.generator.uniform())

            if np.abs(E) >= 1:
                # 探索阶段
                if self.generator.random() >= 0.5:
                    X_rand = self.pop[
                        self.generator.integers(0, self.pop_size)
                    ].solution.copy()
                    pos_new = X_rand - self.generator.uniform() * np.abs(
                        X_rand - 2 * self.generator.uniform() * self.pop[idx].solution
                    )
                else:
                    X_m = np.mean([x.solution for x in self.pop], axis=0)
                    pos_new = (self.g_best.solution - X_m) - self.generator.uniform() * (
                        np.array(self.problem.lb) + self.generator.uniform() * (
                            np.array(self.problem.ub) - np.array(self.problem.lb)
                        )
                    )
                pos_new = self.correct_solution(pos_new)
                pop_new.append(self.generate_empty_agent(pos_new))

            else:
                # 开发阶段
                if self.generator.random() >= 0.5:
                    delta_X = self.g_best.solution - self.pop[idx].solution
                    if np.abs(E) >= 0.5:
                        pos_new = delta_X - E * np.abs(
                            J * self.g_best.solution - self.pop[idx].solution
                        )
                    else:
                        pos_new = self.g_best.solution - E * np.abs(delta_X)
                    pos_new = self.correct_solution(pos_new)
                    pop_new.append(self.generate_empty_agent(pos_new))
                else:
                    LF_D = self.get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1)
                    if np.abs(E) >= 0.5:
                        Y = self.g_best.solution - E * np.abs(
                            J * self.g_best.solution - self.pop[idx].solution
                        )
                    else:
                        X_m = np.mean([x.solution for x in self.pop], axis=0)
                        Y = self.g_best.solution - E * np.abs(
                            J * self.g_best.solution - X_m
                        )
                    pos_Y    = self.correct_solution(Y)
                    target_Y = self.get_target(pos_Y)
                    Z        = Y + self.generator.uniform(
                        self.problem.lb, self.problem.ub
                    ) * LF_D
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

        # 评估 + 贪婪选择
        if self.mode not in self.AVAILABLE_MODES:
            for i, agent in enumerate(pop_new):
                pop_new[i].target = self.get_target(agent.solution)
        else:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = self.greedy_selection_population(
            self.pop, pop_new, self.problem.minmax
        )

        # ── 改进五：精英记忆 + 融合引导 + 停滞翻转变异 ──────────────────
        if not self.enable_elite_memory:
            return

        self._update_memory()
        current_best_fitness = self.g_best.target.fitness

        # 后期精英融合引导
        if epoch > self.beta * self.epoch:
            guided_pos    = self._build_guided_solution()
            guided_pos    = self.correct_solution(guided_pos)
            guided_target = self.get_target(guided_pos)

            if self.compare_target(guided_target, self.g_best.target,
                                   self.problem.minmax):
                self.g_best.solution = guided_pos
                self.g_best.target   = guided_target
                current_best_fitness = guided_target.fitness
                center_for_perturb   = guided_pos
            else:
                center_for_perturb = guided_pos
        else:
            center_for_perturb = self.g_best.solution

        # 停滞检测
        if self._prev_best is not None:
            if abs(self._prev_best - current_best_fitness) < 1e-8:
                self._stag_count += 1
            else:
                self._stag_count = 0
        self._prev_best = current_best_fitness

        # 停滞触发翻转变异
        if self._stag_count >= self.stagnation_patience:
            self._stag_count = 0
            flipped_pos    = self._flip_mutation(center_for_perturb)
            flipped_pos    = self.correct_solution(flipped_pos)
            flipped_target = self.get_target(flipped_pos)

            if self.compare_target(flipped_target, self.g_best.target,
                                   self.problem.minmax):
                self.g_best.solution = flipped_pos
                self.g_best.target   = flipped_target
                self._prev_best      = flipped_target.fitness
