import numpy as np
import math
from scipy.special import gamma
from mealpy.swarm_based.HHO import OriginalHHO


class ImprovedHHO(OriginalHHO):
    """
    改进版 HHO (CL-HHO):
    1. 引入 Logistic 混沌映射进行种群初始化
    2. 引入 Levy Flight 机制防止局部最优
    """

    def __init__(self, epoch=100, pop_size=50, **kwargs):
        super().__init__(epoch, pop_size, **kwargs)

    def initialization(self):
        """
        重写初始化方法：使用 Logistic 混沌映射代替纯随机
        """
        if self.pop is None:
            self.pop = []

        # Logistic Map: x(t+1) = r * x(t) * (1 - x(t))
        # r = 4.0 是完全混沌状态
        r = 4.0
        n_dims = self.problem.n_dims

        # 生成混沌序列矩阵
        chaos_matrix = np.zeros((self.pop_size, n_dims))
        # 随机启动一个向量
        x = np.random.rand(n_dims)

        for i in range(self.pop_size):
            # 迭代混沌方程
            x = r * x * (1 - x)
            chaos_matrix[i, :] = x

        # 将混沌值映射到问题的边界 [lb, ub]
        lb, ub = self.problem.lb, self.problem.ub

        for i in range(self.pop_size):
            # 映射公式: lb + chaos * (ub - lb)
            pos = lb + chaos_matrix[i] * (ub - lb)
            # 生成 Agent
            agent = self.generate_agent(pos)
            self.pop.append(agent)

    def levy_flight(self, agent_pos, beta=1.5):
        """
        莱维飞行步长计算
        """
        sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u, size=len(agent_pos))
        v = np.random.normal(0, sigma_v, size=len(agent_pos))

        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step  # 0.01 是步长控制因子

    def evolve(self, epoch):
        """
        重写进化过程：在标准 HHO 后加入 Levy Flight 突变
        """
        # 1. 执行原版 HHO 的一轮进化 (super().evolve 会更新 self.pop)
        # 注意：mealpy 的 evolve 通常不返回值，而是直接修改 self.pop
        super().evolve(epoch)

        # 2. 改进策略：对部分适应度较差的鹰应用 Levy Flight
        # 只有当当前的鹰比平均水平差，或者随机概率触发时，才进行突变
        pop_new = []
        for idx, agent in enumerate(self.pop):
            # 复制当前代理
            pos_new = agent.solution.copy()

            # 策略：如果这只鹰不是最好的那只 (Rabbit)，有概率进行莱维飞行
            if agent.target.fitness > self.g_best.target.fitness:  # 假设是求最小化
                if np.random.rand() < 0.5:  # 50% 概率触发突变
                    # 生成莱维飞行步长
                    step = self.levy_flight(pos_new)
                    # 更新位置: pos = pos + step * (pos - best_pos)
                    pos_new = pos_new + step * (pos_new - self.g_best.solution)

                    # 边界检查
                    pos_new = self.correct_solution(pos_new)

                    # 重新评估并择优保留
                    agent_new = self.generate_agent(pos_new)
                    if self.compare_agent(agent_new, agent):
                        self.pop[idx] = agent_new

        # 更新全局最优 (Rabbit)
        _, self.g_best = self.get_global_best_agent(self.pop)