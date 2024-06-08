import numpy as np

# 定义Rosenbrock适应度函数
def fitness_function(position):
    # Rosenbrock函数的参数
    a = 1000
    b = 100000
    # 计算Rosenbrock函数值
    x = position[0]
    y = position[1]
    return (a - x)**2 + b * (y - x**2)**2

# 粒子类定义
class Particle:
    def __init__(self, dim):
        # 初始化粒子的位置，范围在[-10, 10]之间
        self.position = np.random.uniform(-10, 10, dim)
        # 初始化粒子的速度，范围在[-1, 1]之间
        self.velocity = np.random.uniform(-1, 1, dim)
        # 初始化粒子的历史最优位置为当前的位置
        self.best_position = self.position.copy()
        # 计算当前适应度值，并将其作为历史最优适应度值
        self.best_fitness = fitness_function(self.position)

# PSO算法定义
class PSO:
    def __init__(self, particle_count, dim, max_iter, w, c1, c2):
        # 初始化粒子群，包含particle_count个粒子，每个粒子的维度为dim
        self.particles = [Particle(dim) for _ in range(particle_count)]
        # 保存粒子的维度
        self.dim = dim
        # 最大迭代次数
        self.max_iter = max_iter
        # 惯性权重
        self.w = w
        # 个体学习因子
        self.c1 = c1
        # 群体学习因子
        self.c2 = c2
        # 初始化全局最优位置和适应度值
        self.global_best_position = np.random.uniform(-10, 10, dim)
        self.global_best_fitness = float('inf')

    def optimize(self):
        # 迭代优化
        for _ in range(self.max_iter):
            for particle in self.particles:
                # 计算当前粒子的适应度值
                fitness = fitness_function(particle.position)
                # 更新粒子的历史最优位置和适应度值
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                # 更新全局最优位置和适应度值
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

                # 更新粒子的速度
                inertia = self.w * particle.velocity
                cognitive = self.c1 * np.random.uniform(0, 1, self.dim) * (particle.best_position - particle.position)
                social = self.c2 * np.random.uniform(0, 1, self.dim) * (self.global_best_position - particle.position)
                particle.velocity = inertia + cognitive + social
                # 更新粒子的位置
                particle.position += particle.velocity

        # 返回全局最优位置和适应度值
        return self.global_best_position, self.global_best_fitness

# 设置PSO参数
particle_count = 30  # 粒子数量
dim = 2  # 维度
max_iter = 100  # 最大迭代次数
w = 0.5  # 惯性权重
c1 = 1.5  # 个体学习因子
c2 = 1.5  # 群体学习因子

# 创建PSO实例并运行优化过程
pso = PSO(particle_count, dim, max_iter, w, c1, c2)
best_position, best_fitness = pso.optimize()

# 输出最佳位置和最佳适应度值
print(f"最佳位置: {best_position}, 最佳适应度值: {best_fitness}")
