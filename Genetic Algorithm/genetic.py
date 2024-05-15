import random
from deap import base, creator, tools, algorithms


# 定义适应度函数
# 目标是计算个体中所有元素的和，返回值是一个元组
def evaluate(individual):
    return sum(individual),


# 设置遗传算法参数
IND_SIZE = 10  # 个体大小（数组长度）
POP_SIZE = 300  # 种群大小
NGEN = 40  # 迭代次数
CXPB = 0.5  # 交叉概率
MUTPB = 0.2  # 变异概率

# 创建适应度类和个体类
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 定义适应度为最大化
creator.create("Individual", list, fitness=creator.FitnessMax)  # 定义个体为列表，并包含适应度属性

# 初始化工具箱
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)  # 定义个体基因为随机浮点数
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)  # 定义个体初始化
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 定义种群初始化

# 注册遗传算法操作
toolbox.register("mate", tools.cxTwoPoint)  # 注册交叉操作，使用两点交叉
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # 注册变异操作，以0.05的概率翻转位
toolbox.register("select", tools.selTournament, tournsize=3)  # 注册选择操作，使用锦标赛选择
toolbox.register("evaluate", evaluate)  # 注册评估函数


def main():
    # 初始化种群
    population = toolbox.population(n=POP_SIZE)

    # 定义统计数据的收集和报告
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(v[0] for v in x) / len(x))  # 记录平均适应度
    stats.register("min", lambda x: min(v[0] for v in x))  # 记录最小适应度
    stats.register("max", lambda x: max(v[0] for v in x))  # 记录最大适应度

    # 运行遗传算法
    algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, verbose=True)

    # 输出结果
    best_ind = tools.selBest(population, 1)[0]  # 选择适应度最高的个体
    print(f'Best individual: {best_ind}')  # 打印最佳个体的基因
    print(f'Best fitness: {best_ind.fitness.values[0]}')  # 打印最佳个体的适应度


# 运行主函数
if __name__ == "__main__":
    main()
