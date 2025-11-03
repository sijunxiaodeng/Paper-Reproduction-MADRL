import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


from env import Attacker,Defender,Host

# ========================
# 算法1实现
# ========================
def algorithm1_single_defender_attacker(
        num_hosts=3,
        defender_resources=6,
        attacker_resources=6,
        host_importances=None,
        T=5,
        num_rounds=10000,
        epsilon=0.8,
        gamma=0.8,
        batch_size=3,
        lr=0.001
):
    if host_importances is None:
        host_importances = np.random.uniform(1, 3, num_hosts)
    hosts = [Host(i, host_importances[i]) for i in range(num_hosts)]

    defender = Defender(
        agent_id=0,
        total_resources=defender_resources,
        num_hosts=num_hosts,
        num_defenders=1,
        num_attackers=1,
        T=T
    )
    attacker = Attacker(
        agent_id=0,
        total_resources=attacker_resources,
        num_hosts=num_hosts,
        host_importances=host_importances,
        T=T
    )

    state_dim = T * (1 * num_hosts + 1 * num_hosts)  # 状态维度
    action_dim = num_hosts  # 动作是长度为num_hosts的策略向量
    input_dim = state_dim + action_dim  # 输入维度=状态+动作
    defender.init_dqn(input_dim, 1, lr=lr)  # 输出维度为1（Q值）


    defender_utilities = []
    losses = []
    ne_utility = 0.0  # 资源相等时的纳什均衡效用

    for round in range(num_rounds):
        # 重置主机资源
        for host in hosts:
            host.reset_resources()

        # 获取状态与选择策略
        current_state = defender.get_current_state()
        defender_strat = defender.select_strategy(epsilon=epsilon)
        attacker_strat = attacker.select_strategy()

        # 执行策略
        for i, host in enumerate(hosts):
            host.add_defense_resource(defender_strat[i])
            host.add_attack_resource(attacker_strat[i])

        # 计算效用与奖励
        group_utility = sum(host.importance * host.get_outcome() for host in hosts)
        defender_utilities.append(group_utility)
        used_resources = np.sum(defender_strat)
        reward = defender.calculate_reward(group_utility, used_resources)

        # 更新历史与经验
        observation = (np.array([defender_strat]), np.array([attacker_strat]))
        defender.update_state_history(observation)
        attacker.update_defense_history(np.array([defender_strat]))
        next_state = defender.get_current_state()
        defender.add_experience(current_state, defender_strat, reward, next_state)

        # 训练与调整探索率
        loss = defender.train_dqn(gamma=gamma, batch_size=batch_size)
        losses.append(loss)
        if round % 1000 == 0 and epsilon > 0.1:
            epsilon *= 0.9

    return defender_utilities, losses, ne_utility


# ========================
# 运行算法1并可视化结果
# ========================
def run_algorithm1():
    # 运行参数
    num_rounds = 1000
    num_hosts = 3
    host_importances = np.array([2.0, 2.5, 1.8])  # 固定主机重要性

    # 执行算法1
    print("运行算法1（单防御者-单攻击者）...")
    utilities, losses, ne_utility = algorithm1_single_defender_attacker(
        num_hosts=num_hosts,
        defender_resources=6,
        attacker_resources=6,
        host_importances=host_importances,
        num_rounds=num_rounds
    )

    # 结果可视化
    plot_results(utilities, losses, ne_utility, num_rounds)
    return utilities, losses


def plot_results(utilities, losses, ne_utility, num_rounds):
    # 设置中文字体，解决中文显示问题
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 平滑处理（100轮移动平均）
    window = 100
    smoothed_utils = np.convolve(utilities, np.ones(window) / window, mode='valid')
    smoothed_losses = np.convolve(losses, np.ones(window) / window, mode='valid')
    x = np.arange(len(smoothed_utils))

    # 绘制效用曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, smoothed_utils, label="防御者效用（平滑后）")
    plt.axhline(y=ne_utility, color='r', linestyle='--', label="纳什均衡效用")
    plt.xlabel(f"训练轮次（每{window}轮平均）")
    plt.ylabel("防御者效用")
    plt.title("算法1：效用变化曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(x, smoothed_losses, label="DQN损失（平滑后）", color='orange')
    plt.xlabel(f"训练轮次（每{window}轮平均）")
    plt.ylabel("损失值")
    plt.title("算法1：损失变化曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 输出关键指标
    final_utils = utilities[-1000:]
    print(f"\n算法1关键指标：")
    print(f"最终1000轮平均效用：{np.mean(final_utils):.2f}")
    print(f"最终1000轮效用标准差：{np.std(final_utils):.2f}")
    print(f"最终1000轮平均损失：{np.mean(losses[-1000:]):.6f}")


if __name__ == "__main__":
    run_algorithm1()