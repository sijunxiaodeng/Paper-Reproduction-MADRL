import numpy as np
from env import Host,Defender,Attacker


def algorithm1_single_defender_attacker(
    num_hosts=3,
    defender_resources=6,
    attacker_resources=6,
    host_importances=None,
    T=5,
    num_rounds=1000,
    epsilon=0.8,
    gamma=0.8,
    batch_size=3,
    lr=0.1
):
    """
    算法1：单防御者-单攻击者场景的深度Q学习
    返回：每轮的防御者效用和损失
    """
    # 初始化主机（默认重要性为2，范围[1,3]）
    if host_importances is None:
        host_importances = np.random.uniform(1, 3, num_hosts)
    hosts = [Host(i, host_importances[i]) for i in range(num_hosts)]

    # 初始化智能体
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

    # 初始化DQN：状态维度 = T*(1*N + 1*N)，动作维度简化为1（策略向量通过采样评估）
    state_dim = T * (1 * num_hosts + 1 * num_hosts)

    action_dim = 1  # 实际动作空间为离散策略集合，此处简化
    defender.init_dqn(state_dim, action_dim, lr=lr)

    # 记录训练过程
    defender_utilities = []
    losses = []
    ne_utility = 0.0  # 纳什均衡效用（论文定理2：资源相等时为0）

    for round in range(num_rounds):
        # 1. 重置主机资源
        for host in hosts:
            host.reset_resources()

        # 2. 获取当前状态
        current_state = defender.get_current_state()

        # 3. 选择策略（算法1 Line7）
        defender_strat = defender.select_strategy(epsilon=epsilon)
        attacker_strat = attacker.select_strategy()

        # 4. 执行策略（算法1 Line8）
        # 防御者分配资源
        for i, host in enumerate(hosts):
            host.add_defense_resource(defender_strat[i])
        # 攻击者分配资源
        for i, host in enumerate(hosts):
            host.add_attack_resource(attacker_strat[i])

        # 5. 计算效用（论文公式3/5）
        defender_group_utility = 0.0
        for host in hosts:
            outcome = host.get_outcome()
            defender_group_utility += host.importance * outcome
        defender_utilities.append(defender_group_utility)

        # 6. 计算奖励（算法1 Line9）
        used_resources = np.sum(defender_strat)
        reward = defender.calculate_reward(defender_group_utility, used_resources)

        # 7. 更新观测历史（算法1 Line12）
        observation = (np.array([defender_strat]), np.array([attacker_strat]))  # (M=1策略, L=1策略)
        defender.update_state_history(observation)
        attacker.update_defense_history(np.array([defender_strat]))

        # 8. 获取下一状态
        next_state = defender.get_current_state()

        # 9. 存储经验（算法1 Line13）
        defender.add_experience(current_state, defender_strat, reward, next_state)

        # 10. 训练DQN（算法1 Line14-16）
        loss = defender.train_dqn(gamma=gamma, batch_size=batch_size)
        losses.append(loss)



    return defender_utilities, losses, ne_utility