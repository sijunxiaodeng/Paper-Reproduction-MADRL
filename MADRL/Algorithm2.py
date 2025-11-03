
from env import *

def algorithm2_multi_defender_attacker(
        num_hosts=3,
        num_defenders=3,
        num_attackers=3,
        defender_resources_range=[5, 7],  # 每个防御者资源范围（平均6）
        attacker_resources_range=[5, 7],  # 每个攻击者资源范围（平均6）
        host_importances=None,
        T=5,
        num_rounds=10000,
        epsilon=0.8,
        gamma=0.8,
        batch_size=3,
        lr=0.1          #学习效率
):
    """
    算法2：多防御者-多攻击者场景的深度Q学习
    返回：每轮的防御者群体效用和各防御者损失
    """
    # 初始化主机（默认重要性为2，范围[1,3]）
    if host_importances is None:
        host_importances = np.random.uniform(1, 3, num_hosts)
    hosts = [Host(i, host_importances[i]) for i in range(num_hosts)]

    # 初始化防御者（资源随机生成自指定范围）
    defenders = []
    for i in range(num_defenders):
        res = np.random.randint(defender_resources_range[0], defender_resources_range[1] + 1)
        defender = Defender(
            agent_id=i,
            total_resources=res,
            num_hosts=num_hosts,
            num_defenders=num_defenders,
            num_attackers=num_attackers,
            T=T
        )
        # 初始化DQN：状态维度 = T*(M*N + L*N)


        state_dim = T * (num_defenders * num_hosts + num_attackers * num_hosts)  # 状态维度
        action_dim = num_hosts  # 动作维度是主机数量（资源分配向量长度）
        input_dim = state_dim + action_dim  # DQN实际输入维度 = 状态 + 动作
        defender.init_dqn(input_dim, 1, lr=lr)  # 输出维度固定为1（Q值）
        defenders.append(defender)

    # 初始化攻击者（资源随机生成自指定范围）
    attackers = []
    for i in range(num_attackers):
        res = np.random.randint(attacker_resources_range[0], attacker_resources_range[1] + 1)
        attacker = Attacker(
            agent_id=i,
            total_resources=res,
            num_hosts=num_hosts,
            host_importances=host_importances,
            T=T
        )
        attackers.append(attacker)

    # 记录训练过程
    group_utilities = []
    all_losses = [[] for _ in defenders]  # 每个防御者的损失
    total_def_res = sum(d.total_resources for d in defenders)
    total_att_res = sum(a.total_resources for a in attackers)
    ne_utility = 0.0 if total_def_res == total_att_res else -1.0  # 纳什均衡效用

    for round in range(num_rounds):
        # 1. 重置主机资源
        for host in hosts:
            host.reset_resources()

        # 2. 获取所有防御者当前状态
        current_states = [d.get_current_state() for d in defenders]

        # 3. 选择策略（算法2 Line8）
        def_strats = []
        for defender in defenders:
            strat = defender.select_strategy(epsilon=epsilon)
            def_strats.append(strat)

        att_strats = []
        for attacker in attackers:
            strat = attacker.select_strategy()
            att_strats.append(strat)

        # 4. 执行策略（算法2 Line9）
        # 防御者分配资源
        for strat in def_strats:
            for i, host in enumerate(hosts):
                host.add_defense_resource(strat[i])
        # 攻击者分配资源
        for strat in att_strats:
            for i, host in enumerate(hosts):
                host.add_attack_resource(strat[i])

        # 5. 计算防御者群体效用（论文公式3）
        group_utility = 0.0
        for host in hosts:
            outcome = host.get_outcome()
            group_utility += host.importance * outcome
        group_utilities.append(group_utility)

        # 6. 计算每个防御者的奖励（算法2 Line10）
        rewards = []
        for i, defender in enumerate(defenders):
            used_res = np.sum(def_strats[i])
            reward = defender.calculate_reward(group_utility, used_res)
            rewards.append(reward)

        # 7. 更新观测历史（算法2 Line13）
        observation = (np.array(def_strats), np.array(att_strats))  # (M策略, L策略)
        for defender in defenders:
            defender.update_state_history(observation)
        for attacker in attackers:
            attacker.update_defense_history(np.array(def_strats))

        # 8. 获取下一状态
        next_states = [d.get_current_state() for d in defenders]

        # 9. 存储经验（算法2 Line14）
        for i, defender in enumerate(defenders):
            defender.add_experience(
                current_states[i],
                def_strats[i],
                rewards[i],
                next_states[i]
            )

        # 10. 经验共享：收集所有防御者的经验（算法2 Line15核心差异）
        shared_experience = []
        for defender in defenders:
            shared_experience.extend(defender.experience_replay)

        # 11. 训练所有防御者的DQN（算法2 Line16-17）
        for i, defender in enumerate(defenders):
            # 为当前防御者过滤可用经验（策略必须在其动作空间内）
            valid_experience = []
            for exp in shared_experience:
                _, action, _, _ = exp
                if defender.validate_strategy(action):
                    valid_experience.append(exp)

            # 临时替换经验池进行训练（模拟经验共享）
            original_replay = defender.experience_replay
            defender.experience_replay = deque(valid_experience, maxlen=10000)

            # 训练
            loss = defender.train_dqn(gamma=gamma, batch_size=batch_size)
            all_losses[i].append(loss)

            # 恢复原始经验池
            defender.experience_replay = original_replay



    return group_utilities, all_losses, ne_utility