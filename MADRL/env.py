import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Host:
    def __init__(self,host_id,importance_value):
        self.host_id = host_id  # 主机ID
        self.importance = importance_value  # 主机重要性（u_k）
        self.defense_resources = 0  # 分配的防御资源总和
        self.attack_resources = 0  # 分配的攻击资源总和

    def reset_resources(self):
        """重置当前轮次的资源分配"""
        self.defense_resources = 0
        self.attack_resources = 0

    def add_defense_resource(self, amount):
        """添加防御资源"""
        self.defense_resources += amount

    def add_attack_resource(self, amount):
        """添加攻击资源"""
        self.attack_resources += amount

    def get_outcome(self):
        """计算当前主机的攻防结果（符号函数）
        返回：1（防御方胜）、-1（攻击方胜）、0（平局）
        """
        diff = self.defense_resources - self.attack_resources
        if diff > 0:
            return 1
        elif diff < 0:
            return -1
        else:
            return 0


class Agent:
    def __init__(self, agent_id, total_resources, num_hosts):
        self.agent_id = agent_id  # 智能体ID
        self.total_resources = total_resources  # 总资源量（B_i/C_j）
        self.num_hosts = num_hosts  # 主机数量
        self.strategy = np.zeros(num_hosts, dtype=int)  # 当前策略（资源分配向量）

    def reset_strategy(self):
        """重置策略"""
        self.strategy = np.zeros(self.num_hosts, dtype=int)

    def validate_strategy(self, strategy):
        """验证策略合法性：资源非负且总和不超过总资源"""
        if np.any(strategy < 0):
            return False
        if np.sum(strategy) > self.total_resources:
            return False
        return True

    def generate_random_strategy(self):
        """生成随机合法策略（离散资源分配）"""
        remaining = self.total_resources
        strategy = np.zeros(self.num_hosts, dtype=int)

        for i in range(self.num_hosts - 1):
            if remaining <= 0:
                break
            # 为当前主机分配0到剩余资源的随机数
            alloc = np.random.randint(0, remaining + 1)
            strategy[i] = alloc
            remaining -= alloc

        # 剩余资源分配给最后一个主机
        strategy[-1] = remaining
        return strategy

    def get_strategy(self):
        """获取当前策略"""
        return self.strategy.copy()

#防御者类，继承Agent加入e贪婪策略等
class Defender(Agent):
    def __init__(self, agent_id, total_resources, num_hosts, num_defenders, num_attackers, T=5):
        super().__init__(agent_id, total_resources, num_hosts)
        self.num_defenders = num_defenders  # 防御者总数（M）
        self.num_attackers = num_attackers  # 攻击者总数（L）
        self.T = T  # 状态窗口大小（历史轮次数量）
        self.experience_replay = deque(maxlen=10000)  # 经验回放池
        self.state_history = deque(maxlen=T)  # 状态历史（最近T轮的观测）
        self.dqn = None  # DQN网络
        self.optimizer = None  # 优化器
        self.loss_fn = None  # 损失函数

    def init_dqn(self, state_dim, action_dim, lr=0.1):
        """初始化DQN网络"""
        self.dqn = nn.Sequential(
            nn.Linear(state_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, action_dim)
        ).to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

    def update_state_history(self, observation):
        """更新状态历史
        observation: 单轮观测，格式为（所有防御者策略，所有攻击者策略）
        """
        self.state_history.append(observation)

    def get_current_state(self):
        """获取当前状态（扁平化的历史观测向量）
        状态维度：T * (M*N + L*N)，其中N为宿主数量
        """
        if len(self.state_history) < self.T:
            # 状态历史不足时，用全零向量填充
            pad_length = self.T - len(self.state_history)
            padded_history = []
            for _ in range(pad_length):
                pad_def = np.zeros((self.num_defenders, self.num_hosts))
                pad_att = np.zeros((self.num_attackers, self.num_hosts))
                padded_history.append((pad_def, pad_att))

            full_history = padded_history + list(self.state_history)
        else:
            full_history = list(self.state_history)

        # 扁平化状态向量
        state_vec = []
        for def_strats, att_strats in full_history:
            state_vec.extend(def_strats.flatten())
            state_vec.extend(att_strats.flatten())

        return np.array(state_vec, dtype=np.float32)

    def select_strategy(self, epsilon=0.8):
        """基于ε-贪婪策略选择动作（算法1/2的核心选择逻辑）"""
        if np.random.random() < epsilon:
            # 探索：随机选择策略
            self.strategy = self.generate_random_strategy()
        else:
            # 利用：DQN选择最优策略
            if self.dqn is None or len(self.state_history) < self.T:
                self.strategy = self.generate_random_strategy()
            else:
                state = self.get_current_state()
                state_tensor = torch.tensor(state, dtype=torch.float32)

                # 获取所有可能策略的Q值（注：实际应用中需优化动作空间表示）
                # 此处为简化实现，采用随机采样策略评估Q值
                candidate_strats = [self.generate_random_strategy() for _ in range(20)]
                q_values = []

                for strat in candidate_strats:
                    # 状态-动作拼接（简化动作表示）
                    state_action = np.concatenate([state, strat])
                    sa_tensor = torch.tensor(state_action, dtype=torch.float32)
                    q_val = self.dqn(sa_tensor).item()
                    q_values.append(q_val)

                # 选择Q值最大的策略
                best_idx = np.argmax(q_values)
                self.strategy = candidate_strats[best_idx]

        return self.strategy.copy()

    def add_experience(self, state, action, reward, next_state):
        """添加经验到回放池"""
        self.experience_replay.append((state, action, reward, next_state))

    def strategy_based_sampling(self, batch_size=3):
        """基于策略相似度的采样（论文提出的新型采样方法）"""
        if len(self.experience_replay) < batch_size:
            return random.sample(self.experience_replay, len(self.experience_replay))

        # 提取所有经验中的动作（策略）
        all_actions = [exp[1] for exp in self.experience_replay]
        all_actions = np.array(all_actions)

        # 计算平均策略
        avg_strategy = np.mean(all_actions, axis=0)

        # 计算每个经验与平均策略的欧氏距离
        distances = [np.linalg.norm(action - avg_strategy) for action in all_actions]

        # 按距离降序排序，选择前batch_size个样本
        sorted_indices = np.argsort(distances)[::-1]
        selected_samples = [self.experience_replay[i] for i in sorted_indices[:batch_size]]

        return selected_samples

    def train_dqn(self, gamma=0.8, batch_size=3):
        """训练DQN网络（算法1/2的更新逻辑）"""
        if len(self.experience_replay) < batch_size or self.dqn is None:
            return 0.0  # 经验不足，不训练

        # 采样批次（算法1：仅自身经验；算法2：所有防御者经验）
        batch = self.strategy_based_sampling(batch_size)

        # 准备训练数据
        states = []
        actions = []
        rewards = []
        next_states = []

        for s, a, r, ns in batch:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)

        # 转换为张量
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)

        # 计算目标Q值
        target_q = []
        for i in range(batch_size):
            # 计算next_state的最大Q值
            next_state = next_states[i]
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            # 采样候选策略评估next_state的Q值
            candidate_strats = [self.generate_random_strategy() for _ in range(20)]
            next_q_values = []
            for strat in candidate_strats:
                ns_action = np.concatenate([next_state, strat])
                ns_action_tensor = torch.tensor(ns_action, dtype=torch.float32)
                next_q = self.dqn(ns_action_tensor).item()
                next_q_values.append(next_q)

            max_next_q = max(next_q_values) if next_q_values else 0.0
            target = rewards[i] + gamma * max_next_q
            target_q.append(target)

        target_q_tensor = torch.tensor(target_q, dtype=torch.float32)

        # 计算当前Q值
        current_q = []
        for i in range(batch_size):
            state_action = np.concatenate([states[i], actions[i]])
            sa_tensor = torch.tensor(state_action, dtype=torch.float32)
            q_val = self.dqn(sa_tensor)
            current_q.append(q_val)

        current_q_tensor = torch.stack(current_q).squeeze()

        # 反向传播优化
        loss = self.loss_fn(current_q_tensor, target_q_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def calculate_reward(self, group_utility, used_resources):
        """计算单轮奖励（论文定义：个体效用/使用资源）"""
        if used_resources == 0:
            return 0.0
        individual_utility = group_utility / self.num_defenders
        return individual_utility / used_resources



  #Attacker
class Attacker(Agent):
        def __init__(self, agent_id, total_resources, num_hosts, host_importances, T=5):
            super().__init__(agent_id, total_resources, num_hosts)
            self.host_importances = host_importances  # 主机重要性列表
            self.T = T  # 历史窗口大小
            self.defense_history = deque(maxlen=T)  # 防御者策略历史

        def update_defense_history(self, defense_strategies):
            """更新防御者策略历史（所有防御者的策略）"""
            self.defense_history.append(defense_strategies)

        def calculate_avg_defense(self):
            """计算防御者对各主机的平均资源分配"""
            if not self.defense_history:
                return np.zeros(self.num_hosts)

            avg_def = np.zeros(self.num_hosts)
            for def_strats in self.defense_history:
                # 计算单轮所有防御者的资源总和
                round_total = np.sum(def_strats, axis=0)
                avg_def += round_total

            avg_def /= len(self.defense_history)
            return avg_def

        def select_strategy(self):
            """基于主机重要性和防御历史选择攻击策略（论文定义的攻击者策略）"""
            avg_def = self.calculate_avg_defense()
            num_hosts = self.num_hosts

            # 按重要性降序排序主机ID
            host_indices = np.argsort(self.host_importances)[::-1]

            remaining = self.total_resources
            self.strategy = np.zeros(num_hosts, dtype=int)

            for host_id in host_indices:
                if remaining <= 0:
                    break

                # 针对重要主机，分配比平均防御多的资源（至少1个）
                needed = max(int(np.ceil(avg_def[host_id])), 1)
                alloc = min(needed, remaining)
                self.strategy[host_id] = alloc
                remaining -= alloc

            # 剩余资源分配给第一个主机
            if remaining > 0:
                first_host = host_indices[0]
                self.strategy[first_host] += remaining

            return self.strategy.copy()