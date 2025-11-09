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
            nn.Linear(state_dim, 1000),#the number of nodes 1000
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

        """但是按照原文有可能出现部分可观测情况暂时未写入算法"""

        """获取当前状态（扁平化的历史观测向量）
        状态维度：T * (M*N + L*N)，其中N为host数量
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

        # 扁平化状态向量  多维->一维
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
                    # 拼接状态和动作，转换为张量并迁移到GPU
                    sa_tensor = torch.tensor(
                        np.concatenate([state, strat]),
                        dtype=torch.float32
                    ).to(device)  # 数据迁移到GPU
                    q_val = self.dqn(sa_tensor).item()  # 模型在GPU上计算
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

    def train_dqn(self, gamma=0.8, batch_size=32):
        if len(self.experience_replay) < batch_size:
            return 0.0

        # 采样批量样本
        batch = self.strategy_based_sampling(batch_size)
        states, actions, rewards, next_states = zip(*batch)  # 解压为列表

        # 1. 列表→numpy数组（解决警告）
        states_np = np.array(states)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards)
        next_states_np = np.array(next_states)

        # 2. numpy数组→张量，并迁移到GPU（解决设备不匹配）
        states_tensor = torch.tensor(states_np, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions_np, dtype=torch.float32).to(device)
        rewards_tensor = torch.tensor(rewards_np, dtype=torch.float32).to(device)
        next_states_tensor = torch.tensor(next_states_np, dtype=torch.float32).to(device)

        # 3. 计算当前Q值（确保输入在GPU）
        current_input = torch.cat([states_tensor, actions_tensor], dim=1)  # 拼接状态和动作
        current_q = self.dqn(current_input).squeeze()  # 模型在GPU上计算

        # 4. 计算目标Q值（确保下一状态的输入也在GPU）
        target_q = []
        for i in range(batch_size):
            next_state = next_states[i]
            # 生成下一状态的候选动作（假设已实现）
            next_candidate_actions = self.generate_candidate_actions(next_state)
            # 计算每个候选动作的Q值（确保输入在GPU）
            next_q_values = []
            for act in next_candidate_actions:
                ns_action_np = np.concatenate([next_state, act])  # 先转numpy
                ns_action_tensor = torch.tensor(ns_action_np, dtype=torch.float32).to(device)  # 迁移到GPU
                next_q = self.dqn(ns_action_tensor).item()
                next_q_values.append(next_q)
            max_next_q = max(next_q_values)
            target_q.append(rewards[i] + gamma * max_next_q)

        # 5. 目标Q值转张量并迁移到GPU
        target_q_tensor = torch.tensor(target_q, dtype=torch.float32).to(device)

        # 6. 计算损失并更新（均在GPU上）
        loss = self.loss_fn(current_q, target_q_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def generate_candidate_actions(self, next_state, num_candidates=10):
        """
        为下一状态生成候选动作（资源分配策略）
        :param next_state: 下一状态（无需使用，仅为接口统一）
        :param num_candidates: 候选策略数量
        :return: 候选策略列表（每个元素是形状为[num_hosts]的资源分配数组）
        """
        candidates = []
        for _ in range(num_candidates):
            # 调用已有的随机策略生成方法，生成合法的资源分配策略
            strategy = self.generate_random_strategy()
            candidates.append(strategy)
        return candidates

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