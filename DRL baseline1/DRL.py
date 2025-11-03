import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class CloudStorageSystem:
    def __init__(self, num_devices: int, total_defense_cpus: int, total_attack_cpus: int):
        """
        初始化云存储系统
        :param num_devices: 存储设备数量 D（论文符号）
        :param total_defense_cpus: 防御者总CPU数 S_M（论文符号）
        :param total_attack_cpus: 攻击者总CPU数 S_N（论文符号）
        """
        self.D = num_devices  # 存储设备数量
        self.S_M = total_defense_cpus  # 防御者总CPU
        self.S_N = total_attack_cpus  # 攻击者总CPU

        # 初始化数据量（论文中B_i^(k)，初始设为1~5的随机值，动态更新）
        self.B = np.random.randint(1, 6, size=self.D).astype(float)
        self.total_data = np.sum(self.B)  # 总数据量 B_hat（论文符号）

    def update_data_dynamic(self, time_step: int, update_interval: int = 1000) -> None:
        """
        动态更新存储设备数据量（论文仿真场景2设定：每1000步更新）
        :param time_step: 当前时间步
        :param update_interval: 数据更新间隔
        """
        if time_step % update_interval == 0 and time_step != 0:
            # 论文设定：每次更新数据量增加1.1~1.2倍
            growth_factor = np.random.uniform(1.1, 1.2)
            self.B = self.B * growth_factor
            self.total_data = np.sum(self.B)

    def generate_apt_attack(self, attack_strategy: str = "epsilon_greedy", epsilon: float = 0.1) -> np.ndarray:
        """
        生成APT攻击者的CPU分配策略（论文中N^(k)）
        :param attack_strategy: 攻击策略（epsilon_greedy/optimal）
        :param epsilon: 探索概率（仅epsilon_greedy策略用）
        :return: N: 攻击者CPU分配向量（D维）
        """
        if attack_strategy == "epsilon_greedy":
            # epsilon-greedy策略：以1-epsilon选择最优分配，epsilon随机分配
            if np.random.rand() < epsilon:
                # 随机分配（满足总CPU约束）
                N = np.random.rand(self.D)
                N = N / np.sum(N) * self.S_N if np.sum(N) != 0 else np.zeros(self.D)
            else:
                # 最优分配：按数据量比例分配（攻击数据量大的设备）
                data_ratio = self.B / self.total_data
                N = data_ratio * self.S_N
        elif attack_strategy == "optimal":
            # 最优攻击策略（基于论文NE）
            if self.S_M == self.S_N:
                # 对称CPU：按论文公式(14)均匀分配
                beta = 2 * self.S_M / self.total_data
                N = np.random.randint(0, int(np.floor(beta * self.B[0])) + 1, size=self.D)
            else:
                # 非对称CPU：按论文公式(24)分配
                N = np.zeros(self.D)
                prob_zero = 1 - self.S_N / self.S_M
                for i in range(self.D):
                    if np.random.rand() < prob_zero:
                        N[i] = 0
                    else:
                        max_alloc = int(np.floor(2 * self.S_M / self.D))
                        N[i] = np.random.randint(1, max_alloc + 1)
        else:
            raise ValueError("未知攻击策略")

        # 确保CPU分配非负且总和不超过S_N
        N = np.maximum(N, 0)
        if np.sum(N) > self.S_N:
            N = N / np.sum(N) * self.S_N
        return N

    def validate_defense_allocation(self, M: np.ndarray) -> bool:
        """
        验证防御者CPU分配是否合法（满足总CPU约束和非负）
        :param M: 防御者CPU分配向量
        :return: 合法则返回True
        """
        return np.all(M >= 0) and np.sum(M) <= self.S_M + 1e-6  # 允许微小浮点数误差


def calculate_data_protection_level(M: np.ndarray, N: np.ndarray, B: np.ndarray, total_data: float) -> float:
    """
    计算数据保护水平 R^(k)（论文公式3）
    :param M: 防御者CPU分配
    :param N: 攻击者CPU分配
    :param B: 设备数据量向量
    :param total_data: 总数据量
    :return: R: 数据保护水平（-1~1）
    """
    sgn = np.where(M > N, 1, np.where(M < N, -1, 0))
    return np.sum(B * sgn) / total_data if total_data != 0 else 0.0

def calculate_defender_utility(M: np.ndarray, N: np.ndarray, B: np.ndarray) -> float:
    """
    计算防御者效用 u_D^(k)（论文公式4）
    :param M: 防御者CPU分配
    :param N: 攻击者CPU分配
    :param B: 设备数据量向量
    :return: u_D: 防御者效用
    """
    sgn = np.where(M > N, 1, np.where(M < N, -1, 0))
    return np.sum(B * sgn)


class HotbootingPHC:
    def __init__(self, num_devices: int, total_defense_cpus: int, alpha: float = 0.9, gamma: float = 0.5,
                 delta: float = 0.02):
        """
        初始化热启动PHC算法
        :param num_devices: 存储设备数量 D
        :param total_defense_cpus: 防御者总CPU S_M
        :param alpha: 学习率（论文设定0.9）
        :param gamma: 折扣因子（论文设定0.5）
        :param delta: 策略更新步长（论文设定0.02）
        """
        self.D = num_devices
        self.S_M = total_defense_cpus
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta

        # 初始化Q函数（状态：(N_prev, B_curr)，动作：M）
        # 为简化存储，将状态离散化：N_prev和B_curr各分为10个区间
        self.state_bins = 10  # 状态离散化区间数
        self.Q = {}  # Q[(N_prev_bin, B_curr_bin)][M_idx] = Q值
        self.policy = {}  # 策略表：pi[(N_prev_bin, B_curr_bin)][M_idx] = 概率

        # 预生成所有可能的防御动作（CPU分配向量）
        self.actions = self._generate_all_actions()
        self.num_actions = len(self.actions)

    def _generate_all_actions(self) -> List[np.ndarray]:
        """
        生成所有合法的防御者CPU分配动作（满足总CPU约束）
        :return: 动作列表（每个元素为D维CPU分配向量）
        """
        actions = []

        # 采用递归方式生成所有分配组合（简化版：整数分配）
        def recursive_generate(idx: int, remaining_cpus: int, current_alloc: List[int]):
            if idx == self.D - 1:
                current_alloc.append(remaining_cpus)
                actions.append(np.array(current_alloc, dtype=float))
                current_alloc.pop()
                return
            for alloc in range(remaining_cpus + 1):
                current_alloc.append(alloc)
                recursive_generate(idx + 1, remaining_cpus - alloc, current_alloc)
                current_alloc.pop()

        recursive_generate(0, self.S_M, [])
        return actions

    def _discretize_state(self, N_prev: np.ndarray, B_curr: np.ndarray) -> Tuple[tuple, tuple]:
        """
        离散化状态（N_prev和B_curr）
        :param N_prev: 上一时刻攻击者CPU分配
        :param B_curr: 当前设备数据量
        :return: 离散化后的状态元组
        """
        # N_prev离散化（0~S_N分为state_bins个区间）
        N_prev_bin = tuple(np.digitize(n, bins=np.linspace(0, self.S_M, self.state_bins)) for n in N_prev)
        # B_curr离散化（0~max_B分为state_bins个区间）
        max_B = np.max(B_curr) if np.max(B_curr) != 0 else 1.0
        B_curr_bin = tuple(np.digitize(b, bins=np.linspace(0, max_B, self.state_bins)) for b in B_curr)
        return N_prev_bin, B_curr_bin

    def hotbooting_initialization(self, cloud_system: CloudStorageSystem, num_simulations: int = 1000) -> None:
        """
        热启动初始化：在相似场景中预训练Q函数和策略（论文Algorithm 2）
        :param cloud_system: 云存储系统实例
        :param num_simulations: 预训练模拟步数
        """
        print(f"开始热启动预训练（{num_simulations}步）...")
        N_prev = cloud_system.generate_apt_attack()  # 初始攻击分配

        for step in range(num_simulations):
            # 1. 观察当前数据量并离散化状态
            B_curr = cloud_system.B
            state = self._discretize_state(N_prev, B_curr)

            # 2. 初始化Q函数和策略（若状态未存在）
            if state not in self.Q:
                self.Q[state] = np.zeros(self.num_actions)
                self.policy[state] = np.ones(self.num_actions) / self.num_actions  # 均匀初始策略

            # 3. 根据当前策略选择动作
            action_idx = np.random.choice(self.num_actions, p=self.policy[state])
            M = self.actions[action_idx]

            # 4. 生成攻击并计算效用
            N_curr = cloud_system.generate_apt_attack()
            utility = calculate_defender_utility(M, N_curr, B_curr)

            # 5. 离散化下一状态
            next_state = self._discretize_state(N_curr, B_curr)
            if next_state not in self.Q:
                self.Q[next_state] = np.zeros(self.num_actions)
                self.policy[next_state] = np.ones(self.num_actions) / self.num_actions

            # 6. 更新Q函数（论文公式27）
            max_next_Q = np.max(self.Q[next_state])
            self.Q[state][action_idx] = (1 - self.alpha) * self.Q[state][action_idx] + \
                                        self.alpha * (utility + self.gamma * max_next_Q)

            # 7. 更新策略（论文公式29）
            best_action_idx = np.argmax(self.Q[state])
            for a_idx in range(self.num_actions):
                if a_idx == best_action_idx:
                    self.policy[state][a_idx] += self.delta
                else:
                    self.policy[state][a_idx] -= self.delta / (self.num_actions - 1)
            # 归一化策略概率
            self.policy[state] = np.maximum(self.policy[state], 0)  # 确保非负
            self.policy[state] /= np.sum(self.policy[state])

            # 8. 更新数据和上一时刻攻击
            cloud_system.update_data_dynamic(step)
            N_prev = N_curr

        print("热启动预训练完成！")

    def select_action(self, N_prev: np.ndarray, B_curr: np.ndarray) -> np.ndarray:
        """
        根据当前状态选择防御动作（论文公式30）
        :param N_prev: 上一时刻攻击者CPU分配
        :param B_curr: 当前设备数据量
        :return: M: 防御者CPU分配向量
        """
        state = self._discretize_state(N_prev, B_curr)
        # 若状态未见过，初始化
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)
            self.policy[state] = np.ones(self.num_actions) / self.num_actions
        # 根据策略选择动作
        action_idx = np.random.choice(self.num_actions, p=self.policy[state])
        return self.actions[action_idx]

    def update(self, N_prev: np.ndarray, B_curr: np.ndarray, M: np.ndarray, utility: float, N_curr: np.ndarray,
               B_next: np.ndarray) -> None:
        """
        更新Q函数和策略（论文公式27、29）
        :param N_prev: 上一时刻攻击分配
        :param B_curr: 当前数据量
        :param M: 选择的动作
        :param utility: 获得的效用
        :param N_curr: 当前攻击分配（下一状态的N_prev）
        :param B_next: 下一时刻数据量（下一状态的B_curr）
        """
        # 获取动作索引
        action_idx = np.argmin(np.sum(np.abs(self.actions - M), axis=1))  # 找到最接近的动作
        # 离散化状态
        state = self._discretize_state(N_prev, B_curr)
        next_state = self._discretize_state(N_curr, B_next)

        # 初始化状态（若未存在）
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)
            self.policy[state] = np.ones(self.num_actions) / self.num_actions
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(self.num_actions)
            self.policy[next_state] = np.ones(self.num_actions) / self.num_actions

        # 更新Q函数
        max_next_Q = np.max(self.Q[next_state])
        self.Q[state][action_idx] = (1 - self.alpha) * self.Q[state][action_idx] + \
                                    self.alpha * (utility + self.gamma * max_next_Q)

        # 更新策略
        best_action_idx = np.argmax(self.Q[state])
        for a_idx in range(self.num_actions):
            if a_idx == best_action_idx:
                self.policy[state][a_idx] += self.delta
            else:
                self.policy[state][a_idx] -= self.delta / (self.num_actions - 1)
        # 归一化策略
        self.policy[state] = np.maximum(self.policy[state], 0)
        self.policy[state] /= np.sum(self.policy[state])


class CNNQNetwork(nn.Module):
    """CNN Q网络（论文表II参数）"""

    def __init__(self, num_devices: int, num_actions: int, state_bins: int = 10):
        """
        初始化CNN Q网络
        :param num_devices: 存储设备数量 D
        :param num_actions: 动作数量
        :param state_bins: 状态离散化区间数
        """
        super(CNNQNetwork, self).__init__()
        # 输入：5x5矩阵（论文图4：经验序列φ^(k)的 reshaped 结果）
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=2, stride=1)  # Conv1: 20个2x2滤波器
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=2, stride=1)  # Conv2: 40个2x2滤波器
        self.relu = nn.ReLU()

        # 全连接层输入维度计算：Conv2输出3x3x40 → 360维
        self.fc1 = nn.Linear(3 * 3 * 40, 180)  # FC1: 360→180
        self.fc2 = nn.Linear(180, num_actions)  # FC2: 180→动作数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: 输入张量（batch_size, 1, 5, 5）
        :return: Q值张量（batch_size, num_actions）
        """
        x = self.relu(self.conv1(x))  # (batch, 20, 4, 4)
        x = self.relu(self.conv2(x))  # (batch, 40, 3, 3)
        x = x.view(x.size(0), -1)  # 展平：(batch, 360)
        x = self.relu(self.fc1(x))  # (batch, 180)
        x = self.fc2(x)  # (batch, num_actions)
        return x


class ReplayMemory:
    """经验回放池（论文图4）"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        """存储经验（φ, a, r, φ', done）"""
        self.memory.append(experience)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样批量经验"""
        batch = random.sample(self.memory, batch_size)
        # 转换为张量
        phi_batch = torch.FloatTensor(np.array([exp[0] for exp in batch])).unsqueeze(1)  # (batch, 1, 5, 5)
        action_batch = torch.LongTensor([exp[1] for exp in batch]).unsqueeze(1)  # (batch, 1)
        reward_batch = torch.FloatTensor([exp[2] for exp in batch]).unsqueeze(1)  # (batch, 1)
        phi_next_batch = torch.FloatTensor(np.array([exp[3] for exp in batch])).unsqueeze(1)  # (batch, 1, 5, 5)
        done_batch = torch.FloatTensor([exp[4] for exp in batch]).unsqueeze(1)  # (batch, 1)
        return phi_batch, action_batch, reward_batch, phi_next_batch, done_batch

    def __len__(self) -> int:
        return len(self.memory)


class HotbootingDQN:
    def __init__(self, num_devices: int, total_defense_cpus: int,
                 batch_size: int = 32, gamma: float = 0.5, epsilon: float = 0.1,
                 lr: float = 1e-3, memory_capacity: int = 10000, W: int = 12):
        """
        初始化热启动DQN算法（论文Algorithm 3）
        :param num_devices: 存储设备数量 D
        :param total_defense_cpus: 防御者总CPU S_M
        :param batch_size: 批量大小（论文未指定，设为32）
        :param gamma: 折扣因子（论文设定0.5）
        :param epsilon: ε-greedy探索概率（论文公式32）
        :param lr: 学习率（论文未指定，设为1e-3）
        :param memory_capacity: 经验回放池容量
        :param W: 经验序列长度（论文设定W=12）
        """
        self.D = num_devices
        self.S_M = total_defense_cpus
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.W = W

        # 生成所有合法动作
        self.actions = self._generate_all_actions()
        self.num_actions = len(self.actions)

        # 初始化Q网络和目标网络
        self.q_network = CNNQNetwork(num_devices, self.num_actions)
        self.target_network = CNNQNetwork(num_devices, self.num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())  # 初始同步
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # 损失函数（论文公式33）

        # 经验回放池
        self.memory = ReplayMemory(memory_capacity)

        # 经验序列φ^(k)：存储最近W个（状态-动作）对
        self.experience_sequence = deque(maxlen=W)

    def _generate_all_actions(self) -> List[np.ndarray]:
        """生成所有合法防御动作（同PHC算法）"""
        actions = []

        def recursive_generate(idx: int, remaining_cpus: int, current_alloc: List[int]):
            if idx == self.D - 1:
                current_alloc.append(remaining_cpus)
                actions.append(np.array(current_alloc, dtype=float))
                current_alloc.pop()
                return
            for alloc in range(remaining_cpus + 1):
                current_alloc.append(alloc)
                recursive_generate(idx + 1, remaining_cpus - alloc, current_alloc)
                current_alloc.pop()

        recursive_generate(0, self.S_M, [])
        return actions

    def _construct_experience_sequence(self) -> np.ndarray:
        """
        构建经验序列φ^(k)（论文公式31）
        :return: φ: 5x5矩阵（经验序列 reshaped 结果）
        """
        # 填充经验序列（不足W个时用0填充）
        seq = list(self.experience_sequence)
        while len(seq) < self.W:
            seq.insert(0, (np.zeros(self.D), np.zeros(self.D), np.zeros(self.D)))  # (N_prev, B_curr, M)

        # 提取特征并reshape为5x5矩阵（论文图4）
        features = []
        for (N_prev, B_curr, M) in seq[:5]:  # 取前5个经验
            features.extend([N_prev[:2], B_curr[:2], M[:2]])  # 每个经验取前2维，共3*2=6维（截断为5维）
        phi = np.array(features[:25]).reshape(5, 5)  # 25个特征→5x5矩阵
        return phi

    def hotbooting_initialization(self, cloud_system: CloudStorageSystem, num_simulations: int = 1000,
                                  H: int = 16) -> None:
        """
        热启动初始化：预训练CNN参数（论文Algorithm 4）
        :param cloud_system: 云存储系统实例
        :param num_simulations: 预训练步数
        :param H: 每次更新迭代次数（论文设定H=16）
        """
        print(f"开始DQN热启动预训练（{num_simulations}步，每次更新{H}次）...")
        N_prev = cloud_system.generate_apt_attack()
        B_curr = cloud_system.B

        for step in range(num_simulations):
            # 1. 构建经验序列φ
            self.experience_sequence.append((N_prev.copy(), B_curr.copy(), np.zeros(self.D)))  # 初始动作设为0
            phi = self._construct_experience_sequence()

            # 2. ε-greedy选择动作（论文公式32）
            if np.random.rand() < self.epsilon:
                action_idx = np.random.randint(self.num_actions)
            else:
                with torch.no_grad():
                    phi_tensor = torch.FloatTensor(phi).unsqueeze(0).unsqueeze(0)  # (1,1,5,5)
                    q_values = self.q_network(phi_tensor)
                    action_idx = torch.argmax(q_values).item()
            M = self.actions[action_idx]

            # 3. 生成攻击并计算效用
            N_curr = cloud_system.generate_apt_attack()
            utility = calculate_defender_utility(M, N_curr, B_curr)

            # 4. 更新经验序列
            self.experience_sequence[-1] = (N_prev.copy(), B_curr.copy(), M.copy())  # 替换初始动作
            phi_next = self._construct_experience_sequence()

            # 5. 存储经验到回放池
            done = (step == num_simulations - 1)  # 最后一步设为done
            self.memory.push((phi, action_idx, utility, phi_next, done))

            # 6. 批量更新网络（论文公式33-35）
            if len(self.memory) >= self.batch_size:
                for _ in range(H):  # 每次更新迭代H次
                    phi_batch, action_batch, reward_batch, phi_next_batch, done_batch = self.memory.sample(
                        self.batch_size)

                    # 计算目标Q值（论文公式34）
                    with torch.no_grad():
                        next_q_values = self.target_network(phi_next_batch)
                        max_next_q = next_q_values.max(1)[0].unsqueeze(1)
                        target_q = reward_batch + self.gamma * max_next_q * (1 - done_batch)

                    # 计算当前Q值
                    current_q = self.q_network(phi_batch).gather(1, action_batch)

                    # 反向传播更新
                    loss = self.loss_fn(current_q, target_q)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # 7. 同步目标网络（每100步同步一次）
            if step % 100 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # 8. 更新数据和状态
            cloud_system.update_data_dynamic(step)
            N_prev = N_curr.copy()
            B_curr = cloud_system.B.copy()

        print("DQN热启动预训练完成！")

    def select_action(self) -> Tuple[np.ndarray, int]:
        """
        选择防御动作（ε-greedy策略，论文公式32）
        :return: M: 防御动作，action_idx: 动作索引
        """
        phi = self._construct_experience_sequence()
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                phi_tensor = torch.FloatTensor(phi).unsqueeze(0).unsqueeze(0)
                q_values = self.q_network(phi_tensor)
                action_idx = torch.argmax(q_values).item()
        return self.actions[action_idx], action_idx

    def update(self, action_idx: int, utility: float, done: bool) -> None:
        """
        更新DQN网络（论文公式33-35）
        :param action_idx: 选择的动作索引
        :param utility: 获得的效用
        :param done: 是否为最后一步
        """
        # 获取当前和下一经验序列
        phi = self._construct_experience_sequence()
        # 临时添加空经验构建下一序列（实际应在环境更新后调用）
        temp_exp = self.experience_sequence[-1]
        self.experience_sequence.append((np.zeros(self.D), np.zeros(self.D), np.zeros(self.D)))
        phi_next = self._construct_experience_sequence()
        self.experience_sequence.pop()
        self.experience_sequence[-1] = temp_exp

        # 存储经验
        self.memory.push((phi, action_idx, utility, phi_next, done))

        # 批量更新
        if len(self.memory) >= self.batch_size:
            phi_batch, action_batch, reward_batch, phi_next_batch, done_batch = self.memory.sample(self.batch_size)

            # 计算目标Q值
            with torch.no_grad():
                next_q_values = self.target_network(phi_next_batch)
                max_next_q = next_q_values.max(1)[0].unsqueeze(1)
                target_q = reward_batch + self.gamma * max_next_q * (1 - done_batch)

            # 计算当前Q值
            current_q = self.q_network(phi_batch).gather(1, action_batch)

            # 反向传播
            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()