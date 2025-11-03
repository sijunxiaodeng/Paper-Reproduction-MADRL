import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# -------------------------- 1. 论文核心状态定义（基于🔶1-109、🔶1-115） --------------------------
@dataclass
class CyberState:
    """
    论文定义：单防御者-单攻击者场景的状态 = 过去T轮攻防策略观测窗口
    观测o^{t-j} = (x^{t-j}, y^{t-j})，其中x为防御者资源分配策略，y为攻击者资源分配策略
    参考🔶1-109：s_i^t = {o_i^{t-T}, ..., o_i^{t-1}}；单智能体场景简化为s^t = {o^{t-T}, ..., o^{t-1}}
    """
    history_strategies: List[Tuple[np.ndarray, np.ndarray]]  # 历史策略窗口：[(x_1,y_1), (x_2,y_2), ..., (x_T,y_T)]
    history_window: int  # T：历史窗口长度（论文未指定固定值，实验中可配置）


# -------------------------- 2. 论文环境建模（基于🔶1-45、🔶1-51、🔶1-57、🔶1-58） --------------------------
class CyberSecurityEnvironment:
    """
    论文核心：单防御者-单攻击者的Colonel Blotto资源分配博弈环境
    关键逻辑：攻防双方同时分配资源→对比每台主机资源→计算控制权→得效用
    参考🔶1-45（博弈模型）、🔶1-51（Colonel Blotto扩展）、🔶1-57（主机控制权规则）、🔶1-58（效用公式）
    """
    def __init__(self, config: Dict):
        # 论文核心参数（🔶1-46、🔶1-49、🔶1-50）
        self.num_hosts = config["num_hosts"]  # N：网络中主机数量
        self.defender_total_res = config["defender_total_res"]  # B：防御者总资源（如CPU、内存等抽象资源）
        self.attacker_total_res = config["attacker_total_res"]  # C：攻击者总资源
        self.host_importance = np.array(config["host_importance"])  # u_k：每台主机的重要性（基于存储数据量）
        self.history_window = config["history_window"]  # T：状态历史窗口长度
        self.max_steps = config["max_steps"]  # 每回合最大步数（论文实验中收敛前均<10000，🔶1-193）
        
        # 初始化动作空间：所有合法的资源分配策略（🔶1-51策略定义）
        # 防御者策略x=(x^1,...,x^N)，满足0≤x^k≤B且∑x^k≤B；攻击者策略y同理
        self.defender_actions = self._generate_resource_allocation_strategies(self.defender_total_res)
        self.attacker_actions = self._generate_resource_allocation_strategies(self.attacker_total_res)
        self.defender_action_dim = len(self.defender_actions)  # 防御者动作数（策略总数）
        self.attacker_action_dim = len(self.attacker_actions)  # 攻击者动作数（策略总数）
        
        # 环境内部状态
        self.current_step = 0
        self.state: CyberState = None
        self.reset()

    def _generate_resource_allocation_strategies(self, total_res: int) -> List[np.ndarray]:
        """
        生成所有合法的资源分配策略（论文🔶1-51策略集合定义）
        输入：总资源量（如防御者B=5）
        输出：策略列表，每个策略为[N,]数组（对应每台主机的资源分配）
        """
        strategies = []
        
        # 递归生成N维资源分配组合（确保∑x^k ≤ total_res）
        def backtrack(remaining_res: int, host_idx: int, current_allocation: List[int]):
            if host_idx == self.num_hosts:
                # 最后一台主机分配剩余所有资源（确保∑x^k = total_res，简化策略空间）
                current_allocation.append(remaining_res)
                strategies.append(np.array(current_allocation, dtype=np.float32))
                return
            # 为当前主机分配0~remaining_res的资源
            for res in range(0, remaining_res + 1):
                backtrack(remaining_res - res, host_idx + 1, current_allocation + [res])
        
        backtrack(total_res, 1, [])
        return strategies

    def reset(self) -> CyberState:
        """
        重置环境：初始化历史策略窗口（论文🔶1-109状态初始化逻辑）
        初始策略：均匀分配资源（论文示例中常用的基础策略，如🔶1-52示例）
        """
        self.current_step = 0
        # 初始防御者策略：均匀分配总资源到N台主机
        init_def_strat = np.ones(self.num_hosts, dtype=np.float32) * (self.defender_total_res / self.num_hosts)
        # 初始攻击者策略：均匀分配总资源到N台主机
        init_att_strat = np.ones(self.num_hosts, dtype=np.float32) * (self.attacker_total_res / self.num_hosts)
        # 填充历史窗口（前T轮均为初始策略）
        init_history = [(init_def_strat, init_att_strat)] * self.history_window
        self.state = CyberState(history_strategies=init_history, history_window=self.history_window)
        return self.state

    def _calculate_utility(self, def_strat: np.ndarray, att_strat: np.ndarray) -> Tuple[float, float]:
        """
        计算攻防双方效用（论文🔶1-58公式）
        防御者效用U = ∑u_k · sgn(∑x_i^k - ∑y_j^k)（单防御者∑x_i^k=x^k，单攻击者∑y_j^k=y^k）
        攻击者效用V = ∑u_k · sgn(∑y_j^k - ∑x_i^k) = -U（零和博弈）
        """
        # 计算每台主机的控制权（🔶1-57规则：资源多者赢，相等则平局）
        host_control = np.sign(def_strat - att_strat)  # sgn(a)=1(a>0), -1(a<0), 0(a=0)
        # 计算效用
        def_utility = np.sum(self.host_importance * host_control)
        att_utility = np.sum(self.host_importance * (-host_control))  # 零和博弈：攻击者效用=防御者效用的负值
        return def_utility, att_utility

    def _update_state_window(self, new_def_strat: np.ndarray, new_att_strat: np.ndarray):
        """
        更新历史状态窗口（论文🔶1-115、Algorithm 1第12行）
        逻辑：滑动窗口→删除最旧观测，添加最新观测：s^{t+1} = s^t ∪ {o^{t+1}} - {o^{t-T}}
        """
        new_history = self.state.history_strategies[1:]  # 删除最旧的1个观测
        new_history.append((new_def_strat, new_att_strat))  # 添加最新的1个观测
        self.state.history_strategies = new_history

    def step(self, defender_action_idx: int, attacker_action_idx: int) -> Tuple[CyberState, float, float, bool, Dict]:
        """
        执行一步博弈（论文Algorithm 1核心步骤：第7-13行）
        输入：防御者/攻击者动作索引（对应资源分配策略）
        输出：新状态、防御者奖励、攻击者奖励、终止标志、信息字典
        """
        self.current_step += 1
        
        # 1. 获取当前攻防策略（从动作索引映射到资源分配方案）
        current_def_strat = self.defender_actions[defender_action_idx]
        current_att_strat = self.attacker_actions[attacker_action_idx]
        
        # 2. 计算效用（论文🔶1-58）
        def_utility, att_utility = self._calculate_utility(current_def_strat, current_att_strat)
        
        # 3. 计算奖励（论文🔶1-115：奖励=效用/资源消耗，单智能体场景M=1、L=1）
        def_resource_used = np.sum(current_def_strat)  # 防御者当前轮资源消耗
        att_resource_used = np.sum(current_att_strat)  # 攻击者当前轮资源消耗
        # 避免除以0（资源消耗为0时奖励为0，代表"无动作"无收益）
        def_reward = def_utility / def_resource_used if def_resource_used > 1e-6 else 0.0
        att_reward = att_utility / att_resource_used if att_resource_used > 1e-6 else 0.0
        
        # 4. 更新历史状态窗口（论文Algorithm 1第12行）
        self._update_state_window(current_def_strat, current_att_strat)
        
        # 5. 检查终止条件（论文隐含逻辑：步数用尽/资源耗尽）
        done = (self.current_step >= self.max_steps 
                or def_resource_used < 1e-6  # 防御者无资源可用
                or att_resource_used < 1e-6)  # 攻击者无资源可用
        
        # 6. 记录关键信息（论文实验关注指标：🔶1-142、🔶1-160）
        info = {
            "step": self.current_step,
            "def_utility": def_utility,
            "att_utility": att_utility,
            "def_strat": current_def_strat,
            "att_strat": current_att_strat,
            "def_resource_used": def_resource_used,
            "att_resource_used": att_resource_used
        }
        
        return self.state, def_reward, att_reward, done, info

    def get_state_vector(self) -> np.ndarray:
        """
        将状态窗口转换为模型输入向量（论文🔶1-109状态表示）
        逻辑：历史T轮观测→每轮含N维防御策略+N维攻击策略→总维度=T*(2*N)
        """
        state_flat = []
        for def_strat, att_strat in self.state.history_strategies:
            state_flat.extend(def_strat)  # 拼接防御者策略（N维）
            state_flat.extend(att_strat)  # 拼接攻击者策略（N维）
        return np.array(state_flat, dtype=np.float32)


# -------------------------- 3. 论文DQN网络架构（基于🔶1-143、🔶1-24） --------------------------
class CyberDQN(nn.Module):
    """
    论文定义的DQN网络：四层全连接神经网络（输入层+2个隐藏层+输出层）
    参考🔶1-143："four-layer fully connected neural network. Each of the two hidden layers has 1000 nodes"
    功能：输入状态→输出每个动作（资源分配策略）的Q值
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1000):
        super().__init__()
        # 论文架构：输入层→隐藏层1（1000节点）→隐藏层2（1000节点）→输出层
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层→隐藏层1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层1→隐藏层2
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 隐藏层2→输出层（动作Q值）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：状态→Q值（论文未明确激活函数，采用DQN常规ReLU，🔶1-101非线性需求）"""
        x = F.relu(self.fc1(x))  # 隐藏层1 + ReLU激活（捕捉非线性关系）
        x = F.relu(self.fc2(x))  # 隐藏层2 + ReLU激活
        return self.fc3(x)  # 输出层：无激活（Q值可为任意实数）


# -------------------------- 4. 论文DQN智能体（基于🔶1-115、Algorithm 1） --------------------------
class CyberDQNAgent:
    """
    单防御者/攻击者DQN智能体（论文Algorithm 1完整实现）
    核心机制：ε-贪心策略、经验回放、目标网络更新、Q值梯度下降
    参考🔶1-115（Algorithm 1）、🔶1-143（参数设置）
    """
    def __init__(self, state_dim: int, action_dim: int, agent_type: str = "defender"):
        self.agent_type = agent_type  # "defender"或"attacker"
        self.state_dim = state_dim    # 输入状态维度（T*(2*N)）
        self.action_dim = action_dim  # 输出动作维度（策略总数）
        
        # 1. 论文DQN双网络架构（主网络+目标网络，🔶1-101、🔶1-115）
        self.q_net = CyberDQN(state_dim, action_dim)  # 主网络：实时更新
        self.target_q_net = CyberDQN(state_dim, action_dim)  # 目标网络：延迟更新
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 初始化目标网络参数
        
        # 2. 论文指定训练参数（🔶1-143：经实验验证的最优参数）
        self.lr = 0.1  # 学习率α=0.1
        self.gamma = 0.8  # 折扣因子γ=0.8
        self.epsilon = 0.8  # 贪心参数ε=0.8（探索概率）
        self.epsilon_min = 0.01  # ε最小阈值（避免完全停止探索）
        self.epsilon_decay = 0.995  # ε衰减率（常规优化，论文未提但需稳定训练）
        self.batch_size = 3  # 采样批次大小m=3
        self.target_update_freq = 100  # 目标网络更新频率（常规优化，论文未提）
        
        # 3. 经验回放池（论文🔶1-115第13行，基于🔶1-101 DQN核心机制）
        self.memory = deque(maxlen=10000)  # 经验池最大容量（论文未指定，取常规值）
        
        # 4. 优化器与训练计数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)  # 论文未指定优化器，Adam为DQN常规选择
        self.update_count = 0  # 主网络更新计数（控制目标网络更新）

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作：ε-贪心策略（论文Algorithm 1第7行）
        训练时：以ε概率随机探索，1-ε概率选Q值最大动作；评估时：仅选Q值最大动作
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作（资源分配策略）
            return random.randint(0, self.action_dim - 1)
        else:
            # 利用：选Q值最大的动作（论文🔶1-115：xt = argMaxx Q(st, x; θ)）
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, state_dim)
            with torch.no_grad():  # 评估时不计算梯度
                q_values = self.q_net(state_tensor)  # (1, action_dim)
            return q_values.argmax(dim=1).item()  # 取Q值最大的动作索引

    def store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        存储经验到回放池（论文Algorithm 1第13行）
        经验格式：(s_t, a_t, r_t, s_{t+1}, done_t)
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float:
        """
        训练主网络（论文Algorithm 1第14-16行）
        步骤：采样经验→计算当前Q值→计算目标Q值→MSE损失→梯度下降
        返回：当前训练损失
        """
        # 经验池样本不足时，不训练（避免随机误差）
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 1. 从经验池采样批次数据（论文Algorithm 1第14行）
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为Tensor（适配PyTorch计算）
        states = torch.FloatTensor(np.array(states))  # (batch_size, state_dim)
        actions = torch.LongTensor(actions).unsqueeze(1)  # (batch_size, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # (batch_size, 1)
        next_states = torch.FloatTensor(np.array(next_states))  # (batch_size, state_dim)
        dones = torch.BoolTensor(dones).unsqueeze(1)  # (batch_size, 1)（终止状态标记）

        # 2. 计算当前Q值（主网络输出，论文Algorithm 1第15行）
        current_q = self.q_net(states).gather(1, actions)  # (batch_size, 1)：仅当前动作的Q值

        # 3. 计算目标Q值（目标网络输出，论文公式：Qj = rj + γ·maxx Q(sj+1, x; θ)）
        with torch.no_grad():  # 目标网络不计算梯度
            next_max_q = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]  # (batch_size, 1)
            target_q = rewards + self.gamma * next_max_q * (~dones)  # 终止状态：γ·next_max_q=0

        # 4. 计算MSE损失（论文Algorithm 1第16行：损失函数=1/m ∑[Qj - Q(sj, xj; θ)]²）
        loss = F.mse_loss(current_q, target_q)

        # 5. 梯度下降更新主网络（论文Algorithm 1第16行）
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 更新主网络参数

        # 6. 衰减ε（控制探索-利用平衡，常规优化）
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 7. 定期更新目标网络（延迟更新，避免训练震荡）
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save_model(self, save_path: str):
        """保存模型参数（论文未提，但为工程化必要功能）"""
        torch.save(self.q_net.state_dict(), save_path)

    def load_model(self, load_path: str):
        """加载模型参数（论文未提，但为工程化必要功能）"""
        self.q_net.load_state_dict(torch.load(load_path))
        self.target_q_net.load_state_dict(self.q_net.state_dict())


# -------------------------- 5. 论文单攻防博弈训练与评估（基于🔶1-142、🔶1-160） --------------------------
class CyberSecurityGame:
    """
    单防御者-单攻击者博弈主类：整合环境与智能体，实现训练、评估、结果可视化
    参考🔶1-142（实验设计）、🔶1-160（评估指标：防御者平均效用）
    """
    def __init__(self, config: Dict):
        self.config = config
        # 初始化环境
        self.env = CyberSecurityEnvironment(config)
        # 计算状态维度：T*(2*N)（历史窗口T，每轮2*N维策略）
        self.state_dim = self.env.history_window * 2 * self.env.num_hosts
        # 初始化防御者与攻击者智能体
        self.defender_agent = CyberDQNAgent(
            state_dim=self.state_dim,
            action_dim=self.env.defender_action_dim,
            agent_type="defender"
        )
        self.attacker_agent = CyberDQNAgent(
            state_dim=self.state_dim,
            action_dim=self.env.attacker_action_dim,
            agent_type="attacker"
        )
        # 训练记录（论文实验关注指标：🔶1-160、🔶1-192）
        self.training_log = {
            "defender_rewards": [],  # 防御者每回合总奖励
            "attacker_rewards": [],  # 攻击者每回合总奖励
            "defender_utilities": [],  # 防御者每回合总效用
            "attacker_utilities": [],  # 攻击者每回合总效用
            "defender_losses": [],  # 防御者每回合平均损失
            "episode_lengths": []  # 每回合步数
        }

    def train_episode(self) -> Dict:
        """训练一个回合（论文实验的基础单位，🔶1-142）"""
        # 重置环境与状态
        state = self.env.reset()
        state_vector = self.env.get_state_vector()
        # 初始化回合统计
        ep_def_reward = 0.0
        ep_att_reward = 0.0
        ep_def_utility = 0.0
        ep_att_utility = 0.0
        ep_def_loss = 0.0
        ep_length = 0
        done = False

        while not done:
            # 1. 选择动作（ε-贪心）
            def_action = self.defender_agent.get_action(state_vector, training=True)
            att_action = self.attacker_agent.get_action(state_vector, training=True)
            
            # 2. 执行动作，获取反馈（环境step）
            next_state, def_reward, att_reward, done, info = self.env.step(def_action, att_action)
            next_state_vector = self.env.get_state_vector()
            
            # 3. 存储经验（防御者与攻击者分别存储）
            self.defender_agent.store_experience(state_vector, def_action, def_reward, next_state_vector, done)
            self.attacker_agent.store_experience(state_vector, att_action, att_reward, next_state_vector, done)
            
            # 4. 训练智能体（仅记录防御者损失，论文重点关注防御方性能）
            def_loss = self.defender_agent.train_step()
            self.attacker_agent.train_step()  # 攻击者训练（不记录损失，论文以防御方为核心）
            
            # 5. 更新回合统计
            state_vector = next_state_vector
            ep_def_reward += def_reward
            ep_att_reward += att_reward
            ep_def_utility += info["def_utility"]
            ep_att_utility += info["att_utility"]
            ep_def_loss += def_loss
            ep_length += 1

        # 计算回合平均损失（仅防御者）
        avg_def_loss = ep_def_loss / ep_length if ep_length > 0 else 0.0
        # 记录回合数据
        self.training_log["defender_rewards"].append(ep_def_reward)
        self.training_log["attacker_rewards"].append(ep_att_reward)
        self.training_log["defender_utilities"].append(ep_def_utility)
        self.training_log["attacker_utilities"].append(ep_att_utility)
        self.training_log["defender_losses"].append(avg_def_loss)
        self.training_log["episode_lengths"].append(ep_length)

        # 返回回合关键信息
        return {
            "episode": len(self.training_log["defender_rewards"]),
            "defender_reward": ep_def_reward,
            "defender_utility": ep_def_utility,
            "defender_loss": avg_def_loss,
            "episode_length": ep_length,
            "system_compromised": info.get("system_compromised", False)  # 兼容扩展，论文单攻防无此指标
        }

    def train(self, num_episodes: int = 1000):
        """训练指定回合数（论文实验训练量，🔶1-193：收敛前<10000回合）"""
        print(f"=== 论文单防御者-单攻击者DQN训练开始（{num_episodes}回合） ===")
        print(f"论文参数：N={self.env.num_hosts}, B={self.env.defender_total_res}, C={self.env.attacker_total_res}, T={self.env.history_window}")
        print(f"训练日志（每100回合输出一次）：")
        
        for episode in range(1, num_episodes + 1):
            # 训练一个回合
            ep_info = self.train_episode()
            
            # 每100回合输出统计（论文实验常用输出频率，🔶1-160）
            if episode % 100 == 0 or episode == 1:
                # 计算最近100回合的平均指标（论文实验分析方式，🔶1-192）
                recent_100_def_reward = np.mean(self.training_log["defender_rewards"][-100:])
                recent_100_def_utility = np.mean(self.training_log["defender_utilities"][-100:])
                recent_100_def_loss = np.mean(self.training_log["defender_losses"][-100:])
                recent_100_length = np.mean(self.training_log["episode_lengths"][-100:])
                
                print(f"回合 {episode:4d} | "
                      f"防御者平均奖励：{recent_100_def_reward:6.2f} | "
                      f"防御者平均效用：{recent_100_def_utility:6.2f} | "
                      f"防御者平均损失：{recent_100_def_loss:6.4f} | "
                      f"平均回合长度：{recent_100_length:4.1f}")
        
        # 训练结束后保存模型
        os.makedirs(self.config["output_dir"], exist_ok=True)
        self.defender_agent.save_model(os.path.join(self.config["output_dir"], "defender_dqn.pth"))
        self.attacker_agent.save_model(os.path.join(self.config["output_dir"], "attacker_dqn.pth"))
        print(f"\n=== 训练结束！模型已保存至 {self.config['output_dir']} ===")

    def evaluate(self, num_episodes: int = 100) -> Dict:
        """评估智能体性能（论文实验评估环节，🔶1-142、🔶1-192）"""
        print(f"\n=== 论文单防御者-单攻击者DQN评估开始（{num_episodes}回合） ===")
        # 加载训练好的模型
        self.defender_agent.load_model(os.path.join(self.config["output_dir"], "defender_dqn.pth"))
        self.attacker_agent.load_model(os.path.join(self.config["output_dir"], "attacker_dqn.pth"))
        
        # 初始化评估统计
        eval_stats = {
            "defender_rewards": [],
            "defender_utilities": [],
            "episode_lengths": []
        }

        for _ in range(num_episodes):
            state = self.env.reset()
            state_vector = self.env.get_state_vector()
            ep_def_reward = 0.0
            ep_def_utility = 0.0
            ep_length = 0
            done = False

            while not done:
                # 评估时不探索（仅选Q值最大动作）
                def_action = self.defender_agent.get_action(state_vector, training=False)
                att_action = self.attacker_agent.get_action(state_vector, training=False)
                # 执行动作
                next_state, def_reward, att_reward, done, info = self.env.step(def_action, att_action)
                # 更新统计
                state_vector = self.env.get_state_vector()
                ep_def_reward += def_reward
                ep_def_utility += info["def_utility"]
                ep_length += 1

            # 记录评估数据
            eval_stats["defender_rewards"].append(ep_def_reward)
            eval_stats["defender_utilities"].append(ep_def_utility)
            eval_stats["episode_lengths"].append(ep_length)

        # 计算评估指标（论文关注的均值与标准差，🔶1-192）
        result = {
            "avg_defender_reward": np.mean(eval_stats["defender_rewards"]),
            "std_defender_reward": np.std(eval_stats["defender_rewards"]),
            "avg_defender_utility": np.mean(eval_stats["defender_utilities"]),
            "std_defender_utility": np.std(eval_stats["defender_utilities"]),
            "avg_episode_length": np.mean(eval_stats["episode_lengths"]),
            "std_episode_length": np.std(eval_stats["episode_lengths"])
        }

        # 输出评估结果（论文实验报告格式，🔶1-192）
        print(f"评估结果：")
        print(f"防御者平均奖励：{result['avg_defender_reward']:6.2f} ± {result['std_defender_reward']:6.2f}")
        print(f"防御者平均效用：{result['avg_defender_utility']:6.2f} ± {result['std_defender_utility']:6.2f}")
        print(f"平均回合长度：{result['avg_episode_length']:4.1f} ± {result['std_episode_length']:4.1f}")
        print(f"=== 评估结束 ===")
        return result

    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """绘制训练曲线（论文实验可视化方式，🔶1-160、🔶1-165）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        episodes = range(1, len(self.training_log["defender_rewards"]) + 1)
        
        # 1. 防御者奖励曲线（论文图2(b)-(d)、图3(b)-(d)风格）
        axes[0, 0].plot(episodes, self.training_log["defender_rewards"], alpha=0.7, label="Defender Reward")
        axes[0, 0].set_title("Defender Episode Reward (Paper Fig. 2-3 Style)", fontsize=12)
        axes[0, 0].set_xlabel("Episode", fontsize=10)
        axes[0, 0].set_ylabel("Total Reward", fontsize=10)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 防御者效用曲线（论文核心评估指标，🔶1-160）
        axes[0, 1].plot(episodes, self.training_log["defender_utilities"], alpha=0.7, color="orange", label="Defender Utility")
        axes[0, 1].set_title("Defender Episode Utility (Paper Key Metric)", fontsize=12)
        axes[0, 1].set_xlabel("Episode", fontsize=10)
        axes[0, 1].set_ylabel("Total Utility", fontsize=10)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 防御者损失曲线（论文未明确，但为训练稳定性分析必要）
        axes[1, 0].plot(episodes, self.training_log["defender_losses"], alpha=0.7, color="red", label="Defender Loss")
        axes[1, 0].set_title("Defender Average Training Loss", fontsize=12)
        axes[1, 0].set_xlabel("Episode", fontsize=10)
        axes[1, 0].set_ylabel("MSE Loss", fontsize=10)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 回合长度曲线（论文图2(b)-(d)辅助指标）
        axes[1, 1].plot(episodes, self.training_log["episode_lengths"], alpha=0.7, color="green", label="Episode Length")
        axes[1, 1].set_title("Episode Length (Paper Auxiliary Metric)", fontsize=12)
        axes[1, 1].set_xlabel("Episode", fontsize=10)
        axes[1, 1].set_ylabel("Step Count", fontsize=10)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 保存图片（论文图分辨率风格，🔶1-17、🔶1-18）
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["output_dir"], save_path), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"训练曲线已保存至：{os.path.join(self.config['output_dir'], save_path)}")


# -------------------------- 6. 主函数（论文实验入口） --------------------------
def main():
    # 解析命令行参数（论文实验可配置项）
    parser = argparse.ArgumentParser(description="Zhu et al. 2023 - Single-Defender Single-Attacker DQN")
    parser.add_argument("--episodes", type=int, default=1000, help="训练回合数（论文实验<10000）")
    parser.add_argument("--eval_episodes", type=int, default=100, help="评估回合数（论文实验常用100）")
    parser.add_argument("--num_hosts", type=int, default=2, help="主机数量N（论文示例用2，🔶1-52）")
    parser.add_argument("--def_res", type=int, default=5, help="防御者总资源B（论文示例用5，🔶1-144）")
    parser.add_argument("--att_res", type=int, default=5, help="攻击者总资源C（论文示例用5，🔶1-144）")
    parser.add_argument("--history_window", type=int, default=5, help="历史窗口T（论文未指定，实验可调）")
    parser.add_argument("--max_steps", type=int, default=100, help="每回合最大步数（论文实验<10000）")
    parser.add_argument("--output_dir", type=str, default="paper_dqn_output", help="输出目录（模型+曲线）")
    parser.add_argument("--do_train", action="store_true", help="执行训练（论文实验核心步骤）")
    parser.add_argument("--do_eval", action="store_true", help="执行评估（论文实验验证步骤）")
    parser.add_argument("--do_plot", action="store_true", help="绘制训练曲线（论文实验可视化）")
    args = parser.parse_args()

    # 构造论文实验配置（严格对应论文参数，🔶1-143、🔶1-144）
    config = {
        "num_hosts": args.num_hosts,
        "defender_total_res": args.def_res,
        "attacker_total_res": args.att_res,
        "history_window": args.history_window,
        "max_steps": args.max_steps,
        "output_dir": args.output_dir,
        "host_importance": [2, 1]  # 主机重要性（论文示例用2和1，🔶1-144）
    }

    # 初始化博弈实例
    game = CyberSecurityGame(config)

    # 执行训练、评估、绘图（论文实验完整流程）
    if args.do_train:
        game.train(num_episodes=args.episodes)
    if args.do_eval:
        game.evaluate(num_episodes=args.eval_episodes)
    if args.do_plot and args.do_train:
        game.plot_training_curves(save_path="paper_training_curves.png")


if __name__ == "__main__":
    main()