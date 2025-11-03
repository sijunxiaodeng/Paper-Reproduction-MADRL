"""
网络安全攻防博弈环境复现 - 更新版本
基于论文: Learning Games for Defending Advanced Persistent Threats in Cyber Systems
实现单防御者-单攻击者场景的Deep Q-Learning算法
兼容现代库版本
"""

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


@dataclass
class CyberState:
    """网络安全状态表示"""
    # 系统状态
    system_compromised: bool = False
    critical_services: int = 0  # 关键服务数量
    network_segments: int = 0     # 网络段数量
    security_level: float = 1.0  # 安全等级 (0-1)
    
    # 攻击者状态
    attacker_position: int = 0    # 攻击者位置
    attack_progress: float = 0.0 # 攻击进度
    compromised_nodes: int = 0    # 被攻陷节点数
    
    # 防御者状态
    defense_resources: int = 10   # 防御资源
    detection_capability: float = 0.5  # 检测能力
    response_time: float = 1.0   # 响应时间


class CyberSecurityEnvironment:
    """网络安全攻防博弈环境"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_steps = config.get('max_steps', 100)
        self.current_step = 0
        self.state = CyberState()
        
        # 环境参数
        self.network_size = config.get('network_size', 10)
        self.critical_services = config.get('critical_services', 3)
        self.initial_defense_resources = config.get('initial_defense_resources', 10)
        
        # 动作空间
        self.defender_actions = config.get('defender_actions', 5)  # 防御动作数量
        self.attacker_actions = config.get('attacker_actions', 4)  # 攻击动作数量
        
        # 奖励参数
        self.reward_weights = config.get('reward_weights', {
            'defense_success': 10.0,
            'attack_success': -5.0,
            'system_compromise': -20.0,
            'resource_cost': -1.0
        })
        
        self.reset()
    
    def reset(self) -> CyberState:
        """重置环境"""
        self.current_step = 0
        self.state = CyberState()
        self.state.defense_resources = self.initial_defense_resources
        return self.state
    
    def step(self, defender_action: int, attacker_action: int) -> Tuple[CyberState, float, float, bool, Dict]:
        """执行一步动作"""
        self.current_step += 1
        
        # 执行防御者动作
        defender_reward = self._execute_defender_action(defender_action)
        
        # 执行攻击者动作
        attacker_reward = self._execute_attacker_action(attacker_action)
        
        # 更新系统状态
        self._update_system_state()
        
        # 计算奖励
        total_reward = defender_reward + attacker_reward
        
        # 检查终止条件
        done = self._check_termination()
        
        info = {
            'step': self.current_step,
            'system_compromised': self.state.system_compromised,
            'defense_resources': self.state.defense_resources,
            'attack_progress': self.state.attack_progress
        }
        
        return self.state, defender_reward, attacker_reward, done, info
    
    def _execute_defender_action(self, action: int) -> float:
        """执行防御者动作"""
        reward = 0.0
        
        if action == 0:  # 加强监控
            self.state.detection_capability = min(1.0, self.state.detection_capability + 0.1)
            self.state.defense_resources -= 1
            reward += 2.0
            
        elif action == 1:  # 隔离网络段
            if self.state.network_segments > 0:
                self.state.network_segments -= 1
                self.state.defense_resources -= 2
                reward += 5.0
                
        elif action == 2:  # 更新安全策略
            self.state.security_level = min(1.0, self.state.security_level + 0.2)
            self.state.defense_resources -= 1
            reward += 3.0
            
        elif action == 3:  # 应急响应
            if self.state.attack_progress > 0:
                self.state.attack_progress = max(0.0, self.state.attack_progress - 0.3)
                self.state.defense_resources -= 3
                reward += 8.0
                
        elif action == 4:  # 等待/观察
            reward += 0.5
            
        return reward
    
    def _execute_attacker_action(self, action: int) -> float:
        """执行攻击者动作"""
        reward = 0.0
        
        if action == 0:  # 侦察
            if random.random() < 0.3:
                self.state.attacker_position += 1
                reward += 1.0
                
        elif action == 1:  # 横向移动
            if self.state.attacker_position > 0:
                self.state.attacker_position += 1
                self.state.attack_progress += 0.1
                reward += 2.0
                
        elif action == 2:  # 权限提升
            if random.random() < 0.4:
                self.state.attack_progress += 0.2
                reward += 3.0
                
        elif action == 3:  # 持久化
            if self.state.attack_progress > 0.5:
                self.state.compromised_nodes += 1
                reward += 5.0
                
        return reward
    
    def _update_system_state(self):
        """更新系统状态"""
        # 检查系统是否被完全攻陷
        if self.state.attack_progress >= 1.0 or self.state.compromised_nodes >= self.critical_services:
            self.state.system_compromised = True
            
        # 更新安全等级
        if self.state.attack_progress > 0:
            self.state.security_level = max(0.0, self.state.security_level - 0.05)
            
        # 更新网络段
        if self.state.attack_progress > 0.3:
            self.state.network_segments = max(0, self.state.network_segments - 1)
    
    def _check_termination(self) -> bool:
        """检查终止条件"""
        return (self.current_step >= self.max_steps or 
                self.state.system_compromised or 
                self.state.defense_resources <= 0)
    
    def get_state_vector(self) -> np.ndarray:
        """获取状态向量"""
        return np.array([
            float(self.state.system_compromised),
            self.state.critical_services / 10.0,
            self.state.network_segments / 10.0,
            self.state.security_level,
            self.state.attacker_position / 10.0,
            self.state.attack_progress,
            self.state.compromised_nodes / 10.0,
            self.state.defense_resources / 20.0,
            self.state.detection_capability,
            self.state.response_time
        ], dtype=np.float32)


class CyberDQN(nn.Module):
    """网络安全Deep Q-Network
    按照论文规范：两个隐藏层，每个1000个神经元
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1000):
        super().__init__()
        # 按照论文规范：两个隐藏层，每个1000个神经元
        self.fc1 = nn.Linear(state_dim, hidden_dim)      # 第一隐藏层：1000个神经元
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)     # 第二隐藏层：1000个神经元
        self.fc3 = nn.Linear(hidden_dim, action_dim)      # 输出层：动作维度
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一隐藏层 + ReLU激活
        x = F.relu(self.fc2(x))  # 第二隐藏层 + ReLU激活
        return self.fc3(x)       # 输出层（无激活函数）


class CyberAgent:
    """网络安全智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, agent_type: str = "defender"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_type = agent_type
        
        # 网络
        self.q_network = CyberDQN(state_dim, action_dim)
        self.target_network = CyberDQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器 - 按照论文参数：学习率 α = 0.1
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.1)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        
        # 训练参数 - 按照论文参数设置
        self.epsilon = 0.8          # 论文参数：greedy parameter = 0.8
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.8            # 论文参数：discount factor γ = 0.8
        self.batch_size = 3         # 论文参数：sampling size m = 3
        self.target_update_freq = 100
        self.update_count = 0
        
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """获取动作"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """训练网络"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 采样批次数据
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


class CyberSecurityGame:
    """网络安全攻防博弈主类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.env = CyberSecurityEnvironment(config)
        
        # 状态维度
        state_dim = 10  # 根据CyberState定义
        
        # 创建智能体
        self.defender = CyberAgent(state_dim, self.env.defender_actions, "defender")
        self.attacker = CyberAgent(state_dim, self.env.attacker_actions, "attacker")
        
        # 训练记录
        self.training_log = {
            'defender_rewards': [],
            'attacker_rewards': [],
            'defender_losses': [],
            'attacker_losses': [],
            'episode_lengths': [],
            'system_compromised': []
        }
    
    def train_episode(self) -> Dict:
        """训练一个回合"""
        state = self.env.reset()
        state_vector = self.env.get_state_vector()
        
        episode_defender_reward = 0
        episode_attacker_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            # 获取动作
            defender_action = self.defender.get_action(state_vector, training=True)
            attacker_action = self.attacker.get_action(state_vector, training=True)
            
            # 执行动作
            next_state, defender_reward, attacker_reward, done, info = self.env.step(
                defender_action, attacker_action
            )
            
            next_state_vector = self.env.get_state_vector()
            
            # 存储经验
            self.defender.store_experience(
                state_vector, defender_action, defender_reward, next_state_vector, done
            )
            self.attacker.store_experience(
                state_vector, attacker_action, attacker_reward, next_state_vector, done
            )
            
            # 训练网络
            defender_loss = self.defender.train()
            attacker_loss = self.attacker.train()
            
            # 更新状态
            state_vector = next_state_vector
            episode_defender_reward += defender_reward
            episode_attacker_reward += attacker_reward
            episode_length += 1
        
        # 记录训练信息
        self.training_log['defender_rewards'].append(episode_defender_reward)
        self.training_log['attacker_rewards'].append(episode_attacker_reward)
        self.training_log['defender_losses'].append(defender_loss if 'defender_loss' in locals() else 0)
        self.training_log['attacker_losses'].append(attacker_loss if 'attacker_loss' in locals() else 0)
        self.training_log['episode_lengths'].append(episode_length)
        self.training_log['system_compromised'].append(info['system_compromised'])
        
        return {
            'defender_reward': episode_defender_reward,
            'attacker_reward': episode_attacker_reward,
            'episode_length': episode_length,
            'system_compromised': info['system_compromised']
        }
    
    def train(self, num_episodes: int = 1000):
        """训练智能体"""
        print("开始训练网络安全攻防博弈...")
        
        for episode in range(num_episodes):
            episode_info = self.train_episode()
            
            if episode % 100 == 0:
                avg_defender_reward = np.mean(self.training_log['defender_rewards'][-100:])
                avg_attacker_reward = np.mean(self.training_log['attacker_rewards'][-100:])
                avg_length = np.mean(self.training_log['episode_lengths'][-100:])
                compromise_rate = np.mean(self.training_log['system_compromised'][-100:])
                
                print(f"Episode {episode}: "
                      f"Defender Reward: {avg_defender_reward:.2f}, "
                      f"Attacker Reward: {avg_attacker_reward:.2f}, "
                      f"Length: {avg_length:.1f}, "
                      f"Compromise Rate: {compromise_rate:.2f}")
        
        print("训练完成！")
    
    def evaluate(self, num_episodes: int = 100) -> Dict:
        """评估智能体性能"""
        print("开始评估...")
        
        evaluation_results = {
            'defender_rewards': [],
            'attacker_rewards': [],
            'episode_lengths': [],
            'system_compromised': []
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            state_vector = self.env.get_state_vector()
            
            episode_defender_reward = 0
            episode_attacker_reward = 0
            episode_length = 0
            
            done = False
            while not done:
                # 评估时不使用探索
                defender_action = self.defender.get_action(state_vector, training=False)
                attacker_action = self.attacker.get_action(state_vector, training=False)
                
                next_state, defender_reward, attacker_reward, done, info = self.env.step(
                    defender_action, attacker_action
                )
                
                state_vector = self.env.get_state_vector()
                episode_defender_reward += defender_reward
                episode_attacker_reward += attacker_reward
                episode_length += 1
            
            evaluation_results['defender_rewards'].append(episode_defender_reward)
            evaluation_results['attacker_rewards'].append(episode_attacker_reward)
            evaluation_results['episode_lengths'].append(episode_length)
            evaluation_results['system_compromised'].append(info['system_compromised'])
        
        # 计算统计信息
        stats = {
            'avg_defender_reward': np.mean(evaluation_results['defender_rewards']),
            'std_defender_reward': np.std(evaluation_results['defender_rewards']),
            'avg_attacker_reward': np.mean(evaluation_results['attacker_rewards']),
            'std_attacker_reward': np.std(evaluation_results['attacker_rewards']),
            'avg_episode_length': np.mean(evaluation_results['episode_lengths']),
            'compromise_rate': np.mean(evaluation_results['system_compromised'])
        }
        
        print(f"评估结果:")
        print(f"防御者平均奖励: {stats['avg_defender_reward']:.2f} ± {stats['std_defender_reward']:.2f}")
        print(f"攻击者平均奖励: {stats['avg_attacker_reward']:.2f} ± {stats['std_attacker_reward']:.2f}")
        print(f"平均回合长度: {stats['avg_episode_length']:.1f}")
        print(f"系统被攻陷率: {stats['compromise_rate']:.2f}")
        
        return stats
    
    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励曲线
        axes[0, 0].plot(self.training_log['defender_rewards'], label='Defender', alpha=0.7)
        axes[0, 0].plot(self.training_log['attacker_rewards'], label='Attacker', alpha=0.7)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 损失曲线
        axes[0, 1].plot(self.training_log['defender_losses'], label='Defender', alpha=0.7)
        axes[0, 1].plot(self.training_log['attacker_losses'], label='Attacker', alpha=0.7)
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 回合长度
        axes[1, 0].plot(self.training_log['episode_lengths'], alpha=0.7)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Length')
        axes[1, 0].grid(True)
        
        # 系统被攻陷率
        axes[1, 1].plot(self.training_log['system_compromised'], alpha=0.7)
        axes[1, 1].set_title('System Compromise Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Compromised')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存到: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='网络安全攻防博弈复现')
    parser.add_argument('--episodes', type=int, default=1000, help='训练回合数')
    parser.add_argument('--eval_episodes', type=int, default=100, help='评估回合数')
    parser.add_argument('--max_steps', type=int, default=100, help='每回合最大步数')
    parser.add_argument('--network_size', type=int, default=10, help='网络大小')
    parser.add_argument('--output_dir', type=str, default='cyber_output', help='输出目录')
    parser.add_argument('--do_train', action='store_true', help='执行训练')
    parser.add_argument('--do_eval', action='store_true', help='执行评估')
    parser.add_argument('--do_plot', action='store_true', help='绘制训练曲线')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置参数
    config = {
        'max_steps': args.max_steps,
        'network_size': args.network_size,
        'critical_services': 3,
        'initial_defense_resources': 10,
        'defender_actions': 5,
        'attacker_actions': 4,
        'reward_weights': {
            'defense_success': 10.0,
            'attack_success': -5.0,
            'system_compromise': -20.0,
            'resource_cost': -1.0
        }
    }
    
    # 创建游戏实例
    game = CyberSecurityGame(config)
    
    if args.do_train:
        print("开始训练...")
        game.train(args.episodes)
        
        # 保存模型
        torch.save(game.defender.q_network.state_dict(), 
                  os.path.join(args.output_dir, 'defender_model.pth'))
        torch.save(game.attacker.q_network.state_dict(), 
                  os.path.join(args.output_dir, 'attacker_model.pth'))
        print("模型已保存")
    
    if args.do_eval:
        print("开始评估...")
        # 加载模型
        if os.path.exists(os.path.join(args.output_dir, 'defender_model.pth')):
            game.defender.q_network.load_state_dict(
                torch.load(os.path.join(args.output_dir, 'defender_model.pth'))
            )
        if os.path.exists(os.path.join(args.output_dir, 'attacker_model.pth')):
            game.attacker.q_network.load_state_dict(
                torch.load(os.path.join(args.output_dir, 'attacker_model.pth'))
            )
        
        stats = game.evaluate(args.eval_episodes)
    
    if args.do_plot:
        print("绘制训练曲线...")
        game.plot_training_curves(os.path.join(args.output_dir, 'training_curves.png'))


if __name__ == "__main__":
    main()

