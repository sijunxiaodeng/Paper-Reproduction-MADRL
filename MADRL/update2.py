
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 导入更新后的模块
from cyber_security_game_updated import CyberSecurityGame, CyberSecurityEnvironment, CyberState






@dataclass
class CyberState:
    """修正：仅存储“历史攻防策略窗口”（论文、）"""
    # 历史策略窗口：list of (defender_allocation, attacker_allocation)
    # defender_allocation: 对N个主机的资源分配（如[N,]数组）
    # attacker_allocation: 对N个主机的资源分配（如[N,]数组）
    history_strategies: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    T: int = 5  # 论文：历史窗口长度（回溯T轮策略）


class CyberSecurityEnvironment:
    """修正：聚焦“资源分配博弈”，删除无关状态（如detection_capability）"""
    def __init__(self, config: Dict):
        self.config = config
        self.max_steps = config.get('max_steps', 100)
        self.current_step = 0
        
        # 论文核心参数（、、）
        self.num_hosts = config.get('num_hosts', 2)  # N：主机数量
        self.defender_total_res = config.get('defender_total_res', 5)  # B：防御者总资源
        self.attacker_total_res = config.get('attacker_total_res', 5)  # C：攻击者总资源
        self.host_importance = np.array(config.get('host_importance', [2, 1]))  # u_k：主机重要性
        self.T = config.get('history_window', 5)  # 论文：状态窗口长度（）
        
        # 动作空间：生成所有合法资源分配策略（论文、）
        self.defender_actions = self._generate_resource_allocation_strategies(self.defender_total_res)
        self.attacker_actions = self._generate_resource_allocation_strategies(self.attacker_total_res)
        self.defender_action_dim = len(self.defender_actions)
        self.attacker_action_dim = len(self.attacker_actions)
        
        self.reset()

    def _generate_resource_allocation_strategies(self, total_res: int) -> List[np.ndarray]:
        """生成所有合法资源分配策略（论文、）
        策略：x=(x1,...,xN)，满足0≤xk≤total_res，∑xk≤total_res（整数资源）
        """
        strategies = []
        # 递归生成N维资源分配组合（以num_hosts=2为例）
        def backtrack(remaining_res: int, idx: int, current: List[int]):
            if idx == self.num_hosts:
                if remaining_res >= 0:
                    strategies.append(np.array(current + [remaining_res]))
                return
            for res in range(0, remaining_res + 1):
                backtrack(remaining_res - res, idx + 1, current + [res])
        backtrack(total_res, 1, [])
        return strategies

    def reset(self) -> CyberState:
        """修正：初始化“历史策略窗口”（论文）"""
        self.current_step = 0
        # 初始策略：均匀分配资源（论文示例）
        init_def_strat = np.ones(self.num_hosts) * (self.defender_total_res // self.num_hosts)
        init_att_strat = np.ones(self.num_hosts) * (self.attacker_total_res // self.num_hosts)
        # 填充初始历史窗口（前T轮均为初始策略）
        history = [(init_def_strat, init_att_strat)] * self.T
        return CyberState(history_strategies=history, T=self.T)

    def _update_system_state(self, def_strat: np.ndarray, att_strat: np.ndarray) -> Tuple[float, float]:
        """修正：按论文计算攻防双方效用（）"""
        # 计算每个主机的控制权（论文）
        def_total_per_host = def_strat  # 单防御者：∑x_i^k = x^k
        att_total_per_host = att_strat  # 单攻击者：∑y_j^k = y^k
        host_control = np.sign(def_total_per_host - att_total_per_host)  # sgn函数
        
        # 防御者效用U（论文公式3）、攻击者效用V（论文公式4）
        def_utility = np.sum(self.host_importance * host_control)
        att_utility = np.sum(self.host_importance * (-host_control))  # V = -U（零和博弈）
        return def_utility, att_utility

    def step(self, defender_action_idx: int, attacker_action_idx: int) -> Tuple[CyberState, float, float, bool, Dict]:
        """修正：执行动作→计算效用→更新历史状态（论文Algorithm 1步骤）"""
        self.current_step += 1
        
        # 1. 获取当前攻防策略（从动作索引映射到资源分配）
        def_strat = self.defender_actions[defender_action_idx]
        att_strat = self.attacker_actions[attacker_action_idx]
        
        # 2. 计算效用与奖励（论文、）
        def_utility, att_utility = self._update_system_state(def_strat, att_strat)
        # 奖励 = 效用 / 资源消耗（单防御者M=1，单攻击者L=1）
        def_resource_used = np.sum(def_strat)
        att_resource_used = np.sum(att_strat)
        def_reward = def_utility / def_resource_used if def_resource_used > 0 else 0.0
        att_reward = att_utility / att_resource_used if att_resource_used > 0 else 0.0
        
        # 3. 更新历史状态窗口（论文Algorithm 1第12行：滑动窗口）
        new_history = self.state.history_strategies[1:]  # 删除最旧观测
        new_history.append((def_strat, att_strat))       # 添加新观测
        self.state.history_strategies = new_history
        
        # 4. 终止条件（论文隐含：步数用尽/资源耗尽）
        done = (self.current_step >= self.max_steps 
                or def_resource_used == 0 
                or att_resource_used == 0)
        
        # 5. 信息记录（聚焦论文核心指标）
        info = {
            'step': self.current_step,
            'def_utility': def_utility,
            'att_utility': att_utility,
            'def_strat': def_strat,
            'att_strat': att_strat
        }
        return self.state, def_reward, att_reward, done, info

    def get_state_vector(self) -> np.ndarray:
        """修正：状态向量=历史T轮策略的扁平化（论文）"""
        # 每个历史观测含（防御者策略[N维] + 攻击者策略[N维]），共T*(2N)维
        state_flat = []
        for def_strat, att_strat in self.state.history_strategies:
            state_flat.extend(def_strat)
            state_flat.extend(att_strat)
        return np.array(state_flat, dtype=np.float32)


        class CyberAgent:
    """修正：仅保留论文相关参数，删除无关逻辑"""
    def __init__(self, state_dim: int, action_dim: int, agent_type: str = "defender"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_type = agent_type
        
        # 论文DQN架构：2个隐藏层各1000节点（）
        self.q_network = CyberDQN(state_dim, action_dim, hidden_dim=1000)
        self.target_network = CyberDQN(state_dim, action_dim, hidden_dim=1000)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 论文参数：学习率α=0.1（）
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.1)
        
        # 经验回放（论文DQN核心机制，）
        self.memory = deque(maxlen=10000)
        
        # 论文参数：ε=0.8、γ=0.8、batch_size=3（）
        self.epsilon = 0.8
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # 论文未提，但为DQN常规优化
        self.gamma = 0.8
        self.batch_size = 3
        self.target_update_freq = 100  # 论文未提，确保目标网络稳定
        self.update_count = 0


class CyberSecurityGame:
    """修正：初始化时基于论文参数生成动作空间"""
    def __init__(self, config: Dict):
        self.config = config
        self.env = CyberSecurityEnvironment(config)
        
        # 状态维度：T*(2*N)（T=历史窗口，N=主机数）
        state_dim = self.env.T * 2 * self.env.num_hosts
        
        # 创建智能体（动作维度=策略总数）
        self.defender = CyberAgent(
            state_dim, self.env.defender_action_dim, "defender"
        )
        self.attacker = CyberAgent(
            state_dim, self.env.attacker_action_dim, "attacker"
        )
        
        # 训练记录：聚焦论文指标（效用、奖励、策略）
        self.training_log = {
            'defender_rewards': [],
            'attacker_rewards': [],
            'defender_utilities': [],
            'attacker_utilities': [],
            'episode_lengths': []
        }

    def train_episode(self) -> Dict:
        """修正：按论文Algorithm 1执行训练步骤"""
        state = self.env.reset()
        state_vector = self.env.get_state_vector()
        
        episode_def_reward = 0.0
        episode_att_reward = 0.0
        episode_def_utility = 0.0
        episode_length = 0
        
        done = False
        while not done:
            # 1. 选动作（ε-贪心，论文Algorithm 1第7行）
            def_action = self.defender.get_action(state_vector, training=True)
            att_action = self.attacker.get_action(state_vector, training=True)
            
            # 2. 执行动作（论文Algorithm 1第8-9行）
            next_state, def_reward, att_reward, done, info = self.env.step(
                def_action, att_action
            )
            next_state_vector = self.env.get_state_vector()
            
            # 3. 存经验（论文Algorithm 1第13行）
            self.defender.store_experience(
                state_vector, def_action, def_reward, next_state_vector, done
            )
            self.attacker.store_experience(
                state_vector, att_action, att_reward, next_state_vector, done
            )
            
            # 4. 训练网络（论文Algorithm 1第14-16行）
            def_loss = self.defender.train()
            att_loss = self.attacker.train()
            
            # 5. 更新状态与记录
            state_vector = next_state_vector
            episode_def_reward += def_reward
            episode_att_reward += att_reward
            episode_def_utility += info['def_utility']
            episode_length += 1
        
        # 记录论文关注的指标（效用、奖励）
        self.training_log['defender_rewards'].append(episode_def_reward)
        self.training_log['attacker_rewards'].append(episode_att_reward)
        self.training_log['defender_utilities'].append(episode_def_utility)
        self.training_log['episode_lengths'].append(episode_length)
        
        return {
            'def_reward': episode_def_reward,
            'att_reward': episode_att_reward,
            'def_utility': episode_def_utility,
            'length': episode_length
        }


        def main():
    parser = argparse.ArgumentParser(description='论文单防御者-单攻击者DQN复现')
    parser.add_argument('--episodes', type=int, default=1000, help='训练回合数')
    parser.add_argument('--max_steps', type=int, default=100, help='每回合最大步数')
    # 论文核心参数（、、）
    parser.add_argument('--num_hosts', type=int, default=2, help='主机数量N')
    parser.add_argument('--def_res', type=int, default=5, help='防御者总资源B')
    parser.add_argument('--att_res', type=int, default=5, help='攻击者总资源C')
    parser.add_argument('--history_window', type=int, default=5, help='状态历史窗口T')
    parser.add_argument('--host_importance', type=list, default=[2,1], help='主机重要性u_k')
    parser.add_argument('--output_dir', type=str, default='cyber_paper_output', help='输出目录')
    parser.add_argument('--do_train', action='store_true', help='执行训练')
    parser.add_argument('--do_eval', action='store_true', help='执行评估')
    parser.add_argument('--do_plot', action='store_true', help='绘制训练曲线')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 修正：仅含论文核心参数的配置（、）
    config = {
        'max_steps': args.max_steps,
        'num_hosts': args.num_hosts,
        'defender_total_res': args.def_res,
        'attacker_total_res': args.att_res,
        'history_window': args.history_window,
        'host_importance': args.host_importance
    }
    
    game = CyberSecurityGame(config)
    # 后续训练/评估逻辑不变（仅基于论文指标输出）