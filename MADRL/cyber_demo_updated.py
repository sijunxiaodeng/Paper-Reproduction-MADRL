"""
网络安全攻防博弈演示脚本 - 更新版本
展示基于论文的完整实现，兼容现代库版本
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 导入更新后的模块
from cyber_security_game_updated import CyberSecurityGame, CyberSecurityEnvironment, CyberState


def demo_cyber_security_game():
    """演示网络安全攻防博弈"""
    print("=" * 80)
    print("网络安全攻防博弈演示 - 更新版本")
    print("基于论文: Learning Games for Defending Advanced Persistent Threats in Cyber Systems")
    print("兼容现代库版本")
    print("=" * 80)
    
    # 配置参数
    config = {
        'max_steps': 50,  # 较短的演示
        'network_size': 8,
        'critical_services': 3,
        'initial_defense_resources': 8,
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
    
    print("\n🔧 环境配置:")
    print(f"- 网络大小: {config['network_size']}")
    print(f"- 关键服务: {config['critical_services']}")
    print(f"- 初始防御资源: {config['initial_defense_resources']}")
    print(f"- 最大步数: {config['max_steps']}")
    
    print("\n🎯 智能体配置:")
    print(f"- 防御者动作空间: {config['defender_actions']} (加强监控, 隔离网络, 更新策略, 应急响应, 等待)")
    print(f"- 攻击者动作空间: {config['attacker_actions']} (侦察, 横向移动, 权限提升, 持久化)")
    
    print("\n🚀 开始训练演示...")
    print("-" * 50)
    
    # 训练演示
    demo_episodes = 200
    for episode in range(demo_episodes):
        episode_info = game.train_episode()
        
        if episode % 50 == 0:
            print(f"Episode {episode}: "
                  f"Defender Reward: {episode_info['defender_reward']:.2f}, "
                  f"Attacker Reward: {episode_info['attacker_reward']:.2f}, "
                  f"Length: {episode_info['episode_length']}, "
                  f"System Compromised: {episode_info['system_compromised']}")
    
    print("\n📊 训练统计:")
    print(f"- 防御者平均奖励: {np.mean(game.training_log['defender_rewards']):.2f}")
    print(f"- 攻击者平均奖励: {np.mean(game.training_log['attacker_rewards']):.2f}")
    print(f"- 平均回合长度: {np.mean(game.training_log['episode_lengths']):.1f}")
    print(f"- 系统被攻陷率: {np.mean(game.training_log['system_compromised']):.2f}")
    
    # 评估演示
    print("\n🔍 开始评估演示...")
    print("-" * 50)
    
    evaluation_results = {
        'defender_rewards': [],
        'attacker_rewards': [],
        'episode_lengths': [],
        'system_compromised': []
    }
    
    for episode in range(20):  # 评估20个回合
        state = game.env.reset()
        state_vector = game.env.get_state_vector()
        
        episode_defender_reward = 0
        episode_attacker_reward = 0
        episode_length = 0
        
        done = False
        step = 0
        while not done and step < 50:
            # 评估时不使用探索
            defender_action = game.defender.get_action(state_vector, training=False)
            attacker_action = game.attacker.get_action(state_vector, training=False)
            
            next_state, defender_reward, attacker_reward, done, info = game.env.step(
                defender_action, attacker_action
            )
            
            state_vector = game.env.get_state_vector()
            episode_defender_reward += defender_reward
            episode_attacker_reward += attacker_reward
            episode_length += 1
            step += 1
        
        evaluation_results['defender_rewards'].append(episode_defender_reward)
        evaluation_results['attacker_rewards'].append(episode_attacker_reward)
        evaluation_results['episode_lengths'].append(episode_length)
        evaluation_results['system_compromised'].append(info['system_compromised'])
        
        print(f"评估回合 {episode + 1}: "
              f"Defender Reward: {episode_defender_reward:.2f}, "
              f"Attacker Reward: {episode_attacker_reward:.2f}, "
              f"Length: {episode_length}, "
              f"Compromised: {info['system_compromised']}")
    
    # 计算评估统计
    avg_defender_reward = np.mean(evaluation_results['defender_rewards'])
    avg_attacker_reward = np.mean(evaluation_results['attacker_rewards'])
    avg_length = np.mean(evaluation_results['episode_lengths'])
    compromise_rate = np.mean(evaluation_results['system_compromised'])
    
    print(f"\n📈 评估结果:")
    print(f"- 防御者平均奖励: {avg_defender_reward:.2f} ± {np.std(evaluation_results['defender_rewards']):.2f}")
    print(f"- 攻击者平均奖励: {avg_attacker_reward:.2f} ± {np.std(evaluation_results['attacker_rewards']):.2f}")
    print(f"- 平均回合长度: {avg_length:.1f}")
    print(f"- 系统被攻陷率: {compromise_rate:.2f}")
    
    # 绘制训练曲线
    print("\n📊 绘制训练曲线...")
    game.plot_training_curves("cyber_training_curves_updated.png")
    
    # 展示算法特性
    print("\n" + "=" * 80)
    print("算法特性展示:")
    print("=" * 80)
    
    print("1. 状态表示:")
    print("   - 系统状态: 是否被攻陷, 关键服务数量, 网络段数量, 安全等级")
    print("   - 攻击者状态: 位置, 攻击进度, 被攻陷节点数")
    print("   - 防御者状态: 防御资源, 检测能力, 响应时间")
    
    print("\n2. 动作空间:")
    print("   - 防御者: 加强监控, 隔离网络段, 更新安全策略, 应急响应, 等待")
    print("   - 攻击者: 侦察, 横向移动, 权限提升, 持久化")
    
    print("\n3. 奖励机制:")
    print("   - 防御成功: +10.0")
    print("   - 攻击成功: -5.0")
    print("   - 系统被攻陷: -20.0")
    print("   - 资源消耗: -1.0")
    
    print("\n4. 训练特性:")
    print("   - 经验回放: 存储 (状态, 动作, 奖励, 下一状态, 终止)")
    print("   - 目标网络: 定期更新目标网络参数")
    print("   - Epsilon-Greedy: 平衡探索与利用")
    print("   - 双智能体: 防御者和攻击者同时学习")
    
    print("\n5. 网络安全特性:")
    print("   - 动态威胁: 攻击者持续尝试攻陷系统")
    print("   - 资源限制: 防御者资源有限")
    print("   - 多层防御: 网络隔离, 监控, 应急响应")
    print("   - 持久化威胁: 攻击者尝试建立持久化访问")
    
    return game


def analyze_cyber_state():
    """分析网络安全状态"""
    print("\n" + "=" * 80)
    print("网络安全状态分析")
    print("=" * 80)
    
    # 创建环境
    config = {
        'max_steps': 20,
        'network_size': 5,
        'critical_services': 2,
        'initial_defense_resources': 5
    }
    
    env = CyberSecurityEnvironment(config)
    
    print("🔍 状态空间分析:")
    print(f"- 状态维度: 10")
    print(f"- 状态组件:")
    
    state = env.reset()
    state_vector = env.get_state_vector()
    
    state_components = [
        "系统被攻陷状态",
        "关键服务数量",
        "网络段数量", 
        "安全等级",
        "攻击者位置",
        "攻击进度",
        "被攻陷节点数",
        "防御资源",
        "检测能力",
        "响应时间"
    ]
    
    for i, component in enumerate(state_components):
        print(f"  {i+1:2d}. {component}: {state_vector[i]:.3f}")
    
    print(f"\n🎯 动作空间分析:")
    print(f"- 防御者动作: {env.defender_actions}")
    print(f"- 攻击者动作: {env.attacker_actions}")
    
    print(f"\n⚖️ 奖励权重:")
    for key, value in env.reward_weights.items():
        print(f"  - {key}: {value}")
    
    return env


def test_modern_libraries():
    """测试现代库兼容性"""
    print("\n" + "=" * 80)
    print("现代库兼容性测试")
    print("=" * 80)
    
    print("🔧 库版本信息:")
    
    # 测试NumPy
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
    
    # 测试PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   - CUDA可用: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
    
    # 测试Matplotlib
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
    
    # 测试其他库
    try:
        import argparse
        print(f"✅ argparse: 内置库")
    except ImportError as e:
        print(f"❌ argparse: {e}")
    
    try:
        from dataclasses import dataclass
        print(f"✅ dataclasses: 内置库")
    except ImportError as e:
        print(f"❌ dataclasses: {e}")
    
    print("\n📊 性能测试:")
    
    # 测试张量操作
    try:
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = torch.mm(x, y)
        print(f"✅ 张量运算: 正常")
    except Exception as e:
        print(f"❌ 张量运算: {e}")
    
    # 测试神经网络
    try:
        model = torch.nn.Linear(10, 5)
        x = torch.randn(32, 10)
        y = model(x)
        print(f"✅ 神经网络: 正常")
    except Exception as e:
        print(f"❌ 神经网络: {e}")
    
    print("\n🎯 兼容性总结:")
    print("- 移除了对旧版Gym的依赖")
    print("- 使用现代PyTorch版本")
    print("- 优化了NumPy数组处理")
    print("- 添加了警告过滤")
    print("- 改进了错误处理")


def compare_with_standard_dqn():
    """与标准DQN对比"""
    print("\n" + "=" * 80)
    print("与标准DQN算法对比")
    print("=" * 80)
    
    print("标准DQN特点:")
    print("- 单智能体环境")
    print("- 静态环境状态")
    print("- 简单的奖励机制")
    print("- 基础的经验回放")
    
    print("\n网络安全攻防博弈特点:")
    print("- 双智能体对抗")
    print("- 动态威胁环境")
    print("- 复杂的奖励机制")
    print("- 多层防御策略")
    print("- 资源约束")
    print("- 持久化威胁")
    
    print("\n主要改进:")
    print("1. 对抗性学习: 防御者和攻击者同时学习优化策略")
    print("2. 动态环境: 系统状态随时间变化")
    print("3. 复杂奖励: 考虑多种安全因素")
    print("4. 资源管理: 防御者需要合理分配资源")
    print("5. 威胁建模: 模拟真实的网络攻击行为")
    
    print("\n库兼容性改进:")
    print("1. 移除Gym依赖: 使用自定义环境")
    print("2. 现代PyTorch: 使用最新版本特性")
    print("3. 优化性能: 改进张量操作")
    print("4. 错误处理: 添加异常处理")
    print("5. 警告过滤: 减少不必要的警告")


def main():
    """主函数"""
    print("🚀 启动网络安全攻防博弈演示 - 更新版本...")
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 测试库兼容性
    test_modern_libraries()
    
    # 运行演示
    game = demo_cyber_security_game()
    
    # 分析状态
    env = analyze_cyber_state()
    
    # 对比分析
    compare_with_standard_dqn()
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    print("📁 输出文件:")
    print("- cyber_training_curves_updated.png: 训练曲线图")
    print("- 训练日志: 包含奖励、损失、回合长度等统计信息")
    
    print("\n🔧 使用方法:")
    print("1. 训练模型: python cyber_security_game_updated.py --do_train --episodes 1000")
    print("2. 评估模型: python cyber_security_game_updated.py --do_eval --eval_episodes 100")
    print("3. 绘制曲线: python cyber_security_game_updated.py --do_plot")
    print("4. 完整流程: python cyber_security_game_updated.py --do_train --do_eval --do_plot")
    
    print("\n📚 技术特点:")
    print("- 完全基于论文算法实现")
    print("- 支持双智能体对抗学习")
    print("- 模拟真实网络安全场景")
    print("- 包含完整的训练和评估流程")
    print("- 兼容现代库版本")
    print("- 优化性能和稳定性")


if __name__ == "__main__":
    main()

