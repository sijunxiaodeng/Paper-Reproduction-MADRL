import matplotlib.pyplot as plt

from Algorithm2 import *

def simulate_scenario2():
    """场景2：变化攻击者数量（2-5），测试鲁棒性"""
    num_attackers_list = [2, 3, 4]
    results = {}

    for num_attackers in num_attackers_list:
        print(f"正在运行场景2：攻击者数量 = {num_attackers}")
        # 算法2：4防御者，变化攻击者数量，资源均为[3,5]
        utils, losses, ne = algorithm2_multi_defender_attacker(
            num_hosts=4,
            num_defenders=4,
            num_attackers=num_attackers,
            defender_resources_range=[3, 5],
            attacker_resources_range=[3, 5],
            num_rounds=1000
        )
        results[num_attackers] = {
            "utilities": utils,
            "losses": losses,
            "ne_utility": ne
        }

    # 可视化结果
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 6))

    # 子图1：不同攻击者数量的效用变化
    plt.subplot(1, 2, 1)
    for num_att, data in results.items():
        avg_utils = np.convolve(data["utilities"], np.ones(100) / 100, mode='valid')
        plt.plot(avg_utils, label=f"攻击者数 = {num_att}")
    plt.axhline(y=results[2]["ne_utility"], color='k', linestyle='--', label='纳什均衡')
    plt.xlabel("训练轮次（×100）")
    plt.ylabel("防御者群体平均效用")
    plt.title("场景2：不同攻击者数量的效用变化")
    plt.legend()
    plt.grid(True)

    # 子图2：收敛速度对比（达到稳定效用的轮次）
    plt.subplot(1, 2, 2)
    convergence_rounds = []
    for num_att, data in results.items():
        # 找到效用稳定的轮次（连续500轮波动<5%）
        utils = data["utilities"]
        conv_round = -1
        for i in range(len(utils) - 500):
            window = utils[i:i + 500]
            if np.max(window) - np.min(window) < 0.05 * np.mean(window):
                conv_round = i
                break
        convergence_rounds.append(conv_round if conv_round != -1 else len(utils))

    plt.bar(num_attackers_list, convergence_rounds, width=0.8)
    plt.xlabel("攻击者数量")
    plt.ylabel("收敛轮次")
    plt.title("场景2：不同攻击者数量的收敛速度")
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

    return results

if __name__ == "__main__":
    # 选择要运行的场景（可单独运行）
    print("=" * 50)
    print("高级持续性威胁（APT）防御仿真代码")
    print("=" * 50)

# 运行场景2：不同攻击者数量
    print("\n2. 运行场景2：不同攻击者数量的鲁棒性测试")
    scenario2_results = simulate_scenario2()