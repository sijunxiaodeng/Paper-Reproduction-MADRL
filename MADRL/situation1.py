from matplotlib.pyplot import xticks

from Algorithm2 import *

def simulate_scenario1():
    """场景1：变化主机数量（3-9），测试可扩展性"""
    num_hosts_list = [3]
    results = {}

    for num_hosts in num_hosts_list:
        print(f"正在运行场景1：主机数量 = {num_hosts}")
        # 算法2：3防御者，3攻击者，资源均为[5,7]
        utils, losses, ne = algorithm2_multi_defender_attacker(
            num_hosts=num_hosts,
            num_defenders=3,
            num_attackers=3,
            num_rounds=10000
        )


        results[num_hosts] = {
            "utilities": utils,
            "losses": losses,
            "ne_utility": ne
        }

    # 可视化结果
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 6))

    # 子图1：不同主机数量的平均效用
    plt.subplot(1, 2, 1)
    for num_hosts, data in results.items():
        # 计算每100轮的平均效用（平滑曲线）
        avg_utils = np.convolve(data["utilities"], np.ones(100) / 100, mode='valid')
        plt.plot(avg_utils, label=f"主机数 = {num_hosts}")
    plt.axhline(y=results[3]["ne_utility"], color='k', linestyle='--', label='纳什均衡')
    plt.xlabel("训练轮次（×100）")
    plt.ylabel("防御者群体平均效用")
    plt.title("场景1：不同主机数量的效用变化")
    plt.legend()
    plt.grid(True)

    # 子图2：最终效用对比
    plt.subplot(1, 2, 2)
    final_utils = [np.mean(results[h]["utilities"][-1000:]) for h in num_hosts_list]
    plt.plot(num_hosts_list, final_utils,marker='o', linestyle='-', linewidth=2, markersize=8 )
    plt.axhline(y=results[3]["ne_utility"], color='k', linestyle='--', label='纳什均衡')
    plt.xlabel("主机数量")
    plt.ylabel("最终1000轮平均效用")
    plt.title("场景1：不同主机数量的最终效用")
    plt.legend()
    plt.grid(True, axis='y')
    plt,xticks(num_hosts_list)
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    # 选择要运行的场景（可单独运行）
    print("=" * 50)
    print("高级持续性威胁（APT）防御仿真代码")
    print("=" * 50)

    # 运行场景1：不同主机数量
    print("\n1. 运行场景1：不同主机数量的可扩展性测试")
    scenario1_results = simulate_scenario1()