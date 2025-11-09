from Algorithm2 import  *

def simulate_scenario3():
    """场景3：变化防御者资源（3-6），测试适应性"""
    avg_resources_list = [3, 4, 5, 6]
    results = {}

    for avg_res in avg_resources_list:
        # 资源范围：[avg-1, avg+1]
        res_range = [avg_res - 1, avg_res + 1]
        print(f"正在运行场景3：防御者平均资源 = {avg_res}（范围{res_range}）")

        utils, losses, ne = algorithm2_multi_defender_attacker(
            num_hosts=4,
            num_defenders=4,
            num_attackers=4,
            defender_resources_range=res_range,
            attacker_resources_range=[3, 5],  # 攻击者资源固定
            num_rounds=1000
        )
        results[avg_res] = {
            "utilities": utils,
            "losses": losses,
            "ne_utility": ne
        }

    # 可视化结果
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 6))

    # 子图1：不同资源的效用变化
    plt.subplot(1, 2, 1)
    for avg_res, data in results.items():
        avg_utils = np.convolve(data["utilities"], np.ones(100) / 100, mode='valid')
        plt.plot(avg_utils, label=f"平均资源 = {avg_res}")
    plt.axhline(y=results[3]["ne_utility"], color='k', linestyle='--', label='纳什均衡')
    plt.xlabel("训练轮次（×100）")
    plt.ylabel("防御者群体平均效用")
    plt.title("场景3：不同防御者资源的效用变化")
    plt.legend()
    plt.grid(True)

    # 子图2：最终效用与资源的关系
    plt.subplot(1, 2, 2)
    final_utils = [np.mean(results[r]["utilities"][-1000:]) for r in avg_resources_list]
    plt.plot(avg_resources_list, final_utils, marker='o', linewidth=2)
    plt.xlabel("防御者平均资源")
    plt.ylabel("最终1000轮平均效用")
    plt.title("场景3：防御者资源与最终效用关系")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    # 选择要运行的场景（可单独运行）
    print("=" * 50)
    print("高级持续性威胁（APT）防御仿真代码")
    print("=" * 50)


    print("\n3. 运行场景3：不同防御者资源的适应性测试")
    scenario3_results = simulate_scenario3()
