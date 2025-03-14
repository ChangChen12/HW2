import matplotlib.pyplot as plt
import numpy as np

# Define algorithms
algos = ["pg", "pgb", "ppo"]

plt.figure(figsize=(10, 6))

for algo in algos:
    try:
        # Load rewards
        rewards = np.load(f"results/rewards_{algo}.npy")

        # 🚀 适配少量数据，防止窗口大小超过数据点数
        window_size = min(10, len(rewards))  
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

        # Plot each algorithm's curve
        plt.plot(smoothed_rewards, label=f"{algo.upper()}")  # 绘制平滑奖励曲线
    except FileNotFoundError:
        print(f"Warning: No data found for {algo}")

plt.xlabel("Episodes")
plt.ylabel("Undiscounted Return")
plt.title("Learning Curves for PG, PGB, PPO in Ant-v4 (50 Episodes)")
plt.legend()
plt.grid()

# Save figure
plt.savefig("results/learning_curve_50.png")  # ✅ 适配 50 轮训练
plt.show()
