import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curve(rewards, algo_name="PG", save_path="results/pg_learning_curve.png", window_size=20):
    """
    绘制训练奖励曲线 (Learning Curve)
    
    :param rewards: list or array, 每个 episode 的总奖励
    :param algo_name: str, 算法名称（用于图例）
    :param save_path: str, 保存图片的路径
    :param window_size: int, 滑动平均窗口大小
    """
    episodes = np.arange(len(rewards))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label=f"{algo_name} (raw)", alpha=0.3)  # 原始数据

    # **确保窗口大小不会超过数据量**
    if len(rewards) >= window_size:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, label=f"{algo_name} (smoothed)", linestyle="--")

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(f"Learning Curve of {algo_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    rewards_path = "results/pg_rewards.npy"

    if os.path.exists(rewards_path):
        rewards = np.load(rewards_path)
        plot_learning_curve(rewards, algo_name="PG", save_path="results/pg_learning_curve.png", window_size=50)
    else:
        print(f"Error: {rewards_path} not found. Please run training first.")
