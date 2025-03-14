import argparse
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import os
from models import PolicyNetwork

# 确保 results 目录存在
os.makedirs("results", exist_ok=True)

def compute_returns(rewards, gamma=0.99):
    """
    计算 Reward-to-Go，并进行归一化
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # 归一化 returns 以减少高方差问题
    mean = returns.mean()
    std = returns.std() + 1e-8  # 避免除零错误
    return (returns - mean) / std

def save_model(policy, path="results/pg_model.pth"):
    """
    保存模型到 results 目录
    """
    torch.save(policy.state_dict(), path)
    print(f"Model saved to {path}")

def save_rewards(rewards, path="results/pg_rewards.npy"):
    """
    保存奖励数据到 results 目录
    """
    np.save(path, rewards)
    print(f"Rewards saved to {path}")

def train_pg(env_name="Ant-v4", num_episodes=1000, gamma=0.99, lr=1e-4, batch_size=5):
    """
    训练 Vanilla Policy Gradient (REINFORCE) 
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []  # 记录每个 episode 的总奖励

    for episode in range(num_episodes):
        batch_log_probs = []
        batch_rewards = []

        # 采样 batch_size 个 trajectories 进行更新
        for _ in range(batch_size):
            state, _ = env.reset()
            log_probs = []
            rewards = []
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action, log_prob = policy.sample_action(state_tensor)
                next_state, reward, terminated, truncated, _ = env.step(action.detach().numpy())
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state

            batch_log_probs.append(torch.stack(log_probs))
            batch_rewards.append(rewards)

        # 计算所有轨迹的 Reward-to-Go
        batch_returns = [compute_returns(r, gamma) for r in batch_rewards]
        batch_returns = torch.cat(batch_returns)  # 转换成 Tensor
        batch_log_probs = torch.cat(batch_log_probs)  # 转换成 Tensor

        # 计算策略梯度损失
        loss = -torch.sum(batch_log_probs * batch_returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录 batch 内所有 episode 的平均 reward
        avg_reward = np.mean([sum(r) for r in batch_rewards])
        episode_rewards.append(avg_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

    env.close()

    # 保存训练结果
    save_model(policy)  
    save_rewards(episode_rewards)

    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="pg", choices=["pg", "pgb", "ppo"],
                        help="Algorithm to run: pg (REINFORCE), pgb (PG with Baseline), ppo (PPO)")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes for training")
    
    args = parser.parse_args()

    if args.algo == "pg":
        episode_rewards = train_pg(num_episodes=args.num_episodes)
