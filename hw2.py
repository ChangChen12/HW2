import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from tqdm import tqdm  # ✅ 添加进度条
from models import PolicyNetwork, ValueNetwork
from utils import compute_returns

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, choices=["pg", "pgb", "ppo"], required=True, help="Choose which RL algorithm to run")
parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
args = parser.parse_args()

# Select environment
env = gym.make("Ant-v4")  # Change to "Ant-v5" if needed
# env = gym.make("CartPole-v1")  # 🚀 Use this for faster testing

# Get state and action dimensions
state_dim = env.observation_space.shape[0]
if isinstance(env.action_space, gym.spaces.Discrete):
    action_dim = env.action_space.n  
else:
    action_dim = env.action_space.shape[0]

print(f"Using environment: {env.spec.id}")
print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

# Initialize policy network
policy = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)  # 🚀 Lower learning rate for stability

# Initialize value network for PGB and PPO
if args.algo in ["pgb", "ppo"]:
    value_net = ValueNetwork(state_dim)
    value_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)  # 🚀 Ensure value learning is stable

# Store rewards for plotting
all_rewards = []

# Training loop with tqdm progress bar ✅
progress_bar = tqdm(range(args.num_episodes), desc=f"Training {args.algo.upper()}", unit="episode")

for episode in progress_bar:
    state, _ = env.reset()
    log_probs, rewards, values = [], [], []

    max_steps = 1000  # Limit maximum steps per episode
    step_count = 0

    while True:
        step_count += 1
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_mean = policy(state_tensor)  # Get mean action
        action_std = torch.ones_like(action_mean) * 0.1  # Set standard deviation
        action_dist = distributions.Normal(action_mean, action_std)  # Define normal distribution
        action = action_dist.sample()  # Sample action
        action = torch.clamp(action, -1, 1)  # Clamp action range

        log_probs.append(action_dist.log_prob(action).sum().unsqueeze(0))  # 🚀 确保维度正确
        state, reward, done, _, _ = env.step(action.detach().numpy())

        rewards.append(reward)

        # Store value estimates for PGB and PPO
        if args.algo in ["pgb", "ppo"]:
            values.append(value_net(state_tensor))  # 🚀 Keep `values` in the computation graph!

        if done or step_count >= max_steps:
            break

    # Compute Reward-to-Go
    returns = compute_returns(rewards)
    returns = torch.tensor(returns, dtype=torch.float32, requires_grad=False)  # Ensure tensor format

    # Compute advantage for PGB and PPO
    if args.algo in ["pgb", "ppo"]:
        values = torch.cat(values)  # 🚀 No `.detach()` so gradients flow properly
        advantages = returns - values  # Compute advantage function
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 🚀 Normalize advantages

    # Compute loss for PG
    if args.algo == "pg":
        loss = -sum([log_p * G for log_p, G in zip(log_probs, returns)])

    # Compute loss for PGB (Policy Gradient with Baseline)
    elif args.algo == "pgb":
        loss = -sum([log_p * A.detach() for log_p, A in zip(log_probs, advantages)])  # 🚀 Detach() 防止二次反向传播

        # Compute value function loss (Mean Squared Error)
        value_loss = ((returns - values) ** 2).mean()

        # 🚀 First update Value Network
        value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)  # ✅ 保留计算图
        value_optimizer.step()

        # 🚀 Then update Policy Network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute loss for PPO (Proximal Policy Optimization)
    elif args.algo == "ppo":
        epsilon = 0.1  # ✅ Reduce clipping range

        # ✅ 计算旧策略的 log_probs
        old_log_probs = torch.stack(log_probs).detach()  

        # ✅ 计算 importance sampling ratio
        ratios = torch.exp(torch.stack(log_probs) - old_log_probs)  # ✅ 修正计算方式

        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()  # ✅ Fix PPO loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Store rewards for visualization
    all_rewards.append(sum(rewards))

    # ✅ Update progress bar with reward info
    progress_bar.set_postfix({"Total Reward": sum(rewards)})

# Save rewards for plotting
np.save(f"results/rewards_{args.algo}.npy", np.array(all_rewards))

print("Training complete!")
