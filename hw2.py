import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from tqdm import tqdm
from models import PolicyNetwork, ValueNetwork
from utils import compute_returns

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, choices=["pg", "pgb", "ppo"], required=True, help="Choose which RL algorithm to run")
parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
args = parser.parse_args()

# Select environment
env = gym.make("Ant-v4")

# Get state and action dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n

print(f"Using environment: {env.spec.id}")
print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

# Initialize policy network
policy = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

# Initialize value network for PGB and PPO
if args.algo in ["pgb", "ppo"]:
    value_net = ValueNetwork(state_dim)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

# Store rewards for plotting
all_rewards = []

# Training loop with tqdm progress bar
progress_bar = tqdm(range(args.num_episodes), desc=f"Training {args.algo.upper()}", unit="episode")

for episode in progress_bar:
    state, _ = env.reset()
    log_probs, rewards, values = [], [], []

    max_steps = 1000
    step_count = 0

    while True:
        step_count += 1
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_mean = policy(state_tensor)
        action_std = torch.ones_like(action_mean) * 0.1
        action_dist = distributions.Normal(action_mean, action_std)
        action = action_dist.sample()
        action = torch.clamp(action, -1, 1)

        log_probs.append(action_dist.log_prob(action).sum().unsqueeze(0))  # ✅ 确保 log_prob 是 1D

        state, reward, done, _, _ = env.step(action.detach().numpy())
        rewards.append(reward)

        if args.algo in ["pgb", "ppo"]:
            values.append(value_net(state_tensor).view(-1))  # ✅ 修正 0D Tensor 问题

        if done or step_count >= max_steps:
            break

    # Compute returns
    returns = compute_returns(rewards).detach()

    # 🚀 确保 `values` 不是空的
    if args.algo in ["pgb", "ppo"]:
        if len(values) == 0:
            raise RuntimeError("Error: `values` is empty. Ensure value_net is correctly computing V(s).")
        values = torch.cat(values)  # ✅ 修正 `torch.cat()` 报错
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Compute loss for PG
    if args.algo == "pg":
        loss = -torch.mean(torch.stack(log_probs) * returns)

    # Compute loss for PGB
    elif args.algo == "pgb":
        loss = -torch.mean(torch.stack(log_probs) * advantages.detach())
        value_loss = ((returns - values) ** 2).mean()

        value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        value_optimizer.step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute loss for PPO
    elif args.algo == "ppo":
        epsilon = 0.2
        log_probs = torch.cat(log_probs)
        old_log_probs = log_probs.detach()
        ratios = torch.exp(log_probs - old_log_probs)

        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    all_rewards.append(sum(rewards))

    progress_bar.set_postfix({"Total Reward": sum(rewards)})

# Save rewards for plotting
np.save(f"results/rewards_{args.algo}.npy", np.array(all_rewards))

print("Training complete!")
