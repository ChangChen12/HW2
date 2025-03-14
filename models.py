import torch
import torch.nn as nn
import torch.distributions as distributions

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出动作均值 (mean)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        
        # 输出 log 标准差 (log_std)（保证 std 始终为正）
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

        self.activation = nn.ReLU()

    def forward(self, state):
        """ 计算策略网络输出的动作分布参数 """
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))

        mean = self.fc_mean(x)  # 输出均值
        log_std = self.fc_log_std(x)  # log(标准差)
        std = torch.exp(log_std)  # 确保标准差为正

        return mean, std

    def sample_action(self, state):
        """ 从高斯分布中采样动作，并计算 log 概率 """
        mean, std = self.forward(state)
        dist = distributions.Normal(mean, std)  # 正态分布
        action = dist.sample()  # 采样动作
        log_prob = dist.log_prob(action).sum(dim=-1)  # 计算 log π_θ(a|s)

        return action, log_prob
