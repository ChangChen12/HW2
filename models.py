import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),  # 🚀 增加隐藏层大小，提高表示能力
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 🚀 限制输出范围在 [-1, 1]
        )

    def forward(self, state):
        return self.fc(state)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),  # 🚀 增加隐藏层大小
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 🚀 输出单个 V(s)
        )

    def forward(self, state):
        return self.fc(state).squeeze()  # 🚀 确保输出是标量
