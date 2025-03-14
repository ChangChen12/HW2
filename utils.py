import torch

def compute_returns(rewards, gamma=0.99, normalize=True):
    """
    Compute discounted rewards (Reward-to-Go) for policy gradient algorithms.
    
    Args:
        rewards (list): List of rewards per episode.
        gamma (float): Discount factor.
        normalize (bool): Whether to normalize the returns.

    Returns:
        torch.Tensor: Discounted returns as a PyTorch tensor.
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns, dtype=torch.float32)  # 🚀 直接转换为 PyTorch Tensor

    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 🚀 归一化，提升稳定性

    return returns
