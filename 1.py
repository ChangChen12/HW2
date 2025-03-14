import numpy as np

data = np.load("results/rewards_pg.npy")
print(data.shape)  # 应该是 (1000,)
print(data[:10])   # 预览前 10 轮的奖励
