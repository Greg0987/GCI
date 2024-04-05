import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=10000)

beta = [0.8, 0.6, 0.4, 0.2]
dataset = ['small', 'big']
prompt = ['lp', 'sp']
#graph_path = f'img_adj/{data}_{p}_{a}.npy'

adj_beta = []
for b in beta:
    # 读取numpy数组
    tmp = np.load(f'img_adj/small_lp_{b}.npy')
    adj_beta.append(tmp)

for i in adj_beta:
    plt.figure(figsize=(10, 10))
    plt.imshow(i, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()




