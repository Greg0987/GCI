
import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm

def cal_pairwise_sim(x):
    '''Input: x: [n, d]
         Return: sim: [n, n]
     '''
    # x = F.normalize(x, p=2, dim=1)
    # sim是（n,n）的矩阵，sim[i,j]表示第i个样本与第j个样本之间的余弦相似度
    print(x.size(0))
    # 建立(n, n)的张量
    # sim = torch.zeros((1000, 1000))
    sim = torch.zeros((x.size(0), x.size(0)))

    for i in tqdm(range(x.size(0))):
        for j in range(i, x.size(0)):
            # 计算第i个样本与第j个样本之间的余弦相似度
            tmp = torch.dot(x[i], x[j]) / (torch.norm(x[i]) * torch.norm(x[j]))
            # 将tmp从[-1, 1]映射到[0, 1]
            sim[i, j] = tmp
            sim[j, i] = tmp


    # 对角线元素设置为1
    # sim = sim + torch.diag(torch.ones(x.size(0)))

    # 判断矩阵是否为对称矩阵
    if not torch.equal(sim, sim.t()):
        print('The similarity matrix is not symmetric!')
    return sim

def train(args):
    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ['small', 'big']
    prompt = ['lp', 'sp']
    beta = [0.8, 0.6, 0.4, 0.2]

    if not os.path.exists(f'img_adj'):
        os.makedirs(f'img_adj')
    for data in dataset:
        for p in prompt:
            if not os.path.exists(f'img_adj/{data}_{p}_ori.npy'):
                img_file = f'image_{data}_{p}_features.txt'
                feats_img = torch.from_numpy(np.genfromtxt(img_file, dtype=float)[:, :])
                feats_img.to(device)
                img_adj = cal_pairwise_sim(feats_img) # -> [n, n]的邻接矩阵，即进行img相似度的计算并构图
                img_adj_ori = torch.FloatTensor(img_adj)

                np.save(f'img_adj/{data}_{p}_ori.npy', img_adj_ori.numpy())
                print(f'Image similarity graph {data}_{p}_ori construction completed!')
            else:
                img_adj_ori = torch.from_numpy(np.load(f'img_adj/{data}_{p}_ori.npy'))
            # np.save(f'img_adj/{data}_{p}_test.npy', img_adj.numpy())

            for b in beta:
                # 讲img_adj的值小于b的值设置为0，大于等于b的值设置为1
                img_adj = img_adj_ori.clone()
                img_adj[img_adj < b] = 0
                img_adj[img_adj >= b] = 1
                # 转换成torch.int类型
                # 保存图，用numpy格式保存
                np.save(f'img_adj/{data}_{p}_{b}.npy', img_adj.numpy())
                print(f'Image similarity graph {data}_{p}_{b} construction completed!')\

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Image Similarity Graph')
    parser.add_argument('--gpu', type=str, default='-1', help='gpu id')

    args = parser.parse_args()
    train(args)