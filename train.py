import os
import copy
import json
import pickle
import argparse

import numpy as np
import scipy.sparse as sp
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1000'

import torch
from torch.utils.tensorboard import SummaryWriter

from util_functions import get_data_split, get_acc, setup_seed, use_cuda
from util_functions import load_data_set, symmetric_normalize_adj

from GCI import GCI
from GCI_run import GCI_run
from eval import eval

def train(args):
    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0

    [c_train, c_val] = args.train_val_class
    # 加载数据集
    """
    idx: list[n, 1]
    labellist: list[n]
    G: np.array[n, n]           # 相当于A
    features: np.array[n, 523]  # 相当于X
    csd_matrix: np.array[c, 512]# 相当于C
    """
    idx, labellist, G, features, csd_matrix = load_data_set(args.dataset)
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())    # 转换为tensor
    # 将labellist的元素转化为整数，转换为torch.longtensor
    labels = [int(x[0]) for x in labellist]
    labels = torch.LongTensor(labels)

    if args.img_feat == 'long':
        img_file = f'preprocess/image_{args.dataset}_lp_features.txt'
        feats_img = torch.from_numpy(np.genfromtxt(img_file, dtype=float)[:, :])
        adj_img = torch.from_numpy(np.load(f'preprocess/img_adj/{args.dataset}_lp_{args.beta}.npy'))
    elif args.img_feat == 'short':
        img_file = f'preprocess/image_{args.dataset}_sp_features.txt'
        feats_img = torch.from_numpy(np.genfromtxt(img_file, dtype=float)[:, :])
        adj_img = torch.from_numpy(np.load(f'preprocess/img_adj/{args.dataset}_sp_{args.beta}.npy'))

    csd_file = 'preprocess/csd_norm.txt'    # 用的是csd_norm
    csd_img = torch.from_numpy(np.genfromtxt(csd_file, dtype=float)[:, :])

    results_path = f'results/{args.name}/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    writer_path = f'results/{args.name}/{args.dataset}_{c_train}{c_val}'

    if not os.path.exists(f'npy_for_pre_recall_f1/{args.name}'):
        os.makedirs(f'npy_for_pre_recall_f1/{args.name}')
    if not os.path.exists(f'models_pt/{args.name}'):
        os.makedirs(f'models_pt/{args.name}')

    accs = []
    for i in range(args.n_trains):
        writer = SummaryWriter(writer_path + f'/{i+1}')
        model_run = GCI_run(args, G, features, labels, c_train, c_val, csd_matrix, adj_img, feats_img, csd_img,
                 cuda=gpu, hidden_size=args.n_hidden, emb_size=args.n_emb, n_layers=args.k, n_epochs=args.n_epochs, seed=args.seed,
                 lr=args.lr, weight_decay=args.wd, dropout=args.dropout, gae=args.use_gae, temperature=args.temperature, warmup=args.warmup,
                 gnnlayer_type=args.gnn, sample_type=args.sample_type, feat_norm='row')
        acc = model_run.train(args, writer)
        accs.append(acc)
    # print(f'mean acc: {np.mean(accs):.4f}, std: {np.std(accs):.4f}')
    if args.acc_only:
        print(f'best acc: {np.max(accs):.4f}')
    else:
        p, r, f1 = eval(args.name, args.dataset, c_train, c_val)

        print(f'best acc: {np.max(accs):.4f}, precision: {p:.4f}, recall: {r:.4f}, f1: {f1:.4f}')
    print('='*30)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--name', type=str, default='GCI', help="exp name")
    # big是Kuairec稀疏大图，small是Kuairec稠密小图
    parser.add_argument('--dataset', type=str, default='small', choices=['cora', 'citeseer', 'C-M10-M', 'small', 'big'], help="dataset")
    # 图像特征种类
    parser.add_argument('--img_feat', type=str, default='long', choices=['long', 'short'], help="image feature type")
    parser.add_argument('--temp', type=float, default=1.0, help="temp for distillation")
    # 图像特征构造新图，是否有阈值
    parser.add_argument('--use_ep', type=bool, default=False, help="use edge prediction or not")
    parser.add_argument("--beta", type=float, default=0.6, choices=[0.2,0.4,0.6,0.8],
                        help="threshold for image feature, range from 0 to 1")
    # parser.add_argument('--threshold', type=float, default=-1.0, help="threshold for image feature, range from 0 to 1, -1 means no threshold")
    # 图补全学习基准：0为原图A，1为新图I（图像特征构造的新图）
    parser.add_argument('--graph_base', type=int, default=0, choices=[0, 1], help="edge prediction learning baseline")
    # 图补全选边的概率apha
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha for edge prediction")
    # 不同的图网络类型
    parser.add_argument('--gnn', type=str, default='gsage', choices=['gcn', 'gat', 'gsage'])
    parser.add_argument('--sample_type', type=str, default='add_sample', choices=['add_sample', 'edge'])
    parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
    # 数据集划分
    parser.add_argument("--train_val_class", type=int, nargs='*', default=[4, 3],
                        help="the first #train_class and #validation classes")

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout probability")
    parser.add_argument('--warmup', type=int, default=0, help="warmup epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--wd', type=float, default=5e-4, help="weight decay")
    parser.add_argument('--n_epochs', type=int, default=500, help="number of training epochs")
    parser.add_argument('--n_hidden', type=int, default=512, help="number of hidden layers", choices=[128, 300])
    parser.add_argument('--n_emb', type=int, default=32, help="number of embedds in EP", choices=[32])
    parser.add_argument('--use_gae', type=bool, default=False, help="use gae or not")

    # k-hop邻居，k-1层GNN
    parser.add_argument("--k", type=int, default=1, help="(k+1)-hop neighbors")

    parser.add_argument('--seed', type=int, default=1234, help="random seed")
    parser.add_argument('--n_trains', type=int, default=1, help="number of training times")

    # lambda系数 λ
    parser.add_argument('--lam_1', type=float, default=1.0, help="weight for node classification loss")
    parser.add_argument('--lam_2', type=float, default=1.0, help="weight for img loss")
    parser.add_argument('--lam_3', type=float, default=1.0, help="weight for edge predict loss")

    parser.add_argument('--acc_only', type=bool, default=False, help="only calculate acc or not")
    args = parser.parse_args()

    print(args)
    train(args)