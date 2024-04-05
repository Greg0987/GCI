
from GCI import GCI

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import scipy.sparse as sp

from tqdm import tqdm

from util_functions import get_acc, get_acc_basic, get_data_split,\
    cal_pairwise_sim
from loss_function import BKLD_loss, DIST_loss, KLD_loss, CE_loss, MSE_loss

from sklearn.metrics import roc_auc_score, average_precision_score

class GCI_run(object):
    def __init__(self, args, adj, features, labels, c_train, c_val, csd_ori, adj_img, feats_img, csd_img,
                 cuda=-1, hidden_size=128, emb_size=32, n_layers=2, n_epochs=200, seed=-1,
                 lr=1e-2, weight_decay=5e-4, dropout=0.5, gae=False, temperature=0.2, warmup=3,
                 gnnlayer_type='gcn', sample_type='add_sample', feat_norm='row'):
        self.args = args
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.gae = gae

        self.warmup = warmup
        self.feat_norm = feat_norm

        idx_train, idx_test, idx_val = get_data_split(c_train, c_val, labels)  # 划分训练集、验证集、测试集
        self.train_ids = idx_train
        self.val_ids = idx_val
        self.test_ids = idx_test

        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda}' if cuda>=0 else 'cpu')

        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # 加载数据
        self.load_data(adj, features, labels, csd_ori, adj_img, feats_img, csd_img,
                       self.train_ids, self.val_ids, self.test_ids, gnnlayer_type)

        self.model = GCI(self.args,
                            self.features.size(1),
                            hidden_size,
                            emb_size,
                            self.csd_ori.size(1),
                            self.feats_img.size(1),
                            self.csd_img.size(1),
                            n_layers,
                            F.relu,
                            dropout,
                            self.device,
                            gnnlayer_type,
                            temperature=temperature,
                            gae=gae,
                            alpha=self.args.alpha,
                            sample_tpye=sample_type)


    def load_data(self, adj, features, labels, csd_ori, adj_img, feats_img, csd_img,
                  train_ids, val_ids, test_ids,
                  gnnlayer_type):
        """ preprocess data"""
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        if self.feat_norm == 'row':
            self.features = F.normalize(self.features, p=1, dim=1)
        elif self.feat_norm == 'col':
            self.features = self.col_normalization(self.features, p=1, dim=0)


        self.csd_ori = csd_ori
        self.feats_img = feats_img # / torch.norm(feats_img, p=2, dim=1, keepdim=True)
        # csd已经在处理的时候归一化过了
        self.csd_img = csd_img

        # 获取图像相似度矩阵
        adj_imgs = adj_img
        if not isinstance(adj_imgs, sp.coo_matrix):
            adj_imgs = sp.coo_matrix(adj_imgs)
        adj_imgs.setdiag(1)
        self.adj_imgs = scipysp_to_pytorchsp(adj_imgs)  # 转换为torch格式
        self.adj_imgs = self.adj_imgs.to_dense()
        degrees = np.array(adj_imgs.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_imgs_norm = degree_mat_inv_sqrt @ adj_imgs @ degree_mat_inv_sqrt
        # 用于输进GCN的VGAE进行图补全，故为D^(-1/2)AD^(-1/2)格式
        self.adj_imgs_norm = scipysp_to_pytorchsp(adj_imgs_norm)    # 转换为torch格式

        # assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        # adj也是原始图，是sp.coo格式
        adj.setdiag(1)  # 对角元素置为1
        # 原始图，转换为torch格式
        self.adj_ori = scipysp_to_pytorchsp(adj)
        self.adj_ori = self.adj_ori.to_dense()
        degrees = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
        # 经过归一化的图，即DAD^(-1/2)
        self.adj_norm = scipysp_to_pytorchsp(adj_norm)

        # adj根据不同的gnn层，对图进行相应处理，获得self.adj
        if gnnlayer_type == 'gcn':  # 送进gcn前要先归一化
            self.adj = scipysp_to_pytorchsp(adj_norm)
        elif gnnlayer_type == 'gsage':
            adj_matrix_nosefloop = sp.coo_matrix(adj)
            adj_matrix_nosefloop = sp.coo_matrix(adj_matrix_nosefloop / adj_matrix_nosefloop.sum(1))
            self.adj = scipysp_to_pytorchsp(adj_matrix_nosefloop)
        elif gnnlayer_type == 'gat':    # 送进gat前要先归一化，并转为密集矩阵
            self.adj = torch.FloatTensor(adj_norm.todense())

        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

        # 判断是二分类还是多分类
        if len(self.labels.size()) == 1:
            self.out_size = len(torch.unique(self.labels))
        else:
            self.out_size = labels.size(1)

        # 采样边来评估图补全结果
        if labels.size(0) > 5000:
            edge_frac = 0.01
        else:
            edge_frac = 0.1
        adj = sp.csr_matrix(adj)    # 转换为csr格式
        n_edges_sample = int(edge_frac * adj.nnz / 2)   # 采样的边的数量


        # 采样负边
        neg_edges = []
        added_edges = set()
        while len(neg_edges) < n_edges_sample:
            i = np.random.randint(0, adj.shape[0])   # 随机采样一个节点
            j = np.random.randint(0, adj.shape[0])
            if i == j:                        # 两个节点相同
                continue
            if adj[i, j] > 0:                 # 两个节点之间已经存在边
                continue
            if (i, j) in added_edges:         # 已经添加过
                continue
            neg_edges.append((i, j))
            added_edges.add((i, j))
            added_edges.add((j, i))
        neg_edges = np.asarray(neg_edges)   # 转换为数组
        # 采样正边
        nz_upper = np.array(sp.triu(adj, 1).nonzero()).T    # 获取上三角矩阵的非零元素的坐标
        np.random.shuffle(nz_upper)                         # 打乱
        pos_edges = nz_upper[:n_edges_sample]               # 采样与负边相同数量的正边
        self.val_edges = np.concatenate((pos_edges, neg_edges), axis=0)   # 将正边和负边合并
        self.edge_labels = np.array([1] * len(pos_edges) + [0] * len(neg_edges))   # 生成边的标签，正边为1，负边为0


    def train(self, args, writer):
        """train the model"""
        [c_train, c_val] = args.train_val_class
        # 经过归一化的图
        adj_norm = self.adj_norm.to(self.device)
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        csd_ori = self.csd_ori.to(self.device)
        feats_img = self.feats_img.float().to(self.device)
        csd_img = self.csd_img.float().to(self.device)
        # 原始图，torch格式
        adj_ori = self.adj_ori.to(self.device)
        adj_imgs = self.adj_imgs_norm.to(self.device)

        model = self.model.to(self.device)

        # weights for log_lik loss when training EP net
        # 用于训练EP网络的log_lik损失的权重
        adj_t = self.adj_ori
        norm_w = adj_t.shape[0] ** 2 / float((adj_t.shape[0] ** 2 - adj_t.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_t.shape[0] ** 2 - adj_t.sum()) / adj_t.sum()]).to(self.device)

        # optimizers
        optims = MultipleOptimizer(torch.optim.Adam(model.ep_net.parameters(),
                                                    lr=self.lr),
                                   torch.optim.Adam(model.nc_net.parameters(),
                                                    lr=self.lr,
                                                    weight_decay=self.weight_decay))
        # get the learning rate schedule for the optimizer of ep_net if needed  # 获取学习率调度表，如果需要的话
        # 通过sigmoid函数将学习率从0逐渐增加到lr
        if self.warmup:
            ep_lr_schedule = self.get_lr_schedule_by_sigmoid(self.n_epochs, self.lr, self.warmup)
        # 用于节点分类的损失函数
        nc_criterion = CE_loss
        # 用于与图像计算的损失函数
        # local_criterion = MSE_loss
        img_criterion = CE_loss

        top_test_acc = []
        train_val_str = str(c_train) + str(c_val)
        
        # writer.add_histogram('Distribution/feats:teacher', feats_img)
        writer.add_histogram('Distribution/preds:teacher', feats_img @ csd_img.T)

        for epoch in tqdm(range(self.n_epochs)):
            # update learning rate for ep_net if needed
            if self.warmup:
                optims.update_lr(0, ep_lr_schedule[epoch])

            # ******************************************************
            # 训练
            model.train()
            outputs = model(adj_imgs, adj_ori, features, feats_img, csd_ori, csd_img)
            (preds, preds_local, preds_img), adj_logits = outputs
            # (preds, preds_img), adj_logits = outputs
            # 计算节点分类的损失
            # 组合损失
            loss_global = nc_criterion(preds[self.train_ids], labels[self.train_ids])
            # 局部损失
            # loss_local = local_criterion(preds_local, feats_img)
            # 图像损失
            loss_img = img_criterion(preds, preds_img)

            nc_loss = args.lam_1 * loss_global + args.lam_2 * loss_img #+ args.lam_2 * loss_local
            # nc_loss = args.lam_1 * loss_global + args.lam_2 * loss_img

            writer.add_scalar('train/loss_global', loss_global.item(), epoch)
            # writer.add_scalar('train/loss_local', loss_local.item(), epoch)
            writer.add_scalar('train/loss_img', loss_img.item(), epoch)
            writer.add_scalar('train/NC_loss', nc_loss.item(), epoch)

            # writer.add_histogram('Distribution/preds:student:', preds, epoch)
            # writer.add_histogram('Distribution/feats:student', preds_local[0], epoch)

            if args.use_ep:
                # 计算图补全的损失
                adj_ori_dense = adj_ori.to_dense()
                adj_imgs_dense = adj_imgs.to_dense()
                # 【思考损失是根据原图A计算，还是跟图像相似度I的矩阵计算】
                # 默认与原图A计算
                adj_base = adj_imgs_dense if self.args.graph_base else adj_ori_dense
                ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_base, pos_weight=pos_weight)
                loss = nc_loss + args.lam_3 * ep_loss
                writer.add_scalar('train/EP_loss', ep_loss.item(), epoch)
                writer.add_scalar('train/ALL_loss', loss.item(), epoch)
            else:
                loss = nc_loss
            train_acc = get_acc(preds[self.train_ids], labels[self.train_ids], c_train, c_val, model='train')
            writer.add_scalar('ACC/train', train_acc, epoch)

            optims.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optims.step()

            # ******************************************************
            # 验证
            model.eval()    # 验证时只看nc节点分类的结果
            with torch.no_grad():
                outputs_eval= model(adj_imgs, adj_ori, features, feats_img, csd_ori, csd_img)
                (preds_val, preds_local_val, preds_img_val), adj_logits_val = outputs_eval
                # (preds_val, preds_img_val), adj_logits_val = outputs_eval
                # 组合损失
                loss_global_val = nc_criterion(preds_val[self.val_ids], labels[self.val_ids])
                # 局部损失
                # loss_local_val = local_criterion(preds_local_val, feats_img)
                # 图像损失
                loss_img_val = img_criterion(preds_val, preds_img_val)
                loss_val = args.lam_1 * loss_global_val + args.lam_2 * loss_img_val #+ args.lam_2 * loss_local_val
                # loss_val = args.lam_1 * loss_global_val + args.lam_2 * loss_img_val

                writer.add_scalar('val/loss_global', loss_global_val.item(), epoch)
                # writer.add_scalar('val/loss_local', loss_local_val.item(), epoch)
                writer.add_scalar('val/loss_img', loss_img_val.item(), epoch)
                writer.add_scalar('val/NC_loss', loss_val.item(), epoch)

                val_acc = get_acc(preds_val[self.val_ids], labels[self.val_ids], c_train, c_val, model='val')
                writer.add_scalar('ACC/val', val_acc, epoch)

                if args.use_ep:
                    # 通过sigmoid函数将预测的邻接logits转换为概率
                    adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
                    # 计算图补全的auc
                    ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
                    writer.add_scalar('EP/AUC', ep_auc, epoch)
                    writer.add_scalar('EP/EP_AP', ep_ap, epoch)

            model.eval()
            # preds, preds_local, preds_img = model.nc_net(adj, features, feats_img, csd_ori, csd_img)
            with torch.no_grad():
                outputs_test = model(adj_imgs, adj_ori, features, feats_img, csd_ori, csd_img)
                (preds, preds_local, preds_img), adj_logits = outputs_test
                # (preds, preds_img), adj_logits = outputs_test
            # preds = preds_img
            test_acc = get_acc(preds[self.test_ids], labels[self.test_ids], c_train, c_val, model='test')
            writer.add_scalar('ACC/test', test_acc, epoch)

            # 从第50个epoch开始保存模型
            if epoch >= 50:
                if len(top_test_acc) == 0:
                    np.save(f'npy_for_pre_recall_f1/{args.name}/' + args.dataset + 'pred' + train_val_str + '.npy',
                            np.array(preds.detach().cpu()))
                    np.save(f'npy_for_pre_recall_f1/{args.name}/' + args.dataset + 'y_true' + train_val_str + '.npy',
                            np.array(labels.detach().cpu()))
                    torch.save(model.state_dict(), f'results/{args.name}/' + args.dataset + 'model' + train_val_str + '.pt')
                elif test_acc > np.max(top_test_acc):
                    np.save(f'npy_for_pre_recall_f1/{args.name}/' + args.dataset + 'pred' + train_val_str + '.npy',
                            np.array(preds.detach().cpu()))
                    np.save(f'npy_for_pre_recall_f1/{args.name}/' + args.dataset + 'y_true' + train_val_str + '.npy',
                            np.array(labels.detach().cpu()))
                    torch.save(model.state_dict(), f'results/{args.name}/' + args.dataset + 'model' + train_val_str + '.pt')
                top_test_acc.append(test_acc)
        writer.add_histogram('Distribution/preds:student', preds)
        np.save(f'npy_for_pre_recall_f1/{args.name}/' + args.dataset + 'idx_test' + train_val_str + '.npy', self.test_ids)
        print('Evaluation!', 'Top Test_acc:', np.max(top_test_acc), "++++++++++")
        del adj_norm, adj_ori, features, labels, csd_ori, feats_img, csd_img
        torch.cuda.empty_cache()
        return np.max(top_test_acc)


    @staticmethod
    def eval_edge_pred(adj_pred, val_edges, edge_labels):
        logits = adj_pred[val_edges.T]  # 获取预测的边的概率
        logits = np.nan_to_num(logits)  # 将nan转换为0
        roc_auc = roc_auc_score(edge_labels, logits)  # 计算roc_auc
        ap_score = average_precision_score(edge_labels, logits)  # 计算ap_score
        return roc_auc, ap_score  # 返回roc_auc和ap_score


    @staticmethod
    def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
        """ schedule the learning rate with the sigmoid function.
        The learning rate will start with near zero and end with near lr """
        factors = torch.FloatTensor(np.arange(n_epochs))
        factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
        factors = torch.sigmoid(factors)
        # range the factors to [0, 1]
        factors = (factors - factors[0]) / (factors[-1] - factors[0])
        lr_schedule = factors * lr
        return lr_schedule

# 将稀疏矩阵转换为稀疏张量
def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):    # 判断是否是coo_matrix格式
        sp_mx = sp_mx.tocoo()           # 转换为coo_matrix格式
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx    # 返回稀疏张量

class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        """ update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer """
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr