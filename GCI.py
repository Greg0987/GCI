import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_functions import adj_to_edge_index
from torch_geometric.nn import SAGEConv

from lazy_random_walk_utils import get_lrw_pre_calculated_feature_list


class GCI(nn.Module):
    def __init__(self,
                 args,
                 dim_feats,  # 输入特征维度：523
                 dim_h,  # 隐藏层维度：128 | 原始nvidgpn是300/500
                 dim_z,  # 图补全隐藏维度： 32
                 dim_csd,  # csd_ori维度：300
                 dim_img,  # 图像特征维度：512
                 dim_csd_img,  # csd_img维度：768
                 k,  # 邻居数：2
                 activation,
                 dropout,
                 device,
                 gnnlayer_type,
                 temperature=1,
                 gae=False,
                 alpha=1,
                 sample_tpye='edge'):
        super(GCI, self).__init__()
        self.args = args
        self.device = device
        self.temperature = temperature
        self.gnnlayer_type = gnnlayer_type
        self.alpha = alpha
        self.sample_tpye = sample_tpye

        self.ep_net = VGAE(dim_img, dim_h, dim_z, activation, gae=gae)
        # self.nc_net = ZNC(dim_feats, dim_h, dim_csd, dim_img, dim_csd_img, k, activation, dropout, gnnlayer_type=gnnlayer_type)
        self.nc_net = DGPN(dim_feats, dim_h, dim_csd, dim_img, dim_csd_img, dropout,
                           self.args.k, self.args.beta)

    # 根据预测的边概率和原始邻接矩阵
    # 选择一部分边及逆行添加或删除，生成一个新的邻接矩阵
    # 【保留原始图结构的同时，引入一些新的边】
    def sample_adj_edge(self, adj_logits, adj_orig, change_frac):
        adj = adj_orig.to_dense() if adj_orig.is_sparse else adj_orig
        n_edges = adj.nonzero().size(0)  # 获取边的数量
        n_change = int(n_edges * change_frac / 2)  # 获取需要改变的边的数量
        # take only the upper triangle  # 取上三角矩阵
        edge_probs = adj_logits.triu(1)  # 获取预测的边概率
        # 使得边概率在0-1之间
        edge_probs = edge_probs - torch.min(edge_probs)
        edge_probs = edge_probs / torch.max(edge_probs)
        # 获取1-adj
        adj_inverse = 1 - adj
        # get edges to be removed
        mask_rm = edge_probs * adj  # 获取需要移除的边
        nz_mask_rm = mask_rm[mask_rm > 0]  # 获取需要移除的边的数量
        if len(nz_mask_rm) > 0:
            n_rm = len(nz_mask_rm) if len(nz_mask_rm) < n_change else n_change
            thresh_rm = torch.topk(mask_rm[mask_rm > 0], n_rm, largest=False)[0][-1]
            mask_rm[mask_rm > thresh_rm] = 0
            mask_rm = CeilNoGradient.apply(mask_rm)
            mask_rm = mask_rm + mask_rm.T
        # remove edges
        adj_new = adj - mask_rm
        # get edges to be added
        mask_add = edge_probs * adj_inverse
        nz_mask_add = mask_add[mask_add > 0]
        if len(nz_mask_add) > 0:
            n_add = len(nz_mask_add) if len(nz_mask_add) < n_change else n_change
            thresh_add = torch.topk(mask_add[mask_add > 0], n_add, largest=True)[0][-1]
            mask_add[mask_add < thresh_add] = 0
            mask_add = CeilNoGradient.apply(mask_add)
            mask_add = mask_add + mask_add.T
        # add edges
        adj_new = adj_new + mask_add
        return adj_new

    # 【直接从预测的边概率采样邻接矩阵】
    def sample_adj(self, adj_logits):
        """ sample an adj from the predicted edge probabilities of ep_net """
        # 从预测的边概率中采样一个邻接矩阵
        # 将预测的边概率转换为概率
        # apply sigmoid function to adj_logits
        adj_logits = torch.sigmoid(adj_logits)
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling  # 采样
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature,
                                                                         probs=edge_probs).rsample()
        # making adj_sampled symmetric  # 使得邻接矩阵对称
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    # 边概率来自重构矩阵和原始邻接矩阵，然后使用伯努利分布采样
    # 【一定程度保留原始图结构】
    def sample_adj_add_bernoulli(self, adj_logits, adj_ori, alpha):
        # print("adj_logits", adj_logits)
        # print("adj_ori", adj_ori)
        # adj_logits = F.softmax(adj_logits, dim=1)
        adj_logits = torch.sigmoid(adj_logits)
        adj_logits = adj_logits / torch.max(adj_logits)
        adj_ori = adj_ori / torch.max(adj_ori)
        edge_probs = alpha * adj_logits + (1 - alpha) * adj_ori
        # 再进行一次归一化
        edge_probs = edge_probs / torch.max(edge_probs)
        # print("edge_probs", edge_probs)
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature,
                                                                         probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    # 对生成的邻接矩阵进行归一化处理，以满足不同类型的图神经网络的输入要求
    def normalize_adj(self, adj):
        if self.gnnlayer_type == 'gcn':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
            D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(self.device)
            adj = D_norm @ adj @ D_norm
        elif self.gnnlayer_type == 'gat':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
        elif self.gnnlayer_type == 'gsage':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            adj = F.normalize(adj, p=1, dim=1)
        return adj

    def forward(self, adj_norm, adj_ori, feats_ori, img_feats, csd_ori, csd_img):
        """
        adj_norm是归一化后的图DAD
        adj_ori是原始图A
        M是根据图像特征I在A的基础上编码后的图矩阵"""

        # 图补全网络，获取图补全的adj_logits的邻接矩阵M
        # 因为ep_net是两层GCN，故先需要归一化再送进去
        if self.args.use_ep:
            adj_logits = self.ep_net(adj_norm, img_feats)  # 根据图像特征，获取graph的logits

            # choice_1 根据图补全，在原始图上A，添加或删除部分边
            if self.sample_tpye == 'edge':
                adj_new = self.sample_adj_edge(adj_logits, adj_ori, self.alpha)
            elif self.sample_tpye == 'add_sample':
            # choice_2 根据图补全，以1-alpha概率保留原图，以alpha概率来自补全过的新图
                if self.alpha == 1:
                    adj_new = self.sample_adj(adj_logits)
                else:
                    adj_new = self.sample_adj_add_bernoulli(adj_logits, adj_ori, self.alpha)
        else:
            adj_new = adj_ori
            adj_logits = adj_ori

        # 将补全后的图进行处理，以满足不同的图神经网络的输入要求
        adj_new = self.normalize_adj(adj_new)

        # 节点分类网络
        output = self.nc_net(adj_new, feats_ori, img_feats, csd_ori, csd_img)  # 根据图像特征，获取graph

        # adj_logits即经过补全的图M需要和新图进行对比，计算loss
        return (output, adj_logits)


# 用于图补全
class VGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """

    def __init__(self, dim_feats, dim_h, dim_z, activation, gae=False):
        super(VGAE, self).__init__()
        self.gae = gae
        self.gcn_base = GCNLayer(dim_feats, dim_h, 1, None, 0, bias=False)
        self.gcn_mean = GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False)
        self.gcn_logstd = GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False)

    def forward(self, adj, features):   # 经过两层GCN获取Z
        # GCN encoder 获取隐藏层特征
        hidden = self.gcn_base(adj, features)
        # 再过一层GCN，获取均值和方差
        self.mean = self.gcn_mean(adj, hidden)
        if self.gae:  # GAE，不对均值进行采样
            # GAE (no sampling at bottleneck)
            Z = self.mean
        else:
            # VGAE  # VGAE，对均值进行采样
            self.logstd = self.gcn_logstd(adj, hidden)  # 获取log方差
            gaussian_noise = torch.randn_like(self.mean)  # 生成高斯噪声
            sampled_Z = gaussian_noise * torch.exp(self.logstd) + self.mean  # 采样
            Z = sampled_Z
        # 进行内积解码
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits  # 返回邻接矩阵M


# 用于节点分类
class ZNC(nn.Module):
    def __init__(self, n_in, n_h, n_csd, n_img, n_csd_img, n_layer, activation, dropout, gnnlayer_type='gcn'):
        super(ZNC, self).__init__()
        heads = [1] * (n_layer)  # 用于gat，不用管

        # 根据type选择不同的图神经网络层
        if gnnlayer_type == 'gcn':
            gnnlayer = GCNLayer
        elif gnnlayer_type == 'gsage':
            gnnlayer = SAGELayer
        elif gnnlayer_type == 'gat':
            gnnlayer = GATLayer
            if n_in in (50, 745, 12047):
                heads = [2] * n_layer + [1]
            else:
                heads = [8] * n_layer + [1]
            n_h = int(n_h / 8)
            activation = F.elu
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(gnnlayer(n_in, n_h, heads[0], activation, dropout))  # n_h=128
        # 隐藏层
        for i in range(n_layer - 1):  # 2条邻居
            self.layers.append(gnnlayer(n_h * heads[i], n_h, heads[i + 1], activation, dropout))
        # output layer # 输出层
        self.layers.append(gnnlayer(n_h * heads[-2], n_h, heads[-1], None, dropout))  # 300是CSD的维度

        self.fc_final_pred_csd = nn.Sequential(
            nn.Linear(n_h, 4 * n_h), nn.ReLU(), nn.Linear(4 * n_h, n_csd_img), nn.Dropout(dropout)
        )


    def forward(self, adj_new, feats_ori, img_feats, csd_ori, csd_img):
        # the local item: 1. get k-hop-gcn by one layer; 2. get local loss
        h = feats_ori   # （, 523)
        # local_h = []
        for layer in self.layers:  # k=2条邻居，3层GNN：0, 1, 2
            h = layer(adj_new, h)
            # local_h.append(h)
        preds = self.fc_final_pred_csd(h)  # (, 300)
        preds = preds @ csd_img.T  # (, 300)

        preds_img = (img_feats @ csd_img.T)

        # 当k为2时，local_list长度为3
        return preds, preds_img

class DGPN(nn.Module):
    def __init__(self, n_in, n_h, n_csd, n_img, n_csd_img, dropout, k, beta):
        super(DGPN, self).__init__()
        # 用于local
        self.k = k
        self.beta = beta


        # self.sage = nn.ModuleList()
        # self.sage.append(GraphSageLayer(n_in, n_h, 1, F.relu, dropout))
        # for i in range(k - 1):
        #     self.sage.append(GraphSageLayer(n_h, n_h, 1, F.relu, dropout))
        # self.sage.append(GraphSageLayer(n_h, n_h, 1, None, dropout))
        #
        # self.conv1 = SAGEConv(n_in, n_h)
        # self.conv2 = SAGEConv(n_h, n_img)

        self.conv1 = GraphSageLayer(n_in, n_h, 1, F.relu, dropout)  # 先relu再dropout
        self.conv2 = GraphSageLayer(n_h, n_img, 1, None, None)

        # self.feat_fnn = nn.Sequential(
        #     nn.Linear(n_img, 4 * n_h), nn.ReLU(), nn.Linear(4 * n_h, n_img), nn.Dropout(dropout))



    def forward(self, adj_new, feats_ori, img_feats, csd_ori, csd_img):
        # # the local item: 1. get k-hop-gcn by one layer; 2. get local loss
        # feats_local = get_lrw_pre_calculated_feature_list(feats_ori, adj_new, self.k, self.beta)

        preds_img = img_feats @ csd_img.T

        # feat_tmp = self.feat_fnn(feats_ori)

        # 图卷积
        feat_tmp = feats_ori
        # for layer in self.sage:
        #     feat_tmp = layer(adj_new, feat_tmp)


        # 图卷积
        # 此处adj_new是边表的形式
        # feat_tmp = F.relu(self.conv1(feat_tmp, adj_new))
        # feat_tmp = F.dropout(feat_tmp, p=0.5, training=self.training)
        # feat_tmp = self.conv2(feat_tmp, adj_new).log_softmax(dim=-1)

        feat_tmp = self.conv1(adj_new, feat_tmp)
        feat_tmp = self.conv2(adj_new, feat_tmp).log_softmax(dim=-1)

        feat_total = feat_tmp

        # feat_total = self.feat_fnn(feat_total)
        preds_total = feat_total @ csd_img.T

        return (preds_total, feat_total, preds_img)



# ****************************************************
# 不同的图神经网络层
class GCNLayer(nn.Module):
    """ one layer of GCN """

    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        h = h.float()   # Ensure that h is of type FloatTensor
        x = h @ self.W
        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        return x

class GraphSageLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GraphSageLayer, self).__init__()
        self.fc = nn.Linear(input_dim * 2, output_dim, bias=bias)
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        neighbor = adj @ h
        # print('neighbor', neighbor.size())
        mean_neigh = neighbor / (adj.sum(dim=1, keepdim=True) + 1e-7)
        # mean_neigh = neighbor / (adj.sum(dim=1, keepdim=True).to_dense() + 1e-7)

        x = torch.cat((h, mean_neigh), dim=1)
        x = self.fc(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class SAGELayer(nn.Module):
    """ one layer of GraphSAGE with gcn aggregator """

    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(SAGELayer, self).__init__()
        self.linear_neigh = nn.Linear(input_dim, output_dim, bias=False)
        # self.linear_self = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        # using GCN aggregator
        if self.dropout:
            h = self.dropout(h)
        x = adj @ h
        x = self.linear_neigh(x)
        # x_neigh = self.linear_neigh(x)
        # x_self = self.linear_self(h)
        # x = x_neigh + x_self
        if self.activation:
            x = self.activation(x)
        # x = F.normalize(x, dim=1, p=2)
        return x


class GATLayer(nn.Module):
    """ one layer of GAT """

    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        self.n_heads = n_heads
        self.attn_l = nn.Linear(output_dim, self.n_heads, bias=False)
        self.attn_r = nn.Linear(output_dim, self.n_heads, bias=False)
        self.attn_drop = nn.Dropout(p=0.6)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W  # torch.Size([2708, 128])
        # calculate attentions, both el and er are n_nodes by n_heads
        el = self.attn_l(x)
        er = self.attn_r(x)  # torch.Size([2708, 8])
        if isinstance(adj, torch.sparse.FloatTensor):
            nz_indices = adj._indices()
        else:
            nz_indices = adj.nonzero().T
        attn = el[nz_indices[0]] + er[nz_indices[1]]  # torch.Size([13264, 8])
        attn = F.leaky_relu(attn, negative_slope=0.2).squeeze()
        # reconstruct adj with attentions, exp for softmax next
        attn = torch.exp(attn)  # torch.Size([13264, 8]) NOTE: torch.Size([13264]) when n_heads=1
        if self.n_heads == 1:
            adj_attn = torch.zeros(size=(adj.size(0), adj.size(1)), device=adj.device)
            adj_attn.index_put_((nz_indices[0], nz_indices[1]), attn)
        else:
            adj_attn = torch.zeros(size=(adj.size(0), adj.size(1), self.n_heads), device=adj.device)
            adj_attn.index_put_((nz_indices[0], nz_indices[1]), attn)  # torch.Size([2708, 2708, 8])
            adj_attn.transpose_(1, 2)  # torch.Size([2708, 8, 2708])
        # edge softmax (only softmax with non-zero entries)
        adj_attn = F.normalize(adj_attn, p=1, dim=-1)
        adj_attn = self.attn_drop(adj_attn)
        # message passing
        x = adj_attn @ x  # torch.Size([2708, 8, 128])
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        if self.n_heads > 1:
            x = x.flatten(start_dim=1)
        return x  # torch.Size([2708, 1024])

# 四舍五入
class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()    # 取整

    @staticmethod
    def backward(ctx, g):
        return g


# 向上取整
class CeilNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil() # 向上取整

    @staticmethod
    def backward(ctx, g):
        return g

