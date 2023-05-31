# @Time   : 2022/3/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
HMLET
################################################
Reference:
    Taeyong Kong et al. "Linear, or Non-Linear, That is the Question!." in WSDM 2022.

Reference code:
    https://github.com/qbxlvnf11/HMLET
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.layers import activation_layer
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv

# 门控单元
class Gating_Net(nn.Module):   # 门控单元
    def __init__(self, embedding_dim, mlp_dims, dropout_p):
        super(Gating_Net, self).__init__()
        self.embedding_dim = embedding_dim   # 嵌入维度

        # 定义层
        fc_layers = []
        for i in range(len(mlp_dims)):
            if i == 0:
                fc = nn.Linear(embedding_dim*2, mlp_dims[i])
                fc_layers.append(fc)
            else:
                fc = nn.Linear(mlp_dims[i-1], mlp_dims[i])
                fc_layers.append(fc)
            if i != len(mlp_dims) - 1:   # 除了最后一层，全部添加`批标准化`,`Dropout`,`ReLU`
                fc_layers.append(nn.BatchNorm1d(mlp_dims[i]))
                fc_layers.append(nn.Dropout(p=dropout_p))
                fc_layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*fc_layers)

    def gumbel_softmax(self, logits, temperature, hard):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        从Gumbel-Softmax分布中取样，可选离散化。
        Args:
          logits: [batch_size, n_class] 非规范log-probs unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y 如果为真，则取argmax，但要对软样本y进行区分。
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
          [batch_size, n_class] Gumbel-Softmax分布的样本。
          如果hard=True，那么返回的样本将是 one-hot 的，否则它将是一个概率分布。是一个概率分布，在不同的类中总和为1。
        """
        y = self.gumbel_softmax_sample(logits, temperature) ## (0.6, 0.2, 0.1.txt,..., 0.11)
        if hard:
            k = logits.size(1)    # k is numb of classes  类的数量？好像没用到
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)  ## (1, 0, 0, ..., 0)
            y_hard = torch.eq(y, torch.max(y, dim=1, keepdim=True)[0]).type_as(y)
            y = (y_hard - y).detach() + y
        return y

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution 从Gumbel-Softmax分布中抽出一个样本"""
        noise = self.sample_gumbel(logits)
        y = (logits + noise) / temperature
        return F.softmax(y, dim=1)

    def sample_gumbel(self, logits):
        """Sample from Gumbel(0, 1) 从Gumbel(0, 1)采样"""
        noise = torch.rand(logits.size())  # 随机采样，shape=[n_item + n_user, 2]
        # 这个地方应该是用反函数法生成的符合gumbel分布的随机数
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return torch.Tensor(noise.float()).to(logits.device)

    def forward(self, feature, temperature, hard):
        x = self.mlp(feature)   # 先对concat的线性和非线性嵌入做MLP计算
        out = self.gumbel_softmax(x, temperature, hard)
        out_value = out.unsqueeze(2)
        gating_out = out_value.repeat(1, 1, self.embedding_dim)
        return gating_out


class HMLET(GeneralGraphRecommender):
    r"""HMLET combines both linear and non-linear propagation layers for general recommendation and yields better performance.
    HMLET结合了线性和非线性传播层，从而获得了更好的性能。
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HMLET, self).__init__(config, dataset)

        # 加载配置参数
        self.latent_dim = config['embedding_size']  # int type:嵌入维度 the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:层数 the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: l2正则化权重 the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization
        self.gate_layer_ids = config['gate_layer_ids']  # list type: 非线性门控的层id layer ids for non-linear gating
        self.gating_mlp_dims = config['gating_mlp_dims']  # list type: 门控模块MLP尺寸列表 list of mlp dimensions in gating module
        self.dropout_ratio = config['dropout_ratio']  # 门控模块MLP的Dropout比 dropout ratio for mlp in gating module
        self.gum_temp = config['ori_temp']   # softmax 温度参数
        self.logger.info(f'Model initialization, gumbel softmax temperature: {self.gum_temp}')

        # 定义层和损失
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)  # 用户嵌入
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)  # 项目嵌入
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)       # LightGCN的图卷机层
        self.activation = nn.ELU() if config['activation_function'] == 'elu' else activation_layer(config['activation_function'])   # 激活函数
        self.gating_nets = nn.ModuleList([      # 门控单元
            Gating_Net(self.latent_dim, self.gating_mlp_dims, self.dropout_ratio) for _ in range(len(self.gate_layer_ids))
        ])

        self.mf_loss = BPRLoss()  # BPR损失
        self.reg_loss = EmbLoss()   # 嵌入损失

        # 存储变量用于全排序求值加速
        self.restore_user_e = None
        self.restore_item_e = None

        # 参数初始化
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e', 'gum_temp']

        # 固定住门控单元的参数
        for gating in self.gating_nets:
            self._gating_freeze(gating, False)

    def _gating_freeze(self, model, freeze_flag):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = freeze_flag

    def __choosing_one(self, features, gumbel_out):
        feature = torch.sum(torch.mul(features, gumbel_out), dim=1)  # batch x embedding_dim (or batch x embedding_dim x layer_num)
        return feature

    def __where(self, idx, lst):
        for i in range(len(lst)):
            if lst[i] == idx:
                return i
        raise ValueError(f'{idx} not in {lst}.')

    def get_ego_embeddings(self):
        r"""获取用户和项目的嵌入，并组合到一个嵌入矩阵。
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()   # 获取所有嵌入
        embeddings_list = [all_embeddings]      # 初始嵌入
        non_lin_emb_list = [all_embeddings]     # 初始非线性嵌入

        for layer_idx in range(self.n_layers):          # 进行 n 层计算
            linear_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)    # 进行LightGCN计算，作为线性嵌入
            if layer_idx not in self.gate_layer_ids:   # 如果不使用门控单元，直接将线性嵌入作为该层嵌入
                all_embeddings = linear_embeddings
            else:           # 使用门控单元
                non_lin_id = self.__where(layer_idx, self.gate_layer_ids)
                last_non_lin_emb = non_lin_emb_list[non_lin_id]   # 取出对应的非线性嵌入
                non_lin_embeddings = self.activation(self.gcn_conv(last_non_lin_emb, self.edge_index, self.edge_weight))  # 进行LightGCN计算，然后套一层激活函数
                stack_embeddings = torch.stack([linear_embeddings, non_lin_embeddings], dim=1)  # stack 线性嵌入和非线性嵌入
                concat_embeddings = torch.cat((linear_embeddings, non_lin_embeddings), dim=-1)  # concat 线性嵌入和非线性嵌入
                gumbel_out = self.gating_nets[non_lin_id](concat_embeddings, self.gum_temp, not self.training)   # 计算gumbel softmax的结果, shape=[n_user + n_item, 2, dim], 其中dim维度是复制的，值是相同的
                all_embeddings = self.__choosing_one(stack_embeddings, gumbel_out)      # 对线性和非线性嵌入加权求和
                non_lin_emb_list.append(all_embeddings)     # 加入作为非线性嵌入
            embeddings_list.append(all_embeddings)    # 加入嵌入列表
        hmlet_all_embeddings = torch.stack(embeddings_list, dim=1)
        hmlet_all_embeddings = torch.mean(hmlet_all_embeddings, dim=1)          # 得到最终的嵌入（多层求均值）

        user_all_embeddings, item_all_embeddings = torch.split(hmlet_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    # 损失函数和LightGCN一致
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    # 和LightGCN一致
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    # 和LightGCN一致
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)