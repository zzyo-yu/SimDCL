# @Time   : 2022/3/8
# @Author : Lanling Xu
# @Email  : xulanling_sherry@163.com

r"""
LightGCN
################################################
Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import torch

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv


class LightGCN(GeneralGraphRecommender):
    r"""
    LightGCN是一个基于GCN的推荐模型，通过PyG实现。
    LightGCN只包括GCN中最重要的部分--邻域聚合--用于协同过滤。
    具体来说，LightGCN通过在用户-项目上线性传播来学习用户和项目的嵌入。
    在用户-项目交互图上传播，并使用所有层学到的嵌入的加权和作为最终嵌入。
    我们按照原作者的思路，用成对的训练模式来实现这个模型。
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)  # 这里从父类里生成了 边索引（end_index） 和 边权重（用度计算得到的）

        # 加载配置信息
        self.latent_dim = config['embedding_size']  # int type: 嵌入维度
        self.n_layers = config['n_layers']  # int type: 图卷积层数（就是做几层聚合）
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization

        # 定义层和损失
        # 嵌入层
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        # LightGCN卷积层
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        # 两个损失函数
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        # 存储变量用于全排序计算的加速
        self.restore_user_e = None
        self.restore_item_e = None

        # 参数初始化
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        # 获取所有的嵌入向量
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        # 进行多次GCN计算
        for layer_idx in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)
        # 对多层表征求均值
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def weight_forward(self):
        # 获取所有的嵌入向量
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        # 进行多次GCN计算
        for layer_idx in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)

        # 对多层表征求均值
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings, embeddings_list


    def calculate_loss(self, interaction):
        # 训练时清空存储变量
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        # 获取用户id，正例项目和反例项目
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # 前向计算
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # 计算BPR损失
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # 计算正则化损失
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)

        # 总损失
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
