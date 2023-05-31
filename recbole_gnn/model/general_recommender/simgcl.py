# -*- coding: utf-8 -*-
r"""
SimGCL
################################################
Reference:
    Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, Quoc Viet Hung Nguyen. "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation." in SIGIR 2022.
"""


import torch
import torch.nn.functional as F

from recbole_gnn.model.general_recommender import LightGCN


class SimGCL(LightGCN):
    def __init__(self, config, dataset):
        super(SimGCL, self).__init__(config, dataset)
        # 加载配置参数
        self.cl_rate = config['lambda']     # 对比损失的权重
        self.eps = config['eps']        # 噪声半径
        self.temperature = config['temperature']        # softmax 温度参数

    def forward(self, perturbed=False):
        all_embs = self.get_ego_embeddings()    # 获取ego嵌入向量
        embeddings_list = []    # 嵌入向量列表

        for layer_idx in range(self.n_layers):      # 进行 n 层计算
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)  # 直接调用的GCN的卷积计算
            if perturbed:   # 对比学习（给嵌入增加噪声）
                random_noise = torch.rand_like(all_embs, device=all_embs.device)        # 生成噪声（这种方式生成的是区间[0,1)上由均匀分布的随机数填充的张量）
                all_embs = all_embs + torch.sign(all_embs) * F.normalize(random_noise, dim=-1) * self.eps  # 将随机噪声加到原嵌入上
            embeddings_list.append(all_embs)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()

    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)   # 直接调用LightGCN的损失计算，即计算BPR损失和正则化损失

        user = torch.unique(interaction[self.USER_ID])      # user
        pos_item = torch.unique(interaction[self.ITEM_ID])  # 正例项目

        perturbed_user_embs_1, perturbed_item_embs_1 = self.forward(perturbed=True)     #
        perturbed_user_embs_2, perturbed_item_embs_2 = self.forward(perturbed=True)

        user_cl_loss = self.calculate_cl_loss(perturbed_user_embs_1[user], perturbed_user_embs_2[user])
        item_cl_loss = self.calculate_cl_loss(perturbed_item_embs_1[pos_item], perturbed_item_embs_2[pos_item])

        return loss + self.cl_rate * (user_cl_loss + item_cl_loss)
