# @Time   : 2022/10/1
# @Author : Yuhao Xu
# @Email  : xyh0811@gmail.com

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss, RegLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv


class SimDCL(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimDCL, self).__init__(config, dataset)

        # 加载配置参数
        self.latent_dim = config["embedding_size"]
        self.n_layers = int(config["n_layers"])
        self.aug_type = config["type"]
        self.dropout_rate = config["dropout"]
        self.noise_dropout_rate = config["noise_dropout"]
        self.ssl_tau = config["ssl_tau"]
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]

        self._user = dataset.inter_feat[dataset.uid_field]  # user列
        self._item = dataset.inter_feat[dataset.iid_field]  # item列

        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.emb_loss = EmbLoss()
        # self.mse_loss = torch.nn.MSELoss(reduction="sum")
        # self.drop_rate = config.dropout
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.noise_dropout = torch.nn.Dropout(p=self.noise_dropout_rate)
        self.mse_loss = torch.nn.MSELoss()
        self.u_LayerNorm = torch.nn.LayerNorm(normalized_shape=(self.n_users, self.latent_dim), eps=1e-12,
                                              dtype=torch.float)
        self.i_LayerNorm = torch.nn.LayerNorm(normalized_shape=(self.n_items, self.latent_dim), eps=1e-12,
                                              dtype=torch.float)
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def forward(self, dropout=False):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embeddings = self.dropout(all_embeddings)
        embeddings_list = [all_embeddings]

        all_embeddings = torch.cat(
            [self.u_LayerNorm(self.user_embedding.weight), self.i_LayerNorm(self.item_embedding.weight)])

        for layer_idx in range(self.n_layers):
            if layer_idx > 0:
                all_embeddings = self.gcn_conv((all_embeddings + embeddings_list[0]) * 0.5, self.edge_index,
                                               self.edge_weight)
            else:
                all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            if dropout:
                all_embeddings = self.noise_dropout(all_embeddings)
            embeddings_list.append(all_embeddings)

        embeddings = torch.stack(embeddings_list, dim=1)  # stack
        embeddings = torch.mean(embeddings, dim=1, keepdim=False)

        user_all_embeddings, item_all_embeddings = torch.split(embeddings, [self.n_users, self.n_items], dim=0)

        return user_all_embeddings, item_all_embeddings, embeddings_list
    

    def calc_bpr_loss(self, user_emd, item_emd, user_list, pos_item_list, neg_item_list):
        r"""Calculate the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.
        Args:
            user_emd (torch.Tensor): Ego embedding of all users after forwarding.
            item_emd (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = user_emd[user_list]
        pi_e = item_emd[pos_item_list]
        ni_e = item_emd[neg_item_list]
        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)
        l1 = torch.sum(-F.logsigmoid(p_scores - n_scores))
        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        l2 = self.emb_loss(u_e_p, pi_e_p, ni_e_p)

        return l1 + l2 * self.reg_weight, p_scores

    def calc_ssl_loss(self, user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2):
        r"""

        Args:
            user_list (torch.Tensor): List of the user.   [batch_size, ]
            pos_item_list (torch.Tensor): List of positive examples.        正例 iid列表（交互的iid）
        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """

        u_emd1 = F.normalize(user_sub1[user_list], dim=1)  # L2正则  [b, dim]
        u_emd2 = F.normalize(user_sub2[user_list], dim=1)  # L2正则  [b, dim]
        all_user2 = F.normalize(user_sub2, dim=1)  #
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[pos_item_list], dim=1)
        i_emd2 = F.normalize(item_sub2[pos_item_list], dim=1)
        all_item2 = F.normalize(item_sub2, dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_item + ssl_user) * self.ssl_weight

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:  # 计算损失前先清空保存的user和item嵌入向量
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]

        user_emd, item_emd, embs_list = self.forward(True)  # 原始图的前向计算（和LightGCN是一样的）
        user_cl, item_cl, embs_cl_list = self.forward(True)
        b_loss_1, emb_s = self.calc_bpr_loss(
            user_emd, item_emd, user_list, pos_item_list, neg_item_list
        )
        b_loss_2, sub_s = self.calc_bpr_loss(
            user_cl, item_cl, user_list, pos_item_list, neg_item_list
        )
        bpr_loss = (b_loss_1 + b_loss_2) * 0.5

        ssl_loss = self.calc_ssl_loss(user_list, pos_item_list, user_emd, user_cl, item_emd, item_cl)
        total_loss = bpr_loss + ssl_loss

        return total_loss

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()

        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()

        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)