# -*- coding: utf-8 -*-
r"""
NCL
################################################
Reference:
    Zihan Lin*, Changxin Tian*, Yupeng Hou*, Wayne Xin Zhao. "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." in WWW 2022.
"""

import torch
import torch.nn.functional as F

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv


class NCL(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NCL, self).__init__(config, dataset)

        # 加载配置参数 load parameters info
        self.latent_dim = config['embedding_size']  # int type: 嵌入维度 the embedding size of the base model
        self.n_layers = config['n_layers']          # int type: 层数 the layer num of the base model
        self.reg_weight = config['reg_weight']      # float32 type: 正则化权重 the weight decay for l2 normalization

        self.ssl_temp = config['ssl_temp']      # 对比损失 Temperature for contrastive loss.
        self.ssl_reg = config['ssl_reg']        # 结构对比损失重量。 The structure-contrastive loss weight.
        self.hyper_layers = config['hyper_layers']      # 控制结构对比损耗的对比范围，例如，当设置为1时，第0层和第2层的一个用户/项的GNN输出将被视为正对。默认为1。

        self.alpha = config['alpha']        # 平衡用户和物品自我监督损失的权重。

        self.proto_reg = config['proto_reg']   # 原型的对比重量。默认为8 e-8。
        self.k = config['num_clusters']     # 原型的数量。默认为1000。

        # 定义层和损失
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)   # user嵌入
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)   # item嵌入
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)       # LightGCN的卷积层
        self.mf_loss = BPRLoss()        # BPR损失
        self.reg_loss = EmbLoss()       # 嵌入损失

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # 权重初始化
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        # 聚类用到的工具对象
        self.user_centroids = None   # 用户嵌入的中心点 (k, dim)
        self.user_2cluster = None    # uid2中心点的列表 （索引是uid）
        self.item_centroids = None   # 项目嵌入的中心店 (k, dim)
        self.item_2cluster = None    # iid2中心点的列表 （索引是iid）

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()  # user嵌入词典矩阵
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()  # item嵌入词典矩阵
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)  # user 中心词, uid2中心词的对应列表
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)  # item 中心词, iid2中心词的对应列表

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        运行k -means算法得到输入张量x的k个簇
        """
        import faiss
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)   # K-means模型
        kmeans.train(x)     # 训练
        cluster_cents = kmeans.centroids       # 获取聚类后的中心点 (k, dim)

        _, I = kmeans.index.search(x, 1)  # 获取离每个uid最近的中心点

        # convert to cuda Tensors for broadcast 转换为cuda张量广播
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)  # 标准化

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)   # squeeze
        return centroids, node2cluster

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    # 前向计算和 LightGCN 完全一致
    def forward(self):
        all_embeddings = self.get_ego_embeddings()  # 固定操作，先获取所有的嵌入
        embeddings_list = [all_embeddings]   # 加入嵌入列表作为第0层嵌入
        # 进行多层gcn的计算
        for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)

        # 算出最终的嵌入
        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, user, item):
        """
            :params:node_embedding  0层的嵌入
            :params:user user列表
            :params:item item列表
        """
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])   # 分离用户嵌入和项目嵌入

        user_embeddings = user_embeddings_all[user]     # [b, dim]  # 输入用户的嵌入
        norm_user_embeddings = F.normalize(user_embeddings)    # 标准化

        user2cluster = self.user_2cluster[user]     # [b,]  根据 输入uid 查找中心词
        user2centroids = self.user_centroids[user2cluster]   # [b, e]  # 获取 输入uid 的中心词嵌入 (已做标准化)
        # 计算user和其中心词的相似度 exp(e_u \cdot c_i | temp)
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)   # (b, )
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)  # (b, )
        # 计算user和所有中心词（这里包括了该user自己的中心词）的相似度 exp(e_u \cdot c_j | temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))   # (b, k)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)   # (b, )

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # 同理，计算item的语义对比损失
        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    # 自监督学习层损失
    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        """
            params:current_embedding 当前层的嵌入     (n_item+n_user, dim)
            params:previous_embedding 0层的嵌入    (n_item+n_user, dim)
            这里就是偶数层 n 层的嵌入和 0 层的嵌入
            params:user 用户id   (b, )
            params:item 项目id   (b, )
        """
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])  # 分离用户嵌入和项目嵌入
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding, [self.n_users, self.n_items])  # 分离用户嵌入和项目嵌入

        # 计算L_S^U  即user的结构化对比损失
        current_user_embeddings = current_user_embeddings[user]     # 获取输入的一批uid的第n层嵌入  (b, dim)
        previous_user_embeddings = previous_user_embeddings_all[user]  # 获取输入的一批uid的第n层嵌入  (b, dim)
        norm_user_emb1 = F.normalize(current_user_embeddings)   # 输入uid n层嵌入归一化  z_u^(k)
        norm_user_emb2 = F.normalize(previous_user_embeddings)  # 输入uid 0嵌入层归一化   z_u^(0)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)   # 相似度计算 z_u^(k) \cdot z_u^(0)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)      # exp(z_u^(k) \cdot z_u^(0) / temp)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)  # 全部user 0层 嵌入归一化  z_j^(0)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))   # 相似度计算 z_u^(k) \cdot z_j^(0)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)  # exp(z_u^(k) \cdot z_j^(0) / temp)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()  # 计算得到 user 结构化嵌入损失

        # 项目的结构化损失，和用户结构化损失计算相同
        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        # 计算结构化损失
        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    # 损失计算
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]            # uid
        pos_item = interaction[self.ITEM_ID]        # 正例item
        neg_item = interaction[self.NEG_ITEM_ID]    # 反例item

        # 进行 LightGCN 前向计算，得到最终的用户嵌入和项目嵌入和每一层的嵌入向量
        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        # 结构性邻居嵌入
        center_embedding = embeddings_list[0]       # 项目和用户本身的嵌入
        context_embedding = embeddings_list[self.hyper_layers * 2]  # 偶数层GNN的输出

        # 计算结构化损失
        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, user, pos_item)
        proto_loss = self.ProtoNCE_loss(center_embedding, user, pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # 计算推荐（BPR）损失
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # 正则化损失（嵌入损失）
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss, ssl_loss, proto_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
