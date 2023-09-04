"""
Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import os
import scipy.sparse as sp


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.__init_weight()

        self.retrain_sparsity = None
        if config["CERP_embedding_bucket_size"]:
            self.use_CERP_embs = True
            if config["path_to_load_CERP_pruned_embs_for_retraining"]:
                self.retrain_sparsity = config["retrain_sparsity"]

            self.embedding_user = self.embedding_item = None
            self.bucket_size = config["CERP_embedding_bucket_size"]
            self.idx_offsets = np.array((0, *np.cumsum([self.num_users, self.num_items])[:-1]), dtype=np.long)
            self.total_entity_num = self.num_users + self.num_items
            self.all_entities_idxes = np.arange(self.total_entity_num)

            self.R_len = self.Q_len = self.bucket_size
            self.Q_v_entity_per_row = int(np.ceil(self.total_entity_num / self.bucket_size))

            self.R_v = torch.nn.Embedding(
                num_embeddings=self.R_len, embedding_dim=self.latent_dim
            )
            self.Q_v = torch.nn.Embedding(
                num_embeddings=self.Q_len, embedding_dim=self.latent_dim
            )
            if not self.retrain_sparsity:
                print(f"create compositional embedding with bucket size {config['CERP_embedding_bucket_size']}")
                nn.init.normal_(self.Q_v.weight, std=0.1)
                nn.init.normal_(self.R_v.weight, std=0.1)
                self.Q_mask = self.R_mask = None
            else:
                print(f"loading CERP pruned embedding with sparsity {self.retrain_sparsity:.4f} for retraining.")
                # load pruned Qv, Rv
                for file_name in ["R_v", "Q_v"]:
                    # load pruned embeddings
                    path = os.path.join(world.RES_PATH, f"{file_name}.npz")
                    locals()[file_name] = torch.from_numpy(sp.load_npz(path).toarray())
                    mask = torch.abs(torch.sign(locals()[file_name]))

                    # load initial embeddings
                    path = os.path.join(os.path.split(world.RES_PATH)[0], f'initial_emb', f"{file_name}.npz")
                    init_emb = torch.from_numpy(sp.load_npz(path).toarray()) * mask
                    # set self.Q_v/self.R_v as initial emb, with mask applied
                    emb = getattr(self, file_name)
                    emb.weight.data.copy_(init_emb)

                    # set self.Q_mask/self.R_mask
                    setattr(self, f"{file_name.split('_')[0]}_mask", mask.to(world.device))
        else:
            self.use_CERP_embs = False

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretrained data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def get_Q_idx(self, original_idxes):
        """
        Obtain the user/item indexes in the quotient matrix

        :return: matrix of the corresponding Q_idx
        """
        return torch.div(original_idxes, self.Q_v_entity_per_row, rounding_mode="trunc")

    def get_R_idx(self, original_idxes):
        """
        Obtain the user/item indexes in the quotient matrix

        :return: matrix of the corresponding R_idx
        """
        return original_idxes % self.bucket_size

    def computer(self):
        """
        propagate methods for lightGCN
        """
        if not self.use_CERP_embs:
            users_emb = self.embedding_user.weight
            items_emb = self.embedding_item.weight
            all_emb = torch.cat([users_emb, items_emb])
        else:
            if self.retrain_sparsity:
                # apply mask to keep 0's elements in Qv and Rv before next computation
                self.Q_v.weight.data.copy_(self.Q_v.weight.data * self.Q_mask)
                self.R_v.weight.data.copy_(self.R_v.weight.data * self.R_mask)

            all_idxes = torch.tensor(self.all_entities_idxes).long()
            if torch.cuda.is_available():
                all_idxes = all_idxes.to(self.R_v.weight.get_device())
            Q_idxes, R_idxes = self.get_Q_idx(all_idxes), self.get_R_idx(all_idxes)
            Q_v = self.Q_v.weight[Q_idxes, :]
            R_v = self.R_v.weight[R_idxes, :]
            all_emb = Q_v + R_v

        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # perform matrix multiplication between graph and all_embedding
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        """
        Used in evaluation

        Get ratings of all items by users supplied in current batch
        """

        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        # final embeddings from Light Graph Convolution
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        # the input layer of user/item embs (here is randomized)
        if not self.use_CERP_embs:
            users_emb_ego = self.embedding_user(users)
            pos_emb_ego = self.embedding_item(pos_items)
            neg_emb_ego = self.embedding_item(neg_items)
        else:
            u_Q_idxes, u_R_idxes = self.get_Q_idx(users), self.get_R_idx(users)
            u_Q_v = self.Q_v.weight[u_Q_idxes, :]
            u_R_v = self.R_v.weight[u_R_idxes, :]
            users_emb_ego = u_Q_v + u_R_v

            new_pos_items_idx = pos_items + self.idx_offsets[1]
            p_Q_idxes, p_R_idxes = self.get_Q_idx(new_pos_items_idx), self.get_R_idx(new_pos_items_idx)
            p_Q_v = self.Q_v.weight[p_Q_idxes, :]
            p_R_v = self.R_v.weight[p_R_idxes, :]
            pos_emb_ego = p_Q_v + p_R_v

            new_neg_items_idx = neg_items + self.idx_offsets[1]
            n_Q_idxes, n_R_idxes = self.get_Q_idx(new_neg_items_idx), self.get_R_idx(new_neg_items_idx)
            n_Q_v = self.Q_v.weight[n_Q_idxes, :]
            n_R_v = self.R_v.weight[n_R_idxes, :]
            neg_emb_ego = n_Q_v + n_R_v
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        """
        Used in training.

        Do calculation of bpr loss
        """
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = self.config["l2_penalty"] * \
                   (userEmb0.norm(2).pow(2) +
                    posEmb0.norm(2).pow(2) +
                    negEmb0.norm(2).pow(2)) / float(len(users))
        # perform dot product
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # loss len: number of (uid, pos, neg) in current batch
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()

        # embs here are from Light Graph Convolution
        users_emb = all_users[users]
        items_emb = all_items[items]

        # dot product
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
