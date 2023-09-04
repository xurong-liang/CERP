import torch
import torch.nn as nn
from torchfm.layer import FactorizationMachine, FeaturesLinear, MultiLayerPerceptron
import numpy as np
import scipy.sparse as sp
from models.com_embedding import CompositionEmbedding
import os


class LR(torch.nn.Module):
    def __init__(self, opt):
        super(LR, self).__init__()
        self.field_dims = opt['field_dims']
        self.linear = FeaturesLinear(self.field_dims)  # linear part

    def forward(self, x):
        """Compute Score"""
        score = self.linear.forward(x)
        return score.squeeze(1)

    def l2_penalty(self, x, lamb):
        return 0

    def R_v_Q_v_loss(self):
        return 0

    def calc_sparsity(self):
        return 0, 0

    def get_thresholds(self):
        return None

    def get_embeddings(self):
        return np.zeros(1)


class FM(torch.nn.Module):
    """Factorization Machines"""

    def __init__(self, opt):
        super(FM, self).__init__()
        self._opt = opt
        self.latent_dim = opt['latent_dim']
        self.field_dims = opt['field_dims']

        self.feature_num = sum(self.field_dims)

        self.embedding = CompositionEmbedding(opt)
        if self.embedding.retrain:
            self.save_masks()

        self.linear = FeaturesLinear(self.field_dims)  # linear part

        self.fm = FactorizationMachine(reduce_sum=True)

        # base emb size: (|Qv| + |Rv|) * d; |Qv| == |Rv| == bucket size
        self.base_param_size = (self.embedding.bucket_size * 2) * self.latent_dim
        if not self.embedding.retrain:
            assert self.base_param_size == opt["max_param"]
        print("BackBone Embedding Parameters: ", self.base_param_size)

    def forward(self, x):
        """
        Forward function for training and pruning.
        """
        linear_score = self.linear.forward(x)

        xv = self.embedding(x)
        fm_score = self.fm.forward(xv)
        linear_score, fm_score = linear_score.to(self._opt["device_id"]), fm_score.to(self._opt["device_id"])

        score = linear_score + fm_score
        return score.squeeze(1)

    def evaluation_forward(self, x):
        """
        Forward function for evaluation
        """
        return self.forward(x)

    def l2_penalty(self, x, lamb):
        xv = self.embedding(x)
        xv_sq = xv.pow(2)
        xv_penalty = xv_sq * lamb
        xv_penalty = xv_penalty.sum()
        return xv_penalty

    def calc_sparsity(self):
        non_zero_values = self.get_total_non_zero_values()
        percentage = 1 - (non_zero_values / self.base_param_size)
        return percentage.item(), non_zero_values.item()

    def get_total_non_zero_values(self):
        return torch.count_nonzero(self.embedding.sparse_Q_v) + \
            torch.count_nonzero(self.embedding.sparse_R_v)

    def get_entity_avg_emb_size(self):
        """
        Returns the avg embedding size for each entity in the dataset
        """
        all_embs = self.embedding.get_all_embeddings_for_gcn()
        return round(torch.count_nonzero(all_embs, dim=1).float().mean().item(), 4)

    def get_average_non_zero_values(self) -> tuple:
        """
        Returns the average non-zero value for each R_v and Q_v embedding

        NOTE: time-consuming process

        :return avg non-zero value for R_v, avg non-zero value for Q_v
        """
        R_v_val = torch.count_nonzero(self.embedding.sparse_R_v, dim=1). \
            float().mean().item()
        Q_v_val = torch.count_nonzero(self.embedding.sparse_Q_v, dim=1). \
            float().mean().item()
        return round(R_v_val, 4), round(Q_v_val, 4)

    def get_thresholds(self):
        """
        :return: R_s, Q_s
        """
        return self.embedding.g(self.embedding.R_s), self.embedding.g(self.embedding.Q_s)

    def get_embeddings(self):
        """
        :return: sparse_R_v, sparse_Q_v
        """
        emb = self.embedding
        sparse_R_v, sparse_Q_v = emb.sparse_R_v.detach().cpu(), emb.sparse_Q_v.detach().cpu()
        # output sparse matrix
        return sp.csr_matrix(sparse_R_v), sp.csr_matrix(sparse_Q_v)

    def save_masks(self):
        # save the PEP retrain masks (if applicable)
        if self.embedding.retrain:
            emb = self.embedding
            Q_mask_path, R_mask_path = os.path.join(self._opt['emb_path'], "pruned_Q_mask.npz"), \
                os.path.join(self._opt['emb_path'], "pruned_R_mask.npz")
            R_mask, Q_mask = sp.csr_matrix(emb.R_mask.detach().cpu()), \
                sp.csr_matrix(emb.Q_mask.detach().cpu())
            sp.save_npz(Q_mask_path, Q_mask)
            sp.save_npz(R_mask_path, R_mask)
            print(f"Pruned mask saved to {self._opt['emb_path']}")


class MLP(FM):
    """
    The model that uses MLP layer from NCF as score function
    """

    def __init__(self, opt):
        super(MLP, self).__init__(opt)
        # disable linear and fm layers in original FM model
        self.linear = self.fm = None

        self.num_layers = opt['num_layers']
        self.dropout = opt['dropout']

        # need to concat (user, item) repr
        mlp_in_dim = 2 * self.latent_dim if opt["Q_v_R_v_concat_method"] == "add" else 4 * self.latent_dim
        mlp_inter_dims = []
        for layer in range(self.num_layers):
            mlp_inter_dims.append(mlp_in_dim * (2 ** (layer + 1)))
        # output squeezed to (batch size, 1)
        self.mlp = MultiLayerPerceptron(input_dim=mlp_in_dim,
                                        embed_dims=mlp_inter_dims,
                                        dropout=self.dropout,
                                        output_layer=True)
        self.init_mlp_weights()

    def init_mlp_weights(self):
        for layer in self.mlp.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        """
        Forward function for training and pruning.

        Concat user and item embeddings
        """
        xv = self.embedding(x)
        concat_embs = torch.cat((xv[:, 0, :], xv[:, 1, :]), dim=1)

        # normalize each concat embedding w.r.t. to their non-zero counts
        concat_embs = concat_embs / torch.count_nonzero(concat_embs, dim=1).unsqueeze(1)
        scores = self.mlp(concat_embs)

        if not self.embedding.retrain:
            splits = xv[:, 0, :].split(6, dim=0)
            user_unique_embs = torch.vstack([_[0] for _ in splits])
            embs_for_batch_pruning_loss = torch.concat((user_unique_embs, xv[:, 1, :]))
            batch_pruning_loss = -torch.tanh(self._opt["K"] * embs_for_batch_pruning_loss).norm(2) ** 2
        else:
            batch_pruning_loss = torch.tensor(0., device=self._opt["device_id"])
        return scores.squeeze(1), batch_pruning_loss

    def evaluation_forward(self, x):
        """
        Identical to forward(), except no batch pruning loss returned
        """
        xv = self.embedding(x)
        concat_embs = torch.cat((xv[:, 0, :], xv[:, 1, :]), dim=1)
        # normalize each concat embedding w.r.t. to their non-zero counts
        concat_embs = concat_embs / torch.count_nonzero(concat_embs, dim=1).unsqueeze(1)
        scores = self.mlp(concat_embs)
        return scores.squeeze(1)


class LightGCN(FM):
    """
    The LightGCN model implemented.
    """

    def __init__(self, opt):
        super(LightGCN, self).__init__(opt)
        # initialize light graph convolution
        self.graph = None
        self.init_sparse_graph()

        # embedding obtained from LGC network
        self.LGC_embedding = None
        # embedding obtained from Q_v, R_v
        self.input_embedding = None

        # number of layers for network propagation
        self.n_layers = opt["num_layers"]

        # the layer for evaluation rating calculation
        self.f = nn.Sigmoid()

    def init_sparse_graph(self):
        """
        Generate pre-adjacency matrix

        Need s_pre_adj_mat.npz to be stored on data_path
        """
        pre_adj_path = os.path.join(self._opt['data_path'], "s_pre_adj_mat.npz")
        if not os.path.exists(pre_adj_path):
            print(f"Pre-adjacency matrix s_pre_adj_mat.npz does not exist "
                  f"in {self._opt['data_path']}. Run LightGCN for this dataset first.")
            exit(5)
        norm_adj = sp.load_npz(pre_adj_path)
        self.graph = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self._opt['device_id'])

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def compute_LGC(self):
        """
        Performs LGC propagation and assigns self.LGC_embedding and self.input_embedding
        """
        all_embs = self.embedding.get_all_embeddings_for_gcn()
        self.input_embedding = all_embs

        embs = [all_embs]
        for layer in range(self.n_layers):
            # perform matrix factorization between graph and all_embedding
            all_embs = torch.sparse.mm(self.graph, all_embs)
            embs.append(all_embs)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        self.LGC_embedding = light_out

    def l2_penalty(self, x, lamb):
        """
        The alternative regularization loss to l2_penalty()

        :param x: [uid, iid]
        :param lamb: the l2 penalty factor
        """
        x_new = x + x.new_tensor(self.embedding.idx_offsets).unsqueeze(0)
        user_embs = self.input_embedding[x_new[:, 0]]
        item_embs = self.input_embedding[x_new[:, 1]]
        return lamb * (user_embs.norm(2).pow(2) + item_embs.norm(2).pow(2))

    def forward(self, x):
        # get latest LGC computed embedding and original embedding assigned
        self.compute_LGC()

        x_new = x + x.new_tensor(self.embedding.idx_offsets).unsqueeze(0)
        user_embs = self.LGC_embedding[x_new[:, 0]]
        item_embs = self.LGC_embedding[x_new[:, 1]]
        assert user_embs.shape == item_embs.shape
        scores = (user_embs * item_embs).sum(dim=1)
        assert len(scores) == len(x)

        if not self.embedding.retrain:
            user_embs = self.input_embedding[torch.unique(x_new[:, 0])]
            item_embs = self.input_embedding[x_new[:, 1]]
            embs_for_batch_pruning_loss = torch.concat((user_embs, item_embs))
            batch_pruning_loss = -torch.tanh(self._opt["K"] * embs_for_batch_pruning_loss).norm(2) ** 2
        else:
            batch_pruning_loss = torch.tensor(0., device=self._opt["device_id"])
        return scores, batch_pruning_loss

    def evaluation_forward(self, x):
        # get latest LGC computed embedding and original embedding assigned
        self.compute_LGC()

        x_new = x + x.new_tensor(self.embedding.idx_offsets).unsqueeze(0)
        user_idxes, item_idxes = x_new[:, 0], x_new[:, 1]
        # check all users in current batch is the same person
        assert (user_idxes[0] == user_idxes).all()
        user_idx = user_idxes[0]
        user_embs = self.LGC_embedding[user_idx].unsqueeze(0)
        item_embs = self.LGC_embedding[item_idxes]

        # (1, dim size) x (dim size, item count) -->  (1, item count)
        rating = self.f(user_embs @ item_embs.T)
        return rating.squeeze()
