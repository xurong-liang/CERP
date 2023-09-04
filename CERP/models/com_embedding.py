import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.stats import powerlaw


class CompositionEmbedding(nn.Module):
    def __init__(self, opt):
        super(CompositionEmbedding, self).__init__()
        self.threshold_type = opt['threshold_type']
        self.latent_dim = opt['latent_dim']
        self._opt = opt

        self.field_dims = opt['field_dims']
        self.feature_num = sum(opt['field_dims'])

        # size of buckets in Qv, Rv
        self.bucket_size = opt["bucket_size"]

        # Q's avg entities per row = \ceiling(#entities / bucket size)
        self.Q_v_entity_per_row = int(np.ceil(self.feature_num / self.bucket_size))

        self.g_type = opt['g_type']
        self.gk = opt['gk']
        init = opt['threshold_init']

        self.g = torch.sigmoid

        # init soft threshold
        self.Q_s = self.init_threshold(init, row_size=self.bucket_size, col_size=self.latent_dim)
        self.R_s = self.init_threshold(init, row_size=self.bucket_size, col_size=self.latent_dim)

        # idx_offset = [idx of first user, idx of first item]; use this since v combines
        # user and item representations altogether
        self.idx_offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.long)

        # all entities index: used by LightGCN to retrieve all entity embeddings
        self.all_entities_idxes = np.arange(self.feature_num)

        if self._opt["GCN_pretrain_emb"]:
            print("Initialize R_v, Q_v by taking LightGCN pretrained embs")
            R_v_path = os.path.join(self._opt["data_path"], "R_v.pt")
            Q_v_path = os.path.join(self._opt["data_path"], "Q_v.pt")
            if not os.path.exists(R_v_path) or not os.path.exists(Q_v_path):
                print(f"files R_v.pt, Q_v.pt not found in {self._opt['GCN_pretrain_emb']}")
                exit(6)

            self.R_v = torch.nn.Parameter(torch.load(R_v_path, map_location="cpu"))
            self.Q_v = torch.nn.Parameter(torch.load(Q_v_path, map_location="cpu"))
            assert self.R_v.data.shape == self.Q_v.data.shape, "Rv, Qv has different shape"
        else:
            # Rv dim = Qv dim = (b, d)
            self.R_v = torch.nn.Parameter(torch.rand(self.bucket_size, self.latent_dim))
            torch.nn.init.xavier_uniform_(self.R_v)
            self.Q_v = torch.nn.Parameter(torch.rand(self.bucket_size, self.latent_dim))
            torch.nn.init.xavier_uniform_(self.Q_v)

        self.Q_mask = self.R_mask = None

        if 'retrain_emb_sparsity' in opt:
            self.retrain = True
            self.init_retrain(opt)
        else:
            self.retrain = False

        self.sparse_R_v, self.sparse_Q_v = self.R_v.data, self.Q_v.data

    def init_threshold(self, init, row_size, col_size):
        requires_scaling = True

        if self.threshold_type == 'global':
            mat = torch.ones(1)
            if self._opt["threshold_init_method"] == "uniform":
                mat = mat * torch.rand(1)
                requires_scaling = False
            elif self._opt["threshold_init_method"] == "normal":
                # N(0, 1)
                mat = mat * torch.normal(mean=0., std=1., size=(1,))
            elif self._opt["threshold_init_method"] == "power_law":
                a = .5
                long_tail_val = torch.tensor(powerlaw.ppf(np.random.random((1,)), a=a))
                mat = mat * long_tail_val
            elif self._opt["threshold_init_method"] == "xavier_uniform":
                raise NotImplementedError
            else:
                requires_scaling = False

            if requires_scaling:
                # apply sigmoid to scale this
                mat = torch.sigmoid(mat)

            mat = mat * init
            s = nn.Parameter(mat)
        elif self.threshold_type == 'element-wise':
            mat = torch.ones([row_size, col_size])
            if self._opt["threshold_init_method"] == "uniform":
                mat = mat * torch.nn.init.uniform_(torch.zeros((row_size, col_size)))
            elif self._opt["threshold_init_method"] == "normal":
                # N(0, 1)
                mat = mat * torch.normal(mean=0., std=1., size=mat.shape)
            elif self._opt["threshold_init_method"] == "power_law":
                a = .5
                long_tail_val = torch.tensor(powerlaw.ppf(np.random.random(size=mat.shape), a=a)).float()
                mat = mat * long_tail_val
            elif self._opt["threshold_init_method"] == "xavier_uniform":
                mat = mat * nn.init.xavier_uniform_(torch.zeros(size=mat.shape))
            else:
                requires_scaling = False

            if requires_scaling:
                # apply min-max scaling to squeeze
                mat_min, _ = mat.min(dim=1, keepdim=True)
                mat_max, _ = mat.max(dim=1, keepdim=True)
                mat = (mat - mat_min) / (mat_max - mat_min)

            assert (0 <= mat).all() and (1 >= mat).all()
            mat = init * mat
            s = nn.Parameter(mat)
        else:
            raise ValueError('Invalid threshold_type: {}'.format(self.threshold_type))
        return s

    def init_retrain(self, opt):
        for file_name in ["R_v", "Q_v"]:
            # load pruned embeddings
            path = os.path.join(opt['emb_path'], f"{file_name}.npz")
            locals()[file_name] = torch.from_numpy(sp.load_npz(path).toarray())
            mask = torch.abs(torch.sign(locals()[file_name]))

            # load initial embeddings
            path = os.path.join(os.path.split(opt['emb_path'])[0], f'initial_emb', f"{file_name}.npz")
            init_emb = torch.from_numpy(sp.load_npz(path).toarray()) * mask
            # set self.Q_v/self.R_v as initial emb, with mask applied
            setattr(self, file_name, torch.nn.Parameter(init_emb))
            # set self.Q_mask/self.R_mask
            setattr(self, f"{file_name.split('_')[0]}_mask", mask.to(opt["device_id"]))
            # set gk = 0 stops the pruning process
            self.gk = 0

    def apply_pruning(self):
        self.sparse_Q_v = torch.sign(self.Q_v) * torch.relu(torch.abs(self.Q_v) - (self.g(self.Q_s) * self.gk))
        self.sparse_R_v = torch.sign(self.R_v) * torch.relu(torch.abs(self.R_v) - (self.g(self.R_s) * self.gk))

        if self.retrain:
            self.sparse_Q_v = self.sparse_Q_v * self.Q_mask
            self.sparse_R_v = self.sparse_R_v * self.R_mask

    def get_Q_idx(self, x_offset):
        """
        Obtain the user/item indexes in the Q matrix

        :param x_offset: matrix with each row being (uid, iid)
        :return: matrix of the corresponding Q_idx
        """
        return torch.div(x_offset, self.Q_v_entity_per_row, rounding_mode="trunc")

    def get_R_idx(self, x_offset):
        """
        Obtain the user/item indexes in the R matrix

        :param x_offset: matrix with each row being (uid, iid)
        :return: matrix of the corresponding R_idx
        """
        return x_offset % self.bucket_size

    def forward(self, x):
        """
        Used by FM and MLP models
        """
        x_new = x + x.new_tensor(self.idx_offsets).unsqueeze(0)
        self.apply_pruning()

        batch_Q_idxes, batch_R_idxes = self.get_Q_idx(x_new), self.get_R_idx(x_new)

        # get user, items' corresponding embedding vectors in Q, R matrices
        batch_Q_v = F.embedding(batch_Q_idxes, self.sparse_Q_v)
        batch_R_v = F.embedding(batch_R_idxes, self.sparse_R_v)
        assert len(batch_R_v) == len(batch_Q_v)

        if self._opt["Q_v_R_v_concat_method"] == 'concat':
            return torch.cat((batch_R_v, batch_Q_v), dim=2)
        else:
            return batch_R_v + batch_Q_v

    def get_all_embeddings_for_gcn(self):
        """
        Alternative forward() function for LightGCN, returns all users and all items embeddings i.e. V

        :return: the whole embedding V \in (|U| + |I|) x latent dim
        """
        all_idxes = torch.tensor(self.all_entities_idxes).to(self._opt['device_id'])
        self.apply_pruning()

        Q_idxes, R_idxes = self.get_Q_idx(all_idxes), self.get_R_idx(all_idxes)

        Q_v = F.embedding(Q_idxes, self.sparse_Q_v)
        R_v = F.embedding(R_idxes, self.sparse_R_v)
        assert len(R_v) == len(Q_v)

        if self._opt["Q_v_R_v_concat_method"] == 'concat':
            # (batch size, 2, latent_dim): concat user/item's emb vectors from R and Q matrices
            return torch.cat((R_v, Q_v), dim=1)
        else:
            return R_v + Q_v
