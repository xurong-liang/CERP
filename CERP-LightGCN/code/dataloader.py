"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
import pickle
from os.path import join
import sys

import scipy.sparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Include graph information
    gowalla dataset
    """

    def __init__(self, config=world.config, path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    # if no interacted items, don't change self.m_item
                    if len(items) != 0:
                        self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset_name} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        # NOTE: the binarized rating matrix for all *train* users and train items of shape (user#, item#)
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()

        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        # all positive training items for each user
        self._allPos = self.getUserPosItems(list(range(self.n_user)))

        # save all train positive items
        all_pos_path = os.path.join(world.RES_PATH, "train_all_pos.pickle")
        save_all_positive_items(all_positive_lists=self._allPos, save_path=all_pos_path)

        # all positive testing items for each user
        self.__testDict = self.__build_test()

        # save interaction matrix
        interaction_mat_save_path = os.path.join(world.RES_PATH, "interact_mat.npz")
        self.save_rating_matrix(save_path=interaction_mat_save_path)

        # save test_pos_item.pickle
        test_pos_item_path = os.path.join(world.RES_PATH, "test_pos_item.pickle")
        self.save_test_pos_item_dict(save_path=test_pos_item_path)

        print(f"{world.dataset_name} is ready to go")

    def save_rating_matrix(self, save_path):
        """
        Save rating matrix (bianrized rating matrix) to world.RES_PATH

        * contains pos items from both train and test dataset
        """
        if os.path.exists(save_path):
            print("interaction matrix already exists")
            return

        # rating matrix contains only positively rated test items
        matrix_test_pos_only = csr_matrix((np.ones(len(self.testUser)), (self.testUser, self.testItem)),
                                          shape=(self.n_user, self.m_item))
        # full rating matrix by performing bitwise or operation
        # on train and test positively rated items
        rating_matrix = self.UserItemNet + matrix_test_pos_only
        assert rating_matrix.min() == 0 and rating_matrix.max() == 1
        scipy.sparse.save_npz(save_path, rating_matrix)
        print(f"{world.dataset_name}'s interaction matrix has been saved to {save_path}")

    def save_test_pos_item_dict(self, save_path):
        """
        Save test positive items dict, in form {user: [positively rated items in test dataset]}
        """
        if os.path.exists(save_path):
            print("test_pos_item.pickle already exists")
            return
        with open(save_path, "wb") as fp:
            pickle.dump(self.testDict, fp)
        print(f"{world.dataset_name}'s test_pos_item.pickle dict has been saved to {save_path}")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        """
        Generate pre-adjacency matrix
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        # graph is a normalized adjacency matrix with all values lie within the range [0, 1]
        return self.Graph

    def __build_test(self):
        """
        construct test dataset

        return:
            dict: {user: [pos rated items in test dataset]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        """
        Get all *training* item IDs positively rated by users

        :return: each user's list of positive *training* item IDs
        e.g. [[1, 2, 3], [3, 5, 6]] means uid 0 has +ve items 1, 2, 3 and
            uid 1 has +ve items 3, 5, 6
        """
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems


def save_train_mapping(training_samples, save_path: str):
    """
    ** This saves all sampled (user, positive) mappings only, meaning
    not all Train positive items are included.

    Create train mapping and save it as a dict, each entry is of format (uid, pos_id): [neg id]

    :param training_samples: of the form {(uid, pos_id): [list of neg ids]}
    :param save_path: The path where train_sample.pickle is to be saved
    """
    if os.path.exists(save_path):
        return

    with open(save_path, "wb") as fp:
        pickle.dump(training_samples, fp)
    print(f"Training samples saved to {save_path}")


def save_all_positive_items(all_positive_lists: list, save_path: str):
    """
    Save list of all Train positive items.

    :param all_positive_lists: self._allPos
    :param save_path: the path where train_all_pos.pickle is to be saved
    """
    if os.path.exists(save_path):
        return

    with open(save_path, "wb") as fp:
        pickle.dump(all_positive_lists, fp)
    print(f"All training positive items are saved to {save_path}")
