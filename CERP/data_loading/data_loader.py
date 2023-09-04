import math
import pickle
import os
import torch
from torch.utils.data import DataLoader
from data_loading.data_record import TrainData, TrainPosRecord
import numpy as np
import scipy.sparse as sp
import random


generator = torch.Generator()
generator.manual_seed(3407)


def setup_generator(opt):
    """Choose different type of sampler for MF & FM"""
    return FMGenerator(opt)


def import_interaction_matrix(data_path):
    """
    In retrain mode, need to import interaction matrix
    """
    interact_mat_path = os.path.join(data_path, "interact_mat.npz")
    if not os.path.exists(interact_mat_path):
        print("No interaction matrix found")
        exit(1)
    else:
        interact_mat = sp.load_npz(interact_mat_path).astype(int).tocoo()
        indices = torch.LongTensor(np.vstack((interact_mat.row, interact_mat.col)))
        vals = torch.FloatTensor(interact_mat.data)
        interact_mat = torch.sparse.LongTensor(indices, vals, torch.Size(interact_mat.shape))
        try:
            print(f"number of non-zeros in the interaction matrix = {torch.count_nonzero(interact_mat)}")
        except NotImplementedError:
            pass
    return interact_mat


def dataloader_seed_worker(worker_id):
    """
    Seed worker to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FMGenerator(object):
    def __init__(self, opt):
        data_path = opt['data_path']
        self.batch_size_train = opt.get('batch_size_train')
        self.batch_size_valid = opt.get('batch_size_valid')
        self.batch_size_test = opt.get('batch_size_test')

        self.opt = opt

        self.retrain = False
        if 'retrain_emb_sparsity' in opt:
            # check if we are in retrain mode
            self.retrain = True

        self.batch_size_train_factor = opt.get("batch_size_train_factor")

        # import interact_mat
        self.interact_mat = import_interaction_matrix(data_path).to(opt["device_id"])
        self.user_num, self.item_num = self.interact_mat.shape

        # dataset loading
        # load {(u, +ve): [5 -ve]} samples
        try:
            with open(os.path.join(data_path, "train_sample.pickle"), "rb") as fp:
                train_sample = pickle.load(fp)
        except FileNotFoundError:
            with open(os.path.join(data_path, "train_samples.pickle"), "rb") as fp:
                train_sample = pickle.load(fp)

        # load all lists of train positive items for all users
        with open(os.path.join(data_path, "train_all_pos.pickle"), "rb") as fp:
            train_all_pos = pickle.load(fp)

        # GCN dataset, dict file in form {uid: [pos rated items in test dataset]}
        # with open(os.path.join(data_path, "test_pos_item.pickle"), "rb") as fp:
        #     test_pos_mapping = pickle.load(fp)
        self.train_data = TrainData(train_sample, {})
        self.test_data = TrainPosRecord(train_mapping=train_all_pos)

        # u: set of items positively rated in train mapping
        self.train_positive_items = self.test_data.train_positive_items

        assert len(self.train_data) % 6 == 0, "number of training instances is not a multiple of 6"
        # the size of each training batch is the multiple of 6
        self.batch_size_train = self.batch_size_train_factor * 6
        # field dimensions takes the shape of rating matrix
        self.field_dims = import_interaction_matrix(data_path).shape

        print('\tNum of fields: {}'.format(len(self.field_dims)))
        print('\tNum of feature values: {}'.format(sum(self.field_dims)))
        # adjust the batch sizes wrt actual data size
        self.num_batches_train = math.ceil(len(self.train_data) / self.batch_size_train)
        self.num_batches_test = math.ceil(len(self.test_data) / self.batch_size_test)

        self._train_epoch = iter([])
        self._valid_epoch = iter([])
        self._test_epoch = iter([])

        print('\tNum of train records: {}'.format(len(self.train_data)))
        print('\tNum of test records: {}'.format(len(self.test_data)))

    @property
    def train_epoch(self):
        """list of training batches"""
        return self._train_epoch

    @train_epoch.setter
    def train_epoch(self, new_epoch):
        self._train_epoch = new_epoch

    @property
    def valid_epoch(self):
        """list of validation batches"""
        return self._valid_epoch

    @valid_epoch.setter
    def valid_epoch(self, new_epoch):
        self._valid_epoch = new_epoch

    @property
    def test_epoch(self):
        """list of test batches"""
        return self._test_epoch

    @test_epoch.setter
    def test_epoch(self, new_epoch):
        self._test_epoch = new_epoch

    def get_epoch(self, epoch_type):
        """
        return:
            list, an epoch of batch field samples of type=['train', 'valid', 'test']
        """
        if epoch_type == 'train':
            return self.train_epoch

        if epoch_type == 'valid':
            return self.valid_epoch

        if epoch_type == 'test':
            return self.test_epoch

    def get_sample(self, sample_type):
        """get training sample or validation sample"""
        epoch = self.get_epoch(sample_type)

        try:
            sample = next(epoch)
        except StopIteration:
            self.set_epoch(sample_type)
            epoch = self.get_epoch(sample_type)
            sample = next(epoch)
            if self.opt['load_in_queue']:
                # continue to queue
                self.cont_queue(sample_type)

        return sample

    def set_epoch(self, data_type):
        """setup batches of type = [training, validation, testing]"""
        cpu_count = self.opt["cpu_count"]
        if data_type == 'train':
            loader = DataLoader(self.train_data, batch_size=self.batch_size_train,
                                num_workers=cpu_count,
                                worker_init_fn=dataloader_seed_worker,
                                generator=generator)
            self.train_epoch = iter(loader)
        elif data_type == 'test':
            loader = DataLoader(self.test_data, batch_size=self.batch_size_test,
                                num_workers=cpu_count, shuffle=False,
                                worker_init_fn=dataloader_seed_worker,
                                generator=generator)
            self.test_epoch = iter(loader)
