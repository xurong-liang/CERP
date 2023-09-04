"""
Defining the DataSet class for datasets from LightGCN.

"""
import torch.utils.data as data
import pandas as pd


class TrainData(data.Dataset):
    """
    The class representation of the dataset to be passed to OffsetLearner in
    training phase

    We convert (user_id, pos_id, 5 neg_ids) to 6 (user_id, neg_id) pairs
    """

    def __init__(self, train_mapping: dict, test_mapping: dict):
        """
        param sample_mapping: the mapping of (u, i) -> [5 j's for u], i is rated, j is unrated
        """
        self.data, self.labels = self.get_dataset(train_mapping)
        self.field_dims = self.get_all_field_dims(train_mapping, test_mapping)

    @staticmethod
    def get_dataset(sample_mapping: dict):
        """
        Return: features: a row is (uid, iid), label: a row is 0/1
        """
        # a label list for each user is 1 pos and 5 negs
        origin_labels = [1, 0, 0, 0, 0, 0]
        users, ids, labels = [], [], []
        for key, neg_id_list in sample_mapping.items():
            # u: user index
            # pos: +ve rated item index
            # neg_ids: for each user, append indices of 5 -ve rated items
            u, pos = key
            users += [u] * 6
            ids.append(pos)
            for neg in neg_id_list:
                ids.append(neg)
            labels += origin_labels

        tbl = pd.DataFrame.from_dict({'user': users, "item": ids, "label": labels})
        features, labels = tbl[["user", "item"]], tbl[["label"]]
        return features.values, labels.values

    @staticmethod
    def get_all_field_dims(train_mapping: dict, test_mapping: dict):
        """
        Get sum of number of user ids and item ids in the train and test sample mapping
        """
        users, items = set(), set()

        for mapping in ["train_mapping", "test_mapping"]:
            mapping = locals()[mapping]
            for key, neg_id_list in mapping.items():
                u, pos = key
                users.add(u)
                items.add(pos)
                for item in neg_id_list:
                    items.add(item)
        return [len(users), len(items)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TrainPosRecord(data.Dataset):
    """
    The class representation of positively rated items for each user in the
    TrainData.

    Each row is a user with all items in the entire dataset
       (+ve items in train loader eliminated later)
    """

    def __init__(self, train_mapping):
        # get all test users from test_pos_neg_mapping
        # train mapping is a list that contains lists of +ve items in training
        assert type(train_mapping) == list
        self.users = list(range(len(train_mapping)))

        self.train_positive_items = dict()
        for u in self.users:
            self.train_positive_items[u] = train_mapping[u]

    @staticmethod
    def get_train_positive_items(train_mapping: dict):
        """
        For user that appeared in train_mapping, find the set of items that are
        positively rated
        """
        train_positive_items = dict()
        for u, i in train_mapping.keys():
            if train_positive_items.get(u):
                train_positive_items[u].add(i)
            else:
                new_set = set()
                new_set.add(i)
                train_positive_items[u] = new_set
        return train_positive_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx]

