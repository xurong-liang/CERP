import torch
from torch.optim.lr_scheduler import ExponentialLR
from copy import deepcopy
from models.modules import LR, FM, MLP, LightGCN
from utils.train import use_optimizer, get_grad_norm


def setup_factorizer(opt):
    new_opt = deepcopy(opt)
    for k, v in opt.items():
        if k.startswith('fm_'):
            new_opt[k[3:]] = v
    return FMFactorizer(new_opt)


def bpr_loss(y_i, y_j):
    """
    param y_i: The predicted label of the (u, i) pair
    param y_j: The 5 predicted labels of the (u, j) paris

    Reference: https://github.com/guoyang9/BPR-pytorch/blob/master/main.py
    """
    return - (y_i - y_j).sigmoid().log().sum()


class Factorizer(object):
    def __init__(self, opt):
        self.opt = opt
        self.clip = opt.get('grad_clip')
        self.batch_size_test = opt.get('batch_size_test')
        self.l2_penalty = opt['l2_penalty']
        self.batch_size_train_factor = opt.get("batch_size_train_factor")

        # use BPR loss for recommendation
        self.recommendation_criterion = bpr_loss

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.param_grad = None
        self.optim_status = None

        self.prev_param = None
        self.param = None

        self._train_step_idx = None
        self._train_episode_idx = None

    @property
    def train_step_idx(self):
        return self._train_step_idx

    @train_step_idx.setter
    def train_step_idx(self, new_step_idx):
        self._train_step_idx = new_step_idx

    @property
    def train_episode_idx(self):
        return self._train_episode_idx

    @train_episode_idx.setter
    def train_episode_idx(self, new_episode_idx):
        self._train_episode_idx = new_episode_idx

    def get_grad_norm(self):
        assert hasattr(self, 'model')
        return get_grad_norm(self.model)

    def get_emb_dims(self):
        return self.model.get_emb_dims()

    def update(self, sampler, gamma):
        if (self.train_step_idx > 0) and (self.train_step_idx % sampler.num_batches_train == 0):
            self.scheduler.step()
        self.train_step_idx += 1
        self.model.train()
        self.optimizer.zero_grad()


class FMFactorizer(Factorizer):
    def __init__(self, opt):
        super(FMFactorizer, self).__init__(opt)
        self.opt = opt

    def init_episode(self):
        opt = self.opt
        if opt['model'] == 'linear':
            self.model = LR(opt)
        elif opt['model'] == 'fm':
            self.model = FM(opt)
        elif opt["model"] == "mlp":
            self.model = MLP(opt)
        elif opt["model"] == "gcn":
            self.model = LightGCN(opt)
        else:
            raise ValueError("Invalid recommendation model type: {}".format(opt['model']))

        self.model.to(opt["device_id"])

        self._train_step_idx = 0
        self.optimizer = use_optimizer(self.model, opt)
        self.scheduler = ExponentialLR(self.optimizer, gamma=opt['lr_exp_decay'], verbose=True)

    def update(self, sampler, gamma):
        """
        update FM model parameters

        :param gamma: the pruning loss control factor,
        """
        super(FMFactorizer, self).update(sampler, gamma)
        data, labels = sampler.get_sample('train')
        data, labels = data.to(self.opt["device_id"]), labels.to(self.opt["device_id"])

        # sparse_Qv, sparse_Rv updated here
        predicted_labels, batch_pruning_loss = self.model.forward(data)
        batch_pruning_loss = gamma * batch_pruning_loss

        # split predicted labels to for each user
        predicted_labels = torch.split(predicted_labels, 6)

        # prediction scores
        c_loss = torch.tensor(0.).to(self.opt["device_id"])
        for user_label in predicted_labels:
            pos_label = user_label[0]
            neg_labels = user_label[1:]

            c_loss += self.recommendation_criterion(y_i=pos_label, y_j=neg_labels)
        non_reg_loss = c_loss / len(predicted_labels)

        l2_loss = self.model.l2_penalty(data, self.l2_penalty) / len(predicted_labels)
        rec_loss = non_reg_loss + l2_loss

        loss = rec_loss + batch_pruning_loss
        loss.backward()

        if self.opt["clip_grad_norm"]:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item(), self.model.embedding.R_s.mean().item(), \
            self.model.embedding.Q_s.mean().item(), batch_pruning_loss.item(), rec_loss.item()
