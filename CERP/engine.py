import os
import timeit
from datetime import timedelta
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import scipy.sparse as sp
from models.factorizer import setup_factorizer
from data_loading.data_loader import setup_generator
from utils.evaluate import evaluate_fm
import torch
from timeit import default_timer as timer

import sys
sys.path.append("..")


class Engine(object):
    """Engine wrapping the training & evaluation
       of adaptive regularized matrix factorization
    """

    def __init__(self, opt):
        self._opt = opt

        if self._opt["data_type"] == "gowalla":
            self._opt["data_path"] = "./data/gowalla"
        else:
            self._opt["data_path"] = "./data/yelp2020"

        self.gamma = self._opt["gamma_init"]
        self.gamma_decay_rate = self._opt["gamma_decay_rate"]

        # use retrain_emb_sparsity to identify whether it is retraining
        self.retrain = 'retrain_emb_sparsity' in opt

        self._sampler = setup_generator(opt)
        self._opt['field_dims'] = self._sampler.field_dims

        self._factorizer = setup_factorizer(opt)

        self._writer = SummaryWriter(log_dir=opt["tensorboard"])
        self._writer.add_text('option', str(opt), 0)
        self._mode = None
        self.early_stop = self._opt.get('early_stop')

        self.prune_max_epoch = self._opt["prune_max_epoch"]

        if self.retrain:
            self.ks = opt.get("ks")
            self.all_items = torch.tensor([_ for _ in range(self._sampler.item_num)]
                                          ).unsqueeze(-1).to(self._opt["device_id"])

            self.ndcg_recall_recorder = open(os.path.join(self._opt["emb_path"], "ndcg_recall_log.txt"), "w")
            text = "epoch\t"
            for metric in ["ndcg", "recall"]:
                for k in self.ks:
                    text += metric + f"_{k}\t"
            print(text, file=self.ndcg_recall_recorder)

        if not self.retrain:
            # not retrain, create a collection to record avg non_zero values for each pruned embedding
            # key: param num, value: [avg_count_R_v, avg_count_Q_v]
            self.embedding_non_zero_vals = dict()
            # key: param num, value: number of epochs used when the embedding is saved
            self.embedding_pruned_epochs = dict()
            # key: param num, value: avg emb size of currently pruned embedding
            self.embedding_avg_emb_size = dict()

            self.pruning_start_time = timer()
        else:
            self.embedding_non_zero_vals = self.pruning_start_time = self.embedding_pruned_epochs = \
                self.embedding_avg_emb_size = None

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        assert new_mode in ['complete', 'partial', None]  # training a complete trajectory or a partial trajectory
        self._mode = new_mode

    def save_initial_embedding(self):
        """
        Save the initial embedding before pruning starts.
        """
        # save initial embeddings
        R_v, Q_v = self._factorizer.model.get_embeddings()
        # R, V saved to emb_path/initial_emb/
        emb_save_path = os.path.join(self._opt['emb_path'], "initial_emb")
        os.makedirs(emb_save_path, exist_ok=True)

        # our dataset, save as sp.csr_matrix
        sp.save_npz(os.path.join(emb_save_path, "R_v.npz"), R_v)
        sp.save_npz(os.path.join(emb_save_path, "Q_v.npz"), Q_v)
        print("*" * 80)
        print("Save the initial embedding tables")
        print("*" * 80)

        # get average non_zero values of initial embedding
        avg_non_zero_R_v, avg_non_zero_Q_v = self._factorizer.model.get_average_non_zero_values()
        init_param = self._factorizer.model.get_total_non_zero_values()
        self.embedding_non_zero_vals[int(init_param)] = [avg_non_zero_R_v, avg_non_zero_Q_v]
        self.embedding_pruned_epochs[int(init_param)] = 0
        self.embedding_avg_emb_size[int(init_param)] = self._factorizer.model.get_entity_avg_emb_size()

    def save_pruned_embedding(self, param, epoch_idx) -> bool:
        """
        :return: whether the pruning is done
        """
        if len(self._opt["candidate_p"]) == 0:
            print("Minimal target parameters achieved, stop pruning.")

            # print embedding avg non_zero values to prepath
            avg_non_zero_val = os.path.join(self._opt["prepath"], "avg_non_zero_val.txt")
            with open(avg_non_zero_val, "w") as fp:
                sentence = f"Average non_zero values for each pruned embedding -- \n" \
                           f"param_num : [avg_count_R_v, avg_count_Q_v]" \
                           f"\n{self.embedding_non_zero_vals}\n\n" \
                           f"number of epochs used when embeddings were saved --\n" \
                           f"param_num: #epochs used\n" \
                           f"{self.embedding_pruned_epochs}\n\n" \
                           f"Average entity embedding size on each pruned embedding --\n" \
                           f"{self.embedding_avg_emb_size}\n\n" \
                           f"Pruning time elapsed = {timedelta(seconds=timer() - self.pruning_start_time)}"
                print(sentence, file=fp)
                print(sentence)

            return True
        else:
            # candidate is sorted in descending order
            max_candidate_p, min_sparsity = self._opt['candidate_p'][0], \
                                            round(self._opt['sparsity_rates'][0], 4)

            if param <= max_candidate_p:
                R_v, Q_v = self._factorizer.model.get_embeddings()
                # R, V saved to emb_path/sparsity_0.xxxx/
                emb_save_path = os.path.join(self._opt['emb_path'],
                                             f"sparsity_{min_sparsity:.4f}")
                os.makedirs(emb_save_path, exist_ok=True)

                # our dataset, save as sp.csr_matrix
                sp.save_npz(os.path.join(emb_save_path, "R_v.npz"), R_v)
                sp.save_npz(os.path.join(emb_save_path, "Q_v.npz"), Q_v)

                # remove the first one from candidate_p and non_sparsity_rates
                self._opt["candidate_p"].pop(0)
                self._opt["sparsity_rates"].pop(0)

                # get average non_zero values
                avg_non_zero_R_v, avg_non_zero_Q_v = self._factorizer.model.get_average_non_zero_values()
                self.embedding_non_zero_vals[param] = [avg_non_zero_R_v, avg_non_zero_Q_v]

                self.embedding_pruned_epochs[param] = int(epoch_idx) + 1

                self.embedding_avg_emb_size[param] = self._factorizer.model.get_entity_avg_emb_size()

                print("*" * 80)
                print(
                    f"Reach the target parameter: {max_candidate_p}, target sparsity: {min_sparsity:.4f},\n"
                    f"save embedding with size: {param},\n"
                    f"avg non_zero values in R_v: {avg_non_zero_R_v:.4f}, "
                    f"avg non_zero values in Q_v: {avg_non_zero_Q_v:.4f}")
                print("*" * 80)
        return False

    def train_an_episode(self, max_steps, episode_idx=''):
        """Train a feature_based recommendation model"""
        train_start = timeit.default_timer()

        assert self.mode in ['partial', 'complete']
        if self.retrain:
            print(f"Early stop Threshold: {self.early_stop}")

        print('-' * 80)
        print('[{} episode {} starts!]'.format(self.mode, episode_idx))
        print('Initializing ...')
        self._factorizer.init_episode()

        log_interval = self._opt.get('log_interval')
        eval_interval = self._opt.get('eval_interval')
        display_interval = self._opt.get('display_interval')

        status = dict()
        # flag used to determine early stop
        flag, test_flag = 0, 0
        train_mf_loss = np.inf

        if self.retrain:
            # record ndcg and recall value: [val, epoch_idx]
            best_test_result = {metric + f"_{k}": (-np.inf, 0)
                                for metric in ["ndcg", "recall"] for k in self.ks}
        else:
            best_test_result = None

        epoch_mf_loss = epoch_rec_loss = epoch_pruning_loss = 0.
        epoch_start = datetime.now()
        for step_idx in range(int(max_steps)):
            # Prepare status for current step
            status['done'] = False
            status['sampler'] = self._sampler

            if step_idx == 0 and not self.retrain:
                self.save_initial_embedding()

            # get training loss here
            train_mf_loss, R_v_threshold_val, Q_v_threshold_val, pruning_loss, rec_loss =\
                self._factorizer.update(self._sampler, self.gamma)
            epoch_mf_loss += train_mf_loss
            epoch_rec_loss += rec_loss
            epoch_pruning_loss += pruning_loss

            # record step wise avg R_s value
            self._writer.add_scalar("train/step_wise/R_v_s_value", R_v_threshold_val, step_idx)
            self._writer.add_scalar("train/step_wise/Q_v_s_value", Q_v_threshold_val, step_idx)

            status['train_mf_loss'] = train_mf_loss

            # Logging & Evaluate on the Evaluate Set
            if self.mode == 'complete' and step_idx % log_interval == 0:
                # enter here when log interval is due

                # An epoch is when all train batches have been gone through
                epoch_idx = int(step_idx / self._sampler.num_batches_train)
                # get sparsity rate and number of non-zero values in the embedding
                sparsity, params = self._factorizer.model.calc_sparsity()

                if step_idx % display_interval == 0:
                    print('[Epoch {}|Step {}|Flag {}|Sparsity {:.4f}|Params {}|'
                          ' Avg R_s threshold {:.4f}| Avg Q_s threshold {:.4f}| '
                          'pruning loss {:.4f}| RecSys Loss {:.4f} | Total loss {:.4f}]'.
                          format(epoch_idx,
                                 step_idx % self._sampler.num_batches_train,
                                 flag, sparsity, params, R_v_threshold_val, Q_v_threshold_val,
                                 pruning_loss, rec_loss, train_mf_loss))

                if not self.retrain and self.save_pruned_embedding(params, epoch_idx):
                    # record terminating epoch stats
                    R_s, Q_s = self._factorizer.model.get_thresholds()
                    # record g(s) threshold as histogram
                    self._writer.add_histogram('threshold/epoch_wise/R_s', R_s, epoch_idx)
                    self._writer.add_histogram("threshold/epoch_wise/Q_s", Q_s, epoch_idx)
                    # record sparsity rate and number of parameters as scalar
                    self._writer.add_scalar('train/epoch_wise/sparsity', sparsity, epoch_idx)
                    self._writer.add_scalar('train/epoch_wise/params', params, epoch_idx)
                    return

                # record train loss, R_v_Q_v_penalty sparsity @ current step index
                self._writer.add_scalar('train/step_wise/mf_loss', train_mf_loss, step_idx)
                self._writer.add_scalar("train/step_wise/rec_loss", rec_loss, step_idx)
                self._writer.add_scalar("train/step_wise/pruning_loss", pruning_loss, step_idx)
                self._writer.add_scalar('train/step_wise/sparsity', sparsity, step_idx)

                if step_idx % self._sampler.num_batches_train == 0:
                    # enter here when we finish an epoch
                    R_s, Q_s = self._factorizer.model.get_thresholds()

                    # record g(s) threshold as histogram
                    self._writer.add_histogram('threshold/epoch_wise/R_s', R_s, epoch_idx)
                    self._writer.add_histogram("threshold/epoch_wise/Q_s", Q_s, epoch_idx)
                    # record sparsity rate and number of parameters as scalar
                    self._writer.add_scalar('train/epoch_wise/sparsity', sparsity, epoch_idx)
                    self._writer.add_scalar('train/epoch_wise/params', params, epoch_idx)
                    # record epoch losses
                    self._writer.add_scalar("train/epoch_wise/mf_loss", epoch_mf_loss, epoch_idx)
                    self._writer.add_scalar("train/epoch_wise/rec_loss", epoch_rec_loss, epoch_idx)
                    self._writer.add_scalar("train/epoch_wise/pruning_loss", epoch_pruning_loss, epoch_idx)

                    # record epoch gamma
                    self._writer.add_scalar("train/epoch_wise/gamma", self.gamma, epoch_idx)

                    if not self.retrain and step_idx != 0:
                        # update gamma for next epoch
                        new_gamma = self.gamma * self.gamma_decay_rate
                        print(f"gamma changed from {self.gamma} to {new_gamma}")
                        self.gamma = new_gamma

                    epoch_mf_loss = epoch_rec_loss = epoch_pruning_loss = 0.

                    if not self.retrain and epoch_idx >= self.prune_max_epoch:
                        # in pruning mode, current epoch idx >= prune_max_epoch, terminate
                        failed_text = f'Epoch idx: {epoch_idx:d}, current sparsity: {sparsity:.4f}, pruning ' \
                                      f'terminated as max pruning epoch is reached.'
                        # print embedding avg non_zero values to prepath
                        avg_non_zero_val = os.path.join(self._opt["prepath"], "avg_non_zero_val.txt")
                        with open(avg_non_zero_val, "w") as fp:
                            sentence = f"Average non_zero values for each pruned embedding -- \n" \
                                       f"param_num : [avg_count_R_v, avg_count_Q_v]" \
                                       f"\n{self.embedding_non_zero_vals}\n\n" \
                                       f"number of epochs used when embeddings were saved --\n" \
                                       f"param_num: #epochs used\n" \
                                       f"{self.embedding_pruned_epochs}\n\n" \
                                       f"Average entity embedding size on each pruned embedding --\n" \
                                       f"{self.embedding_avg_emb_size}\n\n" \
                                       f"Pruning time elapsed = {timedelta(seconds=timer() - self.pruning_start_time)}\n\n"
                            sentence += failed_text
                            print(sentence, file=fp)
                            print(sentence)
                            # record terminating epoch stats
                            R_s, Q_s = self._factorizer.model.get_thresholds()
                            # record g(s) threshold as histogram
                            self._writer.add_histogram('threshold/epoch_wise/R_s', R_s, epoch_idx)
                            self._writer.add_histogram("threshold/epoch_wise/Q_s", Q_s, epoch_idx)
                            # record sparsity rate and number of parameters as scalar
                            self._writer.add_scalar('train/epoch_wise/sparsity', sparsity, epoch_idx)
                            self._writer.add_scalar('train/epoch_wise/params', params, epoch_idx)
                            return

                if (step_idx % self._sampler.num_batches_train == 0) and (
                        epoch_idx % eval_interval == 0) and self.retrain:
                    # enter here when we finish an epoch and current epoch idx is in eval_interval
                    # and in retrain mode

                    # evaluate on test results (modify this when changing to retrain phase)
                    print('Evaluate on test ...')
                    print("*" * 80)
                    start = datetime.now()
                    # our dataset, evaluate ndcg@k and recall@k
                    ndcgs, recalls = evaluate_fm(self._factorizer, self._sampler,
                                                 ks=self.ks, all_items=self.all_items,
                                                 device_id=self._opt["device_id"])

                    # write to ndcg_recall_recorder log
                    text = f"{epoch_idx}\t"
                    for metric in ["ndcgs", "recalls"]:
                        for idx in range(len(self.ks)):
                            text += f"{locals()[metric][idx]:>.8f}\t"
                    print(f"\nrecalls: {recalls}, ndcgs: {ndcgs}", file=self.ndcg_recall_recorder, flush=True)
                    print(text, file=self.ndcg_recall_recorder, end="\n\n", flush=True)

                    # update best results so far
                    text = f"Test result: "
                    for idx in range(len(self.ks)):
                        k_val = self.ks[idx]
                        # write to tensor board
                        self._writer.add_scalar(f'test/epoch_wise/ndcg_{k_val}',
                                                ndcgs[idx], epoch_idx)
                        self._writer.add_scalar(f'test/epoch_wise/recall_{k_val}',
                                                recalls[idx], epoch_idx)

                        for metric in ["ndcg", "recall"]:
                            val = locals()[metric + "s"][idx]

                            text += f"| {metric}@{k_val}: {val:8f} |"

                            # use ndcg@5 as condition for early stopping
                            if metric == "ndcg" and k_val == 5:
                                if val > best_test_result[metric + f"_{k_val}"][0]:
                                    # performance improved, reset flag
                                    print(f"{metric}@{k_val} improved, test_flag reset, "
                                          f"previous test flag was {test_flag}", end=", ")
                                    test_flag = 0
                                else:
                                    print(f"{metric}@{k_val} unchanged, test_flag + 1,"
                                          f" previous test_flag was {test_flag}", end=", ")
                                    test_flag += 1
                                print(f"updated test flag is {test_flag}")

                            if val > best_test_result[metric + f"_{k_val}"][0]:
                                best_test_result[metric + f"_{k_val}"] = val, epoch_idx
                    print(text)

                    end = datetime.now()
                    print('Evaluate Time {} minutes'.format((end - start).total_seconds() / 60))
                    epoch_end = datetime.now()
                    dur = (epoch_end - epoch_start).total_seconds() / 60
                    epoch_start = datetime.now()
                    print('[Epoch {:4d}] train MF loss: {:04.8f}, train rec loss: {:04.8f} '
                          'epoch elapsed time {:04.8f} minutes'.
                          format(epoch_idx, train_mf_loss, rec_loss, dur))
                    print("*" * 80)

            flag = test_flag
            if self.early_stop is not None and flag >= self.early_stop:
                print("Early stop training process")
                print("Best performance on test data: ", best_test_result)

                print("Early stop training process", file=self.ndcg_recall_recorder)
                print("Best performance on test data: ", best_test_result,
                      file=self.ndcg_recall_recorder)

                self._writer.add_text('best_test_result', str(best_test_result), 0)
                break
        train_end = timeit.default_timer()
        text = f"Total elapsed time: {timedelta(seconds=train_end - train_start)}"
        print(text)
        print(text, file=self.ndcg_recall_recorder)
        if self.retrain:
            self.ndcg_recall_recorder.close()

    def train(self):
        self.mode = 'complete'
        self.train_an_episode(self._opt['max_steps'])
