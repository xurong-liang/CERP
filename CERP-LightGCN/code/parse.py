'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument("--optimizer_weight_decay", type=float, default=0.,
                        help="The weight decay (l2 regularization) on Optimizer.")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalization")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset_name', type=str, default='gowalla',
                        choices=["gowalla", "yelp2020"],
                        help="available datasets: [gowalla, yelp2020]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[5, 10, 20, 50]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument("--device_id", type=int, default=0, help="The CUDA device to be used")
    parser.add_argument("--neg_size", type=int, default=5,
                        help="The number of negative samples for each (user, pos_id)")
    parser.add_argument("--save_res_prepath", type=str, default=None,
                        help="The prepath for result saving.")
    parser.add_argument("--CERP_embedding_bucket_size", type=int, default=0,
                        help="The bucket size for CERP compositional embeddings."
                             "If bucket size == 0, compositional embeddings will "
                             "not be deployed.")
    parser.add_argument("--path_to_load_CERP_pruned_embs_for_retraining", type=str,
                        default=None, help="The path specified to import pruned Qv, Rv. If"
                                           "specified, the result texts will be saved to that path as well.")
    parser.add_argument("--retrain_sparsity", type=float, default=0,
                        help="The sparsity of embs to be retrained")
    parser.add_argument("--l2_penalty", type=float, default=.5,
                        help="The l2 penalty factor for LightGCN rating calculation")
    parser.add_argument("--early_stop", type=int, default=-1, help="The early stop threshold.")
    parser.add_argument("--resample_train_sample_each_epoch", type=lambda x: x.lower() == "true", default=False,
                        help="whether to resample training samples at start of each epoch.")
    return parser.parse_args()
