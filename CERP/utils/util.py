"""
The utility file for runner setup.
"""
import os
import torch
import numpy as np
from argparse import ArgumentParser
import multiprocessing
import pickle
import torch.nn.functional as F


def setup_seed(seed: int = 3407):
    """
    Set up random seed for this run
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def setup_args() -> ArgumentParser:
    """
    Setup argument parser for train andd retrain.

    :return: the initialized parser
    """
    # required ones
    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True, help="The dataset to be inputted",
                        choices={"gowalla", "yelp2020"})
    parser.add_argument("--model", required=True, type=str, choices={'fm', "mlp", "gcn"},
                        help="The factorization model to be used."
                             " Available options: fm, mlp, gcn")
    parser.add_argument("--GCN_pretrain_emb", default=False, type=lambda x: x.lower() == "true",
                        help="Whether to import LightGCN pretrained R_v and Q_v embs")
    parser.add_argument("--gamma_init", default=1., type=float,
                        help="The pruning loss control factor")
    parser.add_argument("--gamma_decay_rate", default=.5, type=float,
                        help="The percentage of gamma to be maintained after each epoch.")
    parser.add_argument('--K', default=100, type=float,
                        help='The temperature scalar to be multiplied with tanh() batch pruning loss.')

    #################################################################################
    # default ones
    parser.add_argument("--batch_size_train_factor", type=int, default=10,
                        help="The exponent that controls the number of users to be passed"
                             "in training batch. if factor = 10, then the total number"
                             "of training samples in each batch is 10 *  = 60 instances,"
                             "equivalent to 10 users' performance")
    parser.add_argument('--latent_dim', type=int, default=128,
                        help="The original vector size for each user/item repr in Qv and Rv.")
    parser.add_argument("--device_id", default=0 if torch.cuda.is_available() else "cpu",
                        help="The device id of cuda")
    parser.add_argument("--Q_v_R_v_concat_method", type=str, default="add", choices={'add', 'concat'},
                        help="How the embedding vectors from Q_v and R_v may combine together.")
    parser.add_argument("--threshold_init", type=float, default=-100,
                        help="The initial value of soft threshold.")
    parser.add_argument("--clip_grad_norm", default=True, type=lambda x: (x.lower() == "true"),
                        help="Whether clip grad norm is enabled for training.")
    parser.add_argument("--threshold_type", type=str, default='element-wise',
                        choices={"global", "element-wise"},
                        help="The threshold type for PEP embedding: {'global', 'element-wise'}")
    parser.add_argument('--threshold_init_method', type=str, choices={"normal", "uniform", "power_law",
                                                                      "xavier_uniform", "all_ones"},
                        default="all_ones",
                        help="Whether the initial threshold values are all ones or"
                             " drawn from Normal/Uniform/Long-tail/Xavier Uniform distribution."
                             "For init methods that have any prob distribution applied,"
                             " their values will be scaled to [0, 1]")
    parser.add_argument("--prepath", default=None, type=str,
                        help='The base dir that the results will be stored')
    parser.add_argument("--fm_lr", type=float, default=1e-3, help="The learning rate of FM optimizer.")
    parser.add_argument("--fm_l2_regularization", type=float, default=1e-5,
                        help="The weight decay in optimizer")
    parser.add_argument("--l2_penalty", type=float, default=0.,
                        help="The l2 penalty loss factor")
    parser.add_argument("--prune_max_epoch", type=int, default=50, help="The maximum number of for pruning.")
    parser.add_argument("--seed", type=int, default=3407, help="The random seed to be used.")

    # default ones
    parser.set_defaults(
        ##########
        ## data ##
        ##########
        load_in_queue=False,
        category_only=False,
        rebuild_cache=False,
        ##########################
        ## Devices & Efficiency ##
        ##########################
        early_stop=3,
        log_interval=1,
        display_interval=100,
        eval_interval=5,  # 10 epochs between 2 evaluations
        cpu_count=int(multiprocessing.cpu_count() * .6),
        batch_size_test=1024,
        max_steps=1e8,
        ###########
        ## Model ##
        ###########
        # use this in MLP and LightGCN
        num_layers=3,
        att_dropout=0.4,
        atten_embed_dim=64,
        # MLP
        dropout=0.0,
        # optimizer setting
        fm_optimizer='adam',
        fm_amsgrad=False,
        fm_eps=1e-8,
        fm_betas=(0.9, 0.999),
        fm_grad_clip=100,  # 0.1
        fm_lr_exp_decay=1,
        #########
        ## PEP ##
        #########
        g_type='sigmoid',
        gk=1,
        re_init=False,
        ks=[5, 10, 20, 50]  # k values for evaluation
    )
    return parser


def save_opt(prepath, opt: dict, is_prune: bool = False):
    """
    Save the opt dict in both txt and pickle form

    If it is pruning, the param will be saved to prepath.
    Otherwise, it will be saved to prepath/embedding/sparsity_0.x/

    :param prepath: the path where the 2 files will be saved
    :param is_prune: whether current setting is for pruning
    :param opt: the opt dict used
    """
    if is_prune:
        file_name = "prune_params"
        text_file_name = os.path.join(prepath, f"{file_name}.txt")
        pickle_file_name = os.path.join(prepath, f"{file_name}.pickle")
    else:
        file_name = "retrain_params"
        text_file_name = os.path.join(opt["emb_path"], f"{file_name}.txt")
        pickle_file_name = os.path.join(opt["emb_path"], f"{file_name}.pickle")

    with open(pickle_file_name, "wb") as fp:
        pickle.dump(opt, fp)

    text = ""
    for k, v in opt.items():
        text += f"{k}: {v}\n"
    with open(text_file_name, "w") as fp:
        print(text, file=fp)


