'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing
import sys


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

dataset_name = args.dataset_name

config = vars(args)
all_dataset = ['gowalla', "yelp2020"]
all_models = ['lgn']

config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config["device_id"] = args.device_id
config["neg_size"] = args.neg_size
# the dict that records all users, pos_items, neg_items tensors, when first training
# is run, it will be initialized
config["train_dataset"] = None


ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')

if config["path_to_load_CERP_pruned_embs_for_retraining"]:
    assert config["retrain_sparsity"], "retrain sparsity not specified"
    assert config["CERP_embedding_bucket_size"], "CERP embedding bucket size not specified"
    retrain_sparsity, bucket_size = config["retrain_sparsity"], config["CERP_embedding_bucket_size"]
    print(f"Using CERP pruned embs for retraining, sparsity: {retrain_sparsity:.4f}, bucket size: {bucket_size:d}")
    RES_PATH = os.path.join(config["path_to_load_CERP_pruned_embs_for_retraining"], f"sparsity_{retrain_sparsity:.4f}")
else:
    if not config["save_res_prepath"]:
        RES_PATH = join(ROOT_PATH, "res")
    else:
        os.makedirs(config["save_res_prepath"], exist_ok=True)
        RES_PATH = config["save_res_prepath"]

    save_folder_name = f"{dataset_name}_latent_dim_{config['latent_dim_rec']}"
    if config["CERP_embedding_bucket_size"]:
        save_folder_name += f"_bucket_size_{config['CERP_embedding_bucket_size']}"
    RES_PATH = join(RES_PATH, save_folder_name)

print(f"res will be saved to {RES_PATH}")

EMB_SAVE_PATH = join(RES_PATH)
BOARD_PATH = join(RES_PATH, 'runs')
FILE_PATH = join(RES_PATH, 'checkpoints')
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

# save config params
with open(join(RES_PATH, "hyperparams.txt"), "w") as fp:
    for k, v in config.items():
        print(f"{k}:    {v}", file=fp)

# text file that records all text performance
output_text = open(os.path.join(RES_PATH, "LightGCNres.txt"), "w")
print(f'k values: {args.topks}', file=output_text)
config["output_text"] = output_text

GPU = torch.cuda.is_available()
device = torch.device(config['device_id'] if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

model_name = args.model
if dataset_name not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset_name} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
