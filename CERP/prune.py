"""
The script for embedding pruning (1st step)
"""
import os
import sys
from utils.util import setup_seed, setup_args, save_opt
from datetime import timedelta
from timeit import default_timer as timer
from utils.train import assign_candidate_p, assign_tensorboard_path
from engine import Engine


parser = setup_args()
parser.add_argument("--sparsity_rates", nargs="+", type=float, default=[],
                    help="The target sparsity rates."
                         "e.g. .1 means save the embeddings when .1 of the total params are nullified")
parser.add_argument("--bucket_size", type=int, required=True,
                    help="The bucket size for Rv and Qv. The two has same bucket size.")


opt = parser.parse_args()
opt = vars(opt)

if type(opt['device_id']) != int and opt["device_id"].isnumeric():
    opt["device_id"] = f"cuda:{opt['device_id']}"

if not opt["sparsity_rates"]:
    print('sparsity_rates must be filled.', file=sys.stderr)
    exit(2)

if opt["prepath"] is None:
    # prepath to save embedding and trained model
    prepath = "./res/"
else:
    prepath = opt["prepath"]

# prepath = path to res folder/dataset_name/model_name/
prepath = os.path.join(prepath, opt["data_type"], opt["model"].upper())
os.makedirs(prepath, exist_ok=True)
opt["prepath"] = prepath

# assign save paths
opt["emb_path"] = os.path.join(prepath, "embedding")
# assign tensorboard path and alias name for tensorboard log
opt = assign_tensorboard_path(opt)

# sort sparsity rates in ascending order
opt['sparsity_rates'].sort()
# assign candidate_p array and max_param
opt = assign_candidate_p(opt)

print(opt, end="\n\n")
setup_seed(seed=opt["seed"])

# save pruning parameter settings
save_opt(prepath=prepath, opt=opt, is_prune=True)


if __name__ == '__main__':
    start_time = timer()
    engine = Engine(opt)
    engine.train()
    end_time = timer()
    print(f"{opt['data_type']} pruning complete.\n"
          f"Time Elapsed = {timedelta(seconds=end_time - start_time)}")
