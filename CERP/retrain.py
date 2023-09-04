"""
The retraining script (2nd step) -- only applicable to MLP retraining.

For LightGCN retrain, please use CERP_LightGCN. For detail usage, refer to readme.md
"""
import os
from utils.util import setup_seed, setup_args, save_opt
from datetime import timedelta
from timeit import default_timer as timer
from utils.train import assign_tensorboard_path
from engine import Engine


parser = setup_args()
parser.add_argument("--bucket_size", type=int, required=True,
                    help="The number of buckets used in Qv, Rv. The two has same bucket size.")
parser.add_argument("--retrain_emb_sparsity", required=True, type=float,
                    help="The embeddings of a particular sparsity to be retrained.")

opt = parser.parse_args()
opt = vars(opt)


# use LightGCN original code for retraining
if opt["model"] == "gcn":
    print('Please use original LightGCN code for retraining')
    raise NotImplementedError

if type(opt['device_id']) != int and opt["device_id"].isnumeric():
    opt["device_id"] = f"cuda:{opt['device_id']}"

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
opt["emb_path"] = os.path.join(prepath, "embedding", f'sparsity_{opt["retrain_emb_sparsity"]:.4f}')
# assign tensorboard path and alias name for tensorboard log
opt = assign_tensorboard_path(opt)

print(opt, end="\n\n")
setup_seed(seed=opt["seed"])

# save pruning parameter settings
save_opt(prepath=prepath, opt=opt, is_prune=False)


if __name__ == '__main__':
    start_time = timer()
    engine = Engine(opt)
    engine.train()
    end_time = timer()
    print(f"{opt['data_type']} retraining complete.\n"
          f"Time Elapsed = {timedelta(seconds=end_time - start_time)}")
