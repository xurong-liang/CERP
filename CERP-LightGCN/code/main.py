import world
import utils
from world import cprint
import torch
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

print(f"Use CUDA is {torch.cuda.is_available()}; use multiprocess is {world.config['multicore'] == 1}")

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    early_stop_count = None if world.config["early_stop"] == -1 else 0

    best_ndcg_5 = best_ndcg_epoch = None
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        # evaluate here
        if epoch % 10 == 0:
            # evaluate every 10 epochs
            cprint("[TEST]")
            result_dict = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

            current_ndcg_5 = result_dict["ndcg"][0]
            if best_ndcg_5 is None or current_ndcg_5 > best_ndcg_5:
                # reset early stop
                if early_stop_count is not None:
                    early_stop_count = 0

                # save model state_dict
                torch.save(Recmodel.state_dict(), weight_file)
                # save embedding
                # one single embedding table
                users_emb, items_emb = Recmodel.computer()
                all_embs = torch.cat([users_emb, items_emb])
                torch.save(all_embs, join(world.EMB_SAVE_PATH, "all_embs.pt"))
                if world.config["CERP_embedding_bucket_size"]:
                    if world.config["retrain_sparsity"]:
                        R_v_name, Q_v_name = 'R_v_retrained.pt', "Q_v_retrained.pt"
                    else:
                        R_v_name, Q_v_name = "R_v.pt", "Q_v.pt"

                    # Q_v and R_v
                    R_v, Q_v = Recmodel.R_v.weight, Recmodel.Q_v.weight
                    if world.config["retrain_sparsity"]:
                        R_v = R_v * Recmodel.R_mask
                        Q_v = Q_v * Recmodel.Q_mask

                    torch.save(R_v, join(world.EMB_SAVE_PATH, R_v_name))
                    torch.save(Q_v, join(world.EMB_SAVE_PATH, Q_v_name))

                print(f"Epoch {epoch + 1}: best ndcg@5 change from {best_ndcg_5} to {current_ndcg_5}. Model saved.")
                best_ndcg_5, best_ndcg_epoch = current_ndcg_5, epoch + 1
            else:
                if early_stop_count is not None:
                    # increment early stop
                    early_stop_count += 1
            print(f"early stop flag: {early_stop_count}/{world.config['early_stop']}")

        if early_stop_count is not None and early_stop_count == world.config['early_stop']:
            print(f"early stop threshold {early_stop_count} reached. Now exit.")
            break
        # train here
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
    text = f"Best ndcg@5 at Epoch {best_ndcg_epoch}."
    print(text, file=world.config["output_text"], flush=True)
finally:
    if world.tensorboard:
        w.close()
    world.config["output_text"].close()
