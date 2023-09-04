'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from utils import timer
import model
import multiprocessing
import os


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, w=None):
    """
    The train runner of the program
    """
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    if world.config.get("train_dataset") is None or world.config["resample_train_sample_each_epoch"]:
        # train dataset not created, now create it and save
        with timer(name="Sample"):
            S, training_samples = utils.UniformSample_original(dataset)

        # save train samples i.e. sampled {(train, +ve): [5 -ve items]}
        train_sample_save_path = os.path.join(world.RES_PATH, "train_samples.pickle")
        dataloader.save_train_mapping(training_samples=training_samples,
                                      save_path=train_sample_save_path)

        # users, posItems and negItems are of same length
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()

        users = users.to(world.device)
        posItems = posItems.to(world.device)
        negItems = negItems.to(world.device)
        users, posItems, negItems = utils.shuffle(users, posItems, negItems)

        # record to config["train_dataset"]
        world.config["train_dataset"] = {
            "users": users.detach().clone(),
            "posItems": posItems.detach().clone(),
            "negItems": negItems.detach().clone()
        }
    else:
        # get from config["train_dataset"]
        data = world.config["train_dataset"]
        users, posItems, negItems = data["users"].detach().clone(),\
                                    data["posItems"].detach().clone(),\
                                    data["negItems"].detach().clone()

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    """
    :param X: [(top k rating list, groundTrue_list)] for all users in current batch
    """
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    # r: Test data ranking & topk ranking
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # all positive rated items by batch_users, list of lists of item IDs
            allTrainPos = dataset.getUserPosItems(batch_users)

            # groundTrue: the item IDs positively rated in Test set
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            # dim: batch_user x n_items in the dataset
            rating = Recmodel.getUsersRating(batch_users_gpu)

            # exclude positively rated items in training step
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allTrainPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            # change rating of excluded items to -1024, so never appear in topk result
            rating[exclude_index, exclude_items] = -(1<<10)
            # rating_k: max_k of items IDs that has the highest ranking
            _, rating_K = torch.topk(rating, k=max_K)
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results["recall"], results["precision"], results["ndcg"] =\
            results["recall"].tolist(), results["precision"].tolist(), results['ndcg'].tolist()
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        del results["precision"]

        text = f"[Epoch {epoch + 1}] | [{results}]"
        print(text)
        print(text, file=world.config["output_text"], flush=True)
        return results
