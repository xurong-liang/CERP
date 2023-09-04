from sklearn import metrics
import torch
import numpy as np


def torch_delete(tensor, indices):
    """
    Given a tensor and a list of index,
    the tensor's position of indices will be deleted
    """
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def evaluate_fm(factorizer, sampler, on='test',
                all_items: list = None, ks: list = [5, 10, 20, 50], device_id=None):
    """
    Evaluation function for fm.

    :return tuple (list of mean NDCG@k values, list of Recall@k values)
            It returns NDCG@k and Recall@k, with k = 5, 10, 20, 50.
    """
    ndcgs, recalls = [], []

    res_dict = dict()
    for metric in ["ndcg", "recall"]:
        for k in ks:
            name = metric + f"_{k}"
            res_dict[name] = []

    with torch.no_grad():
        for i in range(sampler.num_batches_test):
            users = sampler.get_sample(on).to(device_id)
            res_dict = evaluate_users(factorizer, sampler, users, ks, all_items, res_dict, device_id)
    for metric in ["ndcg", "recall"]:
        for k in ks:
            # replace nan cell to 0
            res_dict[metric + f"_{k}"] = np.nan_to_num(res_dict[metric + f"_{k}"], nan=0)
            res = np.mean(res_dict[metric + f"_{k}"])
            locals()[metric + "s"].append(res)
    return ndcgs, recalls


def logloss_and_auc(factorizer, sampler, device_id, on='test'):
    all_logloss, all_auc = [], []
    model = factorizer.model
    model.eval()
    for i in range(sampler.num_batches_test):
        data, labels = sampler.get_sample(on)
        print(f"evaluate data shape: {data.shape}")
        print(f"evaluate label shape: {labels.shape}")

        data, labels = data.to(device_id), labels.to(device_id)

        prob_preference = model.forward(data)
        logloss = factorizer.recommendation_criterion(prob_preference, labels.float()) / (data.size()[0])
        all_logloss.append(logloss.detach().cpu().numpy())

        prob_preference = torch.sigmoid(prob_preference).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        auc = metrics.roc_auc_score(labels, prob_preference)
        all_auc.append(auc)

    return np.mean(all_logloss), np.mean(all_auc)


def get_dcg_score(ranked_scores, device_id, method: int = 1):
    """
    Calculate the dcg score of top k items

    :param ranked_scores: The ranked scores
    :param device_id: the gpu device id or "cpu"
    :param method: method 1 or 2 of gain calculation
    :return the calculated dcg scores

    ref: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    if method == 2:
        gain = 2 ** ranked_scores - 1
    else:
        # method 1
        gain = ranked_scores
    discounts = torch.log2(torch.arange(2, len(ranked_scores) + 2, device=device_id))
    return torch.sum(gain / discounts)


def evaluate_users(factorizer, sampler, users, ks: list, all_items, res_dict: dict, device_id):
    """
    Iterate through each user in users and calculate and append ndcg and recall values

    Results all appended to ndcg_k and recall_k lists
    """
    model = factorizer.model
    model.eval()
    for u in users:
        # for each user, check if there exists positively rated items in the training set
        # if so, we eliminate those from our evaluation step
        items_to_eliminate = sampler.train_positive_items.get(int(u))
        if type(items_to_eliminate) == set:
            items_to_eliminate = list(items_to_eliminate)
        its = all_items.detach().clone().to(device_id)

        ground_truth = sampler.interact_mat[u].to_dense().view(-1).to(device_id)
        ground_truth[items_to_eliminate] = 0

        if torch.count_nonzero(ground_truth) == 0:
            print(f"User {u.item()} does not have Test positively rated items.")
            continue

        # make user's shape identical to items before passing them to model
        u = torch.full(its.shape, u, device=device_id)

        data = torch.cat((u, its), dim=1)
        scores = model.evaluation_forward(data)
        scores[items_to_eliminate] = -float("inf")
        _, top_k_indices = torch.topk(scores, max(ks))

        top_k_indices = top_k_indices.to(device_id)

        tp_fn = torch.sum(ground_truth)
        for k in ks:
            # recall calculation
            pred = torch.full(ground_truth.shape, 0, device=device_id)
            pred[top_k_indices[:k]] = 1
            tp = torch.sum(ground_truth * pred)
            recall_val = tp / tp_fn
            res_dict[f"recall_{k}"].append(float(recall_val))
            if torch.isnan(recall_val):
                assert torch.count_nonzero(
                    ground_truth) == 0, "Despite nan but not all ground truth has 0 values"

            # NDCG calculation
            dcg = get_dcg_score(ranked_scores=ground_truth[top_k_indices[:k]], device_id=device_id)

            ideal_scores = torch.zeros(k, device=device_id)
            ideal_scores[:min([k, torch.count_nonzero(ground_truth).item()])] = 1
            idcg = get_dcg_score(ranked_scores=ideal_scores,
                                 device_id=device_id)
            idcg[idcg == 0] = 1

            ndcg_val = dcg / idcg
            ndcg_val[torch.isnan(ndcg_val)] = 0
            res_dict[f"ndcg_{k}"].append(float(ndcg_val))
    return res_dict
